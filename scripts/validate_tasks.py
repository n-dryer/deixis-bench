"""
Validate the task set against the authoring rules.

Runs four programmatic checks (Checks 1, 2, 3, 6 from the rebuild plan).
The semantic checks (Check 4: human identification, Check 5: semantic leakage)
are LLM-driven and run separately during task authoring.

Usage:
    python scripts/validate_tasks.py [--json]

Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

TASKS_PATH = Path("data/tasks.jsonl")
PROMPT_CONDITIONS_PATH = Path("data/prompt_conditions.json")
LOCKFILE_PATH = Path("data/MANIFEST.lock.json")

# Common-object blocklist for image descriptions. Image descriptions must
# NOT name the object directly. This list is non-exhaustive but catches
# obvious cases.
OBJECT_NAME_BLOCKLIST = {
    # Workshop tools
    "hammer",
    "claw hammer",
    "screwdriver",
    "phillips",
    "drill",
    "wrench",
    "ratchet",
    "pliers",
    "saw",
    "handsaw",
    "jigsaw",
    "circular saw",
    "chisel",
    "plane",
    "sander",
    "router",
    "level",
    "mallet",
    "soldering",
    "soldering iron",
    "tape measure",
    "stud finder",
    "clamp",
    "vise",
    # Kitchen
    "pan",
    "saucepan",
    "skillet",
    "frying pan",
    "pot",
    "wok",
    "ladle",
    "spatula",
    "whisk",
    "tongs",
    "knife",
    "chef's knife",
    "paring knife",
    "cutting board",
    "blender",
    "mixer",
    "kettle",
    "toaster",
    # Cleaning / household
    "broom",
    "mop",
    "vacuum",
    "dustpan",
    "sponge",
    "scrub brush",
    # Art / craft
    "paintbrush",
    "pencil",
    "marker",
    "pen",
    "ruler",
    "scissors",
    "needle",
    "knitting needle",
    "crochet hook",
    "loom",
    "easel",
    "palette",
    # Garden
    "trowel",
    "shovel",
    "spade",
    "rake",
    "hoe",
    "shears",
    "pruners",
    "secateurs",
    "pruning shears",
    "watering can",
    "hose",
    "sprayer",
    # Automotive
    "tire",
    "wheel",
    "jack",
    "lug wrench",
    "dipstick",
    "battery",
    "oil filter",
    "spark plug",
    "air filter",
    # Sports / fitness
    "barbell",
    "dumbbell",
    "kettlebell",
    "yoga mat",
    "jump rope",
    "tennis racket",
    "racquet",
    "bat",
    "club",
    "ski",
    "skis",
    # Electronics / digital (some only)
    "laptop",
    "phone",
    "smartphone",
    "tablet",
    "monitor",
    "keyboard",
    "mouse",
}


def word_match(token: str, text: str) -> bool:
    pattern = r"\b" + re.escape(token.lower()) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _load_tasks_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def check_1_token_leakage(tasks):
    """Check 1: No ``current_answers`` or ``prior_answers`` token appears in
    any user speech field (including the optional deictic repair anchor).

    The named repair anchor (``repair_prompt_named``) is exempt because
    it deliberately names the intended and wrong objects to measure
    floor recoverability.
    """
    fails = []
    for sc in tasks:
        sid = sc["task_id"]
        reference_answers = sc.get("reference_answers") or {}
        speech_fields = [
            ("turn_1_user", sc.get("turn_1_user", "") or ""),
            ("turn_2_user", sc.get("turn_2_user", "") or ""),
        ]
        deictic = sc.get("repair_prompt_deictic")
        if deictic:
            speech_fields.append(("repair_prompt_deictic", deictic))
        for field_name, text in speech_fields:
            for token in reference_answers.get("current_answers", []):
                if word_match(token, text):
                    fails.append(
                        {
                            "task_id": sid,
                            "check": "token_leakage",
                            "detail": f"current_answers token {token!r} appears in {field_name}",
                        }
                    )
            for token in reference_answers.get("prior_answers", []):
                if word_match(token, text):
                    fails.append(
                        {
                            "task_id": sid,
                            "check": "token_leakage",
                            "detail": f"prior_answers token {token!r} appears in {field_name}",
                        }
                    )
    return fails


def check_2_object_name_in_images(tasks):
    """Check 2: No common object name appears in any image description."""
    fails = []
    for sc in tasks:
        sid = sc["task_id"]
        image_fields = [
            ("pre_turn_context_scene_description", sc.get("pre_turn_context_scene_description") or ""),
            ("turn_1_scene_description", sc.get("turn_1_scene_description") or ""),
            ("turn_2_scene_description", sc.get("turn_2_scene_description") or ""),
        ]
        for field_name, text in image_fields:
            if not text:
                continue
            for name in OBJECT_NAME_BLOCKLIST:
                if word_match(name, text):
                    fails.append(
                        {
                            "task_id": sid,
                            "check": "object_name_in_image",
                            "detail": f"object name {name!r} appears in {field_name}",
                        }
                    )
    return fails


def check_3_schema_validation(tasks, enforce_distribution: bool = True):
    """Check 3: Required fields present, types correct, IDs unique,
    distributions match.

    ``enforce_distribution`` toggles the shift_type distribution check.
    The frozen 166-task unified set pins exact counts.
    """
    fails = []
    required_task_fields = {
        "task_id",
        "task_set",
        "gold_label",
        "shift_type",
        "domain",
        "referent_complexity",
        "difficulty",
        "pre_turn_context_scene_description",
        "turn_1_scene_description",
        "turn_1_user",
        "turn_2_scene_description",
        "turn_2_user",
        "repair_prompt_named",
        "reference_answers",
    }
    valid_gold_label = {"current", "prior", "clarify", "abstain"}
    valid_shift_type = {
        "object_in_hand",
        "object_state",
        "sequential_task",
        "location",
        "object_in_view",
        "absent_referent",
        "screen_content",
        "cross_session_reference",
    }
    valid_difficulty = {"easy", "medium", "hard"}
    valid_task_set = {"main"}

    seen_ids = set()
    for sc in tasks:
        sid = sc.get("task_id", "<no-id>")
        # Required fields
        missing = required_task_fields - set(sc.keys())
        if missing:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": f"missing fields: {sorted(missing)}",
                }
            )
        # Unique IDs
        if sid in seen_ids:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "duplicate task_id",
                }
            )
        seen_ids.add(sid)
        # Enum validation
        if sc.get("task_set") not in valid_task_set:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": f"invalid task_set: {sc.get('task_set')!r}",
                }
            )
        if sc.get("gold_label") not in valid_gold_label:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": f"invalid gold_label: {sc.get('gold_label')!r}",
                }
            )
        if sc.get("shift_type") not in valid_shift_type:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": f"invalid shift_type: {sc.get('shift_type')!r}",
                }
            )
        if sc.get("difficulty") not in valid_difficulty:
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": f"invalid difficulty: {sc.get('difficulty')!r}",
                }
            )
        # cross_session_reference must have non-null pre_turn_context_scene_description
        if sc.get("shift_type") == "cross_session_reference" and not sc.get("pre_turn_context_scene_description"):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "cross_session_reference tasks must have pre_turn_context_scene_description populated",
                }
            )
        # turn_1_scene_description and turn_2_scene_description must be populated
        if not sc.get("turn_1_scene_description"):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "turn_1_scene_description must be non-null",
                }
            )
        if not sc.get("turn_2_scene_description"):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "turn_2_scene_description must be non-null",
                }
            )
        # reference answers entry must exist
        reference_answers = sc.get("reference_answers")
        if reference_answers is None or not isinstance(reference_answers, dict):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "missing or invalid `reference_answers` field",
                }
            )
            continue
        # Three-category answer rule for current and prior
        label = sc.get("gold_label")
        if label == "current":
            if not reference_answers.get("current_answers"):
                fails.append(
                    {
                        "task_id": sid,
                        "check": "schema",
                        "detail": "current gold_label but current_answers is empty",
                    }
                )
            if len(reference_answers.get("current_answers", [])) < 3:
                fails.append(
                    {
                        "task_id": sid,
                        "check": "schema",
                        "detail": "current_answers must include 3+ items (object name, technique, state) - fewer than 3 found",
                    }
                )
        if label == "prior":
            if not reference_answers.get("prior_answers"):
                fails.append(
                    {
                        "task_id": sid,
                        "check": "schema",
                        "detail": "prior gold_label but prior_answers is empty",
                    }
                )
            if len(reference_answers.get("prior_answers", [])) < 3:
                fails.append(
                    {
                        "task_id": sid,
                        "check": "schema",
                        "detail": "prior_answers must include 3+ items (object name, technique, state) - fewer than 3 found",
                    }
                )
        if label == "abstain" and not reference_answers.get("abstain_indicators"):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "abstain gold_label but abstain_indicators is empty",
                }
            )
        if label == "clarify" and not reference_answers.get("clarify_indicators"):
            fails.append(
                {
                    "task_id": sid,
                    "check": "schema",
                    "detail": "clarify gold_label but clarify_indicators is empty",
                }
            )

    # Distribution checks (main task_set only).
    if enforce_distribution:
        shift_type_counts = Counter(sc.get("shift_type") for sc in tasks)
        expected_shift_type_counts = {
            "object_in_hand": 21,
            "object_state": 22,
            "sequential_task": 18,
            "location": 22,
            "object_in_view": 21,
            "absent_referent": 21,
            "screen_content": 21,
            "cross_session_reference": 20,
        }
        for shift_type, expected_count in expected_shift_type_counts.items():
            if shift_type_counts[shift_type] != expected_count:
                fails.append(
                    {
                        "task_id": "<main>",
                        "check": "schema",
                        "detail": f"shift_type {shift_type} count {shift_type_counts[shift_type]} does not match expected {expected_count}",
                    }
                )

    return fails


def check_7_lockfile_drift():
    """Check 7: Computed asset hashes match the static MANIFEST.lock.json.

    Catches silent mutations to the task set, prompt conditions, or
    judge-prompt template that ship without a coordinated
    benchmark_version bump. To refresh the lockfile after a deliberate
    content change, run ``python scripts/regen_manifest_lock.py``.
    """
    fails = []
    if not LOCKFILE_PATH.exists():
        return [
            {
                "task_id": "<main>",
                "check": "lockfile",
                "detail": (
                    f"missing lockfile at {LOCKFILE_PATH}; run "
                    "scripts/regen_manifest_lock.py to create it"
                ),
            }
        ]
    try:
        lockfile = json.loads(LOCKFILE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            {
                "task_id": "<main>",
                "check": "lockfile",
                "detail": f"lockfile is not valid JSON: {exc}",
            }
        ]

    # Imports here so the validator runs even when the package import
    # path is not set up (e.g., in a slimmed CI environment that only
    # checks tasks). The hash checks above for assets remain the
    # primary signal even if these imports fail.
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from wearable_assistant_context_bench.aggregation import BENCHMARK_VERSION
        from wearable_assistant_context_bench.llm_judge import JUDGE_SYSTEM_PROMPT
    except ImportError as exc:
        return [
            {
                "task_id": "<main>",
                "check": "lockfile",
                "detail": f"could not import benchmark package for lockfile check: {exc}",
            }
        ]

    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expected = {
        "benchmark_version": BENCHMARK_VERSION,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "tasks_sha256": _sha(TASKS_PATH),
        "prompt_conditions_sha256": _sha(PROMPT_CONDITIONS_PATH),
    }
    for key, value in expected.items():
        if lockfile.get(key) != value:
            fails.append(
                {
                    "task_id": "<main>",
                    "check": "lockfile",
                    "detail": (
                        f"{key} mismatch: lockfile={lockfile.get(key)!r}, "
                        f"computed={value!r}. If this drift is intentional, "
                        "bump benchmark_version and regenerate via "
                        "scripts/regen_manifest_lock.py."
                    ),
                }
            )
    return fails


def check_6_duplication(tasks):
    """Check 6: Cross-task near-duplication on T2 user + T2 image +
    (shift_type, gold_label, difficulty) signature."""
    fails = []
    seen_t2_user: dict[str, str] = {}
    seen_t2_image: dict[str, str] = {}
    seen_signatures: Counter[
        tuple[str | None, str | None, str | None, str | None]
    ] = Counter()

    for sc in tasks:
        sid = sc["task_id"]
        t2u = (sc.get("turn_2_user") or "").strip().lower()
        t2i = (sc.get("turn_2_scene_description") or "").strip().lower()
        sig = (
            sc.get("shift_type"),
            sc.get("gold_label"),
            sc.get("difficulty"),
            sc.get("domain"),
        )

        if t2u and t2u in seen_t2_user:
            fails.append(
                {
                    "task_id": sid,
                    "check": "duplication",
                    "detail": f"identical turn_2_user as {seen_t2_user[t2u]}",
                }
            )
        else:
            seen_t2_user[t2u] = sid

        if t2i and t2i in seen_t2_image:
            fails.append(
                {
                    "task_id": sid,
                    "check": "duplication",
                    "detail": f"identical turn_2_scene_description as {seen_t2_image[t2i]}",
                }
            )
        else:
            seen_t2_image[t2i] = sid

        seen_signatures[sig] += 1

    # Flag signatures with >2 instances (some duplication is fine; many
    # is a coverage problem)
    for sig, count in seen_signatures.items():
        if count > 2:
            fails.append(
                {
                    "task_id": "<main>",
                    "check": "duplication",
                    "detail": f"signature {sig} appears {count} times (limit 2)",
                }
            )

    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    all_records = _load_tasks_jsonl(TASKS_PATH)
    main_task_set = [r for r in all_records if r.get("task_set") == "main"]

    all_fails = []
    all_fails.extend(check_1_token_leakage(main_task_set))
    all_fails.extend(check_2_object_name_in_images(main_task_set))
    all_fails.extend(check_3_schema_validation(main_task_set, enforce_distribution=True))
    all_fails.extend(check_6_duplication(main_task_set))
    all_fails.extend(check_7_lockfile_drift())

    if args.json:
        print(json.dumps(all_fails, indent=2, ensure_ascii=False))
    else:
        if not all_fails:
            print(f"All checks passed ({len(main_task_set)} tasks validated).")
        else:
            print(f"{len(all_fails)} validation failure(s):")
            for f in all_fails:
                print(f"  [{f['check']}] {f['task_id']}: {f['detail']}")

    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
