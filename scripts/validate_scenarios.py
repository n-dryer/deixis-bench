"""
Validate the scenario bank against the authoring rules.

Runs four programmatic checks (Checks 1, 2, 3, 6 from the rebuild plan).
The semantic checks (Check 4: human identification, Check 5: semantic leakage)
are LLM-driven and run separately during scenario authoring.

Usage:
    python scripts/validate_scenarios.py [--json]

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

SCENARIOS_PATH = Path("data/scenarios.jsonl")
PROMPT_CONDITIONS_PATH = Path("data/prompt_conditions.json")
LOCKFILE_PATH = Path("data/MANIFEST.lock.json")

# Closed enum for ``activity_domain`` (V1). Single source of truth — tests
# import this set so a domain rename only needs one update. Adding a new
# domain requires editing this list and updating the docs.
VALID_ACTIVITY_DOMAINS: frozenset[str] = frozenset({
    "kitchen",
    "garden",
    "workshop",
    "art_craft",
    "automotive",
    "fitness",
    "household",
    "communication",
    "electronics",
    "office",
    "sports",
    "music",
    "finance",
    "navigation",
})

# ``change_type`` values for which a deictic-only repair anchor cannot
# resolve the reference (V5). The referent isn't visible at Turn 2, so a
# pronoun like "this" doesn't pick it out and the runner falls back to the
# named anchor.
NON_DEICTIC_CHANGE_TYPES: frozenset[str] = frozenset({
    "absent_referent",
    "cross_session_reference",
})

# Floor for ``gold.current_answers`` and ``gold.prior_answers`` lists when
# the scenario's ``target_context`` requires them. Promotes the unwritten
# convention into a hard CI check (V2/V3). Item-count alone doesn't prove
# the three-category rule (object name / technique / state) — that's a
# manual review check — but a floor of 7 makes the convention auditable.
GOLD_LIST_MIN_ITEMS = 7

# Per-``change_type`` bucket floor. Replaces the prior exact-count
# distribution pin that assumed a frozen 50-scenario bank. As the bank
# expands toward 166 scenarios (with ~20 per bucket), this floor will
# rise. Today's minimum is 4 (cross_session_reference).
MIN_CHANGE_TYPE_COUNT = 4

# Common-object blocklist for image descriptions. Image descriptions must
# NOT name the object directly. This list is non-exhaustive but catches
# obvious cases.
OBJECT_NAME_BLOCKLIST = {
    # Workshop tools
    "hammer", "claw hammer", "screwdriver", "phillips", "drill", "wrench",
    "ratchet", "pliers", "saw", "handsaw", "jigsaw", "circular saw",
    "chisel", "plane", "sander", "router", "level", "mallet", "soldering",
    "soldering iron", "tape measure", "stud finder", "clamp", "vise",
    # Kitchen
    "pan", "saucepan", "skillet", "frying pan", "pot", "wok", "ladle",
    "spatula", "whisk", "tongs", "knife", "chef's knife", "paring knife",
    "cutting board", "blender", "mixer", "kettle", "toaster",
    # Cleaning / household
    "broom", "mop", "vacuum", "dustpan", "sponge", "scrub brush",
    # Art / craft
    "paintbrush", "pencil", "marker", "pen", "ruler", "scissors", "needle",
    "knitting needle", "crochet hook", "loom", "easel", "palette",
    # Garden
    "trowel", "shovel", "spade", "rake", "hoe", "shears", "pruners",
    "secateurs", "pruning shears", "watering can", "hose", "sprayer",
    # Automotive
    "tire", "wheel", "jack", "lug wrench", "dipstick", "battery",
    "oil filter", "spark plug", "air filter",
    # Sports / fitness
    "barbell", "dumbbell", "kettlebell", "yoga mat", "jump rope",
    "tennis racket", "racquet", "bat", "club", "ski", "skis",
    "foam roller", "resistance band",
    # Electronics / digital (some only)
    "laptop", "phone", "smartphone", "tablet", "monitor",
    "keyboard", "mouse",
    # Office
    "stapler", "hole punch", "three-hole punch", "paper cutter",
    "letter opener",
    # Measurement / inspection (workshop-adjacent)
    "caliper", "vernier caliper", "digital caliper", "thread gauge",
    "pitch gauge", "feeler gauge", "micrometer", "torque wrench",
    "tire gauge", "multimeter",
}


def word_match(token: str, text: str) -> bool:
    pattern = r"\b" + re.escape(token.lower()) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _load_scenarios_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def check_1_token_leakage(scenarios):
    """Check 1: No ``current_answers`` or ``prior_answers`` token appears in
    any user speech field (including the optional deictic repair anchor).

    The named repair anchor (``turn_3_repair_prompt``) is exempt because
    it deliberately names the intended and wrong objects to measure
    floor recoverability.
    """
    fails = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        gold = sc.get("gold") or {}
        speech_fields = [
            ("turn_1_user", sc.get("turn_1_user", "") or ""),
            ("turn_2_user", sc.get("turn_2_user", "") or ""),
        ]
        deictic = sc.get("turn_3_repair_prompt_deictic")
        if deictic:
            speech_fields.append(("turn_3_repair_prompt_deictic", deictic))
        for field_name, text in speech_fields:
            for token in gold.get("current_answers", []):
                if word_match(token, text):
                    fails.append({
                        "scenario_id": sid,
                        "check": "token_leakage",
                        "detail": f"current_answers token {token!r} appears in {field_name}",
                    })
            for token in gold.get("prior_answers", []):
                if word_match(token, text):
                    fails.append({
                        "scenario_id": sid,
                        "check": "token_leakage",
                        "detail": f"prior_answers token {token!r} appears in {field_name}",
                    })
    return fails


def check_2_object_name_in_images(scenarios):
    """Check 2: No common object name appears in any image description."""
    fails = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        image_fields = [
            ("context_image", sc.get("context_image") or ""),
            ("turn_1_image", sc.get("turn_1_image") or ""),
            ("turn_2_image", sc.get("turn_2_image") or ""),
        ]
        for field_name, text in image_fields:
            if not text:
                continue
            for name in OBJECT_NAME_BLOCKLIST:
                if word_match(name, text):
                    fails.append({
                        "scenario_id": sid,
                        "check": "object_name_in_image",
                        "detail": f"object name {name!r} appears in {field_name}",
                    })
    return fails


def check_3_schema_validation(scenarios, enforce_distribution: bool = True):
    """Check 3: Required fields present, types correct, IDs unique,
    distributions match.

    ``enforce_distribution`` toggles the bank-level cue_type distribution
    check. The frozen 50-scenario bank pins exact counts; the contrast
    pack uses its own distribution and skips this check.
    """
    fails = []
    required_scenario_fields = {
        "scenario_id", "target_context", "change_type", "activity_domain",
        "referent_complexity", "difficulty_tier",
        "context_image", "turn_1_image", "turn_1_user",
        "turn_2_image", "turn_2_user", "turn_3_repair_prompt",
        "gold",
    }
    valid_target_context = {"current", "prior", "clarify", "abstain"}
    valid_change_type = {
        "object_in_hand", "object_state", "sequential_task", "location",
        "object_in_view", "absent_referent", "screen_content",
        "cross_session_reference",
    }
    valid_referent_complexity = {
        "single_referent", "multi_referent", "distractor_present",
        "referent_offscreen",
    }
    valid_difficulty = {"easy", "medium", "hard"}

    seen_ids = set()
    for sc in scenarios:
        sid = sc.get("scenario_id", "<no-id>")
        # Required fields
        missing = required_scenario_fields - set(sc.keys())
        if missing:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": f"missing fields: {sorted(missing)}",
            })
        # Unique IDs
        if sid in seen_ids:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": "duplicate scenario_id",
            })
        seen_ids.add(sid)
        # Enum validation
        if sc.get("target_context") not in valid_target_context:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": f"invalid target_context: {sc.get('target_context')!r}",
            })
        if sc.get("change_type") not in valid_change_type:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": f"invalid change_type: {sc.get('change_type')!r}",
            })
        if sc.get("difficulty_tier") not in valid_difficulty:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": f"invalid difficulty_tier: {sc.get('difficulty_tier')!r}",
            })
        if sc.get("referent_complexity") not in valid_referent_complexity:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": (
                    f"invalid referent_complexity: "
                    f"{sc.get('referent_complexity')!r}; allowed: "
                    f"{sorted(valid_referent_complexity)}"
                ),
            })
        # V1: activity_domain must be in the closed enum.
        if sc.get("activity_domain") not in VALID_ACTIVITY_DOMAINS:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": (
                    f"invalid activity_domain: {sc.get('activity_domain')!r}; "
                    f"allowed: {sorted(VALID_ACTIVITY_DOMAINS)}"
                ),
            })
        # V4: cross_session_reference must have non-null context_image.
        if sc.get("change_type") == "cross_session_reference" and not sc.get("context_image"):
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": "cross_session_reference scenarios must have context_image populated",
            })
        # V5: turn_3_repair_prompt_deictic null/non-null contract.
        # Non-null exactly when target_context=current AND change_type
        # admits a deictic gesture (i.e. the referent is visible at T2).
        deictic_present = bool(sc.get("turn_3_repair_prompt_deictic"))
        deictic_expected = (
            sc.get("target_context") == "current"
            and sc.get("change_type") not in NON_DEICTIC_CHANGE_TYPES
        )
        if deictic_expected and not deictic_present:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": (
                    "turn_3_repair_prompt_deictic must be non-null for "
                    "target_context=current scenarios where the referent "
                    "is visible at Turn 2 "
                    f"(change_type={sc.get('change_type')!r})"
                ),
            })
        if not deictic_expected and deictic_present:
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": (
                    "turn_3_repair_prompt_deictic must be null when the "
                    "referent isn't deictically resolvable "
                    f"(target_context={sc.get('target_context')!r}, "
                    f"change_type={sc.get('change_type')!r})"
                ),
            })
        # turn_1_image and turn_2_image must be populated
        if not sc.get("turn_1_image"):
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": "turn_1_image must be non-null",
            })
        if not sc.get("turn_2_image"):
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": "turn_2_image must be non-null",
            })
        # gold answers entry must exist
        gold = sc.get("gold")
        if gold is None or not isinstance(gold, dict):
            fails.append({
                "scenario_id": sid,
                "check": "schema",
                "detail": "missing or invalid `gold` field",
            })
            continue
        # V2/V3: gold-list floors and the three-category rule.
        # ``current_answers`` and ``prior_answers`` must each include at
        # least ``GOLD_LIST_MIN_ITEMS`` entries spanning the three required
        # vocabulary categories (object name, technique, state).
        # Item-count is auditable here; the three-category split is verified
        # by manual review during authoring.
        target = sc.get("target_context")
        if target == "current":
            n = len(gold.get("current_answers", []))
            if n == 0:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": "current target_context but current_answers is empty",
                })
            elif n < GOLD_LIST_MIN_ITEMS:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": (
                        f"current_answers must include {GOLD_LIST_MIN_ITEMS}+ items "
                        "(object name, technique, state) — "
                        f"only {n} found"
                    ),
                })
        if target == "prior":
            n = len(gold.get("prior_answers", []))
            if n == 0:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": "prior target_context but prior_answers is empty",
                })
            elif n < GOLD_LIST_MIN_ITEMS:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": (
                        f"prior_answers must include {GOLD_LIST_MIN_ITEMS}+ items "
                        "(object name, technique, state) — "
                        f"only {n} found"
                    ),
                })
        if target == "clarify":
            n = len(gold.get("clarify_indicators", []))
            if n == 0:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": "clarify target_context but clarify_indicators is empty",
                })
            elif n < GOLD_LIST_MIN_ITEMS:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": (
                        f"clarify_indicators must include {GOLD_LIST_MIN_ITEMS}+ "
                        f"items — only {n} found"
                    ),
                })
        if target == "abstain":
            n = len(gold.get("abstain_indicators", []))
            if n == 0:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": "abstain target_context but abstain_indicators is empty",
                })
            elif n < GOLD_LIST_MIN_ITEMS:
                fails.append({
                    "scenario_id": sid,
                    "check": "schema",
                    "detail": (
                        f"abstain_indicators must include {GOLD_LIST_MIN_ITEMS}+ "
                        f"items — only {n} found"
                    ),
                })

    # Distribution check: every change_type bucket has at least
    # ``min_change_type_count`` scenarios. Replaces the prior pinned-counts
    # check (which became obsolete when the bank/contrast subset split was
    # retired and the bank started growing). The floor rises as the bank
    # grows toward its 166-scenario target (~20+ per bucket).
    if enforce_distribution:
        cue_counts = Counter(sc.get("change_type") for sc in scenarios)
        min_count = MIN_CHANGE_TYPE_COUNT
        for cue in valid_change_type:
            n = cue_counts.get(cue, 0)
            if n < min_count:
                fails.append({
                    "scenario_id": "<bank>",
                    "check": "schema",
                    "detail": (
                        f"change_type {cue!r} has only {n} scenarios; "
                        f"floor is {min_count}"
                    ),
                })

    return fails


def check_7_lockfile_drift():
    """Check 7: Computed asset hashes match the static MANIFEST.lock.json.

    Catches silent mutations to the scenario bank, prompt conditions, or
    judge-prompt template that ship without a coordinated
    benchmark_version (or judge_prompt_version) bump. To refresh the
    lockfile after a deliberate content change, run
    ``python scripts/regen_manifest_lock.py``.
    """
    fails = []
    if not LOCKFILE_PATH.exists():
        return [{
            "scenario_id": "<bank>",
            "check": "lockfile",
            "detail": (
                f"missing lockfile at {LOCKFILE_PATH}; run "
                "scripts/regen_manifest_lock.py to create it"
            ),
        }]
    try:
        lockfile = json.loads(LOCKFILE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [{
            "scenario_id": "<bank>",
            "check": "lockfile",
            "detail": f"lockfile is not valid JSON: {exc}",
        }]

    # Imports here so the validator runs even when the package import
    # path is not set up (e.g., in a slimmed CI environment that only
    # checks scenarios). The hash checks above for assets remain the
    # primary signal even if these imports fail.
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from wearable_assistant_context_bench.llm_judge import (
            JUDGE_PROMPT_VERSION,
            JUDGE_SYSTEM_PROMPT,
        )
        from wearable_assistant_context_bench.report import BENCHMARK_VERSION, SCHEMA_REVISION
    except ImportError as exc:
        return [{
            "scenario_id": "<bank>",
            "check": "lockfile",
            "detail": f"could not import benchmark package for lockfile check: {exc}",
        }]

    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expected = {
        "benchmark_version": BENCHMARK_VERSION,
        "schema_revision": SCHEMA_REVISION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": hashlib.sha256(
            JUDGE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "scenarios_sha256": _sha(SCENARIOS_PATH),
        "prompt_conditions_sha256": _sha(PROMPT_CONDITIONS_PATH),
    }
    for key, value in expected.items():
        if lockfile.get(key) != value:
            fails.append({
                "scenario_id": "<bank>",
                "check": "lockfile",
                "detail": (
                    f"{key} mismatch: lockfile={lockfile.get(key)!r}, "
                    f"computed={value!r}. If this drift is intentional, "
                    "bump benchmark_version (or judge_prompt_version) "
                    "and regenerate via scripts/regen_manifest_lock.py."
                ),
            })
    return fails


def check_6_duplication(scenarios):
    """Check 6: Cross-scenario near-duplication on T2 user + T2 image +
    (change_type, target_context, difficulty_tier) signature."""
    fails = []
    seen_t2_user = {}
    seen_t2_image = {}
    seen_signatures = Counter()

    for sc in scenarios:
        sid = sc["scenario_id"]
        t2u = (sc.get("turn_2_user") or "").strip().lower()
        t2i = (sc.get("turn_2_image") or "").strip().lower()
        sig = (sc.get("change_type"), sc.get("target_context"), sc.get("difficulty_tier"), sc.get("activity_domain"))

        if t2u and t2u in seen_t2_user:
            fails.append({
                "scenario_id": sid,
                "check": "duplication",
                "detail": f"identical turn_2_user as {seen_t2_user[t2u]}",
            })
        else:
            seen_t2_user[t2u] = sid

        if t2i and t2i in seen_t2_image:
            fails.append({
                "scenario_id": sid,
                "check": "duplication",
                "detail": f"identical turn_2_image as {seen_t2_image[t2i]}",
            })
        else:
            seen_t2_image[t2i] = sid

        seen_signatures[sig] += 1

    # Flag signatures with >2 instances (some duplication is fine; many
    # is a coverage problem)
    for sig, count in seen_signatures.items():
        if count > 2:
            fails.append({
                "scenario_id": "<bank>",
                "check": "duplication",
                "detail": f"signature {sig} appears {count} times (limit 2)",
            })

    return fails


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    all_records = _load_scenarios_jsonl(SCENARIOS_PATH)

    all_fails = []
    all_fails.extend(check_1_token_leakage(all_records))
    all_fails.extend(check_2_object_name_in_images(all_records))
    all_fails.extend(
        check_3_schema_validation(all_records, enforce_distribution=True)
    )
    all_fails.extend(check_6_duplication(all_records))
    all_fails.extend(check_7_lockfile_drift())

    if args.json:
        print(json.dumps(all_fails, indent=2, ensure_ascii=False))
    else:
        if not all_fails:
            print(
                f"All checks passed ({len(all_records)} scenarios validated)."
            )
        else:
            print(f"{len(all_fails)} validation failure(s):")
            for f in all_fails:
                print(f"  [{f['check']}] {f['scenario_id']}: {f['detail']}")

    return 0 if not all_fails else 1


if __name__ == "__main__":
    sys.exit(main())
