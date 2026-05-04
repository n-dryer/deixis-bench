"""Schema and validator tests for the unified scenario file.

These tests treat ``data/scenarios.jsonl`` as the only active
source of truth for scenarios + gold labels. They assert the schema
contract from ``docs/schema.md`` and run the four programmatic
validator checks from ``scripts/validate_scenarios.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCENARIOS_PATH = REPO_ROOT / "data" / "scenarios.jsonl"


REQUIRED_FIELDS = {
    "scenario_id",
    "target_context",
    "change_type",
    "activity_domain",
    "referent_complexity",
    "difficulty_tier",
    "context_image",
    "turn_1_image",
    "turn_1_user",
    "turn_2_image",
    "turn_2_user",
    "turn_3_repair_prompt",
    "gold",
}
OPTIONAL_FIELDS = {"time_gap_bucket", "notes", "pair_id", "turn_3_repair_prompt_deictic"}

ALLOWED_TARGET_CONTEXTS = {"current", "prior", "clarify", "abstain"}
ALLOWED_CUE_TYPES = {
    "object_in_hand",
    "object_state",
    "sequential_task",
    "location",
    "object_in_view",
    "absent_referent",
    "screen_content",
    "cross_session_reference",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_COGNITIVE_LOADS = {
    "single_referent",
    "multi_referent",
    "distractor_present",
    "referent_offscreen",
}
# Closed enum for activity_domain. Source of truth lives next to the validator
# in scripts/validate_scenarios.py (V1 check). Tests load the same set so a
# domain rename only needs one update.
ALLOWED_ACTIVITY_DOMAINS = {
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
}

SCENARIO_ID_PATTERN = re.compile(r"^(sc|adv)-\d{2,}$")


# Make ``scripts/validate_scenarios.py`` importable from tests.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_scenarios  # type: ignore[import-not-found]  # noqa: E402


def _load_records() -> list[dict]:
    out: list[dict] = []
    with SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def all_records() -> list[dict]:
    return _load_records()


# ---------------------------------------------------------------------------
# Schema contract (was tests/test_scenarios_config.py)
# ---------------------------------------------------------------------------


def test_scenarios_jsonl_is_non_empty(all_records: list[dict]) -> None:
    assert isinstance(all_records, list)
    assert len(all_records) > 0


def test_every_scenario_has_required_fields(all_records: list[dict]) -> None:
    for entry in all_records:
        missing = REQUIRED_FIELDS - entry.keys()
        assert not missing, f"scenario {entry.get('scenario_id')!r} missing fields: {missing}"


def test_scenario_ids_are_unique(all_records: list[dict]) -> None:
    ids = [entry["scenario_id"] for entry in all_records]
    assert len(ids) == len(set(ids))


def test_scenario_ids_follow_format(all_records: list[dict]) -> None:
    for entry in all_records:
        sid = entry["scenario_id"]
        assert SCENARIO_ID_PATTERN.match(sid), (
            f"scenario_id {sid!r} does not match `(sc|adv)-NN` format"
        )


def test_target_contexts_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["target_context"] in ALLOWED_TARGET_CONTEXTS, (
            f"{entry['scenario_id']}: unexpected target_context {entry['target_context']!r}"
        )


def test_subset_field_is_absent(all_records: list[dict]) -> None:
    """Post-retirement: the `subset` field should not appear in any row.

    Catches regressions where someone re-introduces a subset split (which
    we deliberately retired in favor of a single unified bank).
    """
    for entry in all_records:
        assert "subset" not in entry, (
            f"{entry['scenario_id']}: `subset` field reappeared "
            "(was retired with the contrast/bank split)"
        )


def test_cue_types_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["change_type"] in ALLOWED_CUE_TYPES, (
            f"{entry['scenario_id']}: unexpected change_type {entry['change_type']!r}"
        )


def test_difficulty_tiers_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["difficulty_tier"] in ALLOWED_DIFFICULTIES, (
            f"{entry['scenario_id']}: unexpected difficulty_tier {entry['difficulty_tier']!r}"
        )


def test_cognitive_loads_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["referent_complexity"] in ALLOWED_COGNITIVE_LOADS, (
            f"{entry['scenario_id']}: unexpected referent_complexity "
            f"{entry['referent_complexity']!r}"
        )


def test_activity_domains_in_allowed_set(all_records: list[dict]) -> None:
    """V1: activity_domain is a closed enum.

    Catches typos and tag drift (e.g. ``home`` vs ``household``) at CI
    time so the long tail of singletons can't grow back.
    """
    for entry in all_records:
        assert entry["activity_domain"] in ALLOWED_ACTIVITY_DOMAINS, (
            f"{entry['scenario_id']}: unexpected activity_domain "
            f"{entry['activity_domain']!r}; allowed: {sorted(ALLOWED_ACTIVITY_DOMAINS)}"
        )


def test_required_string_fields_are_non_empty(all_records: list[dict]) -> None:
    for entry in all_records:
        sid = entry["scenario_id"]
        for field_name in (
            "turn_1_image",
            "turn_1_user",
            "turn_2_image",
            "turn_2_user",
            "turn_3_repair_prompt",
            "activity_domain",
        ):
            value = entry[field_name]
            assert isinstance(value, str) and value, (
                f"{sid}.{field_name}: must be a non-empty string"
            )


def test_context_image_is_string_or_null(all_records: list[dict]) -> None:
    for entry in all_records:
        value = entry["context_image"]
        assert value is None or isinstance(value, str), (
            f"{entry['scenario_id']}.context_image: must be null or str"
        )


def test_cross_session_reference_has_context_image(all_records: list[dict]) -> None:
    """``cross_session_reference`` scenarios must have a non-null context_image."""
    for entry in all_records:
        if entry["change_type"] != "cross_session_reference":
            continue
        sid = entry["scenario_id"]
        assert entry["context_image"], (
            f"{sid}: cross_session_reference scenarios must have a non-null context_image populated"
        )


def test_every_scenario_has_inline_gold(all_records: list[dict]) -> None:
    for entry in all_records:
        sid = entry["scenario_id"]
        gold = entry.get("gold")
        assert isinstance(gold, dict), f"{sid}: gold must be a dict"
        for key in (
            "current_answers",
            "prior_answers",
            "clarify_indicators",
            "abstain_indicators",
        ):
            assert key in gold, f"{sid}: missing gold.{key}"
            assert isinstance(gold[key], list), f"{sid}.gold.{key}: must be a list"


# Floor for ``gold`` answer/indicator lists when ``target_context`` requires
# them. Mirrors ``GOLD_LIST_MIN_ITEMS`` in scripts/validate_scenarios.py.
GOLD_LIST_MIN_ITEMS = 7


def test_current_target_meets_gold_floor(all_records: list[dict]) -> None:
    """V2: target_context=current rows have >= GOLD_LIST_MIN_ITEMS current_answers."""
    for entry in all_records:
        if entry["target_context"] != "current":
            continue
        sid = entry["scenario_id"]
        n = len(entry["gold"]["current_answers"])
        assert n >= GOLD_LIST_MIN_ITEMS, (
            f"{sid}: target_context=current requires "
            f"{GOLD_LIST_MIN_ITEMS}+ items in current_answers (found {n})"
        )


def test_prior_target_meets_gold_floor(all_records: list[dict]) -> None:
    """V2: target_context=prior rows have >= GOLD_LIST_MIN_ITEMS prior_answers."""
    for entry in all_records:
        if entry["target_context"] != "prior":
            continue
        sid = entry["scenario_id"]
        n = len(entry["gold"]["prior_answers"])
        assert n >= GOLD_LIST_MIN_ITEMS, (
            f"{sid}: target_context=prior requires "
            f"{GOLD_LIST_MIN_ITEMS}+ items in prior_answers (found {n})"
        )


def test_clarify_target_meets_indicator_floor(all_records: list[dict]) -> None:
    """V3: target_context=clarify rows have >= GOLD_LIST_MIN_ITEMS clarify_indicators."""
    for entry in all_records:
        if entry["target_context"] != "clarify":
            continue
        sid = entry["scenario_id"]
        n = len(entry["gold"]["clarify_indicators"])
        assert n >= GOLD_LIST_MIN_ITEMS, (
            f"{sid}: target_context=clarify requires "
            f"{GOLD_LIST_MIN_ITEMS}+ items in clarify_indicators (found {n})"
        )


def test_abstain_target_meets_indicator_floor(all_records: list[dict]) -> None:
    """V3: target_context=abstain rows have >= GOLD_LIST_MIN_ITEMS abstain_indicators."""
    for entry in all_records:
        if entry["target_context"] != "abstain":
            continue
        sid = entry["scenario_id"]
        n = len(entry["gold"]["abstain_indicators"])
        assert n >= GOLD_LIST_MIN_ITEMS, (
            f"{sid}: target_context=abstain requires "
            f"{GOLD_LIST_MIN_ITEMS}+ items in abstain_indicators (found {n})"
        )


# ``change_type`` values where a deictic-only repair anchor cannot resolve
# the reference. Mirrors ``NON_DEICTIC_CHANGE_TYPES`` in
# scripts/validate_scenarios.py.
NON_DEICTIC_CHANGE_TYPES = {"absent_referent", "cross_session_reference"}


def test_deictic_repair_null_contract(all_records: list[dict]) -> None:
    """V5: ``turn_3_repair_prompt_deictic`` is non-null exactly when
    ``target_context == 'current'`` and the referent is visible at Turn 2
    (``change_type`` not in ``NON_DEICTIC_CHANGE_TYPES``).
    """
    for entry in all_records:
        sid = entry["scenario_id"]
        deictic = entry.get("turn_3_repair_prompt_deictic")
        is_present = bool(deictic)
        should_be_present = (
            entry["target_context"] == "current"
            and entry["change_type"] not in NON_DEICTIC_CHANGE_TYPES
        )
        if should_be_present:
            assert is_present, (
                f"{sid}: target_context=current with visible referent "
                f"(change_type={entry['change_type']!r}) requires non-null "
                "turn_3_repair_prompt_deictic"
            )
        else:
            assert not is_present, (
                f"{sid}: turn_3_repair_prompt_deictic must be null when "
                "the referent isn't deictically resolvable "
                f"(target_context={entry['target_context']!r}, "
                f"change_type={entry['change_type']!r})"
            )


def test_composition_includes_all_four_contexts(all_records: list[dict]) -> None:
    counts: dict[str, int] = {}
    for entry in all_records:
        counts[entry["target_context"]] = counts.get(entry["target_context"], 0) + 1
    for context in ALLOWED_TARGET_CONTEXTS:
        assert counts.get(context, 0) > 0, (
            f"expected scenario bank to include {context!r} scenarios, got {counts}"
        )


def test_no_removed_fields_present(all_records: list[dict]) -> None:
    removed_fields = {
        "surface",
        "authoring_basis",
        "source_example_id",
        "ambiguity_marker",
        "modality_required",
        "variant",
        "text_proxy_degraded",
    }
    for entry in all_records:
        sid = entry["scenario_id"]
        present = removed_fields & entry.keys()
        assert not present, f"{sid}: removed earlier-design fields reappeared: {sorted(present)}"


# ---------------------------------------------------------------------------
# Programmatic validator checks (was tests/test_schema.py)
# ---------------------------------------------------------------------------


def _format_fails(fails: list[dict]) -> str:
    return "\n".join(f"  [{f['check']}] {f['scenario_id']}: {f['detail']}" for f in fails)


def test_check_1_token_leakage_passes(all_records: list[dict]) -> None:
    fails = validate_scenarios.check_1_token_leakage(all_records)
    assert not fails, (
        f"check_1_token_leakage produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_2_object_names_in_images_passes(all_records: list[dict]) -> None:
    fails = validate_scenarios.check_2_object_name_in_images(all_records)
    assert not fails, (
        f"check_2_object_name_in_images produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_3_schema_validation_passes(all_records: list[dict]) -> None:
    fails = validate_scenarios.check_3_schema_validation(all_records)
    assert not fails, (
        f"check_3_schema_validation produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_6_duplication_passes(all_records: list[dict]) -> None:
    fails = validate_scenarios.check_6_duplication(all_records)
    assert not fails, (
        f"check_6_duplication produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_7_lockfile_drift_passes(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    fails = validate_scenarios.check_7_lockfile_drift()
    assert not fails, (
        f"check_7_lockfile_drift produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_7_lockfile_drift_detects_mutation(monkeypatch, tmp_path) -> None:
    """Mutating scenarios.jsonl without bumping benchmark_version trips the lock."""
    monkeypatch.chdir(REPO_ROOT)
    fake_root = tmp_path
    (fake_root / "data").mkdir(parents=True)
    original_scenarios = SCENARIOS_PATH.read_bytes()
    (fake_root / "data" / "scenarios.jsonl").write_bytes(original_scenarios + b"\n")
    (fake_root / "data" / "prompt_conditions.json").write_bytes(
        (REPO_ROOT / "data" / "prompt_conditions.json").read_bytes()
    )
    (fake_root / "data" / "MANIFEST.lock.json").write_bytes(
        (REPO_ROOT / "data" / "MANIFEST.lock.json").read_bytes()
    )
    monkeypatch.chdir(fake_root)
    fails = validate_scenarios.check_7_lockfile_drift()
    assert any(f["check"] == "lockfile" and "scenarios_sha256" in f["detail"] for f in fails), (
        f"expected a scenarios_sha256 mismatch, got: {fails}"
    )


def test_validator_main_returns_zero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_scenarios.py"])
    monkeypatch.chdir(REPO_ROOT)
    rc = validate_scenarios.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "All checks passed" in out
