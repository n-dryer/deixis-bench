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
    "subset",
    "target_context",
    "shift_type",
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
ALLOWED_SHIFT_TYPES = {
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
    "absent_referent",
    "referent_offscreen",
    "compound_shift",
}
ALLOWED_PACKS = {"main"}

SCENARIO_ID_PATTERN = re.compile(r"^(sc|adv)-\d{2,3}$")


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


@pytest.fixture(scope="module")
def main_subset(all_records: list[dict]) -> list[dict]:
    return [r for r in all_records if r.get("subset") == "main"]


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


def test_packs_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["subset"] in ALLOWED_PACKS, (
            f"{entry['scenario_id']}: unexpected pack {entry['subset']!r}"
        )


def test_shift_types_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["shift_type"] in ALLOWED_SHIFT_TYPES, (
            f"{entry['scenario_id']}: unexpected shift_type {entry['shift_type']!r}"
        )


def test_difficulty_tiers_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["difficulty_tier"] in ALLOWED_DIFFICULTIES, (
            f"{entry['scenario_id']}: unexpected difficulty_tier {entry['difficulty_tier']!r}"
        )


def test_cognitive_loads_in_allowed_set(all_records: list[dict]) -> None:
    for entry in all_records:
        assert entry["referent_complexity"] in ALLOWED_COGNITIVE_LOADS, (
            f"{entry['scenario_id']}: unexpected cognitive_load {entry['referent_complexity']!r}"
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
        if entry["shift_type"] != "cross_session_reference":
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


def test_current_target_has_three_plus_current_answers(
    all_records: list[dict],
) -> None:
    for entry in all_records:
        if entry["target_context"] != "current":
            continue
        sid = entry["scenario_id"]
        assert len(entry["gold"]["current_answers"]) >= 3, (
            f"{sid}: target_context=current requires 3+ items in current_answers"
        )


def test_prior_target_has_three_plus_prior_answers(
    all_records: list[dict],
) -> None:
    for entry in all_records:
        if entry["target_context"] != "prior":
            continue
        sid = entry["scenario_id"]
        assert len(entry["gold"]["prior_answers"]) >= 3, (
            f"{sid}: target_context=prior requires 3+ items in prior_answers"
        )


def test_clarify_target_has_clarify_indicators(all_records: list[dict]) -> None:
    for entry in all_records:
        if entry["target_context"] != "clarify":
            continue
        sid = entry["scenario_id"]
        assert entry["gold"]["clarify_indicators"], (
            f"{sid}: target_context=clarify requires non-empty clarify_indicators"
        )


def test_abstain_target_has_abstain_indicators(all_records: list[dict]) -> None:
    for entry in all_records:
        if entry["target_context"] != "abstain":
            continue
        sid = entry["scenario_id"]
        assert entry["gold"]["abstain_indicators"], (
            f"{sid}: target_context=abstain requires non-empty abstain_indicators"
        )


def test_composition_includes_all_four_contexts(all_records: list[dict]) -> None:
    counts: dict[str, int] = {}
    for entry in all_records:
        counts[entry["target_context"]] = counts.get(entry["target_context"], 0) + 1
    for context in ALLOWED_TARGET_CONTEXTS:
        assert counts.get(context, 0) > 0, (
            f"expected main subset to include {context!r} scenarios, got {counts}"
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


def test_check_1_token_leakage_passes(main_subset: list[dict]) -> None:
    fails = validate_scenarios.check_1_token_leakage(main_subset)
    assert not fails, (
        f"check_1_token_leakage produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_2_object_names_in_images_passes(main_subset: list[dict]) -> None:
    fails = validate_scenarios.check_2_object_name_in_images(main_subset)
    assert not fails, (
        f"check_2_object_name_in_images produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_3_schema_validation_passes(main_subset: list[dict]) -> None:
    fails = validate_scenarios.check_3_schema_validation(main_subset)
    assert not fails, (
        f"check_3_schema_validation produced {len(fails)} failure(s):\n{_format_fails(fails)}"
    )


def test_check_6_duplication_passes(main_subset: list[dict]) -> None:
    fails = validate_scenarios.check_6_duplication(main_subset)
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
