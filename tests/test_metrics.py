"""Math validation: per-trial scoring + report aggregation.

Covers ``wearable_assistant_context_bench.scoring`` (deterministic
per-trial signals) and
``wearable_assistant_context_bench.aggregation`` /
``wearable_assistant_context_bench.rendering`` (aggregation, recall
metrics, manifest rendering, inter-judge agreement).
"""

from __future__ import annotations

import json
import re

import pytest

from wearable_assistant_context_bench.aggregation import (
    AUXILIARY_POLICY_NOTE,
    BENCHMARK_LABEL,
    DEFAULT_RANKING_CONDITION,
    REQUIRED_MANIFEST_KEYS,
    abstain_rate,
    clarify_rate,
    class_recall_under_condition,
    class_recall_with_ci_under_condition,
    code_judge_disagreement_by_task,
    cohens_kappa,
    coverage_rate,
    inter_judge_agreement_summary,
    inter_judge_disagreement_by_task,
    mean_recall_under_condition,
    mean_recall_with_bootstrap_ci_under_condition,
    mean_recall_with_ci_under_condition,
    per_policy_pass_rate_by_condition,
    recall_by_shift_type,
    recall_by_task_set,
    simulated_repair_rate_by_condition,
    task_by_condition_matrix,
    wilson_interval,
)
from wearable_assistant_context_bench.rendering import render_findings_markdown
from wearable_assistant_context_bench.scoring import (
    detect_refusal,
    fuzzy_match,
    has_prior,
    score_response,
    substring_match,
)
from wearable_assistant_context_bench.statistics import bootstrap_ci

# ===========================================================================
# core.scoring - per-trial code signals
# ===========================================================================


def test_fuzzy_match_true_on_close_match() -> None:
    assert fuzzy_match(
        "The meeting has been rescheduled to 3pm sharp.",
        targets=["rescheduled to 3pm"],
    )


def test_fuzzy_match_false_on_non_match() -> None:
    assert not fuzzy_match(
        "The weather is sunny today.",
        targets=["database migration failed"],
    )


def test_substring_match_true_on_case_insensitive_hit() -> None:
    assert substring_match(
        "I don't have access to that data.",
        indicators=["don't have access"],
    )


def test_substring_match_false_on_non_match() -> None:
    assert not substring_match(
        "The meeting starts at 3pm.",
        indicators=["no videos were provided"],
    )


def test_detect_refusal_catches_common_phrasings() -> None:
    assert detect_refusal("I cannot answer that question.")
    assert detect_refusal("I'm unable to determine the current state.")
    assert detect_refusal("I don't have enough information to say.")
    assert detect_refusal("I don't know the answer.")
    assert detect_refusal("I'm not able to provide that.")
    assert not detect_refusal("The answer is 42.")


def test_has_prior_delegates_to_fuzzy_match() -> None:
    assert has_prior(
        "The meeting ran long on Monday morning.",
        prior_answers=["Monday"],
    )
    assert not has_prior(
        "The meeting ran long today.",
        prior_answers=["Friday afternoon"],
    )


def test_score_response_pass_case() -> None:
    result = score_response(
        response="I can see a toothbrush and a mirror on the counter.",
        current_answers=["toothbrush", "mirror"],
        prior_answers=["stove", "kettle"],
    )
    assert result["has_current"] is True
    assert result["has_prior"] is False
    assert result["is_refusal"] is False


def test_score_response_fail_case_with_prior() -> None:
    result = score_response(
        response="On the counter there is a stove and a kettle.",
        current_answers=["toothbrush", "mirror"],
        prior_answers=["stove", "kettle"],
    )
    assert result["has_current"] is False
    assert result["has_prior"] is True


def test_score_response_refusal_case() -> None:
    result = score_response(
        response="I'm unable to determine what is on the counter.",
        current_answers=["toothbrush", "mirror"],
        prior_answers=["stove", "kettle"],
    )
    assert result["is_refusal"] is True


def test_score_response_contrastive_phrasing_suppresses_has_prior() -> None:
    result = score_response(
        response="Earlier there were 12 photos; now there are 13",
        current_answers=["13"],
        prior_answers=["12"],
    )
    assert result["has_current"] is True
    assert result["has_prior"] is False
    assert result["has_prior_raw"] is True


def test_score_response_no_trigger_verb_leaves_has_prior_true() -> None:
    result = score_response(
        response="The album had 12 photos, it now has 13",
        current_answers=["13"],
        prior_answers=["12"],
    )
    assert result["has_current"] is True
    assert result["has_prior"] is True
    assert result["has_prior_raw"] is True


def test_score_response_has_clarify_signal() -> None:
    result = score_response(
        response=("Do you mean the scene you just described, or the one from earlier?"),
        current_answers=[],
        prior_answers=[],
        clarify_indicators=["Do you mean"],
    )
    assert result["has_clarify"] is True
    assert result["has_abstain"] is False


def test_score_response_has_abstain_signal() -> None:
    result = score_response(
        response="I'm sorry, but I don't have access to your uploaded videos.",
        current_answers=[],
        prior_answers=[],
        abstain_indicators=["don't have access"],
    )
    assert result["has_abstain"] is True
    assert result["has_clarify"] is False


def test_score_response_does_not_emit_legacy_stale_keys() -> None:
    result = score_response(
        response="Earlier there were 12 photos; now there are 13",
        current_answers=["13"],
        prior_answers=["12"],
    )
    assert "has_stale" not in result
    assert "has_stale_raw" not in result
    assert "has_prior" in result
    assert "has_prior_raw" in result


def test_score_response_token_length_estimate_is_positive_integer() -> None:
    result = score_response(
        response="one two three four five",
        current_answers=[],
        prior_answers=[],
    )
    assert isinstance(result["response_length_tokens_est"], int)
    assert result["response_length_tokens_est"] > 0


def test_score_response_no_match_baseline_unchanged() -> None:
    result = score_response(
        response="The counter has a toothbrush on it.",
        current_answers=["toothbrush"],
        prior_answers=["stove"],
    )
    assert result["has_current"] is True
    assert result["has_prior"] is False
    assert result["has_prior_raw"] is False


def test_score_response_indicators_default_to_empty() -> None:
    result = score_response(
        response="anything",
        current_answers=[],
        prior_answers=[],
    )
    assert result["has_clarify"] is False
    assert result["has_abstain"] is False


# ===========================================================================
# core.report - aggregation, recall metrics, manifest rendering
# ===========================================================================


def _trial(
    *,
    task_id: str,
    condition: str,
    trial: int,
    gold_label: str,
    turn_2_passed: bool,
    turn_2_judge_label: str | None = None,
    turn_2_code_signals: dict | None = None,
    turn_3_repair_attempted: bool = False,
    turn_3_repair_passed: bool | None = None,
    task_set: str = "main",
    shift_type: str = "object_in_hand",
) -> dict:
    return {
        "task_id": task_id,
        "task_set": task_set,
        "shift_type": shift_type,
        "condition": condition,
        "trial": trial,
        "gold_label": gold_label,
        "turn_2_passed": turn_2_passed,
        "turn_2_judge_label": (
            turn_2_judge_label
            if turn_2_judge_label is not None
            else (gold_label if turn_2_passed else "abstain")
        ),
        "turn_2_code_signals": turn_2_code_signals or {},
        "turn_3_repair_attempted": turn_3_repair_attempted,
        "turn_3_repair_passed": turn_3_repair_passed,
    }


def _fixture_results() -> list[dict]:
    """Mini fixture: one ``prior`` task (task-003) and three ``current``
    tasks (task-001, task-002, task-004). Two trials per cell. All main task_set."""
    results: list[dict] = []
    prior_cells: list[tuple[str, list[bool], list[bool | None]]] = [
        ("baseline", [False, False], [True, False]),
        ("context_selection_instruction", [True, False], [None, True]),
        ("pre_answer_context_scaffold", [True, True], [None, None]),
    ]
    for condition, passes, repairs in prior_cells:
        for i, (passed, repaired) in enumerate(zip(passes, repairs, strict=False)):
            results.append(
                _trial(
                    task_id="task-003",
                    condition=condition,
                    trial=i,
                    gold_label="prior",
                    turn_2_passed=passed,
                    turn_3_repair_attempted=(not passed),
                    turn_3_repair_passed=repaired,
                )
            )
    current_plan = {
        "task-001": {
            "baseline": [True, True],
            "context_selection_instruction": [True, True],
            "pre_answer_context_scaffold": [True, True],
        },
        "task-002": {
            "baseline": [True, False],
            "context_selection_instruction": [True, True],
            "pre_answer_context_scaffold": [True, True],
        },
        "task-004": {
            "baseline": [False, True],
            "context_selection_instruction": [True, False],
            "pre_answer_context_scaffold": [True, True],
        },
    }
    for task_id, conds in current_plan.items():
        for condition, passes in conds.items():
            for i, passed in enumerate(passes):
                results.append(
                    _trial(
                        task_id=task_id,
                        condition=condition,
                        trial=i,
                        gold_label="current",
                        turn_2_passed=passed,
                        turn_3_repair_attempted=(not passed),
                        turn_3_repair_passed=(False if not passed else None),
                    )
                )
    return results


def test_per_policy_grid_includes_all_four_rows() -> None:
    grid = per_policy_pass_rate_by_condition(_fixture_results())
    assert set(grid.keys()) == {"current", "prior", "clarify", "abstain"}


def test_per_policy_grid_clarify_and_abstain_are_diagnostic() -> None:
    grid = per_policy_pass_rate_by_condition(_fixture_results())
    for condition_cell in grid["clarify"].values():
        assert condition_cell.primary_scored is False
        assert condition_cell.rate is None
    for condition_cell in grid["abstain"].values():
        assert condition_cell.primary_scored is False


def test_per_policy_grid_prior_rates_reflect_fixture() -> None:
    grid = per_policy_pass_rate_by_condition(_fixture_results())
    assert grid["prior"]["baseline"].passed == 0
    assert grid["prior"]["baseline"].total == 2
    assert grid["prior"]["context_selection_instruction"].passed == 1
    assert grid["prior"]["context_selection_instruction"].total == 2
    assert grid["prior"]["pre_answer_context_scaffold"].passed == 2
    assert grid["prior"]["pre_answer_context_scaffold"].total == 2


def test_class_recall_under_baseline() -> None:
    results = _fixture_results()
    accs = class_recall_under_condition(results, "baseline")
    assert accs["prior"] == pytest.approx(0.0)
    assert accs["current"] == pytest.approx(4 / 6)


def test_mean_recall_under_baseline_is_mean_of_class_recalls() -> None:
    results = _fixture_results()
    bal = mean_recall_under_condition(results, "baseline")
    assert bal == pytest.approx((0 + 4 / 6) / 2)


def test_mean_recall_returns_none_when_a_class_is_empty() -> None:
    only_current = [
        _trial(
            task_id="task-001",
            condition="baseline",
            trial=0,
            gold_label="current",
            turn_2_passed=True,
        )
    ]
    assert mean_recall_under_condition(only_current, "baseline") is None


def test_mean_recall_penalizes_clarify_abstain_as_wrong() -> None:
    results = [
        _trial(
            task_id="task-003",
            condition="baseline",
            trial=0,
            gold_label="prior",
            turn_2_passed=False,
            turn_2_judge_label="clarify",
        ),
        _trial(
            task_id="task-004",
            condition="baseline",
            trial=0,
            gold_label="current",
            turn_2_passed=True,
        ),
    ]
    accs = class_recall_under_condition(results, "baseline")
    assert accs["prior"] == pytest.approx(0.0)
    assert accs["current"] == pytest.approx(1.0)
    assert mean_recall_under_condition(results, "baseline") == pytest.approx(0.5)


def test_recall_by_task_set_groups_declared_sets() -> None:
    results = _fixture_results()
    # Mark a few trials as a second task set to verify grouping behavior.
    for i in range(0, 4):
        results[i]["task_set"] = "diagnostic"
    by_task_set = recall_by_task_set(results, "baseline")
    assert "main" in by_task_set
    assert "diagnostic" in by_task_set


def test_recall_by_shift_type_buckets_correctly() -> None:
    results = _fixture_results()
    # Tag two trials with a different cue type so the dict has 2 keys.
    for i in range(2):
        results[i]["shift_type"] = "object_state"
    out = recall_by_shift_type(results, "baseline")
    assert "object_in_hand" in out
    assert "object_state" in out


def test_clarify_and_abstain_and_coverage_rates() -> None:
    results = [
        _trial(
            task_id=f"task-{i:03d}",
            condition="baseline",
            trial=0,
            gold_label="current",
            turn_2_passed=False,
            turn_2_judge_label="clarify",
        )
        for i in range(3)
    ] + [
        _trial(
            task_id="task-099",
            condition="baseline",
            trial=0,
            gold_label="current",
            turn_2_passed=True,
        )
    ]
    cl = clarify_rate(results, "baseline")
    ab = abstain_rate(results, "baseline")
    cov = coverage_rate(results, "baseline")
    assert cl is not None and cl[0] == pytest.approx(3 / 4)
    assert ab is not None and ab[0] == pytest.approx(0.0)
    assert cov is not None and cov[0] == pytest.approx(1 / 4)


def test_simulated_repair_rate_uses_failures_as_denominator() -> None:
    repair = simulated_repair_rate_by_condition(_fixture_results())
    assert repair["baseline"].failures == 4
    assert repair["baseline"].repaired == 1
    assert repair["pre_answer_context_scaffold"].failures == 0
    assert repair["pre_answer_context_scaffold"].rate is None


def test_code_judge_disagreement_counts_per_task() -> None:
    results = [
        {
            "task_id": "task-003",
            "condition": "baseline",
            "trial": 0,
            "gold_label": "prior",
            "turn_2_passed": False,
            "turn_2_judge_label": "abstain",
            "turn_2_code_signals": {
                "has_current": False,
                "has_prior": True,
                "has_clarify": False,
                "has_abstain": False,
                "is_refusal": False,
            },
            "turn_3_repair_attempted": False,
            "turn_3_repair_passed": None,
        },
        {
            "task_id": "task-004",
            "condition": "baseline",
            "trial": 0,
            "gold_label": "current",
            "turn_2_passed": True,
            "turn_2_judge_label": "current",
            "turn_2_code_signals": {
                "has_current": True,
                "has_prior": False,
                "has_clarify": False,
                "has_abstain": False,
                "is_refusal": False,
            },
            "turn_3_repair_attempted": False,
            "turn_3_repair_passed": None,
        },
    ]
    disagreements = code_judge_disagreement_by_task(results)
    assert disagreements["task-003"] == 1
    assert disagreements["task-004"] == 0


def test_task_matrix_preserves_trial_order() -> None:
    matrix = task_by_condition_matrix(_fixture_results())
    trials = matrix["task-003"]["baseline"]
    assert [entry["trial"] for entry in trials] == [0, 1]
    assert trials[0]["turn_3_repair_passed"] is True
    assert trials[1]["turn_3_repair_passed"] is False


def test_render_findings_markdown_shape() -> None:
    output = render_findings_markdown(
        _fixture_results(),
        task_policies={
            "task-001": "current",
            "task-002": "current",
            "task-003": "prior",
            "task-004": "current",
        },
        manifest={
            "benchmark_version": "0.1.0a0",
            "tasks_sha256": "abc",
            "prompt_conditions_sha256": "ghi",
            "candidate_model": "claude-sonnet-4-6",
            "judge_model": "gemini-2.5-flash",
            "judge_family": "gemini",
            "trials": 2,
            "temperature": 0.0,
            "ranking_condition": "baseline",
            "timestamp_utc": "2026-04-19T00:00:00+00:00",
            "runner_git_commit": None,
        },
    )
    assert BENCHMARK_LABEL in output
    assert "Benchmark summary" in output
    assert "Per-class pass rate" in output
    assert "Per task-set recall" in output
    assert "Per shift-type recall" in output
    assert "Hedging behavior" in output
    assert "Code-judge disagreement" in output
    assert "Task-by-condition matrix" in output
    assert "Reproducibility manifest" in output
    assert "Primary score" in output
    assert "class recall" in output
    assert AUXILIARY_POLICY_NOTE in output


def test_render_findings_markdown_emits_complete_manifest() -> None:
    manifest = {
        "benchmark_version": "0.1.0a0",
        "tasks_sha256": "sha-tasks",
        "prompt_conditions_sha256": "sha-prompt-conditions",
        "candidate_model": "claude-sonnet-4-6",
        "judge_model": "gemini-2.5-flash",
        "judge_family": "gemini",
        "judge_family_resolution": "auto",
        "trials": 2,
        "temperature": 0.0,
        "ranking_condition": DEFAULT_RANKING_CONDITION,
        "timestamp_utc": "2026-04-19T12:00:00+00:00",
        "runner_git_commit": None,
        "random_seed": None,
        "manifest_warnings": [],
    }
    output = render_findings_markdown(_fixture_results(), manifest=manifest)
    match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in payload
    assert payload["judge_family_resolution"] == "auto"
    assert payload["manifest_warnings"] == []


def test_render_findings_markdown_manifest_fills_missing_keys() -> None:
    output = render_findings_markdown(
        _fixture_results(),
        manifest={"benchmark_version": "0.1.0a0"},
    )
    match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in payload
    warnings = payload["manifest_warnings"]
    assert any("tasks_sha256" in w for w in warnings)
    assert payload["tasks_sha256"] is None


def test_wilson_interval_zero_n_returns_none() -> None:
    assert wilson_interval(0, 0) is None


def test_wilson_interval_typical_case_brackets_rate() -> None:
    triple = wilson_interval(45, 50)
    assert triple is not None
    rate, lo, hi = triple
    assert abs(rate - 0.9) < 1e-9
    assert 0.0 <= lo <= rate <= hi <= 1.0
    assert hi - lo > 0


def test_wilson_interval_full_pass_caps_at_one() -> None:
    triple = wilson_interval(50, 50)
    assert triple is not None
    rate, lo, hi = triple
    assert rate == 1.0
    assert hi == pytest.approx(1.0)
    assert lo < 1.0


def test_class_recall_with_ci_returns_triples_or_none() -> None:
    results = []
    for _ in range(5):
        results.append(
            _trial(
                task_id="task-current",
                condition="baseline",
                trial=0,
                gold_label="current",
                turn_2_passed=True,
            )
        )
    for _ in range(5):
        results.append(
            _trial(
                task_id="task-prior",
                condition="baseline",
                trial=0,
                gold_label="prior",
                turn_2_passed=False,
            )
        )
    out = class_recall_with_ci_under_condition(results, "baseline")
    assert out["current"] is not None and out["current"][0] == 1.0
    assert out["prior"] is not None and out["prior"][0] == 0.0


def test_mean_recall_with_ci_handles_missing_class() -> None:
    results = [
        _trial(
            task_id="task-current",
            condition="baseline",
            trial=0,
            gold_label="current",
            turn_2_passed=True,
        )
    ]
    assert mean_recall_with_ci_under_condition(results, "baseline") is None


def test_mean_recall_bootstrap_ci_brackets_normal_ci() -> None:
    """The percentile bootstrap CI should be in the same neighborhood as
    the normal-approximation CI on the same data."""
    results = []
    for _ in range(20):
        results.append(
            _trial(
                task_id="task-current",
                condition="baseline",
                trial=0,
                gold_label="current",
                turn_2_passed=True,
            )
        )
    for i in range(20):
        results.append(
            _trial(
                task_id="task-prior",
                condition="baseline",
                trial=0,
                gold_label="prior",
                turn_2_passed=(i < 10),
            )
        )
    triple = mean_recall_with_bootstrap_ci_under_condition(results, "baseline", n_iter=2000)
    assert triple is not None
    point, lo, hi = triple
    # Mean of recalls = (1.0 + 0.5) / 2 = 0.75
    assert point == pytest.approx(0.75)
    assert lo <= point <= hi


def test_bootstrap_ci_basic_values() -> None:
    ci = bootstrap_ci([1.0, 1.0, 0.0, 0.0, 1.0], n_iter=500)
    assert ci is not None
    assert ci.point == pytest.approx(0.6)
    assert 0.0 <= ci.lower <= ci.point <= ci.upper <= 1.0


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_cohens_kappa_independent_random_is_zero() -> None:
    a = ["x", "x", "y", "y"]
    b = ["x", "y", "x", "y"]
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_cohens_kappa_below_zero_when_systematically_disagreeing() -> None:
    a = ["x", "x", "y", "y"]
    b = ["y", "y", "x", "x"]
    assert cohens_kappa(a, b) == pytest.approx(-1.0)


def test_cohens_kappa_unequal_lengths_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_inter_judge_agreement_summary_returns_none_without_ranking_labels() -> None:
    assert inter_judge_agreement_summary(_fixture_results()) is None


def test_inter_judge_agreement_summary_with_ranking_labels() -> None:
    results = _fixture_results()
    for i, trial in enumerate(results):
        primary = trial["turn_2_judge_label"]
        if i % 5 == 0 and primary == "current":
            trial["turn_2_ranking_judge_label"] = "prior"
        else:
            trial["turn_2_ranking_judge_label"] = primary
    summary = inter_judge_agreement_summary(results)
    assert summary is not None
    assert summary["trials"] == len(results)
    assert summary["observed_agreement"] < 1.0
    assert summary["kappa"] is not None
    assert -1.0 <= summary["kappa"] <= 1.0


def test_inter_judge_disagreement_by_task_counts_only_paired_trials() -> None:
    results = _fixture_results()
    target_idx = 0
    results[target_idx]["turn_2_ranking_judge_label"] = (
        "prior" if results[target_idx]["turn_2_judge_label"] != "prior" else "current"
    )
    counts = inter_judge_disagreement_by_task(results)
    assert counts.get(results[target_idx]["task_id"]) == 1
    assert sum(counts.values()) == 1


def test_render_findings_markdown_includes_inter_judge_section() -> None:
    output = render_findings_markdown(_fixture_results())
    assert "Inter-judge agreement (cross-LLM)" in output
    # No ranking judge in the fixture -> placeholder text rendered.
    assert "ranking-judge-family" in output


if __name__ == "__main__":
    pytest.main([__file__])
