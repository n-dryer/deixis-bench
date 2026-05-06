"""Markdown findings-report rendering for benchmark results.

:func:`render_findings_markdown` consumes per-trial result dicts plus
the metric helpers in
:mod:`wearable_assistant_context_bench.aggregation` and emits the
``findings.md`` body in 10 sections:

1. **Benchmark summary.** Headline primary score (mean of per-class
   recall under the default comparison condition), per-class recall
   for ``current`` and ``prior``, and a per-condition sensitivity row.
2. **Per task-set recall.** Mean recall sliced by task set.
3. **Per shift-type recall.** Mean recall sliced by ``shift_type``.
4. **Per-class pass rate by condition.** A 4-row internal grid for
   visibility. ``current`` and ``prior`` are the primary classes.
   ``clarify`` and ``abstain`` are auxiliary diagnostic classes and
   are not included in the primary score.
5. **Hedging behavior.** Clarification rate, abstention rate, and the
   coverage metric ``1 - (clarify_rate + abstain_rate)``.
6. **Code-judge disagreement count per task.**
7. **Inter-judge agreement (cross-LLM).**
8. **Task-by-condition matrix.**
9. **Reproducibility manifest.** A JSON block with the task /
    prompt-conditions / judge-prompt SHAs, model strings, trials,
    temperature, and the default comparison condition.
"""

from __future__ import annotations

import json
from typing import Any

from wearable_assistant_context_bench.aggregation import (
    AUXILIARY_POLICY_NOTE,
    BENCHMARK_LABEL,
    BENCHMARK_NAME,
    DEFAULT_RANKING_CONDITION,
    POLICIES,
    REQUIRED_MANIFEST_KEYS,
    SCORED_POLICIES,
    PassRateCell,
    abstain_rate,
    clarify_rate,
    class_recall_with_ci_under_condition,
    code_judge_disagreement_by_task,
    coverage_rate,
    inter_judge_agreement_summary,
    inter_judge_disagreement_by_task,
    mean_recall_with_bootstrap_ci_under_condition,
    mean_recall_with_ci_under_condition,
    per_policy_pass_rate_by_condition,
    recall_by_shift_type,
    recall_by_task_set,
    sorted_conditions,
    task_by_condition_matrix,
    wilson_interval,
)


def render_findings_markdown(
    results: list[dict],
    task_policies: dict[str, str] | None = None,
    manifest: dict[str, Any] | None = None,
    ranking_condition: str = DEFAULT_RANKING_CONDITION,
) -> str:
    """Render the findings.md body from per-trial results.

    Args:
        results: Per-trial result dicts.
        task_policies: Optional task_id -> gold_label
            mapping. When provided, tasks are ordered by this
            map's iteration order.
        manifest: Reproducibility manifest dict. Every required key in
            `REQUIRED_MANIFEST_KEYS` should be present; missing keys
            render as `null` with a `manifest_warnings` note.
        ranking_condition: Condition name to use for the primary score.

    Returns:
        A Markdown string including the benchmark-summary section,
        per-task-set and per-shift-type breakdowns, the per-policy
        pass-rate grid, the task matrix, hedging behavior, and a
        reproducibility manifest block.
    """
    grid = per_policy_pass_rate_by_condition(results)
    disagreements = code_judge_disagreement_by_task(results)
    matrix = task_by_condition_matrix(results)
    conditions = sorted_conditions(results)
    inter_judge_summary = inter_judge_agreement_summary(results)
    inter_judge_disagreements = (
        inter_judge_disagreement_by_task(results) if inter_judge_summary is not None else {}
    )

    primary_score_ci = mean_recall_with_ci_under_condition(results, ranking_condition)
    primary_score_bootstrap = mean_recall_with_bootstrap_ci_under_condition(
        results, ranking_condition
    )
    class_recall_ci = class_recall_with_ci_under_condition(results, ranking_condition)
    per_condition_recall_ci = {
        condition: mean_recall_with_ci_under_condition(results, condition)
        for condition in conditions
    }
    per_task_set_recall_ci = recall_by_task_set(results, ranking_condition)
    per_shift_type_recall_ci = recall_by_shift_type(results, ranking_condition)
    clarify_ci = clarify_rate(results, ranking_condition)
    abstain_ci = abstain_rate(results, ranking_condition)
    coverage_ci = coverage_rate(results, ranking_condition)

    sections = [
        f"# {BENCHMARK_NAME}: Findings",
        "",
        f"**Benchmark:** {BENCHMARK_LABEL}",
        "",
        "## Benchmark summary",
        "",
        _render_benchmark_summary(
            benchmark_label=BENCHMARK_LABEL,
            ranking_condition=ranking_condition,
            primary_score_ci=primary_score_ci,
            primary_score_bootstrap=primary_score_bootstrap,
            class_recall_ci=class_recall_ci,
            per_condition_recall_ci=per_condition_recall_ci,
        ),
        "",
        "## Per task-set recall",
        "",
        _render_per_task_set_table(per_task_set_recall_ci),
        "",
        "## Per shift-type recall",
        "",
        _render_per_shift_type_table(per_shift_type_recall_ci),
        "",
        "## Per-class pass rate by condition",
        "",
        _render_policy_grid(grid, conditions),
        "",
    ]
    sections.extend(
        [
            "## Hedging behavior",
            "",
            _render_hedging_section(clarify_ci, abstain_ci, coverage_ci),
            "",
            "## Code-judge disagreement by task",
            "",
            _render_disagreement_list(disagreements),
            "",
            "## Inter-judge agreement (cross-LLM)",
            "",
            _render_inter_judge_section(inter_judge_summary, inter_judge_disagreements),
            "",
            "## Task-by-condition matrix",
            "",
            _render_task_matrix(matrix, conditions, task_policies),
            "",
            "## Reproducibility manifest",
            "",
            _render_manifest_block(manifest or {}),
            "",
        ]
    )
    return "\n".join(sections)


def _render_benchmark_summary(
    *,
    benchmark_label: str,
    ranking_condition: str,
    primary_score_ci: tuple[float, float, float] | None,
    primary_score_bootstrap: tuple[float, float, float] | None,
    class_recall_ci: dict[str, tuple[float, float, float] | None],
    per_condition_recall_ci: dict[str, tuple[float, float, float] | None],
) -> str:
    def _pct(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{_pct(rate)} (95% CI {_pct(lo)}-{_pct(hi)})"

    lines: list[str] = [
        f"- **Benchmark**: {benchmark_label}",
        f"- **Default comparison condition**: `{ranking_condition}`",
        (
            "- **Primary score** - `mean(current_recall, prior_recall)` "
            "(class recall, not overall accuracy): "
            f"**{_ci(primary_score_ci)}**"
        ),
        (
            "- **Bootstrap 95% CI on primary score**: "
            f"{_ci(primary_score_bootstrap)} (5000 percentile bootstrap iterations)"
        ),
        "- **How to read this run**: compare candidate models on the "
        f"`{ranking_condition}` score below; treat the other conditions as "
        "diagnostic sensitivity checks. CIs are 95% Wilson per class and "
        "95% normal-approximation on the mean recall (with a bootstrap "
        "second opinion).",
        f"- **Per-class recall under `{ranking_condition}`** (TP / (TP + FN)):",
    ]
    for policy in SCORED_POLICIES:
        lines.append(f"    - `{policy}_recall`: {_ci(class_recall_ci.get(policy))}")
    lines.append("")
    lines.append("Condition sensitivity (mean per-class recall):")
    lines.append("")
    lines.append("| Condition | Mean recall (95% CI) |")
    lines.append("| --- | --- |")
    for condition, ci in per_condition_recall_ci.items():
        marker = " (default)" if condition == ranking_condition else ""
        lines.append(f"| {condition}{marker} | {_ci(ci)} |")
    return "\n".join(lines)


def _render_per_task_set_table(
    per_task_set: dict[str, tuple[float, float, float] | None],
) -> str:
    if not per_task_set:
        return "_No trials recorded._"

    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%-{hi * 100:.1f}%)"

    rows = ["| Task set | Mean recall (95% CI) |", "| --- | --- |"]
    for task_set_value in sorted(per_task_set.keys()):
        rows.append(f"| `{task_set_value}` | {_ci(per_task_set[task_set_value])} |")
    return "\n".join(rows)


def _render_per_shift_type_table(
    per_shift_type: dict[str, tuple[float, float, float] | None],
) -> str:
    if not per_shift_type:
        return "_No trials recorded._"

    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%-{hi * 100:.1f}%)"

    rows = ["| Shift type | Pass rate (95% CI) |", "| --- | --- |"]
    for shift_type in sorted(per_shift_type.keys()):
        rows.append(f"| `{shift_type}` | {_ci(per_shift_type[shift_type])} |")
    return "\n".join(rows)


def _render_hedging_section(
    clarify: tuple[float, float, float] | None,
    abstain: tuple[float, float, float] | None,
    coverage: tuple[float, float, float] | None,
) -> str:
    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%-{hi * 100:.1f}%)"

    lines = [
        f"- **Clarification rate**: {_ci(clarify)}",
        f"- **Abstention rate**: {_ci(abstain)}",
        f"- **Coverage** (1 - clarify - abstain): {_ci(coverage)}",
    ]
    if coverage is not None and coverage[0] < 0.6:
        lines.append(
            "- _Coverage below 60% - model is hedging on a majority of "
            "trials. Compare against a less hedge-prone baseline._"
        )
    return "\n".join(lines)


def _render_policy_grid(
    grid: dict[str, dict[str, PassRateCell]],
    conditions: list[str],
) -> str:
    header = "| Class | " + " | ".join(conditions) + " |"
    separator = "| --- | " + " | ".join("---" for _ in conditions) + " |"
    rows = [header, separator]
    for policy in POLICIES:
        label = f"`{policy}`"
        if policy not in SCORED_POLICIES:
            label = f"`{policy}` (auxiliary)"
        cells = [label]
        for condition in conditions:
            cell = grid[policy][condition]
            if cell.total == 0:
                if cell.primary_scored:
                    cells.append("-")
                else:
                    cells.append(AUXILIARY_POLICY_NOTE)
                continue
            if cell.rate is None:
                cells.append("-")
                continue
            ci = wilson_interval(cell.passed, cell.total)
            pct = cell.rate * 100
            if ci is None:
                cells.append(f"{pct:.1f}% ({cell.passed}/{cell.total})")
            else:
                _, lo, hi = ci
                cells.append(
                    f"{pct:.1f}% [95% CI {lo * 100:.1f}-{hi * 100:.1f}] "
                    f"({cell.passed}/{cell.total})"
                )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _render_disagreement_list(disagreements: dict[str, int]) -> str:
    if not disagreements:
        return "_No trials recorded._"
    lines: list[str] = []
    for task_id in sorted(disagreements.keys()):
        lines.append(f"- {task_id}: {disagreements[task_id]} trial(s) with code/judge disagreement")
    return "\n".join(lines)


def _render_inter_judge_section(
    summary: dict[str, Any] | None,
    disagreements: dict[str, int],
) -> str:
    """Render the cross-LLM inter-judge agreement section.

    When no ranking-judge labels are present, render a placeholder
    explaining that this run did not pair a second judge.
    """
    if summary is None:
        return (
            "_No ranking-judge labels in this run. To enable cross-LLM "
            "inter-judge agreement, pass `--ranking-judge-family` to the "
            "runner so every trial is also labeled by a fixed second judge._"
        )
    kappa = summary["kappa"]
    observed = summary["observed_agreement"]
    trials = summary["trials"]
    lines: list[str] = [
        f"- **Trials with both judge labels**: {trials}",
        f"- **Observed agreement**: {observed * 100:.1f}%",
    ]
    if kappa is None:
        lines.append("- **Cohen's kappa**: undefined (single-class degenerate case)")
    else:
        lines.append(f"- **Cohen's kappa**: {kappa:.3f}")
    lines.append("")
    lines.append("Per-task disagreement counts (where the two judges differ):")
    if not disagreements:
        lines.append("")
        lines.append("_No disagreements recorded._")
        return "\n".join(lines)
    lines.append("")
    for task_id in sorted(disagreements.keys()):
        lines.append(
            f"- {task_id}: {disagreements[task_id]} trial(s) where "
            "primary and ranking judges disagreed"
        )
    return "\n".join(lines)


def _render_task_matrix(
    matrix: dict[str, dict[str, list[dict]]],
    conditions: list[str],
    task_policies: dict[str, str] | None,
) -> str:
    header = "| Task | Target context | " + " | ".join(conditions) + " |"
    separator = "| --- | --- | " + " | ".join("---" for _ in conditions) + " |"
    rows = [header, separator]

    if task_policies is not None:
        task_order = list(task_policies.keys())
    else:
        task_order = sorted(matrix.keys())

    for task_id in task_order:
        if task_id not in matrix:
            continue
        gold_label = task_policies[task_id] if task_policies else "?"
        cells = [task_id, f"`{gold_label}`"]
        for condition in conditions:
            trials = matrix[task_id].get(condition, [])
            cells.append(_format_trial_outcomes(trials))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format_trial_outcomes(trials: list[dict]) -> str:
    if not trials:
        return "-"
    return ", ".join("pass" if entry["turn_2_passed"] else "fail" for entry in trials)


def _render_manifest_block(manifest: dict[str, Any]) -> str:
    """Render the reproducibility manifest as a JSON code block.

    Ensures every required key is present (with `null` fallback) and
    attaches any `manifest_warnings` provided by the runner.
    """
    out: dict[str, Any] = {}
    missing: list[str] = []
    for key in REQUIRED_MANIFEST_KEYS:
        if key in manifest:
            out[key] = manifest[key]
        else:
            out[key] = None
            missing.append(key)
    warnings = list(manifest.get("manifest_warnings") or [])
    for key in missing:
        warnings.append(f"manifest key missing: {key}")
    extras = {
        key: value
        for key, value in manifest.items()
        if key not in REQUIRED_MANIFEST_KEYS and key != "manifest_warnings"
    }
    out.update(extras)
    out["manifest_warnings"] = warnings
    return "```json\n" + json.dumps(out, indent=2, sort_keys=False) + "\n```"
