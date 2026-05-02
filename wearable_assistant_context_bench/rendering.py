"""Markdown findings-report rendering for benchmark results.

:func:`render_findings_markdown` consumes per-trial result dicts plus
the metric helpers in
:mod:`wearable_assistant_context_bench.aggregation` and emits the
``findings.md`` body in 11 sections:

1. **Benchmark summary.** Headline primary score (mean of per-class
   recall under the default comparison condition), per-class recall
   for ``current`` and ``prior``, and a per-condition sensitivity row.
2. **Per-pack recall.** Mean recall sliced by pack (``main`` /
   ``contrast``).
3. **Per shift-type recall.** Mean recall sliced by ``shift_type``.
4. **Contrast pair consistency.** Percentage of A/B pairs in the
   contrast pack where both variants pass. Reported only when
   ``pair_id`` metadata is present.
5. **Per-class pass rate by condition.** A 4-row internal grid for
   visibility. ``current`` and ``prior`` are the primary classes.
   ``clarify`` and ``abstain`` are auxiliary diagnostic classes and
   are not included in the primary score.
6. **Simulated repair rate per condition** (only when ``--enable-repair``
   was set).
7. **Hedging behavior.** Clarification rate, abstention rate, and the
   coverage metric ``1 - (clarify_rate + abstain_rate)``.
8. **Code-judge disagreement count per scenario.**
9. **Inter-judge agreement (cross-LLM).**
10. **Scenario-by-condition matrix.**
11. **Reproducibility manifest.** A JSON block with the scenario /
    interventions / judge-prompt SHAs, model strings, trials,
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
    RepairRateCell,
    abstain_rate,
    clarify_rate,
    class_recall_with_ci_under_condition,
    code_judge_disagreement_by_scenario,
    contrast_pair_consistency,
    coverage_rate,
    inter_judge_agreement_summary,
    inter_judge_disagreement_by_scenario,
    mean_recall_with_bootstrap_ci_under_condition,
    mean_recall_with_ci_under_condition,
    per_policy_pass_rate_by_condition,
    recall_by_shift_type,
    recall_by_subset,
    scenario_by_condition_matrix,
    simulated_repair_rate_by_condition,
    sorted_conditions,
    wilson_interval,
)


def render_findings_markdown(
    results: list[dict],
    scenario_policies: dict[str, str] | None = None,
    manifest: dict[str, Any] | None = None,
    ranking_condition: str = DEFAULT_RANKING_CONDITION,
) -> str:
    """Render the findings.md body from per-trial results.

    Args:
        results: Per-trial result dicts.
        scenario_policies: Optional scenario_id -> target_context
            mapping. When provided, scenarios are ordered by this
            map's iteration order.
        manifest: Reproducibility manifest dict. Every required key in
            `REQUIRED_MANIFEST_KEYS` should be present; missing keys
            render as `null` with a `manifest_warnings` note.
        ranking_condition: Condition name to use for the primary score.

    Returns:
        A Markdown string including the benchmark-summary section,
        per-pack and per-shift-type breakdowns, contrast pair
        consistency, the per-policy pass-rate grid, the scenario
        matrix, hedging behavior, and a reproducibility manifest block.
    """
    grid = per_policy_pass_rate_by_condition(results)
    repair = simulated_repair_rate_by_condition(results)
    disagreements = code_judge_disagreement_by_scenario(results)
    matrix = scenario_by_condition_matrix(results)
    conditions = sorted_conditions(results)
    inter_judge_summary = inter_judge_agreement_summary(results)
    inter_judge_disagreements = (
        inter_judge_disagreement_by_scenario(results) if inter_judge_summary is not None else {}
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
    per_subset_recall_ci = recall_by_subset(results, ranking_condition)
    per_cue_recall_ci = recall_by_shift_type(results, ranking_condition)
    pair_consistency = contrast_pair_consistency(results, ranking_condition)
    clarify_ci = clarify_rate(results, ranking_condition)
    abstain_ci = abstain_rate(results, ranking_condition)
    coverage_ci = coverage_rate(results, ranking_condition)
    repair_enabled = any(bool(t.get("turn_3_repair_attempted")) for t in results)

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
        "## Per-subset recall",
        "",
        _render_per_subset_table(per_subset_recall_ci),
        "",
        "## Per shift-type recall",
        "",
        _render_per_cue_table(per_cue_recall_ci),
        "",
        "## Contrast pair consistency",
        "",
        _render_pair_consistency(pair_consistency),
        "",
        "## Per-class pass rate by condition",
        "",
        _render_policy_grid(grid, conditions),
        "",
    ]
    if repair_enabled:
        sections.extend(
            [
                "## Simulated repair rate by condition",
                "",
                _render_repair_table(repair),
                "",
            ]
        )
    sections.extend(
        [
            "## Hedging behavior",
            "",
            _render_hedging_section(clarify_ci, abstain_ci, coverage_ci),
            "",
            "## Code-judge disagreement by scenario",
            "",
            _render_disagreement_list(disagreements),
            "",
            "## Inter-judge agreement (cross-LLM)",
            "",
            _render_inter_judge_section(inter_judge_summary, inter_judge_disagreements),
            "",
            "## Scenario-by-condition matrix",
            "",
            _render_scenario_matrix(matrix, conditions, scenario_policies),
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
        return f"{_pct(rate)} (95% CI {_pct(lo)}–{_pct(hi)})"

    lines: list[str] = [
        f"- **Benchmark**: {benchmark_label}",
        f"- **Default comparison condition**: `{ranking_condition}`",
        (
            "- **Primary score** — `mean(current_recall, prior_recall)` "
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


def _render_per_subset_table(
    per_pack: dict[str, tuple[float, float, float] | None],
) -> str:
    if not per_pack:
        return "_No trials recorded._"

    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%–{hi * 100:.1f}%)"

    rows = ["| Subset | Mean recall (95% CI) |", "| --- | --- |"]
    for pack in sorted(per_pack.keys()):
        rows.append(f"| `{pack}` | {_ci(per_pack[pack])} |")
    return "\n".join(rows)


def _render_per_cue_table(
    per_cue: dict[str, tuple[float, float, float] | None],
) -> str:
    if not per_cue:
        return "_No trials recorded._"

    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%–{hi * 100:.1f}%)"

    rows = ["| Shift type | Pass rate (95% CI) |", "| --- | --- |"]
    for cue in sorted(per_cue.keys()):
        rows.append(f"| `{cue}` | {_ci(per_cue[cue])} |")
    return "\n".join(rows)


def _render_pair_consistency(payload: dict[str, Any]) -> str:
    if payload.get("pairs_evaluated", 0) == 0:
        note = payload.get("note") or "no pair_id metadata available"
        return f"_{note}_"
    rate = payload["consistency_rate"]
    ci = payload.get("ci")
    rate_pct = f"{rate * 100:.1f}%"
    if ci is None:
        ci_part = ""
    else:
        lo, hi = ci
        ci_part = f" (95% CI {lo * 100:.1f}%–{hi * 100:.1f}%)"
    return (
        f"- **Pairs evaluated**: {payload['pairs_evaluated']}\n"
        f"- **Both-correct rate**: {rate_pct}{ci_part}"
    )


def _render_hedging_section(
    clarify: tuple[float, float, float] | None,
    abstain: tuple[float, float, float] | None,
    coverage: tuple[float, float, float] | None,
) -> str:
    def _ci(triple: tuple[float, float, float] | None) -> str:
        if triple is None:
            return "n/a"
        rate, lo, hi = triple
        return f"{rate * 100:.1f}% (95% CI {lo * 100:.1f}%–{hi * 100:.1f}%)"

    lines = [
        f"- **Clarification rate**: {_ci(clarify)}",
        f"- **Abstention rate**: {_ci(abstain)}",
        f"- **Coverage** (1 - clarify - abstain): {_ci(coverage)}",
    ]
    if coverage is not None and coverage[0] < 0.6:
        lines.append(
            "- _Coverage below 60% — model is hedging on a majority of "
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
                    f"{pct:.1f}% [95% CI {lo * 100:.1f}–{hi * 100:.1f}] "
                    f"({cell.passed}/{cell.total})"
                )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _render_repair_table(repair: dict[str, RepairRateCell]) -> str:
    header = "| Condition | Repair rate (95% CI) |"
    separator = "| --- | --- |"
    rows = [header, separator]
    for condition, cell in repair.items():
        if cell.failures == 0:
            rows.append(f"| {condition} | no Turn 2 failures |")
            continue
        pct = cell.rate * 100 if cell.rate is not None else 0.0
        ci = wilson_interval(cell.repaired, cell.failures)
        if ci is None:
            rows.append(f"| {condition} | {pct:.1f}% ({cell.repaired} / {cell.failures}) |")
        else:
            _, lo, hi = ci
            rows.append(
                f"| {condition} | {pct:.1f}% [95% CI {lo * 100:.1f}–{hi * 100:.1f}] "
                f"({cell.repaired} / {cell.failures}) |"
            )
    return "\n".join(rows)


def _render_disagreement_list(disagreements: dict[str, int]) -> str:
    if not disagreements:
        return "_No trials recorded._"
    lines: list[str] = []
    for scenario_id in sorted(disagreements.keys()):
        lines.append(
            f"- {scenario_id}: {disagreements[scenario_id]} trial(s) with code/judge disagreement"
        )
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
    lines.append("Per-scenario disagreement counts (where the two judges differ):")
    if not disagreements:
        lines.append("")
        lines.append("_No disagreements recorded._")
        return "\n".join(lines)
    lines.append("")
    for scenario_id in sorted(disagreements.keys()):
        lines.append(
            f"- {scenario_id}: {disagreements[scenario_id]} trial(s) where "
            "primary and ranking judges disagreed"
        )
    return "\n".join(lines)


def _render_scenario_matrix(
    matrix: dict[str, dict[str, list[dict]]],
    conditions: list[str],
    scenario_policies: dict[str, str] | None,
) -> str:
    header = "| Scenario | Target context | " + " | ".join(conditions) + " |"
    separator = "| --- | --- | " + " | ".join("---" for _ in conditions) + " |"
    rows = [header, separator]

    if scenario_policies is not None:
        scenario_order = list(scenario_policies.keys())
    else:
        scenario_order = sorted(matrix.keys())

    for scenario_id in scenario_order:
        if scenario_id not in matrix:
            continue
        target_context = scenario_policies[scenario_id] if scenario_policies else "?"
        cells = [scenario_id, f"`{target_context}`"]
        for condition in conditions:
            trials = matrix[scenario_id].get(condition, [])
            cells.append(_format_trial_outcomes(trials))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _format_trial_outcomes(trials: list[dict]) -> str:
    if not trials:
        return "-"
    tokens: list[str] = []
    for entry in trials:
        if entry["turn_2_passed"]:
            tokens.append("pass")
            continue
        if not entry["turn_3_repair_attempted"]:
            tokens.append("fail")
            continue
        if entry["turn_3_repair_passed"]:
            tokens.append("fail→repair-pass")
        else:
            tokens.append("fail→repair-fail")
    return ", ".join(tokens)


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
