"""Aggregation and Markdown rendering for benchmark results.

The runner produces one result dict per trial. This module rolls
those up into a findings report for the situated context-tracking
benchmark:

1. **Benchmark summary.** Headline primary score (mean of per-class
   recall under the default comparison condition), per-class recall
   for ``current`` and ``prior``, and a per-condition sensitivity row.
2. **Per-pack recall.** Mean recall sliced by pack (``bank`` /
   ``contrast``).
3. **Per change-type recall.** Mean recall sliced by ``change_type``.
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

Expected per-trial result dict keys:
    scenario_id (str)
    subset (str): "bank" or "contrast"
    pair_id (str | None)
    target_context (str): one of "current", "prior", "clarify", "abstain"
    change_type (str)
    condition (str)
    trial (int)
    turn_2_code_signals (dict)
    turn_2_judge_label (str)
    turn_2_passed (bool): judge_label == target_context
    turn_3_repair_attempted (bool)
    turn_3_repair_passed (bool | None)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from wearable_assistant_context_bench.statistics import wilson_ci

# 95% normal-distribution z-score for Wilson score interval
WILSON_Z_95: float = 1.959964


def wilson_interval(
    passed: int, total: int, z: float = WILSON_Z_95
) -> tuple[float, float, float] | None:
    """Wilson score interval for a binomial proportion as ``(rate, lo, hi)``.

    Thin tuple-returning wrapper over :func:`core.statistics.wilson_ci`
    for callers in this module that pre-date the dataclass API. New
    code should call ``wilson_ci`` directly.

    Returns ``None`` when ``total == 0``. The custom ``z`` argument is
    honored only when it differs from the 95% default; non-default
    confidence levels route through ``wilson_ci``.
    """
    if total <= 0:
        return None
    if z == WILSON_Z_95:
        ci = wilson_ci(passed, total)
        return ci.proportion, ci.lower, ci.upper
    # Custom z: derive a confidence level for wilson_ci.
    from statistics import NormalDist

    confidence = 2 * NormalDist().cdf(z) - 1
    ci = wilson_ci(passed, total, confidence=confidence)
    return ci.proportion, ci.lower, ci.upper


POLICIES: tuple[str, ...] = ("current", "prior", "clarify", "abstain")
SCORED_POLICIES: tuple[str, ...] = ("current", "prior")
CONDITIONS_ORDER: tuple[str, ...] = ("baseline", "condition_a", "condition_b")
AUXILIARY_POLICY_NOTE: str = "auxiliary; not included in the primary current/prior score"

BENCHMARK_NAME: str = "Wearable Assistant Context Bench"
BENCHMARK_VERSION: str = "0.1.0"
BENCHMARK_LABEL: str = (
    "situated context-tracking benchmark for multimodal AI assistants "
    "used actively for advice or coaching (wearable or handheld)"
)
DEFAULT_RANKING_CONDITION: str = "baseline"


@dataclass
class PassRateCell:
    """One cell of the per-policy-by-condition pass-rate grid.

    Attributes:
        passed: Trials in this cell where `turn_2_passed` was True.
        total: Total trials in this cell.
        primary_scored: True when the policy contributes to the primary
            balanced-accuracy metric (`current`, `prior`). Auxiliary
            policies still report rates when scenarios are present.
    """

    passed: int
    total: int
    primary_scored: bool

    @property
    def rate(self) -> float | None:
        if self.total == 0:
            return None
        return self.passed / self.total


@dataclass
class RepairRateCell:
    """One condition's simulated repair rate."""

    repaired: int
    failures: int

    @property
    def rate(self) -> float | None:
        if self.failures == 0:
            return None
        return self.repaired / self.failures


def _policies_with_scenarios(results: list[dict]) -> set[str]:
    return {r["target_context"] for r in results}


def per_policy_pass_rate_by_condition(
    results: list[dict],
) -> dict[str, dict[str, PassRateCell]]:
    """Group per-trial results into a policy x condition grid of pass rates."""
    observed_policies = _policies_with_scenarios(results)
    conditions = sorted({r["condition"] for r in results}, key=_condition_sort_key)
    grid: dict[str, dict[str, PassRateCell]] = {}
    for policy in POLICIES:
        grid[policy] = {}
        for condition in conditions:
            if policy not in observed_policies:
                grid[policy][condition] = PassRateCell(
                    passed=0, total=0, primary_scored=(policy in SCORED_POLICIES)
                )
                continue
            passed = 0
            total = 0
            for trial in results:
                if trial["target_context"] != policy:
                    continue
                if trial["condition"] != condition:
                    continue
                total += 1
                if bool(trial["turn_2_passed"]):
                    passed += 1
            grid[policy][condition] = PassRateCell(
                passed=passed,
                total=total,
                primary_scored=(policy in SCORED_POLICIES),
            )
    return grid


def class_recall_under_condition(
    results: list[dict],
    condition: str,
) -> dict[str, float | None]:
    """Compute per-class Turn 2 recall under one condition.

    These are recall values, not overall accuracy: with four judge
    labels (current/prior/clarify/abstain) a trial is "correct" only
    when ``judge_label == target_context``. The denominator is trials
    whose ``target_context`` equals the named policy (TP + FN); the
    numerator is the subset where ``turn_2_passed`` is True (TP).
    Clarify / abstain trials get their own denominator the same way.

    Returns a dict keyed on the **scored** policies (``prior``,
    ``current``) mapping to recall in [0, 1], or ``None`` if there are
    no trials in that class.
    """
    out: dict[str, float | None] = {}
    for policy in SCORED_POLICIES:
        total = 0
        passed = 0
        for trial in results:
            if trial["condition"] != condition:
                continue
            if trial["target_context"] != policy:
                continue
            total += 1
            if bool(trial["turn_2_passed"]):
                passed += 1
        out[policy] = (passed / total) if total > 0 else None
    return out


def class_recall_with_ci_under_condition(
    results: list[dict],
    condition: str,
) -> dict[str, tuple[float, float, float] | None]:
    """Per-class Turn 2 recall under one condition, with 95% Wilson CIs.

    Returns a dict keyed on ``current`` and ``prior`` mapping to
    ``(rate, lo, hi)`` tuples or ``None`` when no trials exist. The
    rate is class recall, not overall accuracy.
    """
    out: dict[str, tuple[float, float, float] | None] = {}
    for policy in SCORED_POLICIES:
        total = 0
        passed = 0
        for trial in results:
            if trial["condition"] != condition:
                continue
            if trial["target_context"] != policy:
                continue
            total += 1
            if bool(trial["turn_2_passed"]):
                passed += 1
        out[policy] = wilson_interval(passed, total)
    return out


def mean_recall_under_condition(
    results: list[dict],
    condition: str,
) -> float | None:
    """Primary benchmark score under a given default comparison condition.

    Defined as the mean of per-class recall across the two scored
    policies (``prior``, ``current``). Clarify / abstain trials
    contribute to their class's denominator as wrong answers (they
    never pass the target-policy match), matching the rule that they
    count as wrong for the primary score.
    """
    classes = class_recall_under_condition(results, condition)
    usable = [acc for acc in classes.values() if acc is not None]
    if not usable or len(usable) < len(classes):
        # All scored classes must have at least one trial for a score
        # to be defined. If one is missing, return None and let the
        # caller render that as "n/a" in the report.
        return None
    return sum(usable) / len(usable)


def mean_recall_with_ci_under_condition(
    results: list[dict],
    condition: str,
    z: float = WILSON_Z_95,
) -> tuple[float, float, float] | None:
    """Mean per-class recall with a 95% normal-approximation CI on the mean.

    The mean of two independent binomial proportions has variance
    ``(p1(1-p1)/n1 + p2(1-p2)/n2) / 4``. We use the normal
    approximation on that variance to bound the mean-recall point
    estimate. Per-class point estimates use the same formulas as
    :func:`class_recall_under_condition`.

    Returns ``(mean, lo, hi)`` or ``None`` if any scored class has no
    trials in this condition.
    """
    n_passed: dict[str, int] = {}
    n_total: dict[str, int] = {}
    for policy in SCORED_POLICIES:
        n_passed[policy] = 0
        n_total[policy] = 0
    for trial in results:
        if trial["condition"] != condition:
            continue
        policy = trial["target_context"]
        if policy not in SCORED_POLICIES:
            continue
        n_total[policy] += 1
        if bool(trial["turn_2_passed"]):
            n_passed[policy] += 1
    if any(n_total[p] == 0 for p in SCORED_POLICIES):
        return None
    rates = {p: n_passed[p] / n_total[p] for p in SCORED_POLICIES}
    mean = sum(rates.values()) / len(rates)
    var_sum = sum(rates[p] * (1.0 - rates[p]) / n_total[p] for p in SCORED_POLICIES)
    se = math.sqrt(var_sum) / len(SCORED_POLICIES)
    margin = z * se
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def mean_recall_with_bootstrap_ci_under_condition(
    results: list[dict],
    condition: str,
    n_iter: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float] | None:
    """Mean per-class recall with a non-parametric percentile bootstrap CI.

    Bootstraps the per-trial outcome vector for each scored class, then
    computes the mean of the resampled per-class recalls. Useful as a
    second opinion on the normal-approximation CI for small N.

    Returns ``(mean, lo, hi)`` or ``None`` if any scored class has no
    trials in this condition.
    """
    by_class: dict[str, list[float]] = {p: [] for p in SCORED_POLICIES}
    for trial in results:
        if trial["condition"] != condition:
            continue
        policy = trial["target_context"]
        if policy not in SCORED_POLICIES:
            continue
        by_class[policy].append(1.0 if bool(trial["turn_2_passed"]) else 0.0)
    if any(not by_class[p] for p in SCORED_POLICIES):
        return None
    import random

    rng = random.Random(seed)
    bootstraps: list[float] = []
    seqs = {p: list(by_class[p]) for p in SCORED_POLICIES}
    for _ in range(n_iter):
        means = []
        for p in SCORED_POLICIES:
            seq = seqs[p]
            n = len(seq)
            sample = [seq[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)
        bootstraps.append(sum(means) / len(means))
    bootstraps.sort()
    lo_idx = max(0, int(round(0.025 * n_iter)) - 1)
    hi_idx = min(n_iter - 1, int(round(0.975 * n_iter)) - 1)
    point = sum(by_class[p].count(1.0) / len(by_class[p]) for p in SCORED_POLICIES) / len(
        SCORED_POLICIES
    )
    return point, bootstraps[lo_idx], bootstraps[hi_idx]


def recall_by_subset(
    results: list[dict],
    condition: str,
) -> dict[str, tuple[float, float, float] | None]:
    """Mean per-class recall sliced by pack (``bank`` / ``contrast``).

    Returns a dict keyed on pack name. Each value is
    ``(mean, lo, hi)`` from the normal-approximation CI, or ``None``
    when one of the scored classes has no trials in that pack.
    """
    by_pack: dict[str, list[dict]] = defaultdict(list)
    for trial in results:
        pack = trial.get("subset") or "bank"
        by_pack[pack].append(trial)
    return {
        pack: mean_recall_with_ci_under_condition(trials, condition)
        for pack, trials in by_pack.items()
    }


def recall_by_change_type(
    results: list[dict],
    condition: str,
) -> dict[str, tuple[float, float, float] | None]:
    """Mean recall sliced by ``change_type``.

    For each cue type, treats every trial as a single binomial outcome
    (collapses across class). The Wilson CI for the per-cue pass rate
    is reported. Returns ``None`` for a cue type with zero trials in
    the requested condition.
    """
    by_cue: dict[str, list[dict]] = defaultdict(list)
    for trial in results:
        if trial["condition"] != condition:
            continue
        cue = trial.get("change_type") or "unknown"
        by_cue[cue].append(trial)
    out: dict[str, tuple[float, float, float] | None] = {}
    for cue, trials in by_cue.items():
        passed = sum(1 for t in trials if bool(t["turn_2_passed"]))
        out[cue] = wilson_interval(passed, len(trials))
    return out


def contrast_pair_consistency(
    results: list[dict],
    condition: str | None = None,
) -> dict[str, Any]:
    """Pair-consistency rate over the contrast pack.

    Groups contrast-pack trials by ``pair_id`` and computes the share
    of pairs where every member trial passed Turn 2. Trials with no
    ``pair_id`` are ignored. When ``condition`` is non-None, only
    trials in that condition contribute.

    Returns a dict with:
        ``pairs_evaluated`` (int)
        ``consistency_rate`` (float | None)
        ``ci`` (tuple[float, float] | None) — Wilson CI bounds
        ``note`` (str) — set when no pair_id metadata is present
    """
    pairs: dict[str, list[bool]] = defaultdict(list)
    for trial in results:
        if (trial.get("subset") or "bank") != "contrast":
            continue
        if condition is not None and trial["condition"] != condition:
            continue
        pair_id = trial.get("pair_id")
        if not pair_id:
            continue
        pairs[pair_id].append(bool(trial["turn_2_passed"]))
    if not pairs:
        return {
            "pairs_evaluated": 0,
            "consistency_rate": None,
            "ci": None,
            "note": ("no pair_id metadata in current contrast pack; metric not applicable"),
        }
    consistent = sum(1 for outcomes in pairs.values() if all(outcomes))
    total = len(pairs)
    triple = wilson_interval(consistent, total)
    ci = None if triple is None else (triple[1], triple[2])
    return {
        "pairs_evaluated": total,
        "consistency_rate": consistent / total,
        "ci": ci,
        "note": "",
    }


def clarify_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """Share of trials whose primary judge label is ``clarify``."""
    return _label_rate(results, condition, label="clarify")


def abstain_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """Share of trials whose primary judge label is ``abstain``."""
    return _label_rate(results, condition, label="abstain")


def coverage_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """``1 - (clarify_rate + abstain_rate)`` — share of substantive answers.

    Low coverage indicates excessive hedging. Reported as a Wilson CI
    over the binary "substantive" label.
    """
    total = 0
    substantive = 0
    for trial in results:
        if trial["condition"] != condition:
            continue
        total += 1
        label = trial.get("turn_2_judge_label")
        if label not in ("clarify", "abstain"):
            substantive += 1
    return wilson_interval(substantive, total)


def build_run_summary_dict(
    *,
    results: list[dict],
    manifest: dict[str, Any],
    run_label: str,
    ranking_condition: str = DEFAULT_RANKING_CONDITION,
) -> dict[str, Any]:
    """Build the small ``summary.json`` payload for a finished run.

    Companion to :func:`render_findings_markdown`. Produces the same
    aggregate metrics in machine-readable form so downstream tools can
    consume run results without parsing the Markdown report.

    Args:
        results: Per-trial result dicts (the runner's return value).
        manifest: Reproducibility manifest dict (the same one
            :func:`render_findings_markdown` embeds in findings.md).
        run_label: Short human-readable run identifier, typically the
            output directory name.
        ranking_condition: Condition used for the primary score.

    Returns:
        A JSON-serializable dict with primary score, per-class recall,
        per-pack recall, hedging rates, the benchmark version, and a
        small config snapshot.
    """

    def _ci(triple: tuple[float, float, float] | None) -> dict | None:
        if triple is None:
            return None
        rate, lo, hi = triple
        return {"point": rate, "ci_95": [lo, hi]}

    primary = mean_recall_with_ci_under_condition(results, ranking_condition)
    per_class = class_recall_with_ci_under_condition(results, ranking_condition)
    per_pack = recall_by_subset(results, ranking_condition)
    return {
        "run_label": run_label,
        "n_trials": len(results),
        "n_scenarios": len({r["scenario_id"] for r in results}),
        "candidate_model": manifest.get("candidate_model"),
        "judge_model": manifest.get("judge_model"),
        "judge_family": manifest.get("judge_family"),
        "ranking_condition": ranking_condition,
        "primary_score_mean_recall": _ci(primary),
        "per_class_recall": {policy: _ci(triple) for policy, triple in per_class.items()},
        "per_subset_recall": {pack: _ci(triple) for pack, triple in per_pack.items()},
        "clarify_rate": _ci(clarify_rate(results, ranking_condition)),
        "abstain_rate": _ci(abstain_rate(results, ranking_condition)),
        "coverage_rate": _ci(coverage_rate(results, ranking_condition)),
        "benchmark_version": manifest.get("benchmark_version"),
        "config_snapshot": {
            "trials": manifest.get("trials"),
            "temperature": manifest.get("temperature"),
            "subset": manifest.get("subset"),
            "no_camera": manifest.get("camera_injection") is False,
            "enable_repair": manifest.get("enable_repair"),
        },
    }


def _label_rate(
    results: list[dict], condition: str, label: str
) -> tuple[float, float, float] | None:
    total = 0
    matched = 0
    for trial in results:
        if trial["condition"] != condition:
            continue
        total += 1
        if trial.get("turn_2_judge_label") == label:
            matched += 1
    return wilson_interval(matched, total)


def simulated_repair_rate_by_condition(
    results: list[dict],
) -> dict[str, RepairRateCell]:
    """Compute simulated repair rate per condition."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for trial in results:
        by_condition[trial["condition"]].append(trial)

    out: dict[str, RepairRateCell] = {}
    for condition in sorted(by_condition.keys(), key=_condition_sort_key):
        failures = 0
        repaired = 0
        for trial in by_condition[condition]:
            if bool(trial["turn_2_passed"]):
                continue
            failures += 1
            if bool(trial.get("turn_3_repair_passed")):
                repaired += 1
        out[condition] = RepairRateCell(repaired=repaired, failures=failures)
    return out


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """Cohen's kappa for two equal-length sequences of categorical labels.

    Returns ``None`` when fewer than 2 paired observations are present
    or when expected agreement equals 1 (single-class degenerate case,
    where kappa is undefined).
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"cohens_kappa requires equal-length sequences; got {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n < 2:
        return None
    classes = set(labels_a) | set(labels_b)
    matches = sum(1 for a, b in zip(labels_a, labels_b, strict=False) if a == b)
    p_observed = matches / n
    p_expected = 0.0
    for c in classes:
        marginal_a = sum(1 for x in labels_a if x == c) / n
        marginal_b = sum(1 for x in labels_b if x == c) / n
        p_expected += marginal_a * marginal_b
    if p_expected >= 1.0:
        return None
    return (p_observed - p_expected) / (1.0 - p_expected)


def inter_judge_agreement_summary(
    results: list[dict],
) -> dict[str, Any] | None:
    """Compute Cohen's kappa across primary and ranking-judge labels.

    Returns ``None`` when no trials carry the optional
    ``turn_2_ranking_judge_label`` field. Otherwise returns a dict
    with ``kappa``, ``observed_agreement``, ``trials``, and per-class
    ``confusion`` counts.
    """
    paired_a: list[str] = []
    paired_b: list[str] = []
    for trial in results:
        primary = trial.get("turn_2_judge_label")
        ranking = trial.get("turn_2_ranking_judge_label")
        if primary is None or ranking is None:
            continue
        paired_a.append(str(primary))
        paired_b.append(str(ranking))
    if not paired_a:
        return None
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for a, b in zip(paired_a, paired_b, strict=False):
        confusion[(a, b)] += 1
    matches = sum(1 for a, b in zip(paired_a, paired_b, strict=False) if a == b)
    return {
        "kappa": cohens_kappa(paired_a, paired_b),
        "observed_agreement": matches / len(paired_a),
        "trials": len(paired_a),
        "confusion": {f"{a}->{b}": count for (a, b), count in confusion.items()},
    }


def inter_judge_disagreement_by_scenario(
    results: list[dict],
) -> dict[str, int]:
    """Per-scenario count of trials where the two judges disagreed.

    Only counts trials that carry both judge labels.
    """
    counts: dict[str, int] = defaultdict(int)
    for trial in results:
        primary = trial.get("turn_2_judge_label")
        ranking = trial.get("turn_2_ranking_judge_label")
        if primary is None or ranking is None:
            continue
        if primary != ranking:
            counts[trial["scenario_id"]] += 1
    return dict(counts)


def code_judge_disagreement_by_scenario(results: list[dict]) -> dict[str, int]:
    """Count trials where code signals imply a different policy than judge."""
    counts: dict[str, int] = defaultdict(int)
    scenario_ids = {r["scenario_id"] for r in results}
    for scenario_id in scenario_ids:
        counts[scenario_id] = 0
    for trial in results:
        code_policy = _code_implied_policy(trial.get("turn_2_code_signals") or {})
        if code_policy is None:
            continue
        if code_policy != trial.get("turn_2_judge_label"):
            counts[trial["scenario_id"]] += 1
    return dict(counts)


def scenario_by_condition_matrix(
    results: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    """Group per-trial outcomes into a scenario x condition grid."""
    grid: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for trial in results:
        grid[trial["scenario_id"]][trial["condition"]].append(
            {
                "trial": trial["trial"],
                "turn_2_passed": bool(trial["turn_2_passed"]),
                "turn_3_repair_attempted": bool(trial.get("turn_3_repair_attempted", False)),
                "turn_3_repair_passed": trial.get("turn_3_repair_passed"),
            }
        )
    for scenario in grid.values():
        for cell in scenario.values():
            cell.sort(key=lambda entry: entry["trial"])
    return {scenario_id: dict(cells) for scenario_id, cells in grid.items()}


REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "benchmark_version",
    "scenarios_sha256",
    "prompt_conditions_sha256",
    "candidate_model",
    "judge_model",
    "judge_family",
    "trials",
    "temperature",
    "ranking_condition",
    "timestamp_utc",
    "runner_git_commit",
    "random_seed",
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
        per-pack and per-change-type breakdowns, contrast pair
        consistency, the per-policy pass-rate grid, the scenario
        matrix, hedging behavior, and a reproducibility manifest block.
    """
    grid = per_policy_pass_rate_by_condition(results)
    repair = simulated_repair_rate_by_condition(results)
    disagreements = code_judge_disagreement_by_scenario(results)
    matrix = scenario_by_condition_matrix(results)
    conditions = _sorted_conditions(results)
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
    per_cue_recall_ci = recall_by_change_type(results, ranking_condition)
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
        "## Per change-type recall",
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

    rows = ["| Change type | Pass rate (95% CI) |", "| --- | --- |"]
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


def _code_implied_policy(signals: dict) -> str | None:
    """Map a Turn 2 code-signal dict to a single implied policy, if any."""
    if not signals:
        return None
    if signals.get("is_refusal") or signals.get("has_abstain"):
        return "abstain"
    if signals.get("has_clarify"):
        return "clarify"
    has_current = bool(signals.get("has_current"))
    has_prior = bool(signals.get("has_prior"))
    if has_current and not has_prior:
        return "current"
    if has_prior and not has_current:
        return "prior"
    return None


def _sorted_conditions(results: list[dict]) -> list[str]:
    observed = {r["condition"] for r in results}
    return sorted(observed, key=_condition_sort_key)


def _condition_sort_key(condition: str) -> tuple[int, str]:
    if condition in CONDITIONS_ORDER:
        return (CONDITIONS_ORDER.index(condition), condition)
    return (len(CONDITIONS_ORDER), condition)
