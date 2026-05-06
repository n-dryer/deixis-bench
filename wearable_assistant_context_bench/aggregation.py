"""Metric and aggregation surface for benchmark results.

The runner produces one result dict per trial. Functions here roll
those up into per-class recall, per-task-set and per-shift-type
breakdowns, hedging rates, simulated
repair rates, inter-judge agreement, and the task x condition
matrix used by the Markdown renderer in
:mod:`wearable_assistant_context_bench.rendering`.

Expected per-trial result dict keys:
    task_id (str)
    task_set (str): always "main" in the unified task set
    gold_label (str): one of "current", "prior", "clarify", "abstain"
    shift_type (str)
    condition (str)
    trial (int)
    turn_2_code_signals (dict)
    turn_2_judge_label (str)
    turn_2_passed (bool): judge_label == gold_label
    turn_3_repair_attempted (bool)
    turn_3_repair_passed (bool | None)
"""

from __future__ import annotations

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

    Thin tuple-returning wrapper over
    :func:`wearable_assistant_context_bench.statistics.wilson_ci`
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
CONDITIONS_ORDER: tuple[str, ...] = ("baseline", "context_selection_instruction", "pre_answer_context_scaffold")
AUXILIARY_POLICY_NOTE: str = "auxiliary; not included in the primary current/prior score"

BENCHMARK_NAME: str = "Wearable Assistant Context Bench"
BENCHMARK_VERSION: str = "0.1.0a0"
BENCHMARK_LABEL: str = (
    "situated context-tracking benchmark for AI wearable assistants "
    "used actively for advice or coaching"
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
            policies still report rates when tasks are present.
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


def _policies_with_tasks(results: list[dict]) -> set[str]:
    return {r["gold_label"] for r in results}


def per_policy_pass_rate_by_condition(
    results: list[dict],
) -> dict[str, dict[str, PassRateCell]]:
    """Group per-trial results into a policy x condition grid of pass rates."""
    observed_policies = _policies_with_tasks(results)
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
                if trial["gold_label"] != policy:
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
    when ``judge_label == gold_label``. The denominator is trials
    whose ``gold_label`` equals the named policy (TP + FN); the
    numerator is the portion of those trials where ``turn_2_passed`` is
    True (TP).
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
            if trial["gold_label"] != policy:
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
            if trial["gold_label"] != policy:
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
    never pass the policy-label match), matching the rule that they
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
        policy = trial["gold_label"]
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
        policy = trial["gold_label"]
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


def recall_by_task_set(
    results: list[dict],
    condition: str,
) -> dict[str, tuple[float, float, float] | None]:
    """Mean per-class recall sliced by ``task_set``.

    Returns a dict keyed on task_set value. Each value is
    ``(mean, lo, hi)`` from the normal-approximation CI, or ``None``
    when one of the scored classes has no trials. In the unified
    task set, all real trials live in the ``main`` bucket; the
    function still partitions correctly if a future split is added.
    """
    by_task_set: dict[str, list[dict]] = defaultdict(list)
    for trial in results:
        task_set_value = trial.get("task_set") or "main"
        by_task_set[task_set_value].append(trial)
    return {
        task_set_value: mean_recall_with_ci_under_condition(trials, condition)
        for task_set_value, trials in by_task_set.items()
    }


def recall_by_shift_type(
    results: list[dict],
    condition: str,
) -> dict[str, tuple[float, float, float] | None]:
    """Mean recall sliced by ``shift_type``.

    For each shift type, treats every trial as a single binomial
    outcome (collapses across class). The Wilson CI for the
    per-shift-type pass rate is reported. Returns ``None`` for a
    shift type with zero trials in the requested condition.
    """
    by_shift: dict[str, list[dict]] = defaultdict(list)
    for trial in results:
        if trial["condition"] != condition:
            continue
        shift = trial.get("shift_type") or "unknown"
        by_shift[shift].append(trial)
    out: dict[str, tuple[float, float, float] | None] = {}
    for shift, trials in by_shift.items():
        passed = sum(1 for t in trials if bool(t["turn_2_passed"]))
        out[shift] = wilson_interval(passed, len(trials))
    return out


def clarify_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """Share of trials whose primary judge label is ``clarify``."""
    return _label_rate(results, condition, label="clarify")


def abstain_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """Share of trials whose primary judge label is ``abstain``."""
    return _label_rate(results, condition, label="abstain")


def coverage_rate(results: list[dict], condition: str) -> tuple[float, float, float] | None:
    """``1 - (clarify_rate + abstain_rate)`` - share of substantive answers.

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
        per-task-set recall, hedging rates, the benchmark version, and a
        small config snapshot.
    """

    def _ci(triple: tuple[float, float, float] | None) -> dict | None:
        if triple is None:
            return None
        rate, lo, hi = triple
        return {"point": rate, "ci_95": [lo, hi]}

    primary = mean_recall_with_ci_under_condition(results, ranking_condition)
    per_class = class_recall_with_ci_under_condition(results, ranking_condition)
    per_task_set = recall_by_task_set(results, ranking_condition)
    return {
        "run_label": run_label,
        "n_trials": len(results),
        "n_tasks": len({r["task_id"] for r in results}),
        "candidate_model": manifest.get("candidate_model"),
        "judge_model": manifest.get("judge_model"),
        "judge_family": manifest.get("judge_family"),
        "ranking_condition": ranking_condition,
        "primary_score_mean_recall": _ci(primary),
        "per_class_recall": {policy: _ci(triple) for policy, triple in per_class.items()},
        "per_task_set_recall": {
            task_set_value: _ci(triple)
            for task_set_value, triple in per_task_set.items()
        },
        "clarify_rate": _ci(clarify_rate(results, ranking_condition)),
        "abstain_rate": _ci(abstain_rate(results, ranking_condition)),
        "coverage_rate": _ci(coverage_rate(results, ranking_condition)),
        "benchmark_version": manifest.get("benchmark_version"),
        "config_snapshot": {
            "trials": manifest.get("trials"),
            "temperature": manifest.get("temperature"),
            "task_set": manifest.get("task_set"),
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


def inter_judge_disagreement_by_task(
    results: list[dict],
) -> dict[str, int]:
    """Per-task count of trials where the two judges disagreed.

    Only counts trials that carry both judge labels.
    """
    counts: dict[str, int] = defaultdict(int)
    for trial in results:
        primary = trial.get("turn_2_judge_label")
        ranking = trial.get("turn_2_ranking_judge_label")
        if primary is None or ranking is None:
            continue
        if primary != ranking:
            counts[trial["task_id"]] += 1
    return dict(counts)


def code_judge_disagreement_by_task(results: list[dict]) -> dict[str, int]:
    """Count trials where code signals imply a different policy than judge."""
    counts: dict[str, int] = defaultdict(int)
    task_ids = {r["task_id"] for r in results}
    for task_id in task_ids:
        counts[task_id] = 0
    for trial in results:
        code_policy = _code_implied_policy(trial.get("turn_2_code_signals") or {})
        if code_policy is None:
            continue
        if code_policy != trial.get("turn_2_judge_label"):
            counts[trial["task_id"]] += 1
    return dict(counts)


def task_by_condition_matrix(
    results: list[dict],
) -> dict[str, dict[str, list[dict]]]:
    """Group per-trial outcomes into a task x condition grid."""
    grid: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for trial in results:
        grid[trial["task_id"]][trial["condition"]].append(
            {
                "trial": trial["trial"],
                "turn_2_passed": bool(trial["turn_2_passed"]),
                "turn_3_repair_attempted": bool(trial.get("turn_3_repair_attempted", False)),
                "turn_3_repair_passed": trial.get("turn_3_repair_passed"),
            }
        )
    for task in grid.values():
        for cell in task.values():
            cell.sort(key=lambda entry: entry["trial"])
    return {task_id: dict(cells) for task_id, cells in grid.items()}


REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "benchmark_version",
    "tasks_sha256",
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


def sorted_conditions(results: list[dict]) -> list[str]:
    observed = {r["condition"] for r in results}
    return sorted(observed, key=_condition_sort_key)


def _condition_sort_key(condition: str) -> tuple[int, str]:
    if condition in CONDITIONS_ORDER:
        return (CONDITIONS_ORDER.index(condition), condition)
    return (len(CONDITIONS_ORDER), condition)
