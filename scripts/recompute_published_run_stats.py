"""Recompute published-run summary statistics.

For each run under ``data/published-runs/``:

- Replace normal-approximation / Wilson CIs in ``summary.json`` and
  ``summary-codex-judge.json`` with 95% percentile bootstrap CIs over
  1000 resamples (numpy ``default_rng(42)``).
- Add a top-level ``cross_judge_agreement`` field to ``summary.json``
  with the Fleiss kappa between the Gemini and Codex judge labels on
  the shared baseline trials.

The script is idempotent: re-running it produces byte-identical
JSON given the same inputs and seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_RUNS = REPO_ROOT / "data" / "published-runs"

BOOTSTRAP_N_ITER = 1000
BOOTSTRAP_SEED = 42
SCORED_POLICIES = ("current", "prior")
JUDGE_CATEGORIES = ("current", "prior", "clarify", "abstain")
CODEX_JUDGE_MODEL = "gpt-5-codex"


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_class_recall(
    outcomes: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = len(outcomes)
    point = float(outcomes.mean())
    idx = rng.integers(0, n, size=(BOOTSTRAP_N_ITER, n))
    samples = outcomes[idx].mean(axis=1)
    lo, hi = percentile_ci(samples)
    return point, lo, hi


def bootstrap_balanced_recall(
    outcomes_by_class: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[float, float, float]:
    point = float(np.mean([o.mean() for o in outcomes_by_class.values()]))
    means_per_iter = np.zeros(BOOTSTRAP_N_ITER)
    for policy in SCORED_POLICIES:
        outcomes = outcomes_by_class[policy]
        n = len(outcomes)
        idx = rng.integers(0, n, size=(BOOTSTRAP_N_ITER, n))
        means_per_iter += outcomes[idx].mean(axis=1)
    means_per_iter /= len(SCORED_POLICIES)
    lo, hi = percentile_ci(means_per_iter)
    return point, lo, hi


def bootstrap_rate(indicators: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    return bootstrap_class_recall(indicators, rng)


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Fleiss kappa from an N x K rater-count matrix.

    Each row is one item; each column is one category; cells are the
    number of raters that assigned that category. Every row must sum
    to the same n_raters.
    """
    n_items, n_categories = matrix.shape
    n_raters = int(matrix[0].sum())
    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j**2))
    row_agreement = (np.sum(matrix**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(row_agreement.mean())
    if p_e >= 1.0:
        return 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def load_baseline_trials(run_dir: Path) -> list[dict]:
    trials = []
    with (run_dir / "transcripts.jsonl").open() as fh:
        for line in fh:
            trial = json.loads(line)
            if trial["condition"] == "baseline":
                trials.append(trial)
    return trials


def gemini_outcomes(trials: list[dict]) -> dict:
    outcomes_by_class = {
        policy: np.array(
            [bool(t["turn_2_passed"]) for t in trials if t["gold_label"] == policy],
            dtype=float,
        )
        for policy in SCORED_POLICIES
    }
    labels = np.array([t["turn_2_judge_label"] for t in trials])
    return {
        "by_class": outcomes_by_class,
        "clarify": (labels == "clarify").astype(float),
        "abstain": (labels == "abstain").astype(float),
        "coverage": (~np.isin(labels, ["clarify", "abstain"])).astype(float),
    }


def codex_outcomes(verdicts: list[dict]) -> dict:
    outcomes_by_class = {
        policy: np.array(
            [v["codex_judge_label"] == policy for v in verdicts if v["gold_label"] == policy],
            dtype=float,
        )
        for policy in SCORED_POLICIES
    }
    labels = np.array([v["codex_judge_label"] for v in verdicts])
    return {
        "by_class": outcomes_by_class,
        "clarify": (labels == "clarify").astype(float),
        "abstain": (labels == "abstain").astype(float),
        "coverage": (~np.isin(labels, ["clarify", "abstain"])).astype(float),
    }


def update_metric_blocks(summary: dict, outcomes: dict) -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = bootstrap_balanced_recall(outcomes["by_class"], rng)
    summary["primary_score_mean_recall"] = {
        "point": primary[0],
        "ci_95": [primary[1], primary[2]],
    }
    for policy in SCORED_POLICIES:
        triple = bootstrap_class_recall(outcomes["by_class"][policy], rng)
        existing = summary["per_class_recall"].get(policy) or {}
        block = {"point": triple[0], "ci_95": [triple[1], triple[2]]}
        for k in ("tp", "fn", "n"):
            if k in existing:
                block[k] = existing[k]
        summary["per_class_recall"][policy] = block
    summary["per_task_set_recall"]["main"] = {
        "point": primary[0],
        "ci_95": [primary[1], primary[2]],
    }
    for field, key in (
        ("clarify_rate", "clarify"),
        ("abstain_rate", "abstain"),
        ("coverage_rate", "coverage"),
    ):
        triple = bootstrap_rate(outcomes[key], rng)
        summary[field] = {"point": triple[0], "ci_95": [triple[1], triple[2]]}
    summary["ci_method"] = {
        "type": "percentile_bootstrap",
        "n_iter": BOOTSTRAP_N_ITER,
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.random.default_rng",
    }


def compute_cross_judge_agreement(verdicts: list[dict]) -> dict:
    paired = [
        (v["gemini_judge_label"], v["codex_judge_label"])
        for v in verdicts
        if v.get("gemini_judge_label") and v.get("codex_judge_label")
    ]
    matrix = np.zeros((len(paired), len(JUDGE_CATEGORIES)), dtype=float)
    cat_index = {c: i for i, c in enumerate(JUDGE_CATEGORIES)}
    for row, (a, b) in enumerate(paired):
        matrix[row, cat_index[a]] += 1
        matrix[row, cat_index[b]] += 1
    kappa = fleiss_kappa(matrix)
    raw_agreement = sum(1 for a, b in paired if a == b) / len(paired)
    return {
        "fleiss_kappa": kappa,
        "raw_agreement": raw_agreement,
        "n_trials": len(paired),
        "judges": ["gemini/gemini-2.5-flash-lite", CODEX_JUDGE_MODEL],
        "categories": list(JUDGE_CATEGORIES),
    }


def process_run(run_dir: Path) -> None:
    gemini_path = run_dir / "summary.json"
    codex_path = run_dir / "summary-codex-judge.json"

    trials = load_baseline_trials(run_dir)
    gemini_summary = json.loads(gemini_path.read_text())
    update_metric_blocks(gemini_summary, gemini_outcomes(trials))

    codex_summary = None
    if codex_path.exists():
        codex_summary = json.loads(codex_path.read_text())
        if codex_summary.get("judge_model") == "codex-own-model":
            codex_summary["judge_model"] = CODEX_JUDGE_MODEL
        update_metric_blocks(codex_summary, codex_outcomes(codex_summary["baseline_verdicts"]))
        gemini_summary["cross_judge_agreement"] = compute_cross_judge_agreement(
            codex_summary["baseline_verdicts"]
        )

    gemini_path.write_text(json.dumps(gemini_summary, indent=2) + "\n")
    if codex_summary is not None:
        codex_path.write_text(json.dumps(codex_summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=PUBLISHED_RUNS)
    args = parser.parse_args(argv)
    for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        process_run(run_dir)
        print(f"updated {run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
