# Dataset Card: Wearable Assistant Context Bench

## Summary

Wearable Assistant Context Bench is a cross-turn reference-resolution
evaluation set for AI wearable assistants. Each task is a short
conversation where the user's visible situation changes between Turn 1
and Turn 2. The model must answer the Turn 2 question using the
intended scene context.

The active task bank is `data/tasks.jsonl`. It contains 166 tasks in a
single flat `main` task set. There is no train, validation, or test
split because this repo is an evaluation benchmark, not a training
dataset.

## Files

| File | Purpose |
|---|---|
| `tasks.jsonl` | Active task bank. |
| `prompt_conditions.json` | Prompt conditions applied to every task. |
| `config.json` | Default runner configuration. |
| `MANIFEST.lock.json` | Content hashes for reproducibility checks. |

## Fields

Each task has a stable `task_id`, context metadata, Turn 1 and Turn 2
scene-description text, user speech, and judge-only `reference_answers`.
See [`../docs/schema.md`](../docs/schema.md)
for the full field contract.

## Statistics

| Statistic | Value |
|---|---:|
| Total tasks | 166 |
| Task set | `main` |
| Domains | 16 |

### Shift-Type Distribution

| `shift_type` | Count |
|---|---:|
| `object_in_hand` | 22 |
| `object_state` | 22 |
| `sequential_task` | 17 |
| `location` | 22 |
| `object_in_view` | 21 |
| `absent_referent` | 21 |
| `screen_content` | 21 |
| `cross_session_reference` | 20 |

### Gold-Label Distribution

| `gold_label` | Count |
|---|---:|
| `current` | 46 |
| `prior` | 40 |
| `clarify` | 40 |
| `abstain` | 40 |

### Difficulty Distribution

| `difficulty` | Count |
|---|---:|
| `easy` | 30 |
| `medium` | 75 |
| `hard` | 61 |

## Validation

Run:

```bash
uv run python scripts/validate_tasks.py
```

The validator checks schema shape, task ID uniqueness, distribution
counts, token leakage, object-name leakage, near-duplication, and
manifest-lock drift.

## Running

```bash
uv run wac-bench --model <candidate_model_id> --output-dir runs/<run_name>
```

Add `--no-camera` to strip the `[Camera: ...]` scene-description
blocks (camera-channel ablation).

## Published runs

The runs that back the Results table in the top-level `README.md` are
checked in under `data/published-runs/`. Each run directory contains
the candidate transcripts and two judge summaries: one from a
within-family judge (Gemini 2.5 Flash Lite) and one from a cross-family
judge (`gpt-5-codex`).

| Path | Candidate | Within-family judge | Cross-family judge | Notes |
|---|---|---|---|---|
| `data/published-runs/baseline-flash/` | `gemini/gemini-2.5-flash` | `gemini/gemini-2.5-flash-lite` | `gpt-5-codex` | Default prompt, camera on |
| `data/published-runs/baseline-flash-lite/` | `gemini/gemini-2.5-flash-lite` | `gemini/gemini-2.5-flash-lite` | `gpt-5-codex` | Default prompt, camera on, same-family judge baseline |
| `data/published-runs/no-camera-flash-lite/` | `gemini/gemini-2.5-flash-lite` | `gemini/gemini-2.5-flash-lite` | `gpt-5-codex` | Camera channel stripped (`--no-camera`) |

Each directory contains:

| File | Purpose |
|---|---|
| `summary.json` | Aggregate metrics under the within-family (Gemini) judge, plus a `cross_judge_agreement` block with the Fleiss kappa between the two judges. |
| `summary-codex-judge.json` | Aggregate metrics under the cross-family (`gpt-5-codex`) judge, plus the per-trial `baseline_verdicts` table that pairs each judge's label with the gold label. |
| `transcripts.jsonl` | Per-trial candidate inputs, candidate responses, and judge labels. |
| `findings.md` | Human-readable per-class, per-shift-type, per-condition breakdown. |

### Confidence intervals and cross-judge agreement

Each metric in `summary.json` and `summary-codex-judge.json` carries a
`ci_95` field: a 95% percentile bootstrap interval over 1000 resamples
of the trial-level outcomes, drawn with `numpy.random.default_rng(42)`.
Per-class recalls bootstrap each gold class independently; the primary
`mean(current, prior)` recall bootstraps both classes jointly and
takes the mean of resampled per-class recalls. Hedging rates
(`clarify_rate`, `abstain_rate`, `coverage_rate`) bootstrap the
binary indicator across all baseline-condition trials. The `ci_method`
block at the bottom of each summary records the bootstrap parameters.

The `cross_judge_agreement` field on each `summary.json` reports the
Fleiss kappa between the two judges over the 166 paired baseline
trials, treating each judge as one rater across the four-category
label space (`current`, `prior`, `clarify`, `abstain`). Raw label
agreement is stored alongside as `raw_agreement`.

The bootstrap CIs and kappa are recomputed from the trial-level data
by `scripts/recompute_published_run_stats.py`.

Reproduce locally:

```bash
uv run wac-bench --model gemini/gemini-2.5-flash --judge-family gemini --judge-model gemini/gemini-2.5-flash-lite --output-dir runs/baseline-flash
uv run wac-bench --model gemini/gemini-2.5-flash-lite --judge-family gemini --judge-model gemini/gemini-2.5-flash-lite --output-dir runs/baseline-flash-lite
uv run wac-bench --model gemini/gemini-2.5-flash-lite --judge-family gemini --judge-model gemini/gemini-2.5-flash-lite --no-camera --output-dir runs/no-camera-flash-lite
```

`runs/` is the local scratch directory and is gitignored. Promote a
run into `data/published-runs/` when it should be cited.

### Reproducibility and known equivalences

`tests/test_published_runs.py` recomputes the SHA256 of
`data/tasks.jsonl`, `data/prompt_conditions.json`, and the live judge
prompt and compares each value against the SHAs recorded in every
published run's `findings.md` manifest. A run passes if its recorded
SHA matches the live SHA, or if the recorded SHA appears in
`data/published-runs/equivalent_input_sets.json` as content-equivalent
to the live SHA. Each equivalence entry records a `diff_summary`,
`introduced_in_commit`, and `verified_by` reviewer so the equivalence
can be audited; entries are added only when an input-file change is
verifiable as not affecting candidate prompts or judge inputs.

## License

MIT, matching the rest of the repository.
