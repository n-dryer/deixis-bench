# Glossary

Reference for the terms used throughout this repo. Sections collapse — click any heading to expand. Press <kbd>Ctrl</kbd>+<kbd>F</kbd> / <kbd>⌘</kbd>+<kbd>F</kbd> to search the page.

**Jump to:** [Project](#project--names) · [Task](#the-task) · [Dataset](#the-dataset) · [Model inputs](#model-inputs) · [Judging](#the-judge) · [Runner](#the-runner) · [Metrics](#metrics) · [Run outputs](#run-outputs) · [Files & layout](#files--layout)

---

## Project & names

<details>
<summary><strong>What the project is, what it's installed as, and what to call it</strong></summary>

### Wearable assistant
An AI assistant the user is actively engaging with for advice or coaching, typically on smart glasses, an ear-worn device, or a phone. Sees the world through a camera and microphone and replies with text or audio. The benchmark targets this product context, not background or passive assistants.

### Wearable Assistant Context Bench
The benchmark itself. Tests one specific failure mode: when the user's context changes between turns, does the model respond from the current situational evidence, or stay anchored to the prior context?

### `wearable-assistant-context-bench`
The package install name (kebab-case). Maps to the Python package `wearable_assistant_context_bench` (snake_case) at the top level of the repo.

### `wac-bench`
The console-script entry point registered by `pyproject.toml`. `wac-bench --help` works after `pip install -e .` or `uv sync`. Equivalent to `python -m wearable_assistant_context_bench.runner`.

</details>

---

## The task

<details>
<summary><strong>What the benchmark measures and how it relates to standard NLP terminology</strong></summary>

### Cross-turn reference resolution
The task the benchmark measures. Whether the model resolves the user's reference (their "this", "that", "it") to the current video frame instead of staying anchored to an earlier one.

### Deictic reference
A reference whose meaning depends on the situational context — gesture, gaze, position in the scene — rather than naming the thing directly. "This", "that", "the one over there", "what I'm holding" are all deictic. Resolving a deictic reference requires the perceptual frame; that's why the benchmark's `--no-camera` ablation drops accuracy so sharply. The term comes from linguistics (Greek *deixis*, "pointing").

### Scenario
One conversational unit. Three turns: optional pre-conversation frame, Turn 1 (initial state), Turn 2 (after a context shift), and a Turn 3 repair prompt fired only when `--enable-repair` is set. Stored as one JSON line in `data/scenarios.jsonl`.

### `scenario_id`
Unique identifier for a scenario. Format `sc-NN` for the main subset, `adv-NN` for the contrast subset.

### Test scenarios
Synonym for the published scenarios in `data/scenarios.jsonl`. The benchmark is an evaluation set — there is no train/val/test split; every scenario is for inference and labeling.

</details>

---

## The dataset

<details>
<summary><strong>Subsets, scenario fields, and how scenarios are categorized</strong></summary>

### Subset
A grouping of scenarios. The `subset` field on every scenario marks which one it belongs to. Two subsets: `main` and `contrast`. Filterable on the runner via `--subset {main,contrast}`.

### `main` (Main Subset)
The primary 50-scenario subset. Distribution is pinned: 12 / 8 / 6 / 6 / 5 / 5 / 4 / 4 across the 8 shift types.

### `contrast` (Contrast Subset)
A 20-scenario subset of distractor-rich minimal pairs. The earlier object or scene may still be visible at Turn 2.

### `shift_type`
The category of context shift between Turn 1 and Turn 2. One of eight values:

| Value | What changes |
|---|---|
| `object_in_hand` | User puts down one object, picks up another. |
| `object_state` | Same object, different state (cooking progress, paint drying). |
| `sequential_task` | Same task, user advances to a later step. |
| `location` | Whole scene changes (room, work area). |
| `object_in_view` | Attention shifts within a static scene. |
| `absent_referent` | Object the question is about is no longer in frame. |
| `screen_content` | Both turns look at a screen; the screen content changes. |
| `cross_session_reference` | Turn 2 asks about a state that existed before Turn 1. Requires `context_image`. |

### `target_context`
The grounding target a well-functioning assistant should pick. One of `current`, `prior`, `clarify`, `abstain`.

### `referent_complexity`
Internal complexity estimate. One of `single_referent`, `multi_referent`, `distractor_present`, `absent_referent`, `compound_shift`. Affects how hard the scenario is to disambiguate.

### `difficulty_tier`
Author-assigned difficulty. One of `easy`, `medium`, `hard`. The main subset pins distribution at 15 / 20 / 15.

### `time_gap_bucket`
Approximate time between Turn 1 and Turn 2: `seconds`, `minutes`, `hours`, or `next_day`.

### `pair_id`
Optional grouping key for contrast A/B pairs. When populated, the report computes pair consistency.

</details>

---

## Model inputs

<details>
<summary><strong>What the candidate sees: camera blocks, scene descriptions, turns, gold labels</strong></summary>

### Camera block
The runner injects scene descriptions into user messages as a tagged block:

```text
[Camera: <scene description>]
<user speech>
```

This represents what the wearable's vision channel would have produced alongside the user's spoken text. The candidate sees this format directly.

### Scene description
Text describing what a vision system would say about a video frame: shape, material, color, motion, position. Does not name the object directly. Stored in `turn_*_image` fields.

### `context_image`
A scene description for what was visible *before the conversation started*. Injected as a `[Camera: ...]` block before Turn 1, with no spoken text. Required for `cross_session_reference` scenarios; null otherwise.

### Turn 1, Turn 2, Turn 3
- **Turn 1**: establishes the starting state. Image and user speech.
- **Turn 2**: after the context shift. Image has changed; the user's question is deictic and does not announce the change. **The only turn that contributes to the primary metric.**
- **Turn 3**: opt-in repair (see `--enable-repair`). Sent only when the candidate misses Turn 2.

### Camera ablation (`--no-camera`)
Runs the benchmark with `[Camera: ...]` blocks stripped. Measures the contribution of the visual input by comparing scores against a normal run with the same model.

### `gold` field
Inline dict on each scenario record carrying four answer lists. Visible to the judge, never to the candidate.

| List | Used to score |
|---|---|
| `current_answers` | Responses grounded in the Turn 2 (current) context. Three vocabulary categories: object name, technique, state. |
| `prior_answers` | Responses anchored to the Turn 1 / `context_image` (prior) context. Same three-category structure. |
| `clarify_indicators` | `clarify`-target scenarios. |
| `abstain_indicators` | `abstain`-target scenarios. |

</details>

---

## The judge

<details>
<summary><strong>LLM-as-judge: labels, family resolution, and the prompt contract</strong></summary>

### LLM judge / LLM-as-judge
A second model that labels each Turn 2 (and Turn 3 when present) response with one of four labels: `current`, `prior`, `clarify`, `abstain`. Sees the user message, the scene descriptions, the gold answer lists, and a ground-truth section naming the actual objects in frame. Never sees `target_context` or `shift_type`, which would leak the answer.

### Judge label
The judge's verdict for one trial. One of `current`, `prior`, `clarify`, `abstain`. Stored as `selected_label` on the JudgeVerdict and as `turn_2_judge_label` / `turn_3_judge_label` on each transcript row. A trial is correct when the judge label equals the scenario's `target_context`.

### `JudgeVerdict`
Dataclass with `selected_label` and `rationale` (one-sentence justification). Returned by the judge for every trial.

### Judge family
The provider family the judge runs against: `claude`, `gemini`, or `openai`. Set by `--judge-family`.

### Cross-family judging (`--judge-family auto`)
Default behavior. Picks a judge from a different family than the candidate (Claude → Gemini, Gemini → OpenAI, OpenAI → Gemini).

### Shared judge (`--ranking-judge-family`)
Optional second judge held constant across all candidate runs in a comparison sweep.

### Judge prompt
The system prompt that defines the judge's task and JSON output contract. Hashed into every run's manifest. Lives in `wearable_assistant_context_bench/llm_judge.py`. Privileged-field constraint enforced by `tests/test_llm_judge.py`.

</details>

---

## The runner

<details>
<summary><strong>Adapters, the runner CLI, prompt conditions, and the optional repair turn</strong></summary>

### Adapter
A small wrapper that exposes a uniform `query(messages, system, config)` interface over a provider's chat API. The runner doesn't know which provider it's talking to; it only knows about adapters.

### `GeminiAdapter`
Native Google Gemini SDK transport. Used for bare Gemini model IDs (`gemini-2.5-flash`, etc.). File: `wearable_assistant_context_bench/gemini_adapter.py`.

### `LiteLLMAdapter`
LiteLLM-backed adapter. Handles Claude (via `openrouter/anthropic/...`), OpenAI, Hugging Face Inference Providers, DashScope, and any other provider-qualified model ID with a slash. File: `wearable_assistant_context_bench/litellm_adapter.py`.

### Model family
The provider behind a model id: `claude`, `gemini`, `openai`. Inferred from the id by `infer_candidate_family` in `llm_judge.py`. Hugging Face open-weights candidates return `None`, which forces an explicit `--judge-family` choice.

### Candidate model
The model under evaluation in a run. Selected by `--model <model_id>`.

### Runner
The orchestration code that loads scenarios, builds messages, calls the candidate adapter, asks the judge to label each response, and writes outputs. File: `wearable_assistant_context_bench/runner.py`. CLI entry point: `wac-bench`.

### Prompt condition
A system prompt variant applied uniformly to every scenario in a run. Three conditions ship in `data/prompt_conditions.json`: `baseline`, `condition_a`, `condition_b`. Every scenario is run against every condition.

### `baseline` (condition)
The default-comparison prompt condition. The primary metric is reported under this condition.

### Trial
One execution of `(scenario, condition, trial-index)`. `trials_per_cell` defaults to 1. Multiple trials are only meaningful at non-zero temperature.

### Repair turn
Opt-in via `--enable-repair`. When the candidate misses Turn 2, the runner sends a templated correction prompt and the judge labels the Turn 3 response. Off by default; the report omits the repair section when disabled.

### Repair prompt (named vs deictic)
The Turn 3 follow-up message. Two styles, selected by `--repair-style {named,deictic}`:
- **Named** (`turn_3_repair_prompt`, default): names the intended and the wrong objects explicitly. Best-case-recovery metric.
- **Deictic** (`turn_3_repair_prompt_deictic`): pure spatial / temporal pronouns ("this", "what I'm holding now"). Only populated for visible-referent `current`-target scenarios; falls back to named for the rest.

### `--enable-repair`
CLI flag that turns on the Turn 3 repair turn. Off by default.

</details>

---

## Metrics

<details>
<summary><strong>Primary metric, per-class recall, CIs, and the diagnostic rates</strong></summary>

### Primary metric
`mean(current_recall, prior_recall)` under the `baseline` prompt condition. These are class recall values (TP / (TP + FN)), not overall accuracy. With four judge labels, a trial is correct only when the judge label equals the scenario's `target_context`.

### Class recall
Per-class recall. `current_recall` = trials with `target_context == "current"` that the judge labeled `current`, divided by trials with `target_context == "current"`. Same for `prior_recall`.

### Mean recall
Mean of the per-class recalls. The headline number for a run.

### Wilson CI
Wilson score confidence interval for a binomial proportion. Used for per-class recall, per-subset recall, per-shift-type recall. Well-calibrated for small N and proportions near 0 or 1; unlike the Wald (normal) approximation, Wilson never overshoots [0, 1].

### Bootstrap CI
Non-parametric percentile bootstrap. Reported alongside Wilson on the primary metric (mean of two proportions, where Wilson is awkward).

### Per-subset recall
Mean recall sliced by `subset`. Lets you compare how a model holds up on `main` vs `contrast`.

### Per-shift-type recall
Pass rate sliced by `shift_type`. Surfaces categories the model handles well vs poorly.

### Contrast pair consistency
Among contrast-subset scenarios that share a `pair_id`, the share of pairs where every member trial passes. Reported only when `pair_id` metadata is populated.

### Hedging behavior
Three rates that surface how often the candidate refuses to commit:

| Rate | Definition |
|---|---|
| Clarification rate | Share of trials labeled `clarify`. |
| Abstention rate | Share of trials labeled `abstain`. |
| Coverage | `1 − (clarification + abstention)` — share of substantive responses. |

### Cohen's kappa
Inter-rater agreement statistic. Reported when a shared (`ranking`) judge is used, comparing primary-judge and shared-judge labels.

### Code-judge disagreement
Per-scenario count of trials where the deterministic code-signal scorer (`scoring.score_response`) implies a different label than the LLM judge picked. Diagnostic.

### Repair rate
Fraction of Turn 2 misses that pass on Turn 3. Reported only when `--enable-repair` was set.

</details>

---

## Run outputs

<details>
<summary><strong>The three artifacts every run produces</strong></summary>

### Run
One execution of the runner. Identified by its `output_dir`.

### `transcripts.jsonl`
Per-trial result dicts, one per line. Gitignored; regenerate by re-running the matching command.

### `findings.md`
Auto-generated Markdown report rendered by `wearable_assistant_context_bench.rendering.render_findings_markdown`. Includes the benchmark summary, per-class / per-subset / per-shift-type breakdowns, hedging behavior, the scenario × condition matrix, and the reproducibility manifest as a JSON code block.

### `summary.json`
Machine-readable companion to `findings.md`. Aggregate metrics + manifest in JSON for downstream tooling. Committed alongside `findings.md`; transcripts are not.

### Reproducibility manifest
A JSON block embedded in every `findings.md` (and a `benchmark_version` field plus the same fields in every `summary.json`). Names the benchmark version, content hashes (`scenarios_sha256`, `prompt_conditions_sha256`, `judge_prompt_sha256`), candidate / judge model IDs, trials, temperature, subset, and the runner's git commit. Two runs with matching hashes evaluate against the same content.

### Ablation
A category of run that holds everything constant except one variable to measure that variable's contribution.

</details>

---

## Reproducibility

<details>
<summary><strong>Versioning, the lockfile, and the validator</strong></summary>

### `MANIFEST.lock.json`
Static lockfile at `data/MANIFEST.lock.json`. Pins SHA256 hashes of `scenarios.jsonl`, `prompt_conditions.json`, and the judge-prompt template, plus `BENCHMARK_VERSION`. Hash drift without a coordinated version bump fails CI.

### Validator
`scripts/validate_scenarios.py`. Runs five programmatic checks: token leakage, object-name leakage, schema validation, cross-scenario duplication, and manifest-lock drift. Run by CI on every PR.

### `BENCHMARK_VERSION`
String constant in `wearable_assistant_context_bench/aggregation.py`. Currently `0.1.0`. Single source of truth for the benchmark version; matches `pyproject.toml:version`. Bumps coordinated with content changes; reported in every run's manifest.

</details>

---

## Files & layout

<details>
<summary><strong>Top-level directories and what each documentation file covers</strong></summary>

### Top-level directories

| Path | What's there |
|---|---|
| `wearable_assistant_context_bench/` | The single Python package. Adapters, judge, runner, scoring, report, statistics. |
| `data/` | Frozen content used at runtime: `scenarios.jsonl`, `prompt_conditions.json`, `config.json`, `MANIFEST.lock.json`, `README.md` (dataset card). |
| `runs/` | Published baseline run results. One subdirectory per run; each contains `findings.md` and `summary.json`. |
| `tests/` | Five test files mirroring module names: `test_adapters.py`, `test_llm_judge.py`, `test_metrics.py`, `test_runner.py`, `test_schema.py`, plus `conftest.py`. |
| `scripts/` | Two ongoing utilities: `validate_scenarios.py` (CI validator) and `regen_manifest_lock.py` (lockfile refresh). |
| `site/` | Gitignored. Holds presentation/marketing assets that live outside the benchmark surface. |

### Documentation files

| File | Purpose |
|---|---|
| [`README.md`](../README.md) | Project README. Quickstart and code layout. |
| [`docs/benchmark_spec.md`](benchmark_spec.md) | The benchmark specification — task, inputs, scoring, judge. |
| [`docs/schema.md`](schema.md) | The JSON Lines schema reference for `data/scenarios.jsonl`. Field-by-field type and meaning. |
| [`docs/scenario_authoring_rules.md`](scenario_authoring_rules.md) | Authoring rules for new scenarios + the validation checklist. |
| [`docs/api_keys.md`](api_keys.md) | Provider-specific environment-variable setup. |
| [`docs/running_open_weights.md`](running_open_weights.md) | Instructions for running open-weight Hugging Face models via LiteLLM. |
| [`data/README.md`](../data/README.md) | Dataset card (HuggingFace-style). Lives next to the data so GitHub auto-renders it when browsing `data/`. Covers structure, statistics, curation rationale, and reproduction commands. |
| [`CITATION.cff`](../CITATION.cff) | Machine-readable citation metadata. |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Release policy, setup, adapter-extension guide, code style. |

</details>
