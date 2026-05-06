# Contributing

Thanks for your interest in this benchmark.

## Scope

This benchmark tests whether AI wearable assistants update to current
context instead of staying anchored to prior context. It is narrow on
purpose. Contributions that fit that scope are welcome. Contributions
that broaden the benchmark into general assistant evaluation, coaching
quality, or domain expertise are not.

## Release policy

The following stay stable across patch releases so cross-model
comparisons remain valid:

- `data/tasks.jsonl`: task text + inline reference_answers labels
- `data/prompt_conditions.json`: prompt conditions
- The four judge labels (`current`, `prior`, `clarify`, `abstain`)
- The primary scoring rule: `mean(current_recall, prior_recall)`
  under `baseline` (class recall, not overall accuracy)
- The default comparison condition (`baseline`)

Edits that change task meaning, answer-key vocabulary, prompt
text, or scoring semantics are out of scope after the tag.

## What is welcome at any time

- Bug fixes in the runner, adapters, judge, scoring, or report
- New candidate-model adapter support that preserves benchmark
  semantics
- Documentation improvements, typo fixes, broken-link fixes
- Test coverage for existing behavior
- Reproducibility improvements (better manifest fields, cache hygiene,
  run-output clarity)
- Validator improvements (new programmatic checks, better diagnostics)

## How to add a new model adapter

Adapters live in [`wearable_assistant_context_bench/`](wearable_assistant_context_bench). The existing examples are:

- `wearable_assistant_context_bench/gemini_adapter.py`: native Google Gemini SDK adapter
- `wearable_assistant_context_bench/litellm_adapter.py`: LiteLLM-backed adapter for Claude, OpenAI, OpenRouter, and any provider-qualified model IDs

Each adapter exposes a `query(messages, system, config)` method
returning a string. To add a new adapter:

1. Implement the adapter in a new module under `wearable_assistant_context_bench/`.
2. Add tests under `tests/` that stub the underlying client and
   exercise the adapter contract (the existing adapter tests are
   templates).
3. Update `_build_adapter` in `wearable_assistant_context_bench/runner.py` to dispatch to the new family.
4. Update `infer_candidate_family` and the cross-family map in
   `wearable_assistant_context_bench/llm_judge.py` if the new family should participate in
   `--judge-family auto` resolution.

If your adapter is for a routing layer that supports multiple
families (like LiteLLM), keep family detection in
`infer_candidate_family` and reuse the existing `LiteLLMJudgeAdapter`.

## How to write a task

Tasks are written from scratch following the rules in
[`docs/task_authoring.md`](docs/task_authoring.md).
That document is the source of truth for user-message,
scene-description, and reference-answer rules, and it includes the full
validation checklist.

Quick summary:

- **User message** (`turn_*_user`): natural, deictic. No object
  names, no property descriptions, no shift announcements, no answer
  vocabulary.
- **Scene description** (`turn_*_scene_description`): scene-level features only.
  No object names, no functional labels, no technique evaluation.
  Detailed enough that a fresh reader can identify the object with
  high confidence.
- **Gold answers** (`reference_answers.*`): judge-only. Object names permitted.
  Each `current_answers` and `prior_answers` list must cover three
  vocabulary categories: object name, technique vocabulary, state
  descriptors.

New tasks are not accepted into the frozen main task_set.
Authoring rules are published as contributor reference.

## Validation

Before submitting any change to tasks or answer keys, run:

```bash
python scripts/validate_tasks.py
```

This runs five programmatic checks (token leakage, object name
leakage, schema validation, cross-task duplication, and lockfile
drift) over the task set. Two additional semantic checks
(human identification of image descriptions, semantic leakage
isolation test) run during task authoring rather than in CI.

The full 10-point validation checklist is in
[`docs/task_authoring.md`](docs/task_authoring.md).

### Static asset lockfile

`data/MANIFEST.lock.json` pins SHA256 hashes of the task set,
prompt conditions, and the judge-prompt template alongside
`BENCHMARK_VERSION`. The validator's lockfile check fails if any of
those drift. After a content change (with a corresponding
`BENCHMARK_VERSION` bump in code), regenerate the lockfile:

```bash
python scripts/regen_manifest_lock.py
```

This is the only sanctioned way to update the lockfile.

## Setup

With [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync --extra dev
uv run pytest -q
```

With pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

CI tests Python 3.11, 3.12, and 3.13.

## Test requirements

Every PR must pass:

```bash
python -m pytest tests/ -q
python scripts/validate_tasks.py
```

The test suite stubs candidate models and the judge so it works
without API access. CI runs both commands on every PR.

## Code style

- Python 3.11+ with type hints on every public function
- Docstrings on every public function and module-level docstrings on
  files in `wearable_assistant_context_bench/`
- Dataclasses for structured payloads where they improve readability
- No bare `except:` clauses
- Preserve the runner CLI contract unless a change is explicitly
  needed and reviewed

## Build / packaging

[`pyproject.toml`](pyproject.toml) is the single source of truth for
runtime and dev dependencies. The runtime deps are pinned by exact
version under `[project.dependencies]`; pytest sits under
`[project.optional-dependencies]` `dev`.

## PR process

1. Open an issue first for non-trivial changes.
2. Fork and branch from `main`.
3. Make your change. Keep commits atomic and descriptive.
4. Run the full test suite and the validator locally.
5. Open a pull request against `main` and reference the issue.
6. CI must pass before review.

When reporting a runner or judge bug, include:

- The exact CLI invocation
- The manifest block from `findings.md`, or the full file
- The candidate model and judge model IDs
- The Python version
