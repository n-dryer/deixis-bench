# Changelog

All notable changes to this project are recorded in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once a public 1.0.0 is cut.

## [0.1.1]

### Added
- Scenario bank expanded from 70 to **166 scenarios** in a single unified
  set. The 96 new scenarios cover all 8 shift types with a roughly even
  per-category distribution (object_in_hand 21, object_state 22,
  sequential_task 18, location 22, object_in_view 21, absent_referent 21,
  screen_content 21, cross_session_reference 20).
- Audit infrastructure for the scenario bank:
  `wearable_assistant_context_bench/audit_rubric.py` (rule-based audit),
  `scripts/audit_scenarios.py` (runner with optional Gemini judge
  spot-check on disagreements), `scripts/build_cross_reference.py`,
  `tests/test_audit_rubric.py`, and `docs/audit_rubric.md`.
- `data/authoring_grades.json` per-scenario grades.
- Wheel-install support: `wearable_assistant_context_bench/data/`
  subpackage carries frozen copies of `config.json`,
  `prompt_conditions.json`, and `scenarios.jsonl` so
  `pip install wearable-assistant-context-bench` works without a source
  checkout. Source checkouts continue to use the top-level `data/` as the
  single source of truth.
- Python 3.14 added to the CI test matrix.
- `wheel-smoke` CI job: builds the wheel, installs it into a fresh venv
  with no source tree, and runs `wac-bench --help` to catch packaging
  regressions.
- SPDX license expression in `pyproject.toml`
  (`license = "MIT"` + `license-files = ["LICENSE"]`).

### Changed
- The `--subset` CLI flag has been removed. All 166 scenarios live in a
  single unified `subset = "main"` value; the runner always loads the
  full bank. Legacy `adv-NN` ids are kept stable for traceability.
- `referent_complexity` value `absent_referent` was renamed to
  `referent_offscreen` across all 166 scenarios to disambiguate it from
  the `shift_type` value of the same name. The `shift_type` value
  `absent_referent` is unchanged.
- Validator's pinned distribution updated to the new unified totals.
- `BENCHMARK_VERSION` (and the `pyproject.toml` package version) bumped
  from `0.1.0` to `0.1.1`.

### Fixed
- `runner.py` now falls back to `importlib.resources` when the source
  `data/` tree is absent (wheel-install path).

## [0.1.0]

Initial public release.

- Wearable Assistant Context Bench framework for cross-turn reference
  resolution evaluation across AI wearable assistants used actively
  for advice or coaching.
- 70-scenario bank split into a 50-scenario `main` subset and a
  20-scenario `contrast` pack.
- Three prompt conditions (`baseline`, `condition_a`, `condition_b`)
  with a SHA-256-pinned reproducibility manifest in
  `data/MANIFEST.lock.json`.
- Native Gemini SDK adapter and LiteLLM adapter (Claude via OpenRouter,
  OpenAI, Hugging Face).
- LLM-as-judge labeler with `current` / `prior` / `clarify` / `abstain`
  outputs and an optional cross-candidate ranking judge.
- Markdown findings report with per-class recall, per-shift-type and
  per-subset breakdowns, contrast-pair consistency, hedging rates,
  inter-judge agreement, and reproducibility manifest.
- CI: ruff + mypy lint and pytest matrix on Python 3.11 / 3.12 / 3.13.

[0.1.1]: https://github.com/n-dryer/wearable-assistant-context-bench/releases/tag/v0.1.1
[0.1.0]: https://github.com/n-dryer/wearable-assistant-context-bench/releases/tag/v0.1.0
