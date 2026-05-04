<!--
  Thanks for contributing. See CONTRIBUTING.md for the full policy.
  Keep PRs focused: one change per PR.
-->

## Summary

<!-- One or two sentences on what this changes and why. -->

## Scope check

Does this PR change any of the following? Tick all that apply.

- [ ] `data/scenarios.jsonl` (scenario text or inline gold labels)
- [ ] `data/prompt_conditions.json`
- [ ] Judge labels (`current`, `prior`, `clarify`, `abstain`)
- [ ] Primary scoring rule (`mean(current_recall, prior_recall)`)
- [ ] Default comparison condition (`baseline`)

If any box is ticked, this is a benchmark-semantics change. See the
[release policy](../CONTRIBUTING.md#release-policy) — those stay stable
between releases and require a coordinated `BENCHMARK_VERSION` bump
rather than an in-place edit.

## Test plan

- [ ] `uv run pytest -q` passes
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy wearable_assistant_context_bench` passes
- [ ] If touching the scenario set: `python scripts/validate_scenarios.py` passes
- [ ] If touching scoring or judge logic: a published baseline still produces matching aggregates

## Notes for the reviewer

<!-- Anything non-obvious about the approach, alternatives considered, or follow-ups deferred. -->
