# Scripts

Helper scripts for scenario authoring and lockfile management. Both
are standalone — they don't import the package — and are run from the
repo root.

## `validate_scenarios.py`

Runs the programmatic checks against `data/scenarios.jsonl`,
`data/prompt_conditions.json`, and `data/MANIFEST.lock.json`:

- token leakage
- object-name leakage in scene descriptions
- schema validation
- cross-scenario duplication
- lockfile drift

```bash
python scripts/validate_scenarios.py          # human-readable output
python scripts/validate_scenarios.py --json   # machine-readable output
```

Exit code is `0` if all checks pass, `1` otherwise. CI runs this on
every PR. Run it locally before opening a PR that touches the
scenario bank, prompt conditions, or judge prompt.

The two semantic checks (human identification of scene descriptions,
semantic-leakage isolation) are LLM-driven and are run during scenario
authoring rather than in CI. Authoring details live in
[`docs/scenario_authoring_rules.md`](../docs/scenario_authoring_rules.md).

## `regen_manifest_lock.py`

Regenerates `data/MANIFEST.lock.json`, which pins SHA-256 hashes of
the scenario bank, prompt conditions, and judge-prompt template
alongside `BENCHMARK_VERSION`.

```bash
python scripts/regen_manifest_lock.py
```

Run this only after a coordinated content change has been merged
(scenario edits, prompt-condition edits, or a judge-prompt revision)
together with the corresponding version bump in code. The validator's
lockfile-drift check fails CI if hashes change without a coordinated
bump, so this script is the only sanctioned way to update the
lockfile.
