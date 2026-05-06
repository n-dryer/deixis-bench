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
| `object_in_hand` | 21 |
| `object_state` | 22 |
| `sequential_task` | 18 |
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

## License

MIT, matching the rest of the repository.
