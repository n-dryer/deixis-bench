# Schema Reference

`data/tasks.jsonl` is the active task bank. It is JSON Lines: one task
object per line. The same file is packaged under
`wearable_assistant_context_bench/data/tasks.jsonl`.

The candidate model sees user speech plus scene-description text. The
judge additionally receives reference answer lists and a judge-only
ground-truth context. The candidate never sees `gold_label`,
`shift_type`, `notes`, or `reference_answers`.

## Task Fields

| Field | Type | Required | Description |
|---|---:|---:|---|
| `task_id` | string | yes | Stable ID in `task-NNN` format. |
| `task_set` | string | yes | Always `main` for this pre-release flat task set. |
| `gold_label` | enum | yes | Correct grounding label: `current`, `prior`, `clarify`, or `abstain`. |
| `shift_type` | enum | yes | Context-shift category. |
| `domain` | string | yes | Activity area such as `kitchen`, `workshop`, or `finance`. |
| `referent_complexity` | enum | yes | Complexity tag: `single_referent`, `multi_referent`, `distractor_present`, `referent_offscreen`, or `compound_shift`. |
| `difficulty` | enum | yes | `easy`, `medium`, or `hard`. |
| `time_gap_bucket` | enum or null | no | Approximate gap: `seconds`, `minutes`, `hours`, or `next_day`. |
| `pre_turn_context_scene_description` | string or null | yes | Optional scene-description text from before Turn 1. Required for `cross_session_reference` tasks. |
| `turn_1_scene_description` | string | yes | Scene-description text paired with Turn 1. |
| `turn_1_user` | string | yes | Turn 1 user speech. |
| `turn_2_scene_description` | string | yes | Scene-description text paired with the scored Turn 2 question. |
| `turn_2_user` | string | yes | Scored user question. |
| `reference_answers` | object | yes | Judge-only answer lists. |
| `notes` | string | no | Authoring notes, never sent to the candidate or judge prompt. |

## Label Values

| `gold_label` | Meaning |
|---|---|
| `current` | A good answer uses the Turn 2 scene. |
| `prior` | A good answer uses an earlier scene. |
| `clarify` | The available context is ambiguous, so the assistant should ask a clarifying question. |
| `abstain` | The needed information is not visible or otherwise available, so the assistant should avoid guessing. |

## Shift Types

| `shift_type` | Meaning |
|---|---|
| `object_in_hand` | User puts down one object and picks up another. |
| `object_state` | Same object, changed state. |
| `sequential_task` | Same workflow, later step. |
| `location` | User moves to a different scene. |
| `object_in_view` | Attention shifts to another visible object. |
| `absent_referent` | The referenced object is no longer visible. |
| `screen_content` | The visible screen content changes. |
| `cross_session_reference` | Turn 2 asks about a pre-Turn-1 state. |

When a shift could plausibly fit more than one category, follow the
precedence order in
[`task_authoring.md`](task_authoring.md#choosing-between-shift_type-values).
The short version: pick the most specific category. `sequential_task`
is the last resort — only use it when the same surface and object
remain in frame and only the operation has changed.

## Reference Answers

`reference_answers` contains four judge-only lists:

| Field | Meaning |
|---|---|
| `current_answers` | Tokens and phrases that indicate grounding in the Turn 2 scene. |
| `prior_answers` | Tokens and phrases that indicate grounding in an earlier scene. |
| `clarify_indicators` | Phrases that indicate a clarifying question. |
| `abstain_indicators` | Phrases that indicate refusal or inability to answer from available evidence. |

For `current_answers` and `prior_answers`, each non-empty list should
include object-name vocabulary, task-specific action vocabulary, and
state or condition vocabulary. This lets the judge recognize correct
answers that describe the object without using one exact phrase.

## Validation

Run:

```bash
uv run python scripts/validate_tasks.py
```

The validator checks schema shape, task ID uniqueness, label and
metadata enums, object-name leakage in scene descriptions, token
leakage in user speech, near-duplication, distribution counts, and
manifest-lock drift.
