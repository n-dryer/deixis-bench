# Schema Reference

Field definitions for `data/scenarios.jsonl` (one JSON
object per line, with inline gold labels under the `gold` field).

The benchmark uses a split between what the candidate model sees and what the judge sees. The candidate model sees two
of these channels: the audio (user speech, represented as text
transcripts, not raw audio) and the video (scene descriptions,
represented as scene-description text, not real video frames). The
third part, the gold answer keys, is visible only to the judge.

For the rules that govern how each field is written, see
[scenario_authoring_rules.md](scenario_authoring_rules.md).

---

## scenarios.jsonl

JSON Lines: one scenario object per line.

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| `scenario_id` | string | yes | Unique identifier. Format `sc-NN` or `adv-NN`. The two prefixes are historical (the `adv-NN` rows were authored as a separately-tagged "contrast" subset; that subset was retired in favor of a single unified bank). Both forms are first-class scenarios in the bank. | `"sc-01"` |
| `pair_id` | string or null | no | Optional grouping key reserved for a future paired-twin contrast companion. Currently null in all rows. The runner does not use it today. | `null` |
| `gold` | object | yes | Inline gold-label dict. See "gold field" below. Replaces the legacy `expected_answers.json` join. | `{"current_answers": [...], ...}` |
| `target_context` | enum | yes | The correct grounding target for a well-functioning assistant. One of `current`, `prior`, `clarify`, `abstain`. | `"current"` |
| `change_type` | enum | yes | The category of context shift between Turn 1 and Turn 2 (the shift type). See list below. | `"object_in_hand"` |
| `activity_domain` | string | yes | Domain tag (e.g., `workshop`, `kitchen`, `garden`). Used for coverage reporting. | `"workshop"` |
| `referent_complexity` | enum | yes | Internal complexity estimate. One of `single_referent`, `multi_referent`, `distractor_present`, `referent_offscreen`. The `referent_offscreen` value covers cases where the target object isn't in the current view (renamed from the prior `absent_referent` value to avoid collision with the `change_type` value of the same name). | `"single_referent"` |
| `difficulty_tier` | enum | yes | Internal difficulty estimate. One of `easy`, `medium`, `hard`. | `"medium"` |
| `time_gap_bucket` | enum or null | no | Approximate time between Turn 1 and Turn 2. One of `seconds`, `minutes`, `hours`, `next_day`. Null means the gap is short enough to be considered the same continuous segment of activity (no narrative time skip between turns); use this when none of the four buckets meaningfully applies. | `"seconds"` |
| `context_image` | string or null | yes | Video frame description of what was visible **before the conversation started**. Null when `turn_1_image` already establishes the starting state. Required for scenarios where the user asks about a state from before Turn 1. | `null` |
| `turn_1_image` | string or null | yes | Video frame description at the moment Turn 1 is spoken. A scene description; describes physical properties without naming the object. | `"Hand resting on a slim metal handle..."` |
| `turn_1_user` | string | yes | First user message. Natural speech only. Must not narrate visible objects or evaluate technique. | `"How do I get more torque on this?"` |
| `turn_2_image` | string or null | yes | Video frame description at the moment Turn 2 is spoken. Different from `turn_1_image` (this is where the context shift becomes visible). | `"Hand wrapped around a wooden handle..."` |
| `turn_2_user` | string | yes | Second user message after the context change. Natural follow-up. Must not announce the shift. | `"Am I doing this right?"` |
| `turn_3_repair_prompt` | string | yes | Named repair prompt fired after a Turn 2 miss. Maximally specific user correction that names both the intended and the wrong objects. | `"I mean the hammer I'm holding now, not the screwdriver from before."` |
| `turn_3_repair_prompt_deictic` | string or null | no | Deictic-only repair prompt for visible-referent `current`-target scenarios. Used when the runner is invoked with `--repair-style deictic`. Pure spatial/temporal pronouns ("this", "what I'm holding now") with no object names. **Null/non-null contract (CI-enforced):** non-null exactly when `target_context == "current"` AND `change_type` is not in `{absent_referent, cross_session_reference}`; null otherwise. The runner falls back to the named anchor whenever this field is null. | `"I mean this thing in my hand right now."` |
| `notes` | string | no | Authoring commentary. Not used by the runner. | `"Object swap mid-task; Turn 2 deictic."` |

### target_context values

| Value | Meaning |
|---|---|
| `current` | The correct answer refers to what the video shows right now (Turn 2 frame). |
| `prior` | The correct answer refers to something from an earlier scene (Turn 1 or `context_image`). |
| `clarify` | The question is ambiguous given the available context; the assistant should ask for clarification rather than guessing. |
| `abstain` | The needed information is not present in the context; the assistant should decline to answer rather than hallucinating. |

### change_type values

The eight shift-type categories of context shift. Each scenario fits
exactly one. (In prose throughout the docs we call these "shift
types"; the JSON field name remains `change_type`.)

| Value | Description |
|---|---|
| `object_in_hand` | User puts down one object, picks up another. The video shows a different object in the user's hand. |
| `object_state` | Same object, different state (cooking progress, paint drying, etc.). |
| `sequential_task` | Same task, the user has progressed to a later step. |
| `location` | Whole scene changes; user moves to a different room or work area. |
| `object_in_view` | The video stays roughly in place; the user's attention has shifted to a different object visible in the scene. |
| `absent_referent` | The object the question is about is no longer in frame. |
| `screen_content` | Both Turn 1 and Turn 2 are looking at a screen; the screen content has changed. |
| `cross_session_reference` | Requires `context_image`; Turn 2 asks about a state that existed before Turn 1. |

---

## gold field (inline on each scenario)

Each scenario carries its gold labels inline under the `gold` key.
**These labels are judge-only.** The candidate model never sees them.

| Field | Type | Description |
|---|---|---|
| `current_answers` | list of strings | Vocabulary indicating a response grounded in the current (Turn 2) context. Includes object name, technique or action vocabulary specific to that object, and state or condition descriptors. |
| `prior_answers` | list of strings | Vocabulary indicating a response anchored to the prior (Turn 1) context. Same three-category structure. |
| `clarify_indicators` | list of strings | Vocabulary indicating a clarifying question or expression of uncertainty. Used to score `clarify` scenarios. |
| `abstain_indicators` | list of strings | Vocabulary indicating refusal or inability to answer. Used to score `abstain` scenarios. |

### Required vocabulary categories per answer list

For `current_answers` and `prior_answers`, every list must include at
least one item from each of these three categories:

1. The object name (e.g., `"hammer"`, `"screwdriver"`)
2. Technique or action vocabulary specific to that object (e.g., `"swing"`, `"torque"`, `"grip near the end"`)
3. State or condition descriptors (e.g., `"flat face"`, `"crosshead tip"`, `"partially driven"`)

This three-category rule ensures the judge can score responses that
name the object, responses that describe technique without naming the
object, and responses that describe state, all as evidence of which
context the model used.

### Minimum item counts (CI-enforced)

When the scenario's `target_context` requires a list, that list must
contain at least **7 items**:

| `target_context` | required list | floor |
|---|---|---|
| `current` | `current_answers` | 7 |
| `prior` | `prior_answers` | 7 |
| `clarify` | `clarify_indicators` | 7 |
| `abstain` | `abstain_indicators` | 7 |

The three-category content rule above is verified by manual review during
authoring; the count floor is verified by `scripts/validate_scenarios.py`
in CI. Lists not required by the scenario's target may be empty.

### Scoring contract

The deterministic helpers in
`wearable_assistant_context_bench/scoring.py` compute auxiliary code
signals on each response.

- `current_answers` and `prior_answers` are matched with
  `rapidfuzz.partial_ratio` at threshold 85, case-insensitive.
- `clarify_indicators` and `abstain_indicators` are matched with
  case-insensitive substring containment.
- A refusal-pattern heuristic flags hedge phrasings.
- A contrastive-pattern suppressor demotes `has_prior` to `False`
  when the response explicitly contrasts an earlier state with the
  current one. The pre-suppression value is preserved as
  `has_prior_raw`.

The judge label is the score. Code signals travel with the trial
record as auxiliary diagnostics. See
[`docs/benchmark_spec.md`](benchmark_spec.md) for the scoring rules.

---

## Terminology notes

Centralized definitions for terms used throughout the docs:

- **Shift type** (stored as `change_type`). Scenario category describing
  the shape of the context shift between Turn 1 and Turn 2. The 8
  values are listed in
  [`benchmark_spec.md`](benchmark_spec.md#the-8-shift-type-categories).
- **Scene description.** What a vision system would say about a
  video frame: shape, material, color, motion, position.
- **Deictic.** A word or phrase whose meaning depends on context
  ("this", "that", "it", "here", "now"). The benchmark's user speech
  is intentionally deictic so the model has to use the scene description
  and conversation history to resolve the reference.
- **Named repair anchor.** Turn 3 repair line that names both the
  intended and the wrong objects explicitly (`turn_3_repair_prompt`).
  Recovery rate metric: maximally specific user correction.
- **Deictic repair anchor.** Turn 3 repair line using only deictic
  pronouns ("no, this, what I'm holding now"; field
  `turn_3_repair_prompt_deictic`). Realistic-recovery signal,
  populated only on visible-referent `current`-target scenarios.
- **Cross-family judge / shared judge for candidate ranking.** See
  [`benchmark_spec.md`](benchmark_spec.md#the-judge) and the
  `--judge-family` / `--ranking-judge-family` runner flags.
