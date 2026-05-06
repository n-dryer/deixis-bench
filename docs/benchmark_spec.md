# Benchmark Spec: Wearable Assistant Context Bench

## Overview

The Wearable Assistant Context Bench tests one specific failure
mode of AI wearable assistants the user is actively engaging with
for advice or coaching (smart glasses, ear worn devices), with
audio/video/text input and audio/text output. When the user's
situation changes between turns, does the model respond from the
current situational evidence, or does it stay anchored to the prior
context?

The benchmark measures **cross-turn reference resolution**: whether the
model resolves the user's reference (their "this", "that", or "it")
to the current video frame instead of staying anchored to an earlier
one.

An in-the-moment wearable coach lives in a continuous stream. The
user puts down one object and picks up another. The user moves from
the workbench to the kitchen. The pan was empty a minute ago and now
holds simmering sauce. Each turn is a new chance for the model to
either update or stay stuck.

## Scope boundaries

**Out of scope.** The benchmark does not measure these. Evaluate them
separately:

- Coaching advice quality (correctness, safety, domain appropriateness)
- Multi-turn conversation dynamics beyond three turns
- Proactive coaching (assistance offered without a direct question)
- Domain expertise depth
- Latency, cost, and serving characteristics
- Speaker attribution, addressee detection, ambient audio

**Limitations.** The benchmark does not yet measure these:

- Human inter-annotator agreement on judge labels
- Real-video validation on a held-out sample
- Raw-audio validation
- Multi-trial variance reporting beyond the default (1 trial at
  temperature 0)
- Full omnimodal stack (live audio I/O, real-time streaming, voice
  output)

## What the model and judge see

Each task splits its content between what the candidate model sees and what the judge sees:

| Channel | Field(s) | Visible to candidate | Visible to judge |
|---|---|---|---|
| Audio | `turn_1_user`, `turn_2_user` | Yes | Yes |
| Camera | `pre_turn_context_scene_description`, `turn_1_scene_description`, `turn_2_scene_description` | Yes (as `[Camera: ...]` blocks) | Yes |
| Ground truth | `current_answers`, `prior_answers`, `clarify_indicators`, `abstain_indicators` | No | Yes |

The candidate model sees the user's spoken words, plus a scene
description of the video frame at each turn. The user's turns are
**text transcripts**. The video frame is a **scene description in
text** (shape, material, color, motion, position, without naming the
object directly).

The judge sees the user message and the scene description plus the
reference answer keys, which name the actual objects in frame. The
candidate never sees the reference answer keys.

For the rules that govern how each input field is written, see
[`task_authoring.md`](task_authoring.md).

## Task structure

Each task is a two-turn conversation:

1. **Turn 1.** Optional `pre_turn_context_scene_description` injected first (only on
   `cross_session_reference` tasks), then `turn_1_scene_description` plus
   `turn_1_user`. Turn 1 establishes the starting state.
2. **Turn 2.** `turn_2_scene_description` plus `turn_2_user`. The image has
   changed. The user's question is natural and deictic; it does not
   announce the change. **Turn 2 is the only scored turn.**

Field reference is in [`schema.md`](schema.md).

## Video block injection format

The runner builds each user turn as:

```text
[Camera: {turn_N_image}]
{turn_N_user}
```

When `turn_N_image` is null, the `[Camera: ...]` block is omitted and only the
user message is sent. When `pre_turn_context_scene_description` is populated, it is injected
as a `[Camera: ...]` block before Turn 1, with no accompanying user
message.

The implementation lives in `_build_message` and
`_build_pre_turn_context_scene_description_message` in `wearable_assistant_context_bench/runner.py`.

## The 8 shift-type categories

Every task fits exactly one category. Categories describe the
shape of the context shift between Turn 1 and Turn 2. The shift type
is stored as the `shift_type` field in the data files.

| Category | Description |
|---|---|
| `object_in_hand` | User puts down one object, picks up another. The video shows a different object in the user's hand. |
| `object_state` | Same object, different state (cooking progress, paint drying, plant growth). |
| `sequential_task` | Same task, the user has progressed to a later step. |
| `location` | Whole scene changes; user moves to a different room or work area. |
| `object_in_view` | The video stays roughly in place; the user's attention has shifted to a different object visible in the scene. |
| `absent_referent` | The object the question is about is no longer in frame. |
| `screen_content` | Both Turn 1 and Turn 2 are looking at a screen; the screen content has changed. |
| `cross_session_reference` | Requires `pre_turn_context_scene_description`; Turn 2 asks about a state that existed before Turn 1. |

In prose throughout the docs we call these "shift types" or
"task categories." The data field name is `shift_type`.

## Gold Labels

Each task carries a `gold_label` field naming the correct
grounding label. One of:

| Value | Meaning |
|---|---|
| `current` | The correct answer refers to what the video shows right now (Turn 2 frame). |
| `prior` | The correct answer refers to something from an earlier scene (Turn 1 frame, or `pre_turn_context_scene_description`). |
| `clarify` | The question is ambiguous given the available context; the assistant should ask for clarification. |
| `abstain` | The needed information is not present in the context; the assistant should decline. |

## Scoring

For each Turn 2 response, the judge emits exactly one of the four
labels: `current`, `prior`, `clarify`, or `abstain`. A task is
counted correct when the judge's label matches `gold_label`.

The primary metric is **mean per-class recall**: the mean of
`current_recall` and `prior_recall` under the `baseline` prompt
condition. These are class recall values (`TP / (TP + FN)`).

```text
primary_score = mean(current_recall, prior_recall)
```

The primary metric is reported with both a 95% normal-approximation CI
and a 95% percentile bootstrap CI; per-class numbers carry 95% Wilson
CIs. The findings template also reports task-set recall, shift-type
recall, hedging behavior, code-judge disagreement, and inter-judge
agreement when a shared ranking judge is configured.

`clarify` and `abstain` rates appear in the findings output as
auxiliary diagnostic rows. They do not enter the primary number.

The deterministic helpers in
`wearable_assistant_context_bench/scoring.py` compute auxiliary code
signals on each response:

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
record as auxiliary diagnostics.

## The judge

A second LLM labels each Turn 2 response as `current`, `prior`,
`clarify`, or `abstain`. It sees:

1. A neutral task description with the Turn 1 user message and
   the fact that the user's context shifts between turns. It does
   not see the `gold_label` value, the shift type, or the authoring notes.
2. The Turn 2 user message.
3. The candidate's Turn 2 response.
4. The four answer lists.
5. A **ground-truth context section** with plain-language
   descriptions of the Turn 1 and Turn 2 video frames (plus the
   pre-conversation frame for recall tasks).

The judge returns a JSON verdict with one label and a one-sentence
rationale. See `wearable_assistant_context_bench/llm_judge.py` for the prompt and parsing logic.
The judge prompt is hashed into the run manifest. The privileged-field
constraint (no `gold_label`, `shift_type`, or authoring `notes` in
the rendered prompt) is enforced by `tests/test_llm_judge.py`.

`--judge-family auto` picks a judge from a different model family
than the candidate (Claude -> Gemini, Gemini -> OpenAI, OpenAI ->
Gemini); the mapping is in `resolve_judge_family` in
`wearable_assistant_context_bench/llm_judge.py`. Pass
`--judge-family claude|gemini|openai` to override.

## Reproducibility

Each run emits a manifest recorded in the findings output. The
manifest fields include:

- `benchmark_version`: the runner code version
- `camera_injection`: boolean; always `true`
- `task_set`: `"main"` for the flat pre-release task set
- `tasks_sha256`: hash of `data/tasks.jsonl`
- `prompt_conditions_sha256`: hash of `data/prompt_conditions.json`
- `judge_prompt_sha256`: judge prompt identification
- `candidate_model`, `judge_model`, `judge_family`,
  `judge_family_resolution`: model and resolution mode
- `trials`, `temperature`, `ranking_condition`: run parameters
- `timestamp_utc`, `runner_git_commit`: run identity

The repo commits `data/MANIFEST.lock.json` with the SHA256 hashes of
the task set, prompt conditions, and judge prompt template.
`scripts/validate_tasks.py` checks computed hashes against this
lockfile; CI fails if they drift without a coordinated
`BENCHMARK_VERSION` bump.

## Related work

The semantic-leakage check (Check 5 in the validator) is adapted from
MMStar (Chen et al., NeurIPS 2024), which proposed running questions
through text-only models to filter items that don't actually require
vision. The audio/scene description separation is adapted from
VLSBench (Hu et al., ACL 2025), which showed that text queries often
leak the visual content of an image; we apply the same separation
principle to context cues. The non-labeled scene description style
follows Ego4D narrations (Grauman et al., CVPR 2022).

## Configuration and runtime defaults

The runner loads its defaults from `data/config.json`. Key defaults:

- `trials_per_cell: 1`
- `task_set: "main"` for every active task.
- `temperature: 0.0`.

CLI flags override the config file. Use `--config <path>` to point at
an alternate config.

## Reference

- Schema and field-level definitions: [`schema.md`](schema.md)
- Authoring rules and validation checklist:
  [`task_authoring.md`](task_authoring.md)
- Per-run findings: `findings.md` (in the run's output directory)
- Dataset card: [`../data/README.md`](../data/README.md)
