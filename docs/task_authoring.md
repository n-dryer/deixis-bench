# Task Authoring Rules

These rules govern how tasks are written for the Wearable Assistant
Context Bench. Every task must follow them. The validation
script in `scripts/validate_tasks.py` enforces them programmatically;
the writing-rule sections of this document explain what to write and
why.

If you are writing a new task, read this whole document first.
Then check your task against the validation checklist at the bottom
before submitting.

---

## What the benchmark measures

The benchmark tests one specific failure mode: when the user's context
changes between turns, does the model respond from the current
situational evidence, or does it stay anchored to the prior context?

It does **not** measure:

- Whether the coaching advice is correct, safe, or domain-appropriate
- Multi-turn conversation dynamics beyond a three-turn structure
- Performance on actual video frames (these are text proxies)
- Proactive coaching (noticing without being asked)

A model that fails this benchmark is not viable as a wearable coach. A
model that passes still needs separate evaluation for the things this
benchmark does not measure.

---

## What the model and judge see

Each task splits its content between what the candidate model
sees and what the judge sees:

| Field group | Fields | Visible to candidate | Visible to judge |
|---|---|---|---|
| User messages | `turn_1_user`, `turn_2_user` | Yes | Yes |
| Scene descriptions | `pre_turn_context_scene_description`, `turn_1_scene_description`, `turn_2_scene_description` | Yes (as `[Camera: ...]` blocks) | Yes |
| Gold answers | `current_answers`, `prior_answers`, `clarify_indicators`, `abstain_indicators` | No | Yes |

The candidate sees the user message and the scene description. The
judge sees the same plus the reference answer keys, and uses them to
score which context the candidate's response reflects.

---

## User-message rules

The user message is represented as a **text transcript**, not raw
audio. Write what a person would actually say to an AI wearable
assistant they're actively using for advice or coaching: the words a
speech-to-text system would emit. Acoustic grounding, speaker
attribution, addressee detection, and ambient audio cues are out of
scope; do not attempt to encode prosody, pauses, or non-lexical
sounds. The user does not narrate what the video shows.

### Must

- Be task-focused and natural
- Use deictic references for visible objects: `this`, `it`, `here`,
  `that one`, `over there`
- Read as something a real person would say out loud while doing
  something

### Must not

- Name or describe the object the user is holding, looking at, or
  interacting with
- Describe physical properties of anything visible in frame (color,
  shape, material, size)
- Announce that a context shift has occurred (`"I switched to..."`,
  `"Now I'm using..."`, `"I put that down and picked up..."`)
- Contain any vocabulary that appears in `current_answers` or
  `prior_answers`

### Examples

| Wrong | Right |
|---|---|
| "I just picked up a Phillips-head screwdriver." | "How do I get more torque on this?" |
| "I switched to a different tool. Am I holding it correctly?" | "Am I doing this right?" |
| "I'm in the kitchen now looking at the pot on the stove." | "Is this ready?" |
| "Now I'm holding the claw hammer instead." | "What about now?" |

### Special case: prior tasks

For tasks where the correct answer refers back to Turn 1 context
(`gold_label: "prior"`), Turn 2 user speech may use a temporal
reference (`"thinking back to before"`, `"what was the reading
earlier"`). It still must not name the Turn 1 object or describe what
was previously visible.

---

## Scene-description rules

Describe what a video frame would contain at the scene level. Scene
descriptions are what a vision system would say about a video frame:
shape, material, color, motion, position, without naming the
object directly. Physical properties only. No object names. No
technique evaluation.

### Must

- Describe shape, material, color, texture, motion, and spatial
  position
- Include enough distinguishing detail that a person reading the
  description, without any other context, can correctly identify the
  object
- Describe what the user is doing with the object (motion, contact,
  orientation)
- Include relevant environmental context (surface, setting, background
  elements) when it would be visible

### Must not

- Name the object (`hammer`, `screwdriver`, `spatula`, `ratchet`)
- Use functional labels (`the cutting tool`, `the fastening device`)
- Evaluate technique (`correctly positioned`, `good grip`, `improper
  angle`, `well-aligned`)
- Reference or compare to the prior context (`"now holding a different
  tool than before"`)
- Announce the context shift (`"user has switched from X to Y"`)

### Quality bar: the human identification test

Show the description to a person with no task context. They should
be able to identify the object with high confidence. If they cannot,
add more distinguishing scene detail. If they identify it immediately
from one feature, that is acceptable. The description does not need
to be cryptic, only non-labeled.

### Examples

| Wrong | Right |
|---|---|
| "User holding a claw hammer above a nail." | "Hand wrapped around a long wooden handle. Heavy metal head at the top, one face flat, the other split into two curved prongs. Positioned above a nail head protruding from a pine board." |
| "Correctly gripped Phillips-head screwdriver inserted into screw." | "Right hand gripping a slim metal tool with a crosshead tip, inserted into a fastener partially driven into a wood surface. Elbow raised, wrist rotated." |
| "Pan of soup on the stove, looks done." | "Shallow metal pan on a gas burner. Liquid surface showing small bubbles around the edges. Steam rising visibly." |

---

## Context shift rules

The shift between Turn 1 and Turn 2 is visible only in the scene
description. It is never announced in the user message.

- `turn_1_scene_description` and `turn_2_scene_description` describe different situations
  (different object, location, or state)
- `turn_2_user` does not reference the change
- `turn_2_scene_description` does not compare to or reference `turn_1_scene_description`

The shift between states is intentionally implicit in the benchmark.
A real wearable model processes a continuous video stream and observes
transitions naturally; the benchmark represents the moment of Turn 1
and the moment of Turn 2 and lets the model infer that a shift
occurred from the difference.

### When to use `pre_turn_context_scene_description`

Set `pre_turn_context_scene_description` when the task requires the model to access a
state that existed **before Turn 1 was spoken**. This applies to:

- Recall tasks where Turn 2 asks about something from before the
  conversation began
- `cross_session_reference` shift type tasks

For all other tasks, `pre_turn_context_scene_description` is null and `turn_1_scene_description`
establishes the initial state.

---

## Gold-answer rules

`current_answers` and `prior_answers` are for the judge. Object names
and coaching vocabulary are permitted here. The candidate never sees
these fields.

### Three-category requirement

Every `current_answers` and `prior_answers` list must include at least
one item from each of these three vocabulary categories:

1. **Object name.** `"hammer"`, `"screwdriver"`, `"saucepan"`
2. **Technique or action vocabulary.** `"swing from the elbow"`,
   `"steady torque"`, `"flat strike"`
3. **State or condition descriptors.** `"flat face down"`,
   `"crosshead tip seated"`, `"partially driven"`

This ensures the judge can score:

- Responses that name the object directly
- Responses that describe technique without naming the object
- Responses that describe state or condition

All three are valid evidence that the model used the correct context.
A response that hits any one of them counts.

### `clarify_indicators` and `abstain_indicators`

For tasks where the correct response is to ask for clarification
or decline to answer:

- `clarify_indicators` lists phrases that indicate a clarifying
  question (`"which one"`, `"do you mean"`, `"can you clarify"`)
- `abstain_indicators` lists phrases that indicate refusal
  (`"I can't tell"`, `"not enough to go on"`, `"need more info"`)

Both lists may be empty for tasks where `gold_label` is `current` or
`prior`.

---

## Runner injection format

The runner builds each user turn as:

```
[Camera: {turn_N_image}]
{turn_N_user}
```

When the image field is null, the `[Camera: ...]` block is omitted and only the
user message is sent. When `pre_turn_context_scene_description` is populated, it is injected
as a `[Camera: ...]` block before Turn 1, with no user message attached.

The candidate model sees the `[Camera: ...]` block as part of the user turn.
The judge receives the same content plus the ground-truth answer keys
in a separate ground-truth section.

---

## Validation checklist (10 points)

Every task must pass all ten before being committed:

1. T1 user speech contains no object names or visible-property
   descriptions
2. T2 user speech contains no object names, property descriptions, or
   shift announcements
3. T1 image description identifies the object without naming it
4. T2 image description identifies the object without naming it
5. Neither image description contains technique evaluation language
6. T2 image description contains no reference to T1 context
7. Human identification test passes for both image descriptions (a
   fresh reader can correctly identify the object from each
   description alone)
8. No `current_answers` or `prior_answers` token appears in any user
   speech field (word-boundary, case-insensitive)
9. The context shift is detectable only through the difference between
   `turn_1_scene_description` and `turn_2_scene_description`
10. Each `current_answers` and `prior_answers` list includes at least
    one item from each of the three vocabulary categories (object
    name, technique vocabulary, state descriptors)

The validation script enforces points 1, 2, 8, 9, and 10 automatically.
Points 3, 4, 5, 6, and 7 require semantic review.
