# Scenario Authoring Rules

These rules govern how scenarios are written for the Wearable Assistant
Context Bench. Every scenario must follow them. The validation
script in `scripts/validate_scenarios.py` enforces them programmatically;
the writing-rule sections of this document explain what to write and
why.

If you are writing a new scenario, read this whole document first.
Then check your scenario against the validation checklist at the bottom
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

Each scenario splits its content between what the candidate model
sees and what the judge sees:

| Field group | Fields | Visible to candidate | Visible to judge |
|---|---|---|---|
| User messages | `turn_1_user`, `turn_2_user`, `turn_3_repair_prompt` | Yes | Yes |
| Scene descriptions | `context_image`, `turn_1_image`, `turn_2_image` | Yes (as `[Camera: ...]` blocks) | Yes |
| Gold answers | `current_answers`, `prior_answers`, `clarify_indicators`, `abstain_indicators` | No | Yes |

The candidate sees the user message and the scene description. The
judge sees the same plus the gold answer keys, and uses them to
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

### Special case: prior scenarios

For scenarios where the correct answer refers back to Turn 1 context
(`target_context: "prior"`), Turn 2 user speech may use a temporal
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

Show the description to a person with no scenario context. They should
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

## Repair anchor rules

Every scenario must populate `turn_3_repair_prompt` (the named anchor).
Visible-referent `current`-target scenarios additionally populate
`turn_3_repair_prompt_deictic` (the deictic anchor).

### Named anchor

`turn_3_repair_prompt` names both the intended object and the wrong
object explicitly: *"I mean the hammer I'm holding now, not the
screwdriver from before."*

The named anchor is required on every scenario.

### Deictic anchor

`turn_3_repair_prompt_deictic` uses pure spatial or temporal deictic
language: *"I mean this thing in my hand right now"* or *"I mean what
I'm looking at."* It must not name either object.

Populate the deictic field only when **all three** are true:

1. `target_context == "current"` (the user is repairing toward the
   present frame).
2. `change_type` ∈ {`object_in_hand`, `object_in_view`, `object_state`,
   `screen_content`, `sequential_task`, `location`}. These are the
   visible-referent categories where a real wearable's vision system
   could resolve a deictic gesture.
3. The Turn 2 referent is genuinely visible in the video frame
   (i.e. the deictic gesture is technically resolvable).

Leave the field as `null` for `absent_referent`,
`cross_session_reference`, and any scenario whose target is `prior`,
`clarify`, or `abstain`. The runner falls back to the named anchor in
those cases when invoked with `--repair-style deictic`.

The deictic anchor must not contain object-name vocabulary from the
`current_answers` or `prior_answers` lists. The validator's token
leakage check applies to it the same way it applies to user-speech
fields.

---

## Context shift rules

The shift between Turn 1 and Turn 2 is visible only in the scene
description. It is never announced in the user message.

- `turn_1_image` and `turn_2_image` describe different situations
  (different object, location, or state)
- `turn_2_user` does not reference the change
- `turn_2_image` does not compare to or reference `turn_1_image`

The shift between states is intentionally implicit in the benchmark.
A real wearable model processes a continuous video stream and observes
transitions naturally; the benchmark represents the moment of Turn 1
and the moment of Turn 2 and lets the model infer that a shift
occurred from the difference.

### When to use `context_image`

Set `context_image` when the scenario requires the model to access a
state that existed **before Turn 1 was spoken**. This applies to:

- Recall scenarios where Turn 2 asks about something from before the
  conversation began
- `cross_session_reference` shift type scenarios

For all other scenarios, `context_image` is null and `turn_1_image`
establishes the initial state.

### Choosing between `change_type` values

A single shift can plausibly fit more than one category — a person on
a workshop bench moving from sanding to staining "is" a sequential
task, but it's also a state change for the board and an object swap
in the hand. Pick the **most specific** category that fits, in this
order:

1. **`cross_session_reference`** — `context_image` is non-null. Always
   takes precedence (and is enforced at the schema level).
2. **`screen_content`** — both turns are looking at a screen and the
   screen content has changed. Use this even if the user is doing a
   sequential workflow on the screen.
3. **`absent_referent`** — the entity Turn 2 is asking about is no
   longer in frame. Use this even if a later step continues elsewhere.
4. **`location`** — the whole scene/setting changes (different room,
   indoor → outdoor, workbench → cooktop). Use this for whole-scene
   shifts even if a task is continuing.
5. **`object_in_hand`** — the user puts down one object and picks up a
   different object. Use this even if the new object is part of a
   later step in the same task.
6. **`object_state`** — the same primary object stays in frame and
   transitions to a different physical state (paint drying, dough
   rising, body posture changing).
7. **`object_in_view`** — same scene, the camera or user attention
   shifts to a different object that was already in (or adjacent to)
   the frame. No object swap, no state change of the original.
8. **`sequential_task`** — last resort. Reserve for cases where the
   **same primary object/surface** is still in view and the
   **operation itself** has changed to the next step (e.g.
   sand → stain on the same board, knead → bake the same dough). If
   any of the conditions above hold, pick that category instead.

When in doubt, ask: "what is the dominant visual difference between
T1 and T2?" If it's a different object in the hand, that's
`object_in_hand`. If it's a new room, that's `location`. If it's a
different screen, that's `screen_content`. Only when the surface and
object are unchanged but the operation has moved on is the answer
`sequential_task`.

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

For scenarios where the correct response is to ask for clarification
or decline to answer:

- `clarify_indicators` lists phrases that indicate a clarifying
  question (`"which one"`, `"do you mean"`, `"can you clarify"`)
- `abstain_indicators` lists phrases that indicate refusal
  (`"I can't tell"`, `"not enough to go on"`, `"need more info"`)

Both lists may be empty for scenarios where the target is `current` or
`prior`.

### Minimum item count (CI-enforced)

When the scenario's `target_context` requires a list, that list must
contain at least **7 items**:

| `target_context` | required list | floor |
|---|---|---|
| `current` | `current_answers` | 7 |
| `prior` | `prior_answers` | 7 |
| `clarify` | `clarify_indicators` | 7 |
| `abstain` | `abstain_indicators` | 7 |

The three-category content rule above is verified by manual review.
The count floor is verified automatically by
`scripts/validate_scenarios.py` and trips CI if any required list has
fewer than 7 items.

---

## Runner injection format

The runner builds each user turn as:

```
[Camera: {turn_N_image}]
{turn_N_user}
```

When the image field is null, the `[Camera: ...]` block is omitted and only the
user message is sent. When `context_image` is populated, it is injected
as a `[Camera: ...]` block before Turn 1, with no user message attached.

The candidate model sees the `[Camera: ...]` block as part of the user turn.
The judge receives the same content plus the ground-truth answer keys
in a separate ground-truth section.

---

## Validation checklist (10 points)

Every scenario must pass all ten before being committed:

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
   `turn_1_image` and `turn_2_image`
10. Each `current_answers` and `prior_answers` list includes at least
    one item from each of the three vocabulary categories (object
    name, technique vocabulary, state descriptors)
11. The required gold list for the scenario's `target_context` has at
    least 7 items (`current_answers` for `current` target, etc.)
12. `activity_domain` is one of the 14 allowed values (closed enum;
    see `VALID_ACTIVITY_DOMAINS` in `scripts/validate_scenarios.py`)
13. `turn_3_repair_prompt_deictic` is non-null exactly when
    `target_context == "current"` AND `change_type` is not in
    `{absent_referent, cross_session_reference}`; null otherwise
14. `cross_session_reference` scenarios have a non-null `context_image`

The validation script enforces points 1, 2, 8, 9, 11, 12, 13, and 14
automatically. Point 10 (three-category content rule) requires manual
review during authoring. Points 3, 4, 5, 6, and 7 require semantic
review.
