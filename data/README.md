# Dataset Card: Wearable Assistant Context Bench

## Dataset summary

A cross-turn reference-resolution evaluation set for AI wearable assistants
used actively for advice or coaching (smart glasses, ear worn
devices). Each scenario is a three-turn conversation with a
deliberate context shift between Turn 1 and Turn 2 visible only in
the scene description. The model must integrate scene descriptions with
deictic user speech to determine which context the question refers
to.

For product motivation and quickstart, see
[`README.md`](../../README.md). For the full benchmark contract, see
[`docs/benchmark_spec.md`](../../docs/benchmark_spec.md).

## Supported tasks

- **Reference resolution under cross-turn context shift.** The
  primary task. The judge labels each Turn 2 response as `current`,
  `prior`, `clarify`, or `abstain`.
- **Model selection** for deployed multimodal coaching assistants.

## Languages

English.

## Dataset structure

### Data files

All 70 scenarios live in a single `data/scenarios.jsonl` file
(one JSON object per line) with inline `gold` labels. Records are
distinguished by the `subset` field:

| `subset` value | Count | Purpose |
|---|---|---|
| `main` | 166 | The unified scenario set with pinned shift_type distribution. |

The `subset` field is currently always `main`. It is retained for
forward-compatibility if a future split is introduced. There is no
train/val/test split — all 166 scenarios are intended for inference
and labeling, not training.

Scenarios with legacy `adv-NN` ids were originally authored as a
separate distractor-rich contrast pack and have since been folded
into the unified set; their ids remain stable for traceability.

### Data fields

See [`../../docs/schema.md`](../../docs/schema.md) for the full field
reference. The candidate model receives the audio (text-transcript)
and video (scene-description) channels; the judge additionally
receives the answer lists and a ground-truth section naming the
actual objects in frame.

## Statistics (Main Subset)

| Statistic | Value |
|---|---|
| Total scenarios | 50 |
| Distinct activity domains | 16 |

### Shift-type distribution (`shift_type`)

| Category | Count |
|---|---|
| `object_in_hand` | 12 |
| `object_state` | 8 |
| `sequential_task` | 6 |
| `location` | 6 |
| `object_in_view` | 5 |
| `absent_referent` | 5 |
| `screen_content` | 4 |
| `cross_session_reference` | 4 |

### `target_context` distribution

| Label | Count |
|---|---|
| `current` | 33 |
| `prior` | 12 |
| `clarify` | 3 |
| `abstain` | 2 |

### `difficulty_tier` distribution

| Tier | Count |
|---|---|
| `easy` | 15 |
| `medium` | 20 |
| `hard` | 15 |

### Activity-domain coverage

The main subset spans 16 distinct activity domains, including kitchen,
workshop, garden, art and craft, automotive, electronics, sports,
fitness, music, household, office, navigation, finance, and
communication.

## Curation

All 70 scenarios follow the rules in
[`../../docs/scenario_authoring_rules.md`](../../docs/scenario_authoring_rules.md).
Each scenario passes six validation checks before inclusion.

The user message uses natural deictic language without naming
objects, describing visible properties, or announcing context shifts.
The scene description describes scene-level features (shape, material,
color, motion, position) without object names or technique
evaluation. The gold answers (judge-only) use object names,
technique vocabulary, and state descriptors.

## Annotations and validation

Every scenario passes six validation checks before being committed:

1. **Token-leakage scan.** Programmatic word-boundary check that no
   `current_answers` or `prior_answers` token appears in any
   user-speech field.
2. **Object-name scan.** Programmatic word-boundary check that no
   common object name (hammer, screwdriver, pan, etc.) appears in any
   image field.
3. **Schema and structural validation.** All required fields
   present, types correct, scenario IDs unique, category metadata
   matching, distributions matching the targets.
4. **Semantic image-identification review.** Semantic check that
   each image description identifies its object with high confidence
   to a fresh reader.
5. **Semantic-leakage isolation test.** Semantic check that the
   Turn 2 image plus Turn 2 user speech alone (without Turn 1
   context) is not sufficient to answer the question correctly.
6. **Cross-scenario duplication check.** Programmatic textual and
   structural overlap scan to surface near-duplicates.

Checks 1, 2, 3, and 6 run on every PR via
`scripts/validate_scenarios.py`. Checks 4 and 5 run during
authoring.

## Running the benchmark

```bash
wac-bench \
  --model <candidate_model_id> \
  --judge-model <judge_model_id> --judge-family <claude|gemini|openai> \
  --output-dir runs/<run_name>
```

Add `--subset contrast` to run against the contrast subset instead of
the main subset. Add `--no-camera` to strip the `[Camera: ...]` blocks.
Add `--enable-repair` to run Turn 3 repair after a Turn 2 miss.
See `wac-bench --help` for the full flag list.

Each run writes `findings.md` and `summary.json` to the output
directory. `transcripts.jsonl` is gitignored; regenerate by re-running
the same command.

## Out-of-scope by design

The benchmark does not measure these:

- Coaching advice quality (correctness, safety, domain
  appropriateness)
- Multi-turn dynamics beyond three turns
- Proactive coaching (assistance offered without a direct
  question)
- Domain expertise depth
- Latency, cost, serving characteristics
- Speaker attribution, addressee detection, ambient audio

## License

MIT, the same as the rest of the repository. See
[`../../LICENSE`](../../LICENSE).
