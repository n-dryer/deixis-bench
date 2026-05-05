# Scenario Audit Rubric

Independent rule-based audit of `shift_type` and `difficulty_tier`
metadata in `data/scenarios.jsonl`. The audit ignores the metadata
fields it is auditing and re-classifies each scenario from script
content alone (`turn_1_image`, `turn_2_image`, `turn_1_user`,
`turn_2_user`, `context_image`, `gold`).

Implemented in
[`wearable_assistant_context_bench/audit_rubric.py`](../wearable_assistant_context_bench/audit_rubric.py)
(pure functions, unit-tested in
[`tests/test_audit_rubric.py`](../tests/test_audit_rubric.py)) and
driven by [`scripts/audit_scenarios.py`](../scripts/audit_scenarios.py).

## Output

`scripts/audit_scenarios.py` writes `data/scenarios.audit.csv`. Each row
holds:

| column | meaning |
|---|---|
| `scenario_id` | from the bank |
| `metadata_shift_type` | as authored in `scenarios.jsonl` |
| `audit_shift_type` | inferred by the rubric |
| `shift_type_match` | `True` / `False` |
| `metadata_difficulty` | as authored |
| `audit_difficulty` | inferred by the rubric |
| `difficulty_match` | `True` / `False` |
| `audit_signals_shift_type` | JSON list of rule traces that fired |
| `audit_difficulty_score` | integer additive score |
| `audit_difficulty_breakdown` | JSON dict of named signal contributions |
| `audit_target_context_inferred` | derived from `gold` shape + T2 user phrasing |
| `metadata_target_context` | as authored (for context, not audited) |
| `metadata_referent_complexity` | as authored (for context, not audited) |
| `llm_shift_type` | LLM-judge tie-breaking verdict (when `--judge-on-disagreement`) |
| `llm_difficulty` | LLM-judge tie-breaking verdict |
| `llm_rationale` | one-sentence judge justification |

## Usage

```bash
# Rule-based pass on all 166 scenarios (~1 second)
uv run python scripts/audit_scenarios.py

# Plus LLM-judge spot-check on shift_type disagreements only
# (requires OPENROUTER_API_KEY or equivalent)
uv run python scripts/audit_scenarios.py --judge-on-disagreement

# Print summary, skip CSV
uv run python scripts/audit_scenarios.py --summary-only
```

## Rubric: `shift_type`

Eight categories (matching
[`tests/test_schema.py`](../tests/test_schema.py)). Rules are applied in
order; the first match wins. The full trace is recorded in
`audit_signals_shift_type` so reviewers can see why a verdict was
reached.

| # | Rule | Signal |
|---|---|---|
| 1 | `cross_session_reference` | `context_image` is non-null |
| 2 | `screen_content` | both T1 image and T2 image contain a screen cue (`backlit`, `screen showing`, `screen now`, `the screen`, `compose window`, `speech bubbles`, `phone screen`, `tablet screen`, `smartphone screen`, `drawing slate`) |
| 3 | `absent_referent` | T2 user phrasing references a removed entity (`the one from`, `that one`, `that thing`, `the printing`, `the dosage`, `the wattage`, `what torque`, `the gauge actually`, `the volume level`, `the model number`, `the planting depth`, `the instructions say`, `did the instructions`, `the sender on`, …) |
| 4 | `object_in_hand` | hand cues in BOTH turns (`hand gripping`, `hand wrapped`, `hand holding`, `right hand`, `left hand`, `fingers wrapped`, …) |
| 5 | `sequential_task` | known activity-pair appears across `gold.prior_answers` → `gold.current_answers` (e.g. `sand` → `stain`, `dice` → `saute`, `tune` → `chord`, `charge` → `insert`) |
| 6 | `location` | T2 image does NOT open with `Same`/`The same` AND T1/T2 word-Jaccard < 0.30 |
| 7 | `object_in_view` | T2 image opens with `Same`/`The same` AND has a camera/attention shift cue (`camera now`, `now tilted`, `now showing a`, `attention shifts`, …) |
| 8 | `sequential_task` | T2 opens with `Same`/`The same` AND `gold.prior_answers` and `gold.current_answers` are both empty AND T1/T2 word-Jaccard < 0.25 |
| 9 | `object_state` | T2 opens with `Same`/`The same` (default), OR T1/T2 word-Jaccard ≥ 0.30 with no anchor |
| _fallback_ | `object_in_view` | unanchored T2 with mid-Jaccard |

Rule 8 covers the `clarify`/`abstain` sequential cases where no
specific prior/current vocabulary is anchored — empty gold rules out
`object_state` (which always describes prior-and-current state
vocabulary), and a low T1/T2 image Jaccard signals a new
tool/operation introduced in T2 (sequential median ≈ 0.20 vs.
object_in_view ≈ 0.30, object_state ≈ 0.32).

## Rubric: `difficulty_tier`

Additive scoring; bin into tiers.

| Signal | Points |
|---|---|
| audit-derived `target_context` ∈ `{abstain, clarify}` | +2 |
| audit-derived `referent_offscreen` (target ∈ `{prior, abstain}`) | +2 |
| `distractor_present` (T2 image contains `several`, `multiple`, `row of`, `a few`, `various`, `distinct`, …) | +1 |
| audit `shift_type` ∈ `{cross_session_reference, absent_referent, screen_content}` | +1 |
| Jaccard(prior-answer tokens, current-answer tokens) ≥ 0.30 | +2 |
| 0.10 ≤ Jaccard < 0.30 | +1 |
| T2 user has long-time-gap cue (`yesterday`, `last week`, `earlier today`, `next morning`, …) | +1 |
| char-overlap ratio between T1 image and T2 image ≥ 0.70 (subtle scene contrast) | +1 |
| audit `shift_type` == `object_in_hand` AND target == `current` AND no distractor AND not offscreen | −1 |

Bins:

- **easy**: total ≤ 1
- **medium**: 2 ≤ total ≤ 3
- **hard**: total ≥ 4

## Audit-derived `target_context`

The audit doesn't trust the metadata `target_context` field (it's part
of what's being audited), so it derives one from the `gold`
indicator-list shape:

1. `clarify_indicators` non-empty → `clarify`
2. `abstain_indicators` non-empty → `abstain`
3. only `current_answers` non-empty → `current`
4. only `prior_answers` non-empty → `prior`
5. both populated: `prior` if T2 user uses past-reference deixis,
   else `current`

## Calibration

Run on the current 166-scenario bank, the rubric produces:

```
shift_type mismatches: 24 (14%)
difficulty   mismatches: 97 (58%)
any mismatch: 107 (64%)

per metadata shift_type:
    absent_referent: 2/21 (10%)
    cross_session_reference: 0/20 (0%)
    location: 1/21 (5%)
    object_in_hand: 5/21 (24%)
    object_in_view: 3/21 (14%)
    object_state: 3/21 (14%)
    screen_content: 1/20 (5%)
    sequential_task: 9/21 (43%)
```

The `cross_session_reference` rule is dispositive (0% disagreement),
since it's anchored to the schema invariant that `context_image` is
non-null exactly for that category.

`sequential_task` is the hardest category: many scenarios are
`clarify`/`abstain` with empty `prior_answers`/`current_answers`,
removing the activity-pair signal. Genuine boundary cases — like
sc-100 ("Was that one good?") sitting between `sequential_task` and
`absent_referent`, or sc-105 (spreadsheet → chart workflow on a
laptop) sitting between `sequential_task` and `screen_content` — are
surfaced rather than auto-resolved. Reviewers should expect to make
judgment calls on those rows.

The difficulty mismatch rate is high by design. The rubric defines its
own additive scoring; it doesn't try to reproduce the human-graded
distribution exactly. Compare per-row, not at the aggregate.

## LLM-judge spot-check

Optional `--judge-on-disagreement` flag adds a second pass:

- For every `shift_type` disagreement (only — difficulty
  disagreements are skipped, since the rubric explicitly defines its
  own scoring), one structured Sonnet 4.6 call returns
  `{shift_type, difficulty_tier, rationale}`.
- The verdict and rationale land in the CSV's `llm_shift_type`,
  `llm_difficulty`, `llm_rationale` columns.
- Disagreements are capped at 60 (configurable via `JUDGE_DISAGREEMENT_CAP`
  in the script). Above that, the rubric is broken, not the metadata —
  fix the rubric first.

The judge uses `LiteLLMJudgeAdapter` from
`wearable_assistant_context_bench.llm_judge`; on API errors the row's
`llm_rationale` is set to `JUDGE_ERROR: <reason>` and the run continues.
