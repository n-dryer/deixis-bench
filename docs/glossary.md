# Glossary

Concise terminology reference for the active benchmark. Detailed field
definitions live in [`schema.md`](schema.md). Methodology lives in the
README. Task creation rules live in [`task_authoring.md`](task_authoring.md).

| Term | Meaning |
|---|---|
| task | One evaluated conversation unit in `data/tasks.jsonl`. |
| `task_id` | Stable task identifier in `task-NNN` format. |
| task set | Logical grouping for evaluation tasks. The current flat pre-release set uses `task_set: "main"` for every task. |
| `gold_label` | The correct judge label for Turn 2: `current`, `prior`, `clarify`, or `abstain`. |
| `reference_answers` | Judge-only answer lists used to help classify candidate responses. |
| scene-description text | Text proxy for what a vision system would report from a video frame. |
| domain | Activity area for coverage reporting, such as `kitchen`, `workshop`, or `finance`. |
| difficulty | Author-assigned difficulty: `easy`, `medium`, or `hard`. |
| `shift_type` | Context-shift category between Turn 1 and Turn 2. |
| `context_selection_instruction` | Prompt condition that tells the model to choose the relevant context before answering. |
| `pre_answer_context_scaffold` | Prompt condition that asks the model to output a brief relevant-context line before the answer. |
| official result | A curated benchmark run generated after the task set, prompts, judge prompt, and manifest are locked. |
| LLM judge | Model that labels each candidate response as `current`, `prior`, `clarify`, or `abstain`. |
| primary score | Mean of `current` recall and `prior` recall under the `baseline` prompt condition. |
| camera ablation | Run mode using `--no-camera`, which removes `[Camera: ...]` scene-description blocks. |
| manifest | Reproducibility metadata containing benchmark version, hashes, model IDs, trial count, and runner commit. |
