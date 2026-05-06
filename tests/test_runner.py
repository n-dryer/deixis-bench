"""End-to-end and integration tests for the runner.

Covers the trial loop and CLI parsing, the ``[Camera: ...]`` injection
format and ``pre_turn_context_scene_description`` + judge ground-truth wiring, public-doc
framing terms and forbidden vocabulary, and prompt-condition loading
plus policy neutrality.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from wearable_assistant_context_bench import runner as run_module
from wearable_assistant_context_bench.llm_judge import _build_user_prompt
from wearable_assistant_context_bench.models import ModelConfig
from wearable_assistant_context_bench.prompt_conditions import (
    PromptCondition,
    get_prompt_condition,
    load_prompt_conditions,
)
from wearable_assistant_context_bench.runner import (
    AnswerSet,
    Task,
    _build_message,
    _build_pre_turn_context_scene_description_message,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, messages: list[dict], system: str, config: Any) -> str:
        self.calls.append({"messages": list(messages), "system": system})
        return "STUB_RESPONSE"


class _StubJudge:
    family = "stub"
    model_id = "stub-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def label(self, **kwargs: Any) -> Any:
        self.calls.append(
            {
                "response": kwargs.get("response"),
                "turn_2_user": kwargs.get("turn_2_user"),
                "ground_truth_context": kwargs.get("ground_truth_context"),
            }
        )
        from wearable_assistant_context_bench.llm_judge import JudgeVerdict

        return JudgeVerdict(selected_label="current", rationale="stub")


class _CapturingAdapter:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def query(self, messages: list[dict], system: str, config: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        return "STUB_RESPONSE"


class _PassingJudge:
    family = "stub"
    model_id = "stub-model"

    def label(self, **kwargs: Any) -> Any:
        from wearable_assistant_context_bench.llm_judge import JudgeVerdict

        return JudgeVerdict(selected_label="current", rationale="stub")


def _make_task(
    *,
    task_id: str = "task-test",
    gold_label: str = "current",
    shift_type: str = "object_in_hand",
    domain: str = "workshop",
    referent_complexity: str = "single_referent",
    difficulty: str = "easy",
    turn_1_scene_description: str | None = "Hand on a thin metal handle.",
    turn_1_user: str = "How do I use this?",
    turn_2_scene_description: str | None = "Hand on a wooden grip with a heavy head.",
    turn_2_user: str = "What about now?",
    pre_turn_context_scene_description: str | None = None,
    task_set: str = "main",
    reference_answers: AnswerSet | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        gold_label=gold_label,
        shift_type=shift_type,
        domain=domain,
        referent_complexity=referent_complexity,
        difficulty=difficulty,
        turn_1_scene_description=turn_1_scene_description or "",
        turn_1_user=turn_1_user,
        turn_2_scene_description=turn_2_scene_description or "",
        turn_2_user=turn_2_user,
        pre_turn_context_scene_description=pre_turn_context_scene_description,
        task_set=task_set,
        reference_answers=reference_answers or AnswerSet(),
    )


# ---------------------------------------------------------------------------
# Runner trial loop & output routing (was test_run.py)
# ---------------------------------------------------------------------------


def test_run_produces_expected_trial_count_and_jsonl_shape(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    judge = _StubJudge()
    output_dir = tmp_path / "transcripts"
    results = run_module.run(
        adapter=adapter,
        judge=judge,
        config={"output_dir": str(output_dir)},
    )

    task_count = len(run_module.load_tasks(task_set="main"))
    condition_count = len(run_module.load_prompt_conditions(run_module.PROMPT_CONDITIONS_PATH))
    expected_trials = task_count * condition_count * run_module.CONFIG["trials_per_cell"]
    assert len(results) == expected_trials

    transcript_path = output_dir / "transcripts.jsonl"
    assert transcript_path.exists()
    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == expected_trials

    for line in lines:
        payload = json.loads(line)
        for required in (
            "task_id",
            "task_set",
            "condition",
            "trial",
            "gold_label",
            "shift_type",
            "turn_1_user",
            "turn_1_scene_description",
            "turn_1_response",
            "turn_2_user",
            "turn_2_scene_description",
            "turn_2_response",
            "turn_2_code_signals",
            "turn_2_judge_label",
            "turn_2_passed",
        ):
            assert required in payload, f"missing {required} in transcript row"
        assert payload["task_set"] == "main"


def test_run_output_dir_governs_findings_location(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    judge = _StubJudge()
    output_dir = tmp_path / "some_run"
    run_module.run(
        adapter=adapter,
        judge=judge,
        config={"output_dir": str(output_dir)},
    )
    findings_path = output_dir / "findings.md"
    assert findings_path.exists()
    body = findings_path.read_text(encoding="utf-8")
    assert "Reproducibility manifest" in body
    assert "Benchmark summary" in body
    assert "```json" in body


def test_run_emits_summary_json(tmp_path: Path) -> None:
    """The runner must write summary.json alongside transcripts.jsonl
    and findings.md so downstream tooling has a machine-readable
    aggregate of the run."""
    adapter = _StubAdapter()
    judge = _StubJudge()
    output_dir = tmp_path / "summary_run"
    run_module.run(
        adapter=adapter,
        judge=judge,
        config={"output_dir": str(output_dir)},
    )
    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["run_label"] == "summary_run"
    assert payload["n_trials"] > 0
    assert payload["n_tasks"] > 0
    assert "primary_score_mean_recall" in payload
    assert "per_class_recall" in payload
    assert "per_task_set_recall" in payload
    assert "config_snapshot" in payload


def test_parse_args_accepts_all_flags() -> None:
    args = run_module._parse_args(
        [
            "--model",
            "claude-sonnet-4-6",
            "--judge-model",
            "gemini-2.5-flash",
            "--judge-family",
            "openai",
            "--trials",
            "3",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.model == "claude-sonnet-4-6"
    assert args.judge_model == "gemini-2.5-flash"
    assert args.judge_family == "openai"
    assert args.trials == 3
    assert args.output_dir == "/tmp/out"


def test_parse_args_rejects_unknown_judge_family() -> None:
    with pytest.raises(SystemExit):
        run_module._parse_args(["--judge-family", "mistral"])


def test_parse_args_defaults_are_none() -> None:
    args = run_module._parse_args([])
    assert args.model is None
    assert args.judge_model is None
    assert args.judge_family is None
    assert args.trials is None
    assert args.output_dir is None


def test_config_overrides_from_args_only_sets_provided_flags() -> None:
    args = run_module._parse_args(["--model", "claude-sonnet-4-6"])
    overrides = run_module._config_overrides_from_args(args)
    assert overrides == {"model_id": "claude-sonnet-4-6"}


def test_config_overrides_from_args_full() -> None:
    args = run_module._parse_args(
        [
            "--model",
            "openai/gpt-4.1-mini",
            "--judge-model",
            "gemini-2.5-flash",
            "--judge-family",
            "gemini",
            "--trials",
            "1",
            "--output-dir",
            "out/",
        ]
    )
    overrides = run_module._config_overrides_from_args(args)
    assert overrides == {
        "model_id": "openai/gpt-4.1-mini",
        "judge_model_id": "gemini-2.5-flash",
        "judge_family": "gemini",
        "trials_per_cell": 1,
        "output_dir": "out/",
    }


def test_parse_args_no_longer_exposes_task_set_flag() -> None:
    """The ``--task_set`` CLI flag is not part of the flat pre-release CLI."""
    with pytest.raises(SystemExit):
        run_module._parse_args(["--task_set", "main"])
    with pytest.raises(SystemExit):
        run_module._parse_args(["--task_set", "paired"])


def test_task_ids_use_final_task_prefix(tmp_path: Path) -> None:
    """Committed tasks use the final ``task-###`` ID format."""
    adapter = _StubAdapter()
    judge = _StubJudge()
    output_dir = tmp_path / "unified"
    results = run_module.run(
        adapter=adapter,
        judge=judge,
        config={"output_dir": str(output_dir)},
    )
    ids = {r["task_id"] for r in results}
    assert ids
    assert all(sid.startswith("task-") for sid in ids)
    findings = (output_dir / "findings.md").read_text(encoding="utf-8")
    import re as _re

    match = _re.search(r"```json\n(.*?)\n```", findings, _re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["task_set"] == "main"


def test_manifest_records_run_metadata(tmp_path: Path) -> None:
    adapter = _StubAdapter()
    judge = _StubJudge()
    output_dir = tmp_path / "manifest_run"
    run_module.run(
        adapter=adapter,
        judge=judge,
        config={"output_dir": str(output_dir)},
    )
    findings_body = (output_dir / "findings.md").read_text(encoding="utf-8")
    import re as _re

    match = _re.search(r"```json\n(.*?)\n```", findings_body, _re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["camera_injection"] is True
    assert payload["task_set"] == "main"
    assert "ranking_judge_model" in payload
    assert "ranking_judge_family" in payload
    assert payload["ranking_judge_model"] is None
    assert payload["ranking_judge_family"] is None
    # Positive: benchmark_version is recorded at the top level.
    assert payload["benchmark_version"] == "0.1.0a0"
    # Negative regression guards: removed fields must not reappear.
    assert "schema_revision" not in payload
    assert "judge_prompt_version" not in payload


class _StubRankingJudge(_StubJudge):
    family = "stub-ranking"
    model_id = "stub-ranking-model"

    def label(self, **kwargs: Any) -> Any:
        from wearable_assistant_context_bench.llm_judge import JudgeVerdict

        self.calls.append(kwargs)
        return JudgeVerdict(selected_label="prior", rationale="ranking-stub")


def test_run_records_ranking_judge_fields_when_ranking_judge_provided(
    tmp_path: Path,
) -> None:
    adapter = _StubAdapter()
    judge = _StubJudge()
    ranking_judge = _StubRankingJudge()
    output_dir = tmp_path / "ranking"
    results = run_module.run(
        adapter=adapter,
        judge=judge,
        ranking_judge=ranking_judge,
        config={"output_dir": str(output_dir)},
    )
    for r in results:
        assert "turn_2_ranking_judge_label" in r
        assert r["turn_2_ranking_judge_label"] == "prior"
        assert r["turn_2_ranking_judge_rationale"] == "ranking-stub"
        assert r["turn_2_ranking_passed"] == (r["gold_label"] == "prior")
    assert len(ranking_judge.calls) >= len(results)
    findings_body = (output_dir / "findings.md").read_text(encoding="utf-8")
    import re as _re

    match = _re.search(r"```json\n(.*?)\n```", findings_body, _re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload["ranking_judge_model"] == "stub-ranking-model"
    assert payload["ranking_judge_family"] == "stub-ranking"
    assert "Cohen's kappa" in findings_body
    assert "Observed agreement" in findings_body


def test_parse_args_accepts_ranking_judge_flags() -> None:
    args = run_module._parse_args(
        [
            "--ranking-judge-family",
            "claude",
            "--ranking-judge-model",
            "openrouter/anthropic/claude-sonnet-4.6",
        ]
    )
    assert args.ranking_judge_family == "claude"
    assert args.ranking_judge_model == "openrouter/anthropic/claude-sonnet-4.6"
    overrides = run_module._config_overrides_from_args(args)
    assert overrides == {
        "ranking_judge_family": "claude",
        "ranking_judge_model_id": "openrouter/anthropic/claude-sonnet-4.6",
    }


def test_load_runtime_config_reads_json_file() -> None:
    """``load_runtime_config`` reads ``data/config.json`` and
    returns a dict that includes the expected keys."""
    cfg = run_module.load_runtime_config()
    assert cfg["trials_per_cell"] == 1
    assert cfg["task_set"] == "main"


# ---------------------------------------------------------------------------
# Camera injection (was test_camera_injection.py)
# ---------------------------------------------------------------------------


def test_camera_block_present_when_image_set() -> None:
    msg = _build_message(
        role="user",
        text="Am I doing this right?",
        image="Hand wrapped around a wooden handle. Heavy metal head with a flat striking face.",
    )
    assert msg["role"] == "user"
    content = msg["content"]
    assert content.startswith("[Camera: ")
    assert content.endswith("Am I doing this right?")
    assert "]\nAm I doing this right?" in content


def test_no_camera_block_when_image_null() -> None:
    msg = _build_message(role="user", text="Where do I start?", image=None)
    assert msg == {"role": "user", "content": "Where do I start?"}


def test_no_camera_block_when_image_empty_string() -> None:
    msg = _build_message(role="user", text="Just a question.", image="")
    assert msg == {"role": "user", "content": "Just a question."}


def test_camera_block_format_is_exact() -> None:
    msg = _build_message(role="user", text="USER TEXT", image="IMAGE TEXT")
    assert msg["content"] == "[Camera: IMAGE TEXT]\nUSER TEXT"


def test_pre_turn_context_scene_description_message_is_user_only_camera_block() -> None:
    msg = _build_pre_turn_context_scene_description_message(
        "A workbench with several tools laid out."
    )
    assert msg["role"] == "user"
    assert msg["content"] == "[Camera: A workbench with several tools laid out.]"
    assert "\n" not in msg["content"]


def test_pre_turn_context_scene_description_injected_as_separate_message_before_t1() -> None:
    task = _make_task(
        pre_turn_context_scene_description="A long workbench with a vise on the left and several "
        "tools laid out on a pegboard above.",
    )
    answers = AnswerSet(
        current_answers=["a", "b", "c"],
        prior_answers=[],
        clarify_indicators=[],
        abstain_indicators=[],
    )
    condition = PromptCondition(name="test", description="d", system_prompt="sys", token_count=0)
    adapter = _CapturingAdapter()
    judge = _PassingJudge()

    run_module._run_one_trial(
        task=task,
        answers=answers,
        condition=condition,
        trial=0,
        adapter=adapter,
        judge=judge,
        model_config=ModelConfig(model_id="stub", temperature=0.0),
    )

    turn_1_messages = adapter.calls[0]
    assert len(turn_1_messages) == 2
    leading = turn_1_messages[0]
    assert leading["role"] == "user"
    assert leading["content"].startswith("[Camera: ")
    assert leading["content"] == (
        "[Camera: A long workbench with a vise on the left and several "
        "tools laid out on a pegboard above.]"
    )
    assert "\n" not in leading["content"]
    t1_user = turn_1_messages[1]
    assert t1_user["content"].startswith("[Camera: ")
    assert t1_user["content"].endswith(task.turn_1_user)


def test_no_pre_turn_context_scene_description_message_when_field_is_null() -> None:
    task = _make_task(pre_turn_context_scene_description=None)
    answers = AnswerSet(
        current_answers=["a", "b", "c"],
        prior_answers=[],
        clarify_indicators=[],
        abstain_indicators=[],
    )
    condition = PromptCondition(name="test", description="d", system_prompt="sys", token_count=0)
    adapter = _CapturingAdapter()
    run_module._run_one_trial(
        task=task,
        answers=answers,
        condition=condition,
        trial=0,
        adapter=adapter,
        judge=_PassingJudge(),
        model_config=ModelConfig(model_id="stub", temperature=0.0),
    )
    turn_1_messages = adapter.calls[0]
    assert len(turn_1_messages) == 1
    assert turn_1_messages[0]["content"].startswith("[Camera: ")


def test_t2_message_contains_camera_block_when_image_set() -> None:
    task = _make_task(
        turn_1_scene_description="Hand on a slim cylindrical metal shaft.",
        turn_2_scene_description="Hand on a flat plastic grip with three colored buttons.",
    )
    answers = AnswerSet(
        current_answers=["a", "b", "c"],
        prior_answers=[],
        clarify_indicators=[],
        abstain_indicators=[],
    )
    condition = PromptCondition(name="test", description="d", system_prompt="sys", token_count=0)
    adapter = _CapturingAdapter()
    run_module._run_one_trial(
        task=task,
        answers=answers,
        condition=condition,
        trial=0,
        adapter=adapter,
        judge=_PassingJudge(),
        model_config=ModelConfig(model_id="stub", temperature=0.0),
    )
    turn_2_messages = adapter.calls[1]
    last = turn_2_messages[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("[Camera: ")
    assert "Hand on a flat plastic grip" in last["content"]
    assert last["content"].endswith(task.turn_2_user)


def test_judge_receives_ground_truth_context() -> None:
    prompt = _build_user_prompt(
        response="The hammer should hit straight.",
        task_description="Object swap; target current.",
        turn_2_user="Am I doing this right?",
        current_answers=["hammer", "swing"],
        prior_answers=["screwdriver"],
        clarify_indicators=[],
        abstain_indicators=[],
        ground_truth_context=(
            "Cue type: object_in_hand. Turn 1 was a screwdriver, Turn 2 is a hammer."
        ),
    )
    assert "GROUND TRUTH" in prompt
    assert "judge-only" in prompt.lower()
    assert "Turn 1 was a screwdriver" in prompt
    assert "Turn 2 is a hammer" in prompt


def test_judge_omits_ground_truth_section_when_not_provided() -> None:
    prompt = _build_user_prompt(
        response="The hammer should hit straight.",
        task_description="Object swap.",
        turn_2_user="Am I doing this right?",
        current_answers=["hammer"],
        prior_answers=["screwdriver"],
        clarify_indicators=[],
        abstain_indicators=[],
    )
    assert "GROUND TRUTH" not in prompt


def test_runner_passes_ground_truth_to_judge() -> None:
    task = _make_task(
        turn_1_scene_description="Hand on a slim metal shaft with a phillips tip.",
        turn_2_scene_description="Hand wrapped around a wooden handle with a heavy steel head.",
    )
    answers = AnswerSet(
        current_answers=["a", "b", "c"],
        prior_answers=[],
        clarify_indicators=[],
        abstain_indicators=[],
    )

    captured: dict[str, Any] = {}

    class _RecordingJudge:
        family = "stub"
        model_id = "stub-model"

        def label(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            from wearable_assistant_context_bench.llm_judge import JudgeVerdict

            return JudgeVerdict(selected_label="current", rationale="stub")

    condition = PromptCondition(name="test", description="d", system_prompt="sys", token_count=0)
    adapter = _CapturingAdapter()
    run_module._run_one_trial(
        task=task,
        answers=answers,
        condition=condition,
        trial=0,
        adapter=adapter,
        judge=_RecordingJudge(),
        model_config=ModelConfig(model_id="stub", temperature=0.0),
    )
    assert "ground_truth_context" in captured
    gt = captured["ground_truth_context"]
    assert isinstance(gt, str) and gt
    assert "Turn 1 camera state" in gt
    assert "Turn 2 camera state" in gt
    assert task.turn_1_scene_description in gt
    assert task.turn_2_scene_description in gt


# ---------------------------------------------------------------------------
# Prompt conditions (was test_interventions.py)
# ---------------------------------------------------------------------------


PROJECT_PROMPT_CONDITIONS = REPO_ROOT / "data" / "prompt_conditions.json"


EXPECTED_BASELINE = "You are an assistant helping a user with an ongoing project."
EXPECTED_CONDITION_A_FRAGMENT = "visual context"
EXPECTED_CONDITION_B_FRAGMENT = "RELEVANT CONTEXT"


def test_load_prompt_conditions_from_valid_fixture(prompt_conditions_sample_path: Path) -> None:
    conditions = load_prompt_conditions(prompt_conditions_sample_path)
    assert len(conditions) == 3
    names = [c.name for c in conditions]
    assert names == ["baseline", "context_selection_instruction", "pre_answer_context_scaffold"]
    assert all(isinstance(c, PromptCondition) for c in conditions)
    assert all(c.system_prompt for c in conditions)
    assert all(c.token_count > 0 for c in conditions)


def test_prompt_context_selection_instructionliases_match_existing_loader(
    prompt_conditions_sample_path: Path,
) -> None:
    prompt_conditions = load_prompt_conditions(prompt_conditions_sample_path)
    prompt_conditions_again = load_prompt_conditions(prompt_conditions_sample_path)
    assert prompt_conditions == prompt_conditions_again
    assert all(isinstance(c, PromptCondition) for c in prompt_conditions)


def test_load_prompt_conditions_raises_on_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_prompt_conditions(bad)


def test_get_prompt_condition_returns_context_selection_instruction(
    prompt_conditions_sample_path: Path,
) -> None:
    conditions = load_prompt_conditions(prompt_conditions_sample_path)
    condition = get_prompt_condition(conditions, "context_selection_instruction")
    assert condition.name == "context_selection_instruction"
    assert "visual context" in condition.system_prompt.lower()


def test_get_prompt_condition_returns_pre_answer_context_scaffold(
    prompt_conditions_sample_path: Path,
) -> None:
    conditions = load_prompt_conditions(prompt_conditions_sample_path)
    condition = get_prompt_condition(conditions, "pre_answer_context_scaffold")
    assert condition.name == "pre_answer_context_scaffold"
    assert "relevant context" in condition.system_prompt.lower()


def test_get_prompt_condition_raises_on_unknown(
    prompt_conditions_sample_path: Path,
) -> None:
    conditions = load_prompt_conditions(prompt_conditions_sample_path)
    with pytest.raises(ValueError):
        get_prompt_condition(conditions, "does_not_exist")


def test_project_prompt_conditions_json_loads_three_conditions() -> None:
    conditions = load_prompt_conditions(PROJECT_PROMPT_CONDITIONS)
    assert [c.name for c in conditions] == [
        "baseline",
        "context_selection_instruction",
        "pre_answer_context_scaffold",
    ]


def test_project_prompt_conditions_baseline_matches_expected_text() -> None:
    conditions = load_prompt_conditions(PROJECT_PROMPT_CONDITIONS)
    by_name = {c.name: c.system_prompt for c in conditions}
    assert by_name["baseline"] == EXPECTED_BASELINE
    assert EXPECTED_CONDITION_A_FRAGMENT in by_name["context_selection_instruction"].lower()
    assert EXPECTED_CONDITION_B_FRAGMENT in by_name["pre_answer_context_scaffold"]


def test_prompt_conditions_are_policy_neutral() -> None:
    conditions = load_prompt_conditions(PROJECT_PROMPT_CONDITIONS)
    forbidden_snippets = (
        "always answer based on the user's current state",
        "only that current state as your ground truth",
        "only that current state as ground truth",
    )
    for condition in conditions:
        lowered = condition.system_prompt.lower()
        for snippet in forbidden_snippets:
            assert snippet not in lowered, (
                f"{condition.name} contains policy-forcing snippet: {snippet!r}"
            )


# ---------------------------------------------------------------------------
# Public-doc terminology (was test_terminology.py)
# ---------------------------------------------------------------------------


PUBLIC_PATHS = (
    "README.md",
    "docs/benchmark_spec.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_public_docs_avoid_legacy_reference_state_language() -> None:
    forbidden = (
        "reference-state",
        "reference-state selection",
        "v1 runnable slice",
        "sub-capability",
        "deictic grounding",
        "referent arbitration",
        "context-policy arbitration",
        "multimodal situational grounding",
        "post-shift",
        "implicit context tracking",
        "with-prior-q",
        # Version-anchored prose. Public docs use generic language ("the
        # current version", "the benchmark", "currently") rather than
        # naming a specific release. The three version constants and the
        # formal metadata fields carry version strings; prose does not.
        "v1 release",
        "v1 publishes",
        "v1 release runs",
        "v1.x",
        "v2-dev",
        "v0.1 ships",
        "v0.1 uses",
        "as of v0.1",
        "in v0.1",
        "in v1",
    )
    for path in PUBLIC_PATHS:
        lowered = _read(path).lower()
        for term in forbidden:
            assert term not in lowered, f"{path} still contains {term!r}"


def test_public_docs_avoid_legacy_task_set_files() -> None:
    """The legacy difficult/hard task-set files are no longer accepted."""
    legacy = ("difficult pack", "tasks_difficult.json")
    for path in ("README.md", "docs/benchmark_spec.md"):
        lowered = _read(path).lower()
        for term in legacy:
            assert term not in lowered, f"{path} still contains {term!r}"


def test_public_docs_use_correct_framing() -> None:
    required = (
        "wearable",
        "camera",
        "current",
        "prior",
        "clarify",
        "abstain",
    )
    for path in ("README.md", "docs/benchmark_spec.md"):
        lowered = _read(path).lower()
        for phrase in required:
            assert phrase in lowered, f"{path} is missing required phrase {phrase!r}"


# ---------------------------------------------------------------------------
# Packaged runtime data (wheel-install path)
# ---------------------------------------------------------------------------


def test_runtime_data_is_available_as_package_resources() -> None:
    """The runtime JSON/JSONL files must ship inside the installed package.

    A wheel install lacks the source ``data/`` tree at the repo root, so
    the runner falls back to ``importlib.resources``. This test asserts
    that fallback is wired correctly and that the packaged copies match
    the source-of-truth files byte-for-byte.
    """
    resource_root = files("wearable_assistant_context_bench").joinpath("data")
    for name in ("config.json", "prompt_conditions.json", "tasks.jsonl"):
        packaged = resource_root.joinpath(name)
        source = REPO_ROOT / "data" / name
        assert packaged.is_file(), f"missing packaged runtime data: {name}"
        assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
