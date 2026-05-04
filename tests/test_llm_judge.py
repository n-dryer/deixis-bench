"""Tests for the LLM judge: label parsing, family selection, and the
no-leakage constraints on what the judge prompt may contain.

The judge labels each Turn 2 response without seeing the scenario's
``target_context`` label, the ``shift_type`` shift category, or the
authoring ``notes``. Those fields would tell the judge the answer it
is being asked to produce. These tests verify the prompt-building
helpers respect those constraints.
"""

from __future__ import annotations

from typing import Any

import pytest

from wearable_assistant_context_bench.llm_judge import (
    ALLOWED_POLICIES,
    JUDGE_MODEL_ID_CLAUDE,
    JUDGE_MODEL_ID_GEMINI,
    JUDGE_MODEL_ID_OPENAI,
    JUDGE_SYSTEM_PROMPT,
    GeminiJudgeAdapter,
    JudgeAdapterBase,
    LiteLLMJudgeAdapter,
    LLMJudge,
    OpenAIJudgeAdapter,
    _build_user_prompt,
    build_judge,
    infer_candidate_family,
    parse_verdict,
    resolve_judge_family,
)
from wearable_assistant_context_bench.runner import (
    SCENARIOS_PATH,
    _build_ground_truth_context,
    _build_scenario_description,
    load_scenarios,
)

# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


def test_judge_prompt_mentions_all_four_policies() -> None:
    for policy in ALLOWED_POLICIES:
        assert f"`{policy}`" in JUDGE_SYSTEM_PROMPT, (
            f"policy {policy!r} should appear in the judge rubric"
        )


def test_parse_verdict_extracts_current_policy() -> None:
    raw = 'reasoning...\n{"selected_label": "current", "rationale": "reflects new scene."}'
    verdict = parse_verdict(raw)
    assert verdict.selected_label == "current"
    assert "reflects new scene" in verdict.rationale


def test_parse_verdict_accepts_all_four_policies() -> None:
    for policy in ALLOWED_POLICIES:
        raw = f'{{"selected_label": "{policy}", "rationale": "ok"}}'
        assert parse_verdict(raw).selected_label == policy


def test_parse_verdict_rejects_unknown_policy() -> None:
    raw = '{"selected_label": "nonsense", "rationale": "ok"}'
    with pytest.raises(ValueError):
        parse_verdict(raw)


def test_parse_verdict_falls_back_to_abstain_when_no_json_or_label_line() -> None:
    verdict = parse_verdict("no JSON object here, just bare prose")
    assert verdict.selected_label == "abstain"
    assert "no-verdict" in verdict.rationale.lower() or "fallback" in verdict.rationale.lower()


def test_parse_verdict_does_not_flip_on_contrastive_prose() -> None:
    raw = (
        "Reasoning: the response discusses prior context briefly, but "
        "ultimately grounds in the current frame.\n"
        "(no JSON block, no selected_policy line)"
    )
    verdict = parse_verdict(raw)
    assert verdict.selected_label == "abstain"


def test_parse_verdict_recovers_from_explicit_label_line() -> None:
    raw = "Reasoning: the response describes the new frame.\nselected_label: current"
    verdict = parse_verdict(raw)
    assert verdict.selected_label == "current"
    assert "recovered" in verdict.rationale.lower()


def test_parse_verdict_falls_back_on_malformed_json() -> None:
    verdict = parse_verdict('{"selected_label": "prior", "rationale": }')
    assert verdict.selected_label == "abstain"


def test_parse_verdict_falls_back_to_abstain_when_malformed_json_lacks_label_word() -> None:
    verdict = parse_verdict('{"selected_label": , "rationale": }')
    assert verdict.selected_label == "abstain"


def test_parse_verdict_takes_last_object_when_multiple_present() -> None:
    raw = (
        '{"selected_label": "prior", "rationale": "first"}\n'
        '{"selected_label": "current", "rationale": "second"}'
    )
    assert parse_verdict(raw).selected_label == "current"


# ---------------------------------------------------------------------------
# Family inference and auto resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("claude-sonnet-4-6", "claude"),
        ("claude-opus-4-7", "claude"),
        ("claude-haiku-4-5", "claude"),
        ("Sonnet-4.6", "claude"),
        ("gemini-2.5-flash", "gemini"),
        ("gemini-2.5-pro", "gemini"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("Gemini-2.5-Flash", "gemini"),
        ("openai/gpt-4.1-mini", "openai"),
        ("openrouter/openai/gpt-4.1-mini", "openai"),
        ("gpt-4.1-mini", "openai"),
        ("huggingface/together/openai/gpt-oss-120b", "openai"),
        ("huggingface/fireworks-ai/openai/gpt-oss-20b", "openai"),
        ("huggingface/together/Qwen/Qwen2.5-VL-7B-Instruct", None),
        ("huggingface/together/meta-llama/Llama-3.2-11B-Vision-Instruct", None),
        ("huggingface/fireworks-ai/Qwen/Qwen2.5-VL-72B-Instruct", None),
        ("something-unknown", None),
        ("", None),
    ],
)
def test_infer_candidate_family(model_id: str, expected: str | None) -> None:
    assert infer_candidate_family(model_id) == expected


def test_resolve_judge_family_auto_errors_for_open_hf_candidate() -> None:
    with pytest.raises(ValueError, match="could not infer the candidate family"):
        resolve_judge_family(
            "auto",
            "huggingface/together/Qwen/Qwen2.5-VL-7B-Instruct",
        )


def test_resolve_judge_family_explicit_returns_requested() -> None:
    family, mode = resolve_judge_family("claude", "gemini-2.5-flash")
    assert family == "claude"
    assert mode == "explicit"

    family, mode = resolve_judge_family("gemini", "claude-sonnet-4-6")
    assert family == "gemini"
    assert mode == "explicit"

    family, mode = resolve_judge_family("openai", "gemini-2.5-flash")
    assert family == "openai"
    assert mode == "explicit"


def test_resolve_judge_family_auto_picks_opposite_family() -> None:
    family, mode = resolve_judge_family("auto", "claude-sonnet-4-6")
    assert family == "gemini"
    assert mode == "auto"

    family, mode = resolve_judge_family("auto", "gemini-2.5-flash")
    assert family == "openai"
    assert mode == "auto"

    family, mode = resolve_judge_family("auto", "openai/gpt-4.1-mini")
    assert family == "gemini"
    assert mode == "auto"


def test_resolve_judge_family_auto_errors_on_unknown_candidate() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_judge_family("auto", "mystery-model-v7")
    assert "could not infer" in str(excinfo.value)


def test_resolve_judge_family_rejects_unknown_requested_value() -> None:
    with pytest.raises(ValueError):
        resolve_judge_family("mistral", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Judge adapter routing
# ---------------------------------------------------------------------------


class _StubAdapter(JudgeAdapterBase):
    family = "stub"

    def __init__(self, canned: str) -> None:
        self.canned = canned
        self.calls: list[dict[str, Any]] = []

    def call(self, *, system: str, user: str, model_id: str) -> str:
        self.calls.append({"system": system, "user": user, "model_id": model_id})
        return self.canned


def test_judge_label_round_trips_through_stub_adapter() -> None:
    stub = _StubAdapter('{"selected_label": "prior", "rationale": "answered from earlier clips"}')
    judge = LLMJudge(adapter=stub, model_id="stub-model")
    verdict = judge.label(
        response="I'll summarize the three clips from yesterday.",
        scenario_description="Turn 1 set up three uploaded clips.",
        turn_2_user="Tell me about my day yesterday.",
        current_answers=[],
        prior_answers=["art museum", "farmers market"],
        clarify_indicators=[],
        abstain_indicators=["don't have access"],
    )
    assert verdict.selected_label == "prior"
    assert "earlier" in verdict.rationale
    assert len(stub.calls) == 1
    payload = stub.calls[0]["user"]
    assert "TURN 2 USER MESSAGE" in payload
    assert "CURRENT ANSWERS" in payload
    assert "PRIOR ANSWERS" in payload
    assert "CLARIFY INDICATORS" in payload
    assert "ABSTAIN INDICATORS" in payload


def test_build_judge_defaults_to_family_default_model() -> None:
    judge = build_judge(family="claude", adapter=_StubAdapter("x"))
    assert judge.model_id == JUDGE_MODEL_ID_CLAUDE
    judge = build_judge(family="gemini", adapter=_StubAdapter("x"))
    assert judge.model_id == JUDGE_MODEL_ID_GEMINI
    judge = build_judge(family="openai", adapter=_StubAdapter("x"))
    assert judge.model_id == JUDGE_MODEL_ID_OPENAI


def test_build_judge_accepts_explicit_model_id() -> None:
    judge = build_judge(
        family="claude",
        model_id="claude-haiku-4-5",
        adapter=_StubAdapter("x"),
    )
    assert judge.model_id == "claude-haiku-4-5"


@pytest.mark.parametrize(
    "family,model_id",
    [
        ("claude", "openrouter/anthropic/claude-3.5-haiku"),
        ("gemini", "openrouter/google/gemini-2.5-flash"),
    ],
)
def test_build_judge_uses_litellm_for_provider_qualified_models(family: str, model_id: str) -> None:
    judge = build_judge(family=family, model_id=model_id)
    assert judge.model_id == model_id
    assert isinstance(judge._adapter, LiteLLMJudgeAdapter)
    assert judge.family == family


def test_build_judge_rejects_unknown_family() -> None:
    with pytest.raises(ValueError):
        build_judge(family="mistral")


def test_gemini_judge_adapter_family() -> None:
    assert GeminiJudgeAdapter.family == "gemini"


def test_openai_judge_adapter_family() -> None:
    assert OpenAIJudgeAdapter.family == "openai"


def test_litellm_judge_adapter_family_is_configurable() -> None:
    assert LiteLLMJudgeAdapter(family="claude").family == "claude"


# ---------------------------------------------------------------------------
# No-leakage constraints on what the judge can see
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(SCENARIOS_PATH)


def _label_naming_phrases(target_context: str) -> tuple[str, ...]:
    return (
        f"target context for turn 2 is `{target_context}`",
        f"target context for turn 2 is {target_context}",
        f"target context: {target_context}",
        f"target_context: {target_context}",
        f"target context is `{target_context}`",
        f"the correct label is {target_context}",
        f"correct policy is {target_context}",
        f"the answer is {target_context}",
    )


def test_scenario_description_does_not_name_target_context(scenarios) -> None:
    leaks: list[str] = []
    for scenario in scenarios:
        rendered = _build_scenario_description(scenario).lower()
        for phrase in _label_naming_phrases(scenario.target_context):
            if phrase in rendered:
                leaks.append(f"{scenario.scenario_id}: {phrase!r} found in scenario_description")
    assert not leaks, "scenario_description names target_context for:\n  - " + "\n  - ".join(leaks)


def test_ground_truth_context_omits_target_cue_and_notes(scenarios) -> None:
    leaks: list[str] = []
    for scenario in scenarios:
        rendered = _build_ground_truth_context(scenario)
        rendered_lower = rendered.lower()
        for phrase in _label_naming_phrases(scenario.target_context):
            if phrase in rendered_lower:
                leaks.append(f"{scenario.scenario_id}: {phrase!r} found in ground_truth_context")
        if scenario.shift_type and scenario.shift_type.lower() in rendered_lower:
            leaks.append(
                f"{scenario.scenario_id}: shift_type {scenario.shift_type!r} found in "
                f"ground_truth_context"
            )
        if scenario.notes and len(scenario.notes) >= 8:
            if scenario.notes.lower() in rendered_lower:
                leaks.append(
                    f"{scenario.scenario_id}: authoring notes appear in ground_truth_context"
                )
    assert not leaks, "ground_truth_context exposes privileged fields for:\n  - " + "\n  - ".join(
        leaks
    )


def test_full_rendered_judge_prompt_omits_privileged_fields(scenarios) -> None:
    """End-to-end: the full rendered judge user message contains no privileged fields."""
    sample_response = "An assistant response that mentions some object."
    leaks: list[str] = []
    for scenario in scenarios:
        rendered = _build_user_prompt(
            response=sample_response,
            scenario_description=_build_scenario_description(scenario),
            turn_2_user=scenario.turn_2_user,
            current_answers=scenario.gold.current_answers,
            prior_answers=scenario.gold.prior_answers,
            clarify_indicators=scenario.gold.clarify_indicators,
            abstain_indicators=scenario.gold.abstain_indicators,
            ground_truth_context=_build_ground_truth_context(scenario),
        )
        rendered_lower = rendered.lower()
        for phrase in _label_naming_phrases(scenario.target_context):
            if phrase in rendered_lower:
                leaks.append(f"{scenario.scenario_id}: {phrase!r} found in rendered judge prompt")
        if scenario.shift_type and scenario.shift_type.lower() in rendered_lower:
            leaks.append(
                f"{scenario.scenario_id}: shift_type {scenario.shift_type!r} found in "
                f"rendered judge prompt"
            )
        if scenario.notes and len(scenario.notes) >= 8:
            if scenario.notes.lower() in rendered_lower:
                leaks.append(
                    f"{scenario.scenario_id}: authoring notes appear in rendered judge prompt"
                )
    assert not leaks, "Rendered judge prompt exposes privileged fields for:\n  - " + "\n  - ".join(
        leaks
    )
