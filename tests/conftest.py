"""Pytest setup: add the project root to sys.path and define shared fixtures.

The fixtures here back the consolidated tests under ``tests/``. Adapter
and judge stubs are reused by the runner-facing tests in
``test_runner.py`` and the camera-injection assertions there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Inline prompt-conditions sample used by prompt-condition-loading tests.
_PROMPT_CONDITIONS_SAMPLE = [
    {
        "name": "baseline",
        "description": (
            "Minimal system prompt. Default comparison condition for the primary benchmark score."
        ),
        "system_prompt": ("You are an assistant helping a user with an ongoing project."),
        "token_count": 13,
    },
    {
        "name": "condition_a",
        "description": (
            "Direct policy-selection instruction. Asks the model to "
            "answer from the correct visual context (prior or current) "
            "based on the question, without hard-coding a direction."
        ),
        "system_prompt": (
            "You are an assistant helping a user with an ongoing project.\n"
            "The user's visual context may shift during the conversation - "
            "the assistant may see a new scene, a new object, or a new "
            "location between turns. Each question refers to one specific "
            "visual context: sometimes an earlier frame, sometimes the "
            "current frame. Before answering, decide which visual context "
            "the question actually refers to based on the wording, and "
            "answer from that context. Do not default to the most recent "
            "context if the question is clearly about an earlier one, and "
            "do not default to an earlier context if the question is "
            "clearly about the current one."
        ),
        "token_count": 130,
    },
    {
        "name": "condition_b",
        "description": (
            "Pre-answer scaffold. Requires the model to identify the "
            "relevant visual context (prior or current) before answering."
        ),
        "system_prompt": (
            "You are an assistant helping a user with an ongoing project."
            "\n\n"
            "Before answering any question, first decide which visual "
            "context the question refers to: the current frame (the one "
            "the user is showing you right now), or a prior frame (one "
            "from earlier in the conversation). Output a one-line summary "
            "naming the relevant context, then answer the question using "
            "only that context.\n\n"
            "Format your response exactly as:\n"
            "RELEVANT CONTEXT: [current | prior] - [one-line summary of "
            "which scene this answer is grounded in]\n"
            "ANSWER: [your answer]"
        ),
        "token_count": 112,
    },
]


@pytest.fixture
def prompt_conditions_sample_path(tmp_path: Path) -> Path:
    """Write the inline prompt-conditions sample to tmp_path and return the path.

    Replaces the on-disk ``tests/fixtures/prompt_conditions_sample.json``
    from earlier revisions; consolidating it into conftest removes a
    one-file subdir.
    """
    target = tmp_path / "prompt_conditions_sample.json"
    target.write_text(
        json.dumps(_PROMPT_CONDITIONS_SAMPLE, indent=2),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Provider-adapter stubs (used by tests/test_adapters.py)
# ---------------------------------------------------------------------------


class _StubResponse:
    """Shape-compatible with a `google.genai` GenerateContentResponse."""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubModels:
    """Records `generate_content` calls and returns canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else ""
        return _StubResponse(text)


class _StubGeminiClient:
    """Mimics `genai.Client` with a `models.generate_content(...)` surface."""

    def __init__(self, responses: list[str]) -> None:
        self.models = _StubModels(responses)


class _StubCompletion:
    """Callable stub that records completion kwargs and returns canned text."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else ""
        return {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def stub_gemini_client():
    return _StubGeminiClient


@pytest.fixture
def stub_completion():
    return _StubCompletion


@pytest.fixture
def stub_response():
    return _StubResponse


# ---------------------------------------------------------------------------
# Runner-facing stubs (used by tests/test_runner.py and others)
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Returns a deterministic canned response for every query."""

    def __init__(self, response: str = "STUB_RESPONSE") -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def query(self, messages: list[dict], system: str, config: Any) -> str:
        self.calls.append({"messages": list(messages), "system": system})
        return self._response


class _CapturingAdapter:
    """Records every call's `messages` list verbatim so tests can inspect
    the conversation history that was constructed."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def query(self, messages: list[dict], system: str, config: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        return "STUB_RESPONSE"


def _make_judge_verdict(policy: str = "current", rationale: str = "stub") -> Any:
    from wearable_assistant_context_bench.llm_judge import JudgeVerdict

    return JudgeVerdict(selected_label=policy, rationale=rationale)


class _StubJudge:
    """Always labels `current`. Exposes `.calls` for inspection."""

    family = "stub"
    model_id = "stub-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def label(
        self,
        response: str,
        scenario_description: str,
        turn_2_user: str,
        current_answers: list[str],
        prior_answers: list[str],
        clarify_indicators: list[str],
        abstain_indicators: list[str],
        ground_truth_context: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "response": response,
                "turn_2_user": turn_2_user,
                "ground_truth_context": ground_truth_context,
            }
        )
        return _make_judge_verdict("current", "stub")


@pytest.fixture
def stub_adapter_factory():
    return _StubAdapter


@pytest.fixture
def capturing_adapter_factory():
    return _CapturingAdapter


@pytest.fixture
def stub_judge_factory():
    return _StubJudge


# Shared sample sentences used in the per-class recall tests.
@pytest.fixture
def sample_trial_factory():
    def _make(
        *,
        scenario_id: str,
        condition: str = "baseline",
        trial: int = 0,
        target_context: str = "current",
        turn_2_passed: bool = True,
        turn_2_judge_label: str | None = None,
        turn_2_code_signals: dict | None = None,
        turn_3_repair_attempted: bool = False,
        turn_3_repair_passed: bool | None = None,
        pack: str = "main",
        pair_id: str | None = None,
        shift_type: str = "object_in_hand",
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "subset": pack,
            "pair_id": pair_id,
            "shift_type": shift_type,
            "condition": condition,
            "trial": trial,
            "target_context": target_context,
            "turn_2_passed": turn_2_passed,
            "turn_2_judge_label": (
                turn_2_judge_label
                if turn_2_judge_label is not None
                else (target_context if turn_2_passed else "abstain")
            ),
            "turn_2_code_signals": turn_2_code_signals or {},
            "turn_3_repair_attempted": turn_3_repair_attempted,
            "turn_3_repair_passed": turn_3_repair_passed,
        }

    return _make


@pytest.fixture(scope="session")
def scenarios_by_id() -> dict[str, dict]:
    """Map scenario_id -> scenario dict, loaded from data/scenarios.jsonl.

    Used by audit-rubric tests that exercise rules against real bank
    entries instead of inline fixtures.
    """
    scenarios_path = ROOT / "data" / "scenarios.jsonl"
    out: dict[str, dict] = {}
    with scenarios_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["scenario_id"]] = entry
    return out
