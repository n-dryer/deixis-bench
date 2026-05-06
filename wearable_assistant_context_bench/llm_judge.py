"""LLM-as-judge with support for multiple provider families.

The judge labels each Turn 2 response with exactly one label:
`current`, `prior`, `clarify`, or `abstain`. `prior` and
`current` are the scored labels; `clarify` and `abstain` stay visible
for diagnostics but count as wrong for the primary score (see
docs/benchmark_spec.md).

A `JudgeAdapter` abstraction lets the judge run against native Claude
or Gemini adapters plus a LiteLLM-backed OpenAI-family path. The
CLI's default `--judge-family auto`
resolves to a family **different** from the candidate's (see
`resolve_judge_family` and `infer_candidate_family`). Explicit
`--judge-family claude|gemini|openai` overrides that default.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from wearable_assistant_context_bench.litellm_adapter import LiteLLMAdapter
from wearable_assistant_context_bench.models import ModelConfig

_logger = logging.getLogger(__name__)


JUDGE_MODEL_ID_CLAUDE = "openrouter/anthropic/claude-sonnet-4.6"
JUDGE_MODEL_ID_GEMINI = "gemini-2.5-flash"
JUDGE_MODEL_ID_OPENAI = "openai/gpt-4.1-mini"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024


ALLOWED_POLICIES: tuple[str, ...] = ("current", "prior", "clarify", "abstain")


JUDGE_SYSTEM_PROMPT = """You are an evaluator labeling an AI assistant's \
response with exactly one judge label.

The available labels are defined as follows. Each scored response is \
labeled with exactly one:

- `current`: answer from the newest relevant state.
- `prior`: answer from an earlier state on purpose.
- `clarify`: ask a question because the intended context is \
ambiguous.
- `abstain`: decline because the available evidence is insufficient.

You will be given:
1. A brief description of the task and the state shift that \
occurred between Turn 1 and Turn 2.
2. The Turn 2 user message.
3. The assistant's Turn 2 response being labeled.
4. Four reference answer lists for the task (CURRENT, PRIOR, \
CLARIFY indicators, ABSTAIN indicators). These are provided so you can \
see which concrete tokens anchor each label for this task. They \
are not a scoring shortcut: a response can match tokens from one list \
and still be better described by another label.

Reason briefly about which label best describes the response, grounded \
in the definitions above.

Then emit a single JSON object on the final line with this exact shape:
{"selected_label": "<current|prior|clarify|abstain>", \
"rationale": "<one-sentence justification>"}

Output no text after the JSON object. `selected_label` MUST be exactly \
one of the four label names."""


@dataclass
class JudgeVerdict:
    """Structured verdict returned by the LLM judge.

    Attributes:
        selected_label: One of `current`, `prior`, `clarify`, `abstain`.
        rationale: One-sentence justification from the judge.
    """

    selected_label: str
    rationale: str


class JudgeAdapterBase(ABC):
    """Abstract judge adapter. Implementations wrap a single chat API call."""

    family: str

    @abstractmethod
    def call(self, *, system: str, user: str, model_id: str) -> str:
        """Send a single-shot system+user chat and return raw text."""


class GeminiJudgeAdapter(JudgeAdapterBase):
    """Gemini-backed judge adapter using the native google-genai SDK.

    Wraps `wearable_assistant_context_bench.gemini_adapter.GeminiAdapter` for single-shot
    system+user judge calls. Used when the resolved judge family is
    `gemini` and the model id is a bare Gemini name (no provider
    prefix); provider-qualified ids like
    ``openrouter/google/gemini-2.5-flash`` route through
    `LiteLLMJudgeAdapter` instead.
    """

    family = "gemini"

    def __init__(
        self,
        adapter: Any | None = None,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        from wearable_assistant_context_bench.gemini_adapter import GeminiAdapter

        self._adapter = adapter if adapter is not None else GeminiAdapter()
        self._temperature = temperature
        self._max_tokens = max_tokens

    def call(self, *, system: str, user: str, model_id: str) -> str:
        config = ModelConfig(
            model_id=model_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return self._adapter.query(
            messages=[{"role": "user", "content": user}],
            system=system,
            config=config,
        )


class OpenAIJudgeAdapter(JudgeAdapterBase):
    """LiteLLM-backed OpenAI-family judge adapter."""

    family = "openai"

    def __init__(
        self,
        adapter: LiteLLMAdapter | None = None,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        self._adapter = adapter if adapter is not None else LiteLLMAdapter()
        self._temperature = temperature
        self._max_tokens = max_tokens

    def call(self, *, system: str, user: str, model_id: str) -> str:
        config = ModelConfig(
            model_id=model_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return self._adapter.query(
            messages=[{"role": "user", "content": user}],
            system=system,
            config=config,
        )


class LiteLLMJudgeAdapter(JudgeAdapterBase):
    """LiteLLM-backed judge adapter for provider-qualified model IDs.

    This keeps explicit judge model strings like
    `openrouter/anthropic/...` or `openrouter/google/...` on the
    unified LiteLLM transport instead of forcing them through the
    native provider adapters.
    """

    def __init__(
        self,
        family: str,
        adapter: LiteLLMAdapter | None = None,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        self.family = family
        self._adapter = adapter if adapter is not None else LiteLLMAdapter()
        self._temperature = temperature
        self._max_tokens = max_tokens

    def call(self, *, system: str, user: str, model_id: str) -> str:
        config = ModelConfig(
            model_id=model_id,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return self._adapter.query(
            messages=[{"role": "user", "content": user}],
            system=system,
            config=config,
        )


class LLMJudge:
    """LLM judge for the four allowed labels. Family defaults to `auto`.

    The judge itself does not decide the family; the runner resolves it
    with `resolve_judge_family` and instantiates the correct
    `JudgeAdapterBase` subclass. This class just formats the prompt,
    dispatches through the adapter, and parses the verdict.
    """

    def __init__(
        self,
        adapter: JudgeAdapterBase | None = None,
        model_id: str = JUDGE_MODEL_ID_CLAUDE,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        """Construct the judge.

        Args:
            adapter: A JudgeAdapterBase instance. Defaults to a
                LiteLLM-backed Claude judge via OpenRouter, matching
                the historical default family. Tests inject a stub
                adapter.
            model_id: Judge model identifier. Must match the adapter's
                family; the runner enforces this via `resolve_judge_family`.
            temperature: Sampling temperature.
            max_tokens: Upper bound on judge response length.
        """
        self._adapter: JudgeAdapterBase = (
            adapter
            if adapter is not None
            else LiteLLMJudgeAdapter(
                family="claude",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self._model_id = model_id

    @property
    def family(self) -> str:
        return self._adapter.family

    @property
    def model_id(self) -> str:
        return self._model_id

    def label(
        self,
        response: str,
        task_description: str,
        turn_2_user: str,
        current_answers: list[str],
        prior_answers: list[str],
        clarify_indicators: list[str],
        abstain_indicators: list[str],
        ground_truth_context: str | None = None,
    ) -> JudgeVerdict:
        """Label the Turn 2 response with one of the four policies.

        Args:
            ground_truth_context: Optional plain-language description of
                the actual T1 vs. T2 situation, including the names of
                objects in frame. The judge uses this to determine
                whether the response reflects T2 (current) or T1 (prior)
                context. The candidate model never sees this - only the
                judge does.
        """
        user_message = _build_user_prompt(
            response=response,
            task_description=task_description,
            turn_2_user=turn_2_user,
            current_answers=current_answers,
            prior_answers=prior_answers,
            clarify_indicators=clarify_indicators,
            abstain_indicators=abstain_indicators,
            ground_truth_context=ground_truth_context,
        )
        raw = self._adapter.call(
            system=JUDGE_SYSTEM_PROMPT,
            user=user_message,
            model_id=self._model_id,
        )
        return parse_verdict(raw)


def infer_candidate_family(model_id: str) -> str | None:
    """Infer the candidate model's family from its `--model` string.

    Recognises native and provider-qualified ids for Claude, Gemini,
    and OpenAI. Routing-layer prefixes (``openrouter/...``,
    ``huggingface/<provider>/...``) are stripped recursively so the
    underlying family can be detected.

    Open-weights families served via Inference Providers (Llama, Qwen,
    Mistral, DeepSeek, Gemma, etc.) are intentionally returned as
    ``None`` - the cross-family judge map only covers Claude/Gemini/
    OpenAI today, so HF candidates must pass ``--judge-family``
    explicitly. The runner surfaces a clear error in that case.

    Returns:
        ``"claude"``, ``"gemini"``, ``"openai"``, or ``None``. The
        runner errors out on ``None`` in ``--judge-family auto``.
    """
    if not model_id:
        return None
    lowered = model_id.lower()
    if "/" in lowered:
        provider, remainder = lowered.split("/", 1)
        if provider in {"anthropic", "claude"}:
            return "claude"
        if provider in {"gemini", "google"}:
            return "gemini"
        if provider == "openai":
            return "openai"
        if provider == "openrouter" and remainder:
            return infer_candidate_family(remainder)
        if provider in {"vertexai", "vertex_ai"} and "gemini" in remainder:
            return "gemini"
        if provider == "huggingface" and remainder:
            # huggingface/<inference_provider>/<hf_org>/<hf_model>.
            # Strip the inference-provider segment so we can detect a
            # closed-family model (e.g. huggingface/together/openai/
            # gpt-oss-120b -> openai). For open-weights candidates
            # (Llama, Qwen, Mistral, etc.), the recursion returns None
            # and the caller sees a "pass --judge-family explicitly"
            # error.
            if "/" in remainder:
                _, inner = remainder.split("/", 1)
                inferred = infer_candidate_family(inner)
                if inferred is not None:
                    return inferred
            return None
    claude_markers = ("claude", "sonnet", "opus", "haiku")
    gemini_markers = ("gemini",)
    openai_prefixes = ("gpt-", "o1", "o3", "o4")
    if any(marker in lowered for marker in claude_markers):
        return "claude"
    if any(marker in lowered for marker in gemini_markers):
        return "gemini"
    if lowered.startswith(openai_prefixes):
        return "openai"
    return None


def resolve_judge_family(
    requested: str,
    candidate_model_id: str,
) -> tuple[str, str]:
    """Resolve the judge family from the CLI flag and the candidate model.

    Args:
        requested: Value of `--judge-family`: `auto`, `claude`,
            `gemini`, or `openai`.
        candidate_model_id: The `--model` string for the candidate.

    Returns:
        `(resolved_family, resolution_mode)` where `resolution_mode` is
        `"explicit"` or `"auto"`. Used by the report manifest.

    Raises:
        ValueError: If `requested` is not one of the allowed values, or
            if `auto` cannot infer a family for the candidate.
    """
    if requested not in ("auto", "claude", "gemini", "openai"):
        raise ValueError(
            f"--judge-family must be auto, claude, gemini, or openai; got {requested!r}"
        )
    if requested in ("claude", "gemini", "openai"):
        return requested, "explicit"
    candidate_family = infer_candidate_family(candidate_model_id)
    if candidate_family is None:
        raise ValueError(
            f"--judge-family auto could not infer the candidate family "
            f"from model string {candidate_model_id!r}. Pass --judge-family "
            "claude, gemini, or openai explicitly."
        )
    cross = {"claude": "gemini", "gemini": "openai", "openai": "gemini"}
    return (cross[candidate_family], "auto")


def build_judge(
    family: str,
    model_id: str | None = None,
    adapter: JudgeAdapterBase | None = None,
) -> LLMJudge:
    """Factory: build an LLMJudge for the given family.

    Args:
        family: `"claude"`, `"gemini"`, or `"openai"`.
        model_id: Optional explicit model string; defaults to the
            family's default judge model.
        adapter: Optional pre-built adapter (for tests).

    Returns:
        An `LLMJudge` wired to the correct adapter and model.
    """
    if family == "claude":
        resolved_model = model_id or JUDGE_MODEL_ID_CLAUDE
        if adapter is not None:
            adapter_ = adapter
        else:
            adapter_ = LiteLLMJudgeAdapter(family=family)
    elif family == "gemini":
        resolved_model = model_id or JUDGE_MODEL_ID_GEMINI
        if adapter is not None:
            adapter_ = adapter
        elif "/" in resolved_model:
            adapter_ = LiteLLMJudgeAdapter(family=family)
        else:
            adapter_ = GeminiJudgeAdapter()
    elif family == "openai":
        resolved_model = model_id or JUDGE_MODEL_ID_OPENAI
        if adapter is not None:
            adapter_ = adapter
        elif "/" in resolved_model:
            adapter_ = LiteLLMJudgeAdapter(family=family)
        else:
            adapter_ = OpenAIJudgeAdapter()
    else:
        raise ValueError(f"Unknown judge family: {family!r}")
    return LLMJudge(adapter=adapter_, model_id=resolved_model)


def _build_user_prompt(
    *,
    response: str,
    task_description: str,
    turn_2_user: str,
    current_answers: list[str],
    prior_answers: list[str],
    clarify_indicators: list[str],
    abstain_indicators: list[str],
    ground_truth_context: str | None = None,
) -> str:
    """Format the judge user message.

    The optional `ground_truth_context` field gives the judge plain-language
    information about what's actually in the T1 vs. T2 frames (object
    names, scene descriptions). The candidate model never sees this; the
    judge uses it to decide whether the response reflects current or
    prior context.
    """
    current_block = _format_list(current_answers)
    prior_block = _format_list(prior_answers)
    clarify_block = _format_list(clarify_indicators)
    abstain_block = _format_list(abstain_indicators)
    ground_truth_section = ""
    if ground_truth_context:
        ground_truth_section = (
            f"GROUND TRUTH (judge-only, not visible to the candidate):\n{ground_truth_context}\n\n"
        )
    return (
        f"TASK (Turn 1 context and state shift):\n{task_description}\n\n"
        f"{ground_truth_section}"
        f"TURN 2 USER MESSAGE:\n{turn_2_user}\n\n"
        f"RESPONSE TO LABEL:\n{response}\n\n"
        f"CURRENT ANSWERS (anchors for the `current` policy):\n{current_block}\n\n"
        f"PRIOR ANSWERS (anchors for the `prior` policy):\n{prior_block}\n\n"
        f"CLARIFY INDICATORS (anchors for the `clarify` policy):\n{clarify_block}\n\n"
        f"ABSTAIN INDICATORS (anchors for the `abstain` policy):\n{abstain_block}\n\n"
        "Reason briefly, then emit the JSON verdict."
    )


def _format_list(items: list[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\"selected_label\"[^{}]*\}", re.DOTALL)
# Strict final-line patterns the judge can emit instead of (or after) a
# JSON block. Anchored to "selected_label" so we never confuse a
# casual occurrence of "prior" or "current" inside reasoning prose
# with the verdict.
_LABEL_LINE_RE = re.compile(
    r"selected[_\s-]?label\s*[:=]\s*['\"`]?(current|prior|clarify|abstain)\b",
    re.IGNORECASE,
)


def _strict_label_line(raw: str) -> str | None:
    """Extract a label only from an explicit ``selected_label: X`` line.

    Unlike the previous backwards-scan heuristic, this never flips on
    contrastive prose like *"the response uses prior context but I will
    select current"* - the only signal it accepts is the judge naming
    the field by name.
    """
    if not raw:
        return None
    matches = list(_LABEL_LINE_RE.finditer(raw))
    if not matches:
        return None
    # Use the *last* explicit "selected_label: X" mention; if the judge
    # restated the verdict, the final restatement is the conclusion.
    return matches[-1].group(1).lower()


def parse_verdict(raw: str) -> JudgeVerdict:
    """Extract a JudgeVerdict from the raw judge response text.

    The judge is instructed to end with a JSON object containing
    ``selected_label`` and ``rationale``. The parser tolerates
    step-by-step reasoning before the JSON block.

    When no JSON block is found, the parser falls back to a strict
    ``selected_label: <label>`` regex anchored to the field name. It
    deliberately does NOT scan for bare label words anywhere in the
    text, because contrastive reasoning like *"the response uses prior
    context but I will select current"* would otherwise return the
    wrong label depending on word order. If neither a JSON block nor
    a ``selected_label:`` line is present, the parser returns
    ``abstain`` with a logged warning rather than guessing.

    Raises:
        ValueError: If a JSON block is parsed but its ``selected_label``
            value is not one of the allowed labels.
    """
    if not (raw or "").strip():
        return JudgeVerdict(
            selected_label="abstain",
            rationale=("(no-response fallback - candidate or judge returned empty)"),
        )

    matches = list(_JSON_OBJECT_RE.finditer(raw))
    if matches:
        try:
            payload: dict[str, Any] = json.loads(matches[-1].group(0))
        except json.JSONDecodeError:
            # JSON-shaped block but malformed (e.g. stray escape). Fall
            # through to the strict label-line check below.
            payload = {}
        if payload:
            selected_label = payload.get("selected_label")
            if not isinstance(selected_label, str) or selected_label not in ALLOWED_POLICIES:
                raise ValueError(
                    f"Judge verdict 'selected_label' must be one of "
                    f"{ALLOWED_POLICIES}, got {selected_label!r}"
                )
            rationale = str(payload.get("rationale", "")).strip()
            return JudgeVerdict(selected_label=selected_label, rationale=rationale)

    label = _strict_label_line(raw)
    if label is not None:
        _logger.warning(
            "judge verdict had no JSON block; recovered from "
            "'selected_label: %s' line. Raw length=%d.",
            label,
            len(raw),
        )
        return JudgeVerdict(
            selected_label=label,
            rationale="(recovered from selected_label line; no JSON block)",
        )

    _logger.warning(
        "judge verdict had no JSON block and no 'selected_label:' line; "
        "falling back to abstain. Raw preview=%r",
        (raw[:200] + "...") if len(raw) > 200 else raw,
    )
    return JudgeVerdict(
        selected_label="abstain",
        rationale="(no-verdict fallback - neither JSON nor selected_label line found)",
    )
