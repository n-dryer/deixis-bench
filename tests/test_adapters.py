"""Adapter unit tests (Gemini + LiteLLM).

These tests never hit a real provider API - they inject stub clients
that record their calls and return canned responses. The shared
``_StubGeminiClient`` and ``_StubCompletion`` stubs live in
``tests/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from wearable_assistant_context_bench.gemini_adapter import GeminiAdapter
from wearable_assistant_context_bench.litellm_adapter import LiteLLMAdapter
from wearable_assistant_context_bench.models import ModelConfig


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "adapter-cache"


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------


def test_gemini_instantiates_without_api_call(stub_gemini_client) -> None:
    adapter = GeminiAdapter(client=stub_gemini_client([]))
    assert adapter._client is not None


def test_gemini_query_returns_canned_text(stub_gemini_client, cache_dir: Path) -> None:
    stub = stub_gemini_client(responses=["hello from gemini"])
    adapter = GeminiAdapter(client=stub, cache_dir=cache_dir)
    out = adapter.query(
        messages=[{"role": "user", "content": "hi"}],
        system="SYS",
        config=ModelConfig(model_id="gemini-2.5-flash"),
    )
    assert out == "hello from gemini"
    assert len(stub.models.calls) == 1
    assert stub.models.calls[0]["model"] == "gemini-2.5-flash"


def test_gemini_repeat_call_hits_cache(stub_gemini_client, cache_dir: Path) -> None:
    stub = stub_gemini_client(responses=["cached"])
    adapter = GeminiAdapter(client=stub, cache_dir=cache_dir)
    config = ModelConfig(model_id="gemini-2.5-flash")
    messages = [{"role": "user", "content": "ping"}]
    first = adapter.query(messages=messages, system="SYS", config=config)
    second = adapter.query(messages=messages, system="SYS", config=config)
    assert first == "cached"
    assert second == "cached"
    assert len(stub.models.calls) == 1


def test_gemini_flash_lite_disables_thinking(stub_gemini_client, cache_dir: Path) -> None:
    stub = stub_gemini_client(responses=["x"])
    adapter = GeminiAdapter(client=stub, cache_dir=cache_dir)
    adapter.query(
        messages=[{"role": "user", "content": "hi"}],
        system="SYS",
        config=ModelConfig(model_id="gemini-2.5-flash-lite", max_tokens=512),
    )
    gen_config = stub.models.calls[0]["config"]
    assert gen_config.thinking_config is not None
    assert gen_config.thinking_config.thinking_budget == 0


def test_gemini_non_flash_lite_uses_default_thinking_with_headroom(
    stub_gemini_client, cache_dir: Path
) -> None:
    stub = stub_gemini_client(responses=["x"])
    adapter = GeminiAdapter(client=stub, cache_dir=cache_dir)
    adapter.query(
        messages=[{"role": "user", "content": "hi"}],
        system="SYS",
        config=ModelConfig(model_id="gemini-2.5-pro", max_tokens=512),
    )
    gen_config = stub.models.calls[0]["config"]
    assert gen_config.thinking_config is None
    assert gen_config.max_output_tokens >= 8192


# ---------------------------------------------------------------------------
# LiteLLMAdapter
# ---------------------------------------------------------------------------


def test_litellm_query_returns_canned_text(stub_completion, cache_dir: Path) -> None:
    stub = stub_completion(["hello from litellm"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    out = adapter.query(
        messages=[{"role": "user", "content": "hi"}],
        system="SYS",
        config=ModelConfig(model_id="openai/gpt-4.1-mini"),
    )
    assert out == "hello from litellm"
    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == "openai/gpt-4.1-mini"
    assert stub.calls[0]["timeout"] == 120


def test_litellm_repeat_call_hits_cache(stub_completion, cache_dir: Path) -> None:
    stub = stub_completion(["cached"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    config = ModelConfig(model_id="openai/gpt-4.1-mini")
    messages = [{"role": "user", "content": "ping"}]
    first = adapter.query(messages=messages, system="SYS", config=config)
    second = adapter.query(messages=messages, system="SYS", config=config)
    assert first == "cached"
    assert second == "cached"
    assert len(stub.calls) == 1


def test_litellm_system_prompt_is_prepended_as_message(
    stub_completion, cache_dir: Path
) -> None:
    stub = stub_completion(["x"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    adapter.query(
        messages=[{"role": "user", "content": "hi"}],
        system="SYS",
        config=ModelConfig(model_id="openai/gpt-4.1-mini"),
    )
    assert stub.calls[0]["messages"][0] == {"role": "system", "content": "SYS"}


class _FlakyCompletion:
    """Stub that raises a sequence of transient errors before succeeding."""

    def __init__(
        self, errors: list[Exception], final_text: str = "ok-after-retries"
    ) -> None:
        self._errors = list(errors)
        self._final_text = final_text
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return {"choices": [{"message": {"content": self._final_text}}]}


def test_litellm_retry_on_transient_503(cache_dir: Path, monkeypatch) -> None:
    """LiteLLM service-unavailable errors are retried with backoff."""
    import wearable_assistant_context_bench.litellm_adapter as la

    monkeypatch.setattr(la.time, "sleep", lambda *_: None)
    flaky = _FlakyCompletion(
        errors=[
            Exception("503 UNAVAILABLE: high demand"),
            Exception("RateLimitError: rate limit exceeded"),
        ]
    )
    adapter = LiteLLMAdapter(client=flaky, cache_dir=cache_dir)
    out = adapter.query(
        messages=[{"role": "user", "content": "ping"}],
        system="SYS",
        config=ModelConfig(model_id="gemini/gemini-2.5-flash-lite"),
    )
    assert out == "ok-after-retries"
    assert len(flaky.calls) == 3  # two retries + the final success


def test_litellm_no_retry_on_auth_error(cache_dir: Path, monkeypatch) -> None:
    """Non-transient errors (auth, model-not-found) re-raise immediately."""
    import wearable_assistant_context_bench.litellm_adapter as la

    monkeypatch.setattr(la.time, "sleep", lambda *_: None)

    class _Boom:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **kwargs: Any) -> Any:
            self.calls += 1
            raise Exception("AuthenticationError: invalid api key")

    boom = _Boom()
    adapter = LiteLLMAdapter(client=boom, cache_dir=cache_dir)
    with pytest.raises(Exception, match="invalid api key"):
        adapter.query(
            messages=[{"role": "user", "content": "ping"}],
            system="SYS",
            config=ModelConfig(model_id="openai/gpt-4o-mini"),
        )
    assert boom.calls == 1


def test_litellm_dashscope_intl_prefix_rewrites_to_openai_with_api_base(
    stub_completion, cache_dir: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-key-12345")
    stub = stub_completion(["routed"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    out = adapter.query(
        messages=[{"role": "user", "content": "ping"}],
        system="SYS",
        config=ModelConfig(model_id="dashscope-intl/qwen3-vl-plus"),
    )
    assert out == "routed"
    call = stub.calls[0]
    assert call["model"] == "openai/qwen3-vl-plus"
    assert call["api_base"] == (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    assert call["api_key"] == "sk-test-key-12345"


def test_litellm_dashscope_intl_prefix_errors_when_key_missing(
    stub_completion, cache_dir: Path, monkeypatch
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    stub = stub_completion(["unused"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        adapter.query(
            messages=[{"role": "user", "content": "ping"}],
            system="SYS",
            config=ModelConfig(model_id="dashscope-intl/qwen3-vl-plus"),
        )
    assert len(stub.calls) == 0


def test_litellm_retry_exhausted_reraises_last(cache_dir: Path, monkeypatch) -> None:
    import wearable_assistant_context_bench.litellm_adapter as la

    monkeypatch.setattr(la.time, "sleep", lambda *_: None)
    flaky = _FlakyCompletion(
        errors=[Exception(f"503 UNAVAILABLE attempt {i}") for i in range(10)]
    )
    adapter = LiteLLMAdapter(client=flaky, cache_dir=cache_dir)
    with pytest.raises(Exception, match="503 UNAVAILABLE"):
        adapter.query(
            messages=[{"role": "user", "content": "ping"}],
            system="SYS",
            config=ModelConfig(model_id="gemini/gemini-2.5-flash-lite"),
        )
    assert len(flaky.calls) == 4


@pytest.mark.parametrize(
    "model_id",
    [
        "huggingface/together/openai/gpt-oss-120b",
        "huggingface/together/Qwen/Qwen2.5-VL-7B-Instruct",
        "huggingface/together/meta-llama/Llama-3.2-11B-Vision-Instruct",
        "huggingface/fireworks-ai/Qwen/Qwen2.5-VL-72B-Instruct",
    ],
)
def test_litellm_huggingface_inference_providers_model_id_passes_through(
    stub_completion, cache_dir: Path, model_id: str
) -> None:
    stub = stub_completion(["ok"])
    adapter = LiteLLMAdapter(client=stub, cache_dir=cache_dir)
    out = adapter.query(
        messages=[{"role": "user", "content": "ping"}],
        system="SYS",
        config=ModelConfig(model_id=model_id),
    )
    assert out == "ok"
    assert stub.calls[0]["model"] == model_id
