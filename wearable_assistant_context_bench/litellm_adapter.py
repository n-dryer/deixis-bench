"""LiteLLM-backed adapter for provider-qualified candidate and judge models.

This adapter keeps the same `query(messages, system, config) -> str`
interface used by the existing native adapters while routing calls
through LiteLLM's unified completion API. It is used for broader
provider coverage, including OpenAI, OpenRouter, and Hugging Face
Inference-backed chat-completion models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wearable_assistant_context_bench.models import ModelConfig

_logger = logging.getLogger(__name__)

DEFAULT_LITELLM_CACHE_DIR = Path(".cache/litellm_models")
DEFAULT_LITELLM_TIMEOUT_SECONDS = 120
DEFAULT_LITELLM_RETRY_ATTEMPTS = 4
# Exponential backoff base (seconds): 2, 4, 8, 16 by default.
DEFAULT_LITELLM_RETRY_BASE_SECONDS = 2.0

# Substrings in exception messages that signal a transient server-side
# condition worth retrying with backoff. Anything not matching falls
# through immediately so genuine errors (auth, missing model, etc.)
# don't waste retries.
_TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "503",
    "unavailable",
    "high demand",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "overloaded",
    "service_unavailable",
    "internal server error",
    "502",
    "504",
    "timeout",
    "timed out",
)


def _is_transient_error(exc: BaseException) -> bool:
    text = (str(exc) + " " + type(exc).__name__).lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


# Custom-endpoint prefix routing. Each entry maps a virtual model-id
# prefix to (litellm_model_prefix, api_base, env_var_for_api_key).
# This keeps the CLI ergonomic for OpenAI-compatible endpoints whose
# default LiteLLM provider points at the wrong region or schema.
_CUSTOM_ENDPOINT_ROUTES: dict[str, tuple[str, str, str]] = {
    # Alibaba Cloud Model Studio - Singapore region (DashScope
    # International). OpenAI-compatible endpoint; key in DASHSCOPE_API_KEY.
    "dashscope-intl/": (
        "openai/",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
}


def _resolve_custom_endpoint(model_id: str) -> dict[str, Any] | None:
    """If ``model_id`` carries a custom-endpoint prefix, return the
    extra kwargs LiteLLM needs (rewritten ``model``, ``api_base``,
    ``api_key``). Returns ``None`` for ordinary provider-qualified
    ids that LiteLLM can route on its own."""
    import os

    for prefix, (litellm_prefix, api_base, env_var) in _CUSTOM_ENDPOINT_ROUTES.items():
        if model_id.startswith(prefix):
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"{prefix!r} model requires {env_var} env var to be set"
                )
            return {
                "model": litellm_prefix + model_id[len(prefix):],
                "api_base": api_base,
                "api_key": api_key,
            }
    return None


class LiteLLMAdapter:
    """Thin cached wrapper around `litellm.completion`."""

    def __init__(
        self,
        client: Callable[..., Any] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._cache_dir = (
            cache_dir if cache_dir is not None else DEFAULT_LITELLM_CACHE_DIR
        )

    @property
    def client(self) -> Callable[..., Any]:
        if self._client is None:
            from litellm import completion

            self._client = completion
        return self._client

    def _cache_key(
        self,
        messages: list[dict],
        system: str,
        config: ModelConfig,
    ) -> str:
        payload = {
            "backend": "litellm",
            "messages": messages,
            "system": system,
            "config": asdict(config),
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> str | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("response")

    def _store_cached(self, key: str, response: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._cache_path(key).open("w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False)

    def query(
        self,
        messages: list[dict],
        system: str,
        config: ModelConfig,
    ) -> str:
        """Send a multi-turn message history through LiteLLM."""
        cache_key = self._cache_key(messages, system, config)
        cached = self._load_cached(cache_key)
        if cached is not None:
            return cached

        payload_messages = list(messages)
        if system:
            payload_messages = [
                {"role": "system", "content": system},
                *payload_messages,
            ]

        call_kwargs: dict[str, Any] = {
            "model": config.model_id,
            "messages": payload_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": DEFAULT_LITELLM_TIMEOUT_SECONDS,
            "stream": False,
        }
        custom = _resolve_custom_endpoint(config.model_id)
        if custom is not None:
            call_kwargs.update(custom)
        response = self._call_with_retry(**call_kwargs)

        text = _extract_text(response)
        self._store_cached(cache_key, text)
        return text

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call ``self.client`` with exponential backoff on transient errors.

        Retries on 5xx, rate-limit, and timeout-style messages. Other
        exceptions (auth, model-not-found, malformed request) re-raise
        immediately. After the final attempt, the original exception
        propagates.
        """
        last_exc: BaseException | None = None
        for attempt in range(DEFAULT_LITELLM_RETRY_ATTEMPTS):
            try:
                return self.client(**kwargs)
            except Exception as exc:  # noqa: BLE001 - we re-raise non-transient
                if not _is_transient_error(exc):
                    raise
                last_exc = exc
                if attempt == DEFAULT_LITELLM_RETRY_ATTEMPTS - 1:
                    break
                sleep_for = DEFAULT_LITELLM_RETRY_BASE_SECONDS * (2 ** attempt)
                _logger.warning(
                    "litellm transient error on attempt %d/%d (%s): %.150s; "
                    "sleeping %.1fs",
                    attempt + 1,
                    DEFAULT_LITELLM_RETRY_ATTEMPTS,
                    type(exc).__name__,
                    str(exc),
                    sleep_for,
                )
                time.sleep(sleep_for)
        assert last_exc is not None  # only reachable after a transient failure
        raise last_exc


def _extract_text(response: Any) -> str:
    """Pull plain text from a LiteLLM/OpenAI-style chat response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    first = choices[0]
    message = None
    if isinstance(first, dict):
        message = first.get("message") or first.get("delta")
    else:
        message = getattr(first, "message", None) or getattr(first, "delta", None)
    if message is None:
        return ""

    content = message.get("content") if isinstance(message, dict) else getattr(
        message, "content", None
    )
    return _normalize_content(content)


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                    continue
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                    continue
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(content)
