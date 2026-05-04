"""Shared model configuration types used across adapters.

This module holds the small dataclasses that the runner and the
adapters share. The actual API integrations live in the
provider-specific adapter modules:

- ``wearable_assistant_context_bench.gemini_adapter`` for the native
  Gemini SDK transport
- ``wearable_assistant_context_bench.litellm_adapter`` for the unified
  LiteLLM transport (used for
  Claude, OpenAI, OpenRouter-routed models, and any other provider
  qualified by a slash in the model id)

Claude family models are reached through the LiteLLM transport
(for example ``openrouter/anthropic/claude-sonnet-4.6``); there is no
native Anthropic adapter in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MODEL_ID = "openrouter/anthropic/claude-sonnet-4.6"


@dataclass
class ModelConfig:
    """Configuration for a single model query.

    Attributes:
        model_id: Pinned model identifier. The runner picks an adapter
            based on the prefix; provider-qualified ids like
            ``openrouter/...`` and ``gemini/...`` route through
            LiteLLM, while bare Gemini ids route through the native
            Gemini adapter.
        temperature: Sampling temperature. Runs use 0.0 for
            reproducibility.
        max_tokens: Upper bound on response length in tokens.
    """

    model_id: str = DEFAULT_MODEL_ID
    temperature: float = 0.0
    max_tokens: int = 1024
