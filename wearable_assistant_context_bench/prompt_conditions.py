"""Prompt-condition loading.

The benchmark stores its three prompt conditions in
``data/prompt_conditions.json``. Each condition is a system prompt
applied uniformly to every scenario in a run; the runner iterates over
all conditions per scenario.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path


@dataclass
class PromptCondition:
    """A single prompt condition applied to every scenario.

    Attributes:
        name: Short identifier such as ``baseline`` or ``condition_a``.
        description: One-line summary of the prompt strategy.
        system_prompt: Full system prompt sent to the candidate model.
        token_count: Approximate prompt length used for prompt-overhead reporting.
    """

    name: str
    description: str
    system_prompt: str
    token_count: int


def load_prompt_conditions(path: Path | Traversable) -> list[PromptCondition]:
    """Load prompt conditions from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise TypeError(
            f"Expected top-level JSON list of conditions, got {type(raw).__name__}"
        )

    conditions: list[PromptCondition] = []
    for entry in raw:
        conditions.append(
            PromptCondition(
                name=entry["name"],
                description=entry["description"],
                system_prompt=entry["system_prompt"],
                token_count=int(entry["token_count"]),
            )
        )
    return conditions


def get_prompt_condition_by_name(
    conditions: list[PromptCondition], name: str
) -> PromptCondition:
    """Return the prompt condition with the given name."""
    for condition in conditions:
        if condition.name == name:
            return condition
    known = ", ".join(c.name for c in conditions) or "(none loaded)"
    raise ValueError(f"Unknown prompt condition name {name!r}. Known: {known}")
