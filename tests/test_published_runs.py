"""Reproducibility check for runs under ``data/published-runs/``.

Each published run records a SHA256 hash of the inputs that fed it
(task set, prompt conditions, judge prompt) inside the
``## Reproducibility manifest`` block of its ``findings.md``. This test
recomputes those hashes from the live repo files and fails if they
have drifted, so a published run can never silently disagree with the
data it claims to come from.

The allowlist at ``data/published-runs/equivalent_input_sets.json``
exists because content-equivalent edits to the input files (e.g.,
dropping unused fields from ``data/tasks.jsonl``) legitimately change
the bytes without changing what the candidate or the judge see. When
that happens, the prior SHA is added to the equivalence record with a
documented ``diff_summary``, ``introduced_in_commit``, and
``verified_by`` so a reader can audit why the equivalence holds. A
recorded SHA passes if it matches either the live SHA or an
equivalence entry whose ``equivalent_to`` matches the live SHA.

If a run is missing the manifest block or has no SHA fields, the
parametrized case is skipped rather than failed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from wearable_assistant_context_bench.llm_judge import JUDGE_SYSTEM_PROMPT

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_RUNS_DIR = REPO_ROOT / "data" / "published-runs"
EQUIVALENCE_PATH = PUBLISHED_RUNS_DIR / "equivalent_input_sets.json"
TASKS_PATH = REPO_ROOT / "data" / "tasks.jsonl"
PROMPT_CONDITIONS_PATH = REPO_ROOT / "data" / "prompt_conditions.json"

_MANIFEST_BLOCK_RE = re.compile(
    r"##\s+Reproducibility manifest\s*\n+```json\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)

# Map manifest field name -> equivalence-record top-level key.
_EQUIVALENCE_GROUPS = {
    "tasks_sha256": "tasks",
    "prompt_conditions_sha256": "prompt_conditions",
    "judge_prompt_sha256": "judge_prompts",
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _judge_prompt_sha256() -> str:
    return hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _published_run_dirs() -> list[Path]:
    if not PUBLISHED_RUNS_DIR.is_dir():
        return []
    return sorted(p for p in PUBLISHED_RUNS_DIR.iterdir() if p.is_dir())


def _load_manifest(run_dir: Path) -> dict | None:
    findings = run_dir / "findings.md"
    if not findings.is_file():
        return None
    match = _MANIFEST_BLOCK_RE.search(findings.read_text(encoding="utf-8"))
    if not match:
        return None
    return json.loads(match.group("body"))


def _load_equivalences() -> dict[str, dict[str, dict]]:
    if not EQUIVALENCE_PATH.is_file():
        return {}
    raw = json.loads(EQUIVALENCE_PATH.read_text(encoding="utf-8"))
    return {key: raw.get(key, {}) for key in _EQUIVALENCE_GROUPS.values()}


def _is_equivalent(
    recorded: str,
    live: str,
    equivalences: dict[str, dict],
) -> bool:
    entry = equivalences.get(recorded)
    return bool(entry) and entry.get("equivalent_to") == live


@pytest.mark.parametrize(
    "run_dir",
    _published_run_dirs(),
    ids=lambda p: p.name,
)
def test_published_run_input_hashes_match_live_repo(run_dir: Path) -> None:
    manifest = _load_manifest(run_dir)
    if manifest is None:
        pytest.skip(f"{run_dir.name}: no reproducibility manifest in findings.md")

    recorded = {
        field: manifest.get(field)
        for field in _EQUIVALENCE_GROUPS
        if manifest.get(field) is not None
    }
    if not recorded:
        pytest.skip(f"{run_dir.name}: manifest has no SHA fields")

    live = {
        "tasks_sha256": _file_sha256(TASKS_PATH),
        "prompt_conditions_sha256": _file_sha256(PROMPT_CONDITIONS_PATH),
        "judge_prompt_sha256": _judge_prompt_sha256(),
    }
    equivalences = _load_equivalences()

    mismatches: list[str] = []
    for field, recorded_sha in recorded.items():
        live_sha = live[field]
        if recorded_sha == live_sha:
            continue
        group = equivalences.get(_EQUIVALENCE_GROUPS[field], {})
        if _is_equivalent(recorded_sha, live_sha, group):
            continue
        mismatches.append(f"  {field}: recorded {recorded_sha}, live {live_sha}")

    assert not mismatches, (
        f"Reproducibility drift in {run_dir.name}:\n"
        + "\n".join(mismatches)
        + "\nEither re-run the benchmark, restore the prior data file, or"
        " add a documented entry to data/published-runs/equivalent_input_sets.json."
    )
