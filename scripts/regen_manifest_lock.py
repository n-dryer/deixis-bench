"""Regenerate the static asset lockfile at ``data/MANIFEST.lock.json``.

The lockfile pins SHA256 hashes of the unified task set,
prompt-condition definitions, and the judge-prompt template, alongside
the benchmark version. ``scripts/validate_tasks.py`` compares the
computed hashes against this lockfile and fails CI if any drift without
a coordinated ``BENCHMARK_VERSION`` bump.

Run this script after a coordinated content change is merged to refresh
the lockfile.

Usage:
    python scripts/regen_manifest_lock.py

Note: if multiple manifest lock files accumulate over time (e.g. one
per benchmark version archived for reproducibility), consider moving
them under ``data/manifests/`` rather than the data root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
LOCKFILE_PATH = DATA_DIR / "MANIFEST.lock.json"
TASKS_PATH = DATA_DIR / "tasks.jsonl"
PROMPT_CONDITIONS_PATH = DATA_DIR / "prompt_conditions.json"

LOCKFILE_NOTE = (
    "Static lockfile. scripts/validate_tasks.py compares computed "
    "hashes against this lock and fails CI if they drift without a "
    "coordinated benchmark_version bump. Regenerate by running "
    "scripts/regen_manifest_lock.py."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lockfile() -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from wearable_assistant_context_bench.aggregation import BENCHMARK_VERSION
    from wearable_assistant_context_bench.llm_judge import JUDGE_SYSTEM_PROMPT

    lockfile = {
        "benchmark_version": BENCHMARK_VERSION,
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "tasks_sha256": _sha256(TASKS_PATH),
        "prompt_conditions_sha256": _sha256(PROMPT_CONDITIONS_PATH),
    }
    lockfile["_note"] = LOCKFILE_NOTE
    return lockfile


def main() -> int:
    lockfile = build_lockfile()
    LOCKFILE_PATH.write_text(json.dumps(lockfile, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LOCKFILE_PATH.relative_to(REPO_ROOT)}")
    for key, value in lockfile.items():
        if key == "_note":
            continue
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
