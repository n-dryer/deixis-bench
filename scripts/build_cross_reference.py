"""Generate ``data/scenarios.cross_reference.csv`` from ``wacb.jsonl``.

A flat per-scenario row of category + attribute columns plus truncated user
speech. Used to spot near-duplicates and to track distribution targets as
the bank grows. Regenerate after every authoring batch.

When ``data/authoring_grades.json`` exists, per-scenario grades from §18b
of the plan are merged into the spreadsheet.

Usage:
    python scripts/build_cross_reference.py

Writes ``data/scenarios.cross_reference.csv`` (UTF-8, comma-delimited).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REPO_ROOT / "data" / "wacb.jsonl"
GRADES_PATH = REPO_ROOT / "data" / "authoring_grades.json"
OUTPUT_PATH = REPO_ROOT / "data" / "scenarios.cross_reference.csv"

GRADE_DIMENSIONS = (
    "token_leakage",
    "object_names_in_images",
    "three_category_gold",
    "genuine_ambiguity",
    "genuinely_missing_info",
    "scene_plausibility",
    "deictic_naturalness",
    "distinctness",
    "domain_coverage_note",
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_grades(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate(text: str, n: int = 60) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _row_for_scenario(entry: dict, grade: dict | None) -> dict:
    gold = entry.get("gold") or {}
    out = {
        "scenario_id": entry["scenario_id"],
        "batch": (grade or {}).get("batch", "0"),
        "shift_type": entry["shift_type"],
        "target_context": entry["target_context"],
        "referent_complexity": entry["referent_complexity"],
        "activity_domain": entry["activity_domain"],
        "difficulty_tier": entry["difficulty_tier"],
        "time_gap_bucket": entry.get("time_gap_bucket") or "",
        "is_cross_session_reference": str(
            entry["shift_type"] == "cross_session_reference"
        ),
        "has_deictic_repair": str(
            bool(entry.get("turn_3_repair_prompt_deictic"))
        ),
        "current_answers_count": len(gold.get("current_answers") or []),
        "prior_answers_count": len(gold.get("prior_answers") or []),
        "clarify_indicators_count": len(gold.get("clarify_indicators") or []),
        "abstain_indicators_count": len(gold.get("abstain_indicators") or []),
        "t1_user": _truncate(entry.get("turn_1_user", "")),
        "t2_user": _truncate(entry.get("turn_2_user", "")),
        "pair_id": entry.get("pair_id") or "",
    }
    for dim in GRADE_DIMENSIONS:
        out[f"grade_{dim}"] = (grade or {}).get(dim, "")
    return out


def main() -> None:
    scenarios = _load_jsonl(SCENARIOS_PATH)
    grades = _load_grades(GRADES_PATH)

    rows = [_row_for_scenario(s, grades.get(s["scenario_id"])) for s in scenarios]
    if not rows:
        print(f"No scenarios in {SCENARIOS_PATH}; nothing to write.")
        return

    fieldnames = list(rows[0].keys())
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_graded = sum(1 for r in rows if r["batch"] != "0")
    print(
        f"Wrote {OUTPUT_PATH} with {len(rows)} scenarios "
        f"({n_graded} carry authoring grades)."
    )


if __name__ == "__main__":
    main()
