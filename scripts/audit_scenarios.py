"""Audit the scenario bank's ``change_type`` and ``difficulty_tier`` metadata.

Two-pass audit:

1. Rule-based: applies the rubric in
   ``wearable_assistant_context_bench.audit_rubric`` to every scenario
   in ``data/scenarios.jsonl``. Pure-functional, deterministic, fast.
2. LLM-judge spot-check (optional, ``--judge-on-disagreement``): for any
   scenario where the audit disagrees with metadata on either field,
   one structured judge call produces a tie-breaking verdict + rationale.

Output: a per-scenario diff CSV. The audit ignores the metadata fields
being audited (``change_type``, ``difficulty_tier``) when forming its
verdict, then compares against them in the report.

Usage:
    python scripts/audit_scenarios.py
    python scripts/audit_scenarios.py --judge-on-disagreement
    python scripts/audit_scenarios.py --in data/scenarios.jsonl --out data/scenarios.audit.csv

Exits 0 always. Mismatches are surfaced in the CSV, not as failures, so
this script can be invoked without breaking CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wearable_assistant_context_bench.audit_rubric import (  # noqa: E402
    audit_change_type,
    audit_difficulty,
    audit_target_context,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "scenarios.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "scenarios.audit.csv"

JUDGE_DISAGREEMENT_CAP = 60

_logger = logging.getLogger("audit_scenarios")


# --- LLM-judge spot-check ------------------------------------------------


_AUDIT_JUDGE_SYSTEM_PROMPT = """You are auditing a benchmark scenario's category and difficulty labels from the script content alone.

Eight categories (`change_type`):
- object_in_hand: A hand grasps one object in Turn 1 and a different object in Turn 2.
- object_state: The same object appears in both turns but in a different state (e.g., heating up vs. boiling).
- sequential_task: Same task surface, a later step is shown in Turn 2 (e.g., sand → stain).
- location: The whole scene/setting changes between turns.
- object_in_view: Same scene; the camera or attention shifts to a different object within it.
- absent_referent: An object present in Turn 1 is no longer in frame in Turn 2.
- screen_content: Both turns show a screen/display whose content changed.
- cross_session_reference: A `context_image` (pre-Turn-1 state) is provided and Turn 2 references it.

Three difficulty tiers:
- easy: minimal cognitive load — single referent, clear scene contrast, target=current.
- medium: moderate — clarify or distractor or moderate prior/current vocab overlap.
- hard: high — abstain target, offscreen referent, high prior/current vocab overlap, or subtle scene contrast.

Reason briefly. Then emit a single JSON object on the final line with this exact shape:
{"change_type": "<one of the eight>", "difficulty_tier": "<easy|medium|hard>", "rationale": "<one-sentence justification>"}

Output no text after the JSON object."""


_JSON_TAIL = re.compile(r"\{[^{}]*\}\s*$", re.MULTILINE | re.DOTALL)


def _build_judge_user_prompt(scenario: dict) -> str:
    gold = scenario.get("gold") or {}
    parts = [
        f"scenario_id: {scenario['scenario_id']}",
        f"context_image: {scenario.get('context_image') or '(none)'}",
        f"turn_1_image: {scenario.get('turn_1_image') or ''}",
        f"turn_1_user: {scenario.get('turn_1_user') or ''}",
        f"turn_2_image: {scenario.get('turn_2_image') or ''}",
        f"turn_2_user: {scenario.get('turn_2_user') or ''}",
        f"gold.current_answers: {gold.get('current_answers') or []}",
        f"gold.prior_answers: {gold.get('prior_answers') or []}",
        f"gold.clarify_indicators: {gold.get('clarify_indicators') or []}",
        f"gold.abstain_indicators: {gold.get('abstain_indicators') or []}",
    ]
    return "\n".join(parts)


def _parse_judge_response(raw: str) -> dict[str, str]:
    """Pull the trailing JSON object out of the judge response."""
    match = _JSON_TAIL.search(raw)
    if not match:
        return {"change_type": "", "difficulty_tier": "", "rationale": "PARSE_ERROR"}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"change_type": "", "difficulty_tier": "", "rationale": "PARSE_ERROR"}
    return {
        "change_type": str(obj.get("change_type", "")).strip(),
        "difficulty_tier": str(obj.get("difficulty_tier", "")).strip(),
        "rationale": str(obj.get("rationale", "")).strip(),
    }


def _make_judge_adapter() -> Any:
    """Lazy import so the rule-based pass works even without API creds."""
    from wearable_assistant_context_bench.llm_judge import LiteLLMJudgeAdapter

    return LiteLLMJudgeAdapter(family="claude")


def _call_judge(adapter: Any, scenario: dict, model_id: str) -> dict[str, str]:
    user = _build_judge_user_prompt(scenario)
    raw = adapter.call(
        system=_AUDIT_JUDGE_SYSTEM_PROMPT,
        user=user,
        model_id=model_id,
    )
    return _parse_judge_response(raw)


# --- bank loading ------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# --- audit pipeline ------------------------------------------------------


def _audit_one(scenario: dict) -> dict[str, Any]:
    """Run the rule-based audit on a single scenario."""
    audit_ct, ct_signals = audit_change_type(scenario)
    audit_tc = audit_target_context(scenario.get("gold") or {}, scenario.get("turn_2_user") or "")
    audit_diff, diff_score, diff_breakdown = audit_difficulty(
        scenario,
        target_context=audit_tc,
        change_type=audit_ct,
    )
    metadata_ct = scenario.get("change_type")
    metadata_diff = scenario.get("difficulty_tier")
    return {
        "scenario_id": scenario["scenario_id"],
        "metadata_change_type": metadata_ct,
        "audit_change_type": audit_ct,
        "change_type_match": str(metadata_ct == audit_ct),
        "metadata_difficulty": metadata_diff,
        "audit_difficulty": audit_diff,
        "difficulty_match": str(metadata_diff == audit_diff),
        "audit_signals_change_type": json.dumps(ct_signals, ensure_ascii=False),
        "audit_difficulty_score": diff_score,
        "audit_difficulty_breakdown": json.dumps(diff_breakdown, ensure_ascii=False),
        "audit_target_context_inferred": audit_tc,
        "metadata_target_context": scenario.get("target_context"),
        "metadata_referent_complexity": scenario.get("referent_complexity"),
        "llm_change_type": "",
        "llm_difficulty": "",
        "llm_rationale": "",
    }


def _disagreement(row: dict[str, Any]) -> bool:
    """Either field disagreement is a 'disagreement' for reporting."""
    return row["change_type_match"] == "False" or row["difficulty_match"] == "False"


def _change_type_disagreement(row: dict[str, Any]) -> bool:
    """Only change_type disagreements get the LLM-judge spot-check.

    Difficulty disagreements are expected by design — the rule-based
    rubric defines its own additive scoring, which won't reproduce the
    human-graded distribution exactly. Difficulty is for direct
    reviewer inspection, not LLM tie-breaking.
    """
    return row["change_type_match"] == "False"


def _summarize(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    ct_mismatch = sum(1 for r in rows if r["change_type_match"] == "False")
    diff_mismatch = sum(1 for r in rows if r["difficulty_match"] == "False")
    any_mismatch = sum(1 for r in rows if _disagreement(r))

    out = [
        f"Audited {n} scenarios.",
        f"  change_type mismatches: {ct_mismatch} ({ct_mismatch / n:.0%})",
        f"  difficulty   mismatches: {diff_mismatch} ({diff_mismatch / n:.0%})",
        f"  any mismatch: {any_mismatch} ({any_mismatch / n:.0%})",
    ]

    # Per-change_type mismatch rate (using metadata as the grouping key
    # so the table reads like "of the X scenarios labeled object_state,
    # the audit disagreed on Y").
    per_ct: dict[str, list[int]] = {}
    for r in rows:
        ct = r["metadata_change_type"] or "<unknown>"
        bucket = per_ct.setdefault(ct, [0, 0])
        bucket[0] += 1
        if r["change_type_match"] == "False":
            bucket[1] += 1
    out.append("  per metadata change_type:")
    for ct, (total, mism) in sorted(per_ct.items()):
        out.append(f"    {ct}: {mism}/{total} ({mism / total:.0%})")

    # Audit difficulty distribution vs metadata distribution.
    audit_dist = Counter(r["audit_difficulty"] for r in rows)
    meta_dist = Counter(r["metadata_difficulty"] for r in rows)
    out.append(f"  metadata difficulty distribution: {dict(meta_dist)}")
    out.append(f"  audit    difficulty distribution: {dict(audit_dist)}")

    return "\n".join(out)


# --- main ------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to scenarios.jsonl (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the diff CSV (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--judge-on-disagreement",
        action="store_true",
        help="On any audit/metadata disagreement, call the LLM judge for a tie-breaking verdict.",
    )
    parser.add_argument(
        "--judge-model",
        default="openrouter/anthropic/claude-sonnet-4.6",
        help="LiteLLM model id for the judge call.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary to stdout but skip writing the CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scenarios = _load_jsonl(args.input_path)
    rows = [_audit_one(s) for s in scenarios]

    if args.judge_on_disagreement:
        disagreements = [r for r in rows if _change_type_disagreement(r)]
        n_dis = len(disagreements)
        if n_dis > JUDGE_DISAGREEMENT_CAP:
            _logger.warning(
                "Disagreement count %d exceeds cap %d; skipping LLM-judge "
                "spot-check. The rule-based audit, not the metadata, is "
                "likely the source of drift. Investigate the rubric "
                "before re-running.",
                n_dis,
                JUDGE_DISAGREEMENT_CAP,
            )
        else:
            _logger.info("Running LLM-judge spot-check on %d disagreements...", n_dis)
            adapter = _make_judge_adapter()
            scenario_by_id = {s["scenario_id"]: s for s in scenarios}
            row_by_id = {r["scenario_id"]: r for r in rows}
            for i, row in enumerate(disagreements, 1):
                sid = row["scenario_id"]
                try:
                    verdict = _call_judge(adapter, scenario_by_id[sid], args.judge_model)
                except Exception as exc:  # surface the failure but keep going
                    verdict = {
                        "change_type": "",
                        "difficulty_tier": "",
                        "rationale": f"JUDGE_ERROR: {type(exc).__name__}: {exc}",
                    }
                row_by_id[sid]["llm_change_type"] = verdict["change_type"]
                row_by_id[sid]["llm_difficulty"] = verdict["difficulty_tier"]
                row_by_id[sid]["llm_rationale"] = verdict["rationale"]
                if i % 5 == 0:
                    _logger.info("  judged %d/%d", i, n_dis)

    if not args.summary_only:
        fieldnames = list(rows[0].keys())
        with args.output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        _logger.info("Wrote %d rows to %s", len(rows), args.output_path)

    print(_summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
