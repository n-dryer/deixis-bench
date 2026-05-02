"""Runner for the Wearable Assistant Context Bench.

Implements the cross-turn reference-resolution task over the scenario
bank. Each scenario is a 2-turn conversation; on Turn 2 the runner
labels the candidate's response with the LLM judge. When
``enable_repair`` is set, a Turn 2 failure triggers a templated Turn 3
repair prompt and the response is labeled again. Per-trial transcripts
are written as JSONL alongside ``findings.md`` and ``summary.json``,
all containing a reproducibility manifest.

Candidate and judge models are selected via CLI flags (``--model``,
``--judge-model``, ``--judge-family``, ``--trials``, ``--output-dir``,
``--config``). All flags are optional; defaults live in
``data/config.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()  # idempotent; reads .env in cwd if present
except ImportError:
    # python-dotenv is in pyproject.toml; this branch only fires in
    # slimmed environments. Keys must come from the shell in that case.
    pass

from wearable_assistant_context_bench.aggregation import (
    BENCHMARK_VERSION,
    DEFAULT_RANKING_CONDITION,
    build_run_summary_dict,
)
from wearable_assistant_context_bench.gemini_adapter import GeminiAdapter
from wearable_assistant_context_bench.litellm_adapter import LiteLLMAdapter
from wearable_assistant_context_bench.llm_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    LLMJudge,
    build_judge,
    infer_candidate_family,
    resolve_judge_family,
)
from wearable_assistant_context_bench.models import ModelConfig
from wearable_assistant_context_bench.prompt_conditions import (
    PromptCondition,
    load_prompt_conditions,
)
from wearable_assistant_context_bench.rendering import render_findings_markdown
from wearable_assistant_context_bench.scoring import score_response

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "latest"
SCENARIOS_PATH = DATA_DIR / "scenarios.jsonl"
PROMPT_CONDITIONS_PATH = DATA_DIR / "prompt_conditions.json"
DEFAULT_CONFIG_PATH = DATA_DIR / "config.json"

# In-memory default. Always overridden by ``data/config.json`` at
# startup; kept here so tests that pass a custom config dict stay
# self-contained.
CONFIG: dict[str, Any] = {
    "model_id": "claude-sonnet-4-6",
    "judge_model_id": None,
    "judge_family": "auto",
    # Optional second judge used for cross-candidate ranking comparability.
    # The auto-resolved judge handles per-run integrity (cross-family,
    # self-preference-free). When `ranking_judge_family` is set, every
    # trial is also labeled by this fixed judge so the ranking metric is
    # constant across candidates. Cohen's kappa across the two judges is
    # then reported as cross-LLM inter-judge agreement.
    "ranking_judge_family": None,
    "ranking_judge_model_id": None,
    "temperature": 0.0,
    # Default to a single trial. Multiple trials are only meaningful at
    # non-zero temperature; when used, variance is reported via Wilson
    # CIs over the trial outcomes per scenario/condition cell.
    "trials_per_cell": 1,
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "ranking_condition": DEFAULT_RANKING_CONDITION,
    "no_camera": False,
    # `named` uses scenario.turn_3_repair_prompt (floor metric:
    # maximally specific user correction). `deictic` uses
    # scenario.turn_3_repair_prompt_deictic when populated and falls
    # back to named for scenarios where a deictic gesture cannot
    # resolve the reference (absent_referent, cross_session_reference,
    # target_context != current).
    "repair_style": "named",
    # Turn 3 repair is opt-in. When False (default), a Turn 2 failure
    # is recorded as-is with no repair attempt. When True, the runner
    # fires the templated repair anchor and labels the Turn 3 response.
    "enable_repair": False,
    # Scenario subset to evaluate. `bank` is the frozen 50-scenario
    # bank. `contrast` is the separately-tagged 20-scenario
    # distractor-rich subset of controlled minimal pairs.
    "subset": "bank",
}


def load_runtime_config(path: Path | None = None) -> dict[str, Any]:
    """Load the JSON config file, falling back to the in-memory CONFIG."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return dict(CONFIG)
    raw = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(CONFIG)
    merged.update(raw)
    return merged


@dataclass
class AnswerSet:
    """Per-scenario gold-answer lists.

    Carried inline on the :class:`Scenario` dataclass via the ``gold``
    field rather than loaded from a separate file.
    """

    current_answers: list[str] = field(default_factory=list)
    prior_answers: list[str] = field(default_factory=list)
    clarify_indicators: list[str] = field(default_factory=list)
    abstain_indicators: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    """One scenario record loaded from ``scenarios.jsonl``.

    JSON line schema (one object per line):
        scenario_id: str
        subset: str   # "bank" or "contrast"
        pair_id: str | None  # optional contrast-pair grouping key
        target_context: str  # current | prior | clarify | abstain
        change_type: str        # one of the eight change_type values
        activity_domain: str
        referent_complexity: str
        difficulty_tier: str  # easy | medium | hard
        time_gap_bucket: str | None
        context_image: str | None  # pre-T1 camera state, null when unused
        turn_1_image: str          # camera description at T1
        turn_1_user: str
        turn_2_image: str          # camera description at T2
        turn_2_user: str
        turn_3_repair_prompt: str  # named repair (floor metric)
        turn_3_repair_prompt_deictic: str | None  # deictic-only repair
            for visible-referent scenarios; None when a deictic gesture
            wouldn't resolve the reference (absent_referent,
            cross_session_reference, target_context other than current)
        notes: str  # optional
        gold:
            current_answers: list[str]
            prior_answers: list[str]
            clarify_indicators: list[str]
            abstain_indicators: list[str]
    """

    scenario_id: str
    target_context: str
    change_type: str
    activity_domain: str
    referent_complexity: str
    difficulty_tier: str
    turn_1_image: str
    turn_1_user: str
    turn_2_image: str
    turn_2_user: str
    turn_3_repair_prompt: str
    subset: str = "bank"
    pair_id: str | None = None
    context_image: str | None = None
    time_gap_bucket: str | None = None
    notes: str = ""
    turn_3_repair_prompt_deictic: str | None = None
    gold: AnswerSet = field(default_factory=AnswerSet)


def load_scenarios(path: Path = SCENARIOS_PATH, subset: str | None = None) -> list[Scenario]:
    """Load scenarios from ``scenarios.jsonl``, optionally filtered by subset.

    Each line is one JSON object. When ``subset`` is non-None, only
    records whose ``subset`` field matches are returned.
    """
    scenarios: list[Scenario] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if subset is not None and entry.get("subset") != subset:
                continue
            gold_raw = entry.get("gold") or {}
            gold = AnswerSet(
                current_answers=list(gold_raw.get("current_answers") or []),
                prior_answers=list(gold_raw.get("prior_answers") or []),
                clarify_indicators=list(gold_raw.get("clarify_indicators") or []),
                abstain_indicators=list(gold_raw.get("abstain_indicators") or []),
            )
            scenarios.append(
                Scenario(
                    scenario_id=entry["scenario_id"],
                    subset=entry.get("subset", "bank"),
                    pair_id=entry.get("pair_id"),
                    target_context=entry["target_context"],
                    change_type=entry["change_type"],
                    activity_domain=entry["activity_domain"],
                    referent_complexity=entry["referent_complexity"],
                    difficulty_tier=entry["difficulty_tier"],
                    turn_1_image=entry["turn_1_image"],
                    turn_1_user=entry["turn_1_user"],
                    turn_2_image=entry["turn_2_image"],
                    turn_2_user=entry["turn_2_user"],
                    turn_3_repair_prompt=entry["turn_3_repair_prompt"],
                    context_image=entry.get("context_image"),
                    time_gap_bucket=entry.get("time_gap_bucket"),
                    notes=entry.get("notes", ""),
                    turn_3_repair_prompt_deictic=entry.get("turn_3_repair_prompt_deictic"),
                    gold=gold,
                )
            )
    return scenarios


def _sha256_of_file(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _current_git_commit() -> str:
    """Return the current git HEAD SHA, or "unknown" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def _build_adapter(model_id: str) -> Any:
    """Pick a candidate adapter based on the model string and family.

    Bare Gemini model ids route through the native Gemini adapter.
    Everything else routes through LiteLLM, which handles Claude (via
    ``openrouter/anthropic/...`` or ``anthropic/...``), OpenAI, and
    any other provider-qualified id with a slash.
    """
    family = infer_candidate_family(model_id)
    if "/" in model_id:
        return LiteLLMAdapter()
    if family == "gemini":
        return GeminiAdapter()
    if family in ("claude", "openai"):
        return LiteLLMAdapter()
    raise ValueError(
        f"Unsupported candidate model family for model_id={model_id!r}. "
        "Supported families: claude, gemini, openai, plus provider-qualified "
        "LiteLLM-backed model IDs such as openrouter/... and huggingface/...."
    )


def _build_manifest(
    *,
    effective_config: dict[str, Any],
    resolved_judge: LLMJudge,
    judge_resolution_mode: str,
    ranking_judge: LLMJudge | None = None,
) -> dict[str, Any]:
    """Construct the reproducibility manifest dict."""
    warnings: list[str] = []

    def _sha_or_warn(path: Path, key: str) -> str | None:
        value = _sha256_of_file(path)
        if value is None:
            warnings.append(f"{key} could not be hashed from {path}")
        return value

    judge_prompt_sha = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    pack = effective_config.get("subset", "bank")

    manifest: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "subset": pack,
        "camera_injection": not bool(effective_config.get("no_camera", False)),
        "scenarios_sha256": _sha_or_warn(SCENARIOS_PATH, "scenarios_sha256"),
        "prompt_conditions_sha256": _sha_or_warn(
            PROMPT_CONDITIONS_PATH, "prompt_conditions_sha256"
        ),
        "judge_prompt_sha256": judge_prompt_sha,
        "candidate_model": effective_config["model_id"],
        "judge_model": resolved_judge.model_id,
        "judge_family": resolved_judge.family,
        "judge_family_resolution": judge_resolution_mode,
        "ranking_judge_model": (ranking_judge.model_id if ranking_judge is not None else None),
        "ranking_judge_family": (ranking_judge.family if ranking_judge is not None else None),
        "trials": int(effective_config["trials_per_cell"]),
        "temperature": float(effective_config["temperature"]),
        "ranking_condition": effective_config["ranking_condition"],
        "enable_repair": bool(effective_config.get("enable_repair", False)),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runner_git_commit": _current_git_commit(),
        "random_seed": None,
    }
    manifest["manifest_warnings"] = warnings
    return manifest


def run(
    adapter: Any | None = None,
    judge: LLMJudge | None = None,
    config: dict[str, Any] | None = None,
    ranking_judge: LLMJudge | None = None,
) -> list[dict]:
    """Run the full benchmark and return the per-trial result list.

    Callers that want real API calls pass no arguments. Tests pass a
    stub `adapter` and `judge` so the loop runs without network.

    Args:
        adapter: Candidate adapter. Defaults to the family-appropriate
            adapter resolved from `model_id`.
        judge: `LLMJudge`. Defaults to the auto resolution against
            `CONFIG["model_id"]`.
        config: Overrides for CONFIG. Unrecognized keys are ignored.
        ranking_judge: Optional second `LLMJudge` used for
            cross-candidate ranking comparability. When provided, every
            trial is also labeled by this fixed judge and the result
            dict carries both verdicts. Defaults to the family resolved
            from `ranking_judge_family` / `ranking_judge_model_id`
            config keys; ``None`` if those are unset.

    Returns:
        Per-trial result dicts ready for `core.report` aggregation.
    """
    effective_config = {**CONFIG, **(config or {})}

    pack = effective_config.get("subset", "bank")
    scenarios = load_scenarios(SCENARIOS_PATH, subset=pack)
    conditions = load_prompt_conditions(PROMPT_CONDITIONS_PATH)

    model_config = ModelConfig(
        model_id=effective_config["model_id"],
        temperature=effective_config["temperature"],
    )
    adapter_ = adapter if adapter is not None else _build_adapter(effective_config["model_id"])

    if judge is None:
        family, resolution_mode = resolve_judge_family(
            effective_config["judge_family"],
            effective_config["model_id"],
        )
        judge_ = build_judge(
            family=family,
            model_id=effective_config["judge_model_id"],
        )
    else:
        judge_ = judge
        resolution_mode = "explicit"

    ranking_judge_ = ranking_judge
    if ranking_judge_ is None and effective_config.get("ranking_judge_family"):
        ranking_judge_ = build_judge(
            family=effective_config["ranking_judge_family"],
            model_id=effective_config.get("ranking_judge_model_id"),
        )

    output_dir = Path(effective_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcripts.jsonl"

    enable_repair = bool(effective_config.get("enable_repair", False))

    results: list[dict] = []
    with transcript_path.open("w", encoding="utf-8") as transcript_file:
        for scenario in scenarios:
            for condition in conditions:
                for trial in range(effective_config["trials_per_cell"]):
                    result = _run_one_trial(
                        scenario=scenario,
                        answers=scenario.gold,
                        condition=condition,
                        trial=trial,
                        adapter=adapter_,
                        judge=judge_,
                        ranking_judge=ranking_judge_,
                        model_config=model_config,
                        no_camera=bool(effective_config.get("no_camera", False)),
                        repair_style=effective_config.get("repair_style", "named"),
                        enable_repair=enable_repair,
                    )
                    results.append(result)
                    transcript_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    manifest = _build_manifest(
        effective_config=effective_config,
        resolved_judge=judge_,
        judge_resolution_mode=resolution_mode,
        ranking_judge=ranking_judge_,
    )

    findings = render_findings_markdown(
        results,
        scenario_policies={s.scenario_id: s.target_context for s in scenarios},
        manifest=manifest,
        ranking_condition=effective_config["ranking_condition"],
    )

    findings_path = output_dir / "findings.md"
    findings_path.write_text(findings, encoding="utf-8")

    summary = build_run_summary_dict(
        results=results,
        manifest=manifest,
        run_label=output_dir.name,
        ranking_condition=effective_config["ranking_condition"],
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return results


def _resolve_repair_anchor(scenario: Scenario, repair_style: str) -> tuple[str, str]:
    """Return ``(anchor_text, resolved_style)`` for the Turn 3 repair.

    ``repair_style="named"`` always uses ``scenario.turn_3_repair_prompt``.
    ``repair_style="deictic"`` uses ``scenario.turn_3_repair_prompt_deictic``
    when populated and falls back to the named anchor otherwise (for
    scenarios where a deictic gesture cannot resolve the reference,
    e.g. absent_referent or cross_session_reference).
    """
    if repair_style == "deictic" and scenario.turn_3_repair_prompt_deictic:
        return scenario.turn_3_repair_prompt_deictic, "deictic"
    return scenario.turn_3_repair_prompt, "named"


def _run_one_trial(
    *,
    scenario: Scenario,
    answers: AnswerSet,
    condition: PromptCondition,
    trial: int,
    adapter: Any,
    judge: LLMJudge,
    model_config: ModelConfig,
    no_camera: bool = False,
    ranking_judge: LLMJudge | None = None,
    repair_style: str = "named",
    enable_repair: bool = False,
) -> dict:
    """Run one (scenario, condition, trial) cell end-to-end.

    When ``no_camera`` is True, the runner strips the ``[Camera: ...]``
    blocks from every user message and skips injecting the
    ``context_image`` standalone message. The candidate sees only the
    user's spoken text. Used for camera channel ablation runs.

    When ``enable_repair`` is False (the default), a Turn 2 failure is
    recorded as-is and no Turn 3 message is sent. When True, the runner
    fires the templated repair anchor and labels the Turn 3 response.
    """
    messages: list[dict[str, str]] = []
    # Pre-conversation camera state (only set on cross_session_reference scenarios)
    if scenario.context_image and not no_camera:
        messages.append(_build_context_image_message(scenario.context_image))
    messages.append(
        _build_message(
            role="user",
            text=scenario.turn_1_user,
            image=None if no_camera else scenario.turn_1_image,
        )
    )
    turn_1_response = adapter.query(
        messages=messages, system=condition.system_prompt, config=model_config
    )
    messages.append({"role": "assistant", "content": turn_1_response})
    messages.append(
        _build_message(
            role="user",
            text=scenario.turn_2_user,
            image=None if no_camera else scenario.turn_2_image,
        )
    )
    turn_2_response = adapter.query(
        messages=messages, system=condition.system_prompt, config=model_config
    )

    code_signals = score_response(
        response=turn_2_response,
        current_answers=answers.current_answers,
        prior_answers=answers.prior_answers,
        clarify_indicators=answers.clarify_indicators,
        abstain_indicators=answers.abstain_indicators,
    )

    scenario_description = _build_scenario_description(scenario)

    ground_truth_context = _build_ground_truth_context(scenario)

    judge_verdict = judge.label(
        response=turn_2_response,
        scenario_description=scenario_description,
        turn_2_user=scenario.turn_2_user,
        current_answers=answers.current_answers,
        prior_answers=answers.prior_answers,
        clarify_indicators=answers.clarify_indicators,
        abstain_indicators=answers.abstain_indicators,
        ground_truth_context=ground_truth_context,
    )

    turn_2_passed = judge_verdict.selected_label == scenario.target_context

    turn_2_ranking_verdict: JudgeVerdict | None = None
    if ranking_judge is not None:
        turn_2_ranking_verdict = ranking_judge.label(
            response=turn_2_response,
            scenario_description=scenario_description,
            turn_2_user=scenario.turn_2_user,
            current_answers=answers.current_answers,
            prior_answers=answers.prior_answers,
            clarify_indicators=answers.clarify_indicators,
            abstain_indicators=answers.abstain_indicators,
            ground_truth_context=ground_truth_context,
        )

    turn_3_response: str | None = None
    turn_3_verdict: JudgeVerdict | None = None
    turn_3_passed: bool | None = None
    repair_attempted = False
    turn_3_ranking_verdict: JudgeVerdict | None = None

    repair_anchor_text, resolved_repair_style = _resolve_repair_anchor(scenario, repair_style)

    if enable_repair and not turn_2_passed:
        repair_attempted = True
        messages.append({"role": "assistant", "content": turn_2_response})
        messages.append({"role": "user", "content": repair_anchor_text})
        turn_3_response = adapter.query(
            messages=messages, system=condition.system_prompt, config=model_config
        )
        turn_3_verdict = judge.label(
            response=turn_3_response,
            scenario_description=scenario_description,
            turn_2_user=repair_anchor_text,
            current_answers=answers.current_answers,
            prior_answers=answers.prior_answers,
            clarify_indicators=answers.clarify_indicators,
            abstain_indicators=answers.abstain_indicators,
            ground_truth_context=ground_truth_context,
        )
        turn_3_passed = turn_3_verdict.selected_label == scenario.target_context
        if ranking_judge is not None:
            turn_3_ranking_verdict = ranking_judge.label(
                response=turn_3_response,
                scenario_description=scenario_description,
                turn_2_user=repair_anchor_text,
                current_answers=answers.current_answers,
                prior_answers=answers.prior_answers,
                clarify_indicators=answers.clarify_indicators,
                abstain_indicators=answers.abstain_indicators,
                ground_truth_context=ground_truth_context,
            )

    result: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "subset": scenario.subset,
        "pair_id": scenario.pair_id,
        "condition": condition.name,
        "trial": trial,
        "target_context": scenario.target_context,
        "change_type": scenario.change_type,
        "activity_domain": scenario.activity_domain,
        "difficulty_tier": scenario.difficulty_tier,
        "context_image": scenario.context_image,
        "turn_1_user": scenario.turn_1_user,
        "turn_1_image": scenario.turn_1_image,
        "turn_1_response": turn_1_response,
        "turn_2_user": scenario.turn_2_user,
        "turn_2_image": scenario.turn_2_image,
        "turn_2_response": turn_2_response,
        "turn_2_code_signals": code_signals,
        "turn_2_judge_label": judge_verdict.selected_label,
        "turn_2_judge_rationale": judge_verdict.rationale,
        "turn_2_passed": turn_2_passed,
        "turn_3_repair_attempted": repair_attempted,
        "turn_3_repair_prompt": (repair_anchor_text if repair_attempted else None),
        "turn_3_repair_style": (resolved_repair_style if repair_attempted else None),
        "turn_3_response": turn_3_response,
        "turn_3_judge_label": (turn_3_verdict.selected_label if turn_3_verdict else None),
        "turn_3_judge_rationale": (turn_3_verdict.rationale if turn_3_verdict else None),
        "turn_3_repair_passed": turn_3_passed,
    }
    if ranking_judge is not None:
        result["turn_2_ranking_judge_label"] = (
            turn_2_ranking_verdict.selected_label if turn_2_ranking_verdict else None
        )
        result["turn_2_ranking_judge_rationale"] = (
            turn_2_ranking_verdict.rationale if turn_2_ranking_verdict else None
        )
        result["turn_2_ranking_passed"] = (
            (turn_2_ranking_verdict.selected_label == scenario.target_context)
            if turn_2_ranking_verdict
            else None
        )
        result["turn_3_ranking_judge_label"] = (
            turn_3_ranking_verdict.selected_label if turn_3_ranking_verdict else None
        )
        result["turn_3_ranking_judge_rationale"] = (
            turn_3_ranking_verdict.rationale if turn_3_ranking_verdict else None
        )
        result["turn_3_ranking_repair_passed"] = (
            (turn_3_ranking_verdict.selected_label == scenario.target_context)
            if turn_3_ranking_verdict
            else None
        )
    return result


def _build_scenario_description(scenario: Scenario) -> str:
    """Construct the scenario description shown to the judge.

    Names neither the target_context nor the change_type. Both would
    leak the answer the judge is being asked to produce. The judge
    is pointed at the Turn 2 user message and camera frame as the
    perceptual fields that determine what the assistant should now
    be answering about.
    """
    return (
        f"Turn 1 context:\n{scenario.turn_1_user}\n\n"
        f"Between Turn 1 and Turn 2 the user's context shifts. "
        f"The Turn 2 user message and camera frame describe what "
        f"the assistant should now be answering about."
    )


def _build_ground_truth_context(scenario: Scenario) -> str:
    """Construct the judge-only ground-truth description.

    The candidate model sees perceptual camera descriptions only (no
    object names). The judge sees the perceptual T1/T2 frames plus
    the activity domain so it can determine whether the response
    reflects T2 or T1 context.

    Deliberately omits target_context, change_type, and authoring notes.
    Those would either name or category-hint the answer the judge is
    being asked to produce.
    """
    parts: list[str] = [f"Activity domain: {scenario.activity_domain}."]
    if scenario.context_image:
        parts.append(f"Pre-conversation camera state: {scenario.context_image}")
    parts.append(f"Turn 1 camera state: {scenario.turn_1_image}")
    parts.append(f"Turn 2 camera state: {scenario.turn_2_image}")
    return "\n\n".join(parts)


def _build_message(*, role: str, text: str, image: str | None) -> dict[str, str]:
    """Build a single chat message with optional camera channel injection.

    The benchmark uses a perceptual-text proxy for the camera frame. When
    `image` is non-null, it is prepended to the user message as a tagged
    `[Camera: ...]` block, simulating what a vision backbone would have
    returned alongside the transcribed user speech.

    Format:
        [Camera: {image}]
        {text}

    The `[Camera:]` block represents content the candidate would have
    received from the wearable's vision channel; the user's spoken words
    follow on the next line.
    """
    if image:
        content = f"[Camera: {image}]\n{text}"
        return {"role": role, "content": content}
    return {"role": role, "content": text}


def _build_context_image_message(image: str) -> dict[str, str]:
    """Build a standalone camera channel message for `context_image`.

    `context_image` represents what the wearable's camera saw before any
    user speech began. It is injected as a user-role message containing
    only the `[Camera: ...]` block, with no spoken text. This precedes
    the T1 message in the conversation.
    """
    return {"role": "user", "content": f"[Camera: {image}]"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the Wearable Assistant Context Bench."),
        epilog=(
            "Example: python -m wearable_assistant_context_bench.runner "
            "--model claude-sonnet-4-6 --judge-model gemini-2.5-flash"
        ),
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=None,
        help=("path to the runtime JSON config; defaults to data/config.json"),
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help=(f"candidate model ID; default is {CONFIG['model_id']}"),
    )
    parser.add_argument(
        "--judge-model",
        dest="judge_model",
        default=None,
        help=("judge model ID; defaults to the family-specific judge chosen for the run"),
    )
    parser.add_argument(
        "--judge-family",
        dest="judge_family",
        default=None,
        choices=["auto", "claude", "gemini", "openai"],
        help=(
            "judge family override; default is auto, which picks a judge "
            "from a different model family when candidate family inference succeeds"
        ),
    )
    parser.add_argument(
        "--ranking-judge-family",
        dest="ranking_judge_family",
        default=None,
        choices=["claude", "gemini", "openai"],
        help=(
            "Optional second judge family used for cross-candidate ranking "
            "comparability. When set, every trial is also labeled by this "
            "fixed judge so candidate quality is isolated from judge "
            "strictness when comparing two candidates. Cohen's kappa across "
            "the two judges is reported as cross-LLM inter-judge agreement."
        ),
    )
    parser.add_argument(
        "--ranking-judge-model",
        dest="ranking_judge_model",
        default=None,
        help=(
            "Optional ranking-judge model id; defaults to the family-specific "
            "default judge model from wearable_assistant_context_bench.llm_judge."
        ),
    )
    parser.add_argument(
        "--trials",
        dest="trials",
        type=int,
        default=None,
        help=(f"trials per (scenario, condition) cell; default is {CONFIG['trials_per_cell']}"),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help=(
            f"directory for transcripts, findings, and manifest; defaults to {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--no-camera",
        dest="no_camera",
        action="store_true",
        default=False,
        help=(
            "ablation flag: strip [Camera: ...] blocks from every user "
            "message and skip injecting context_image. Run with this flag "
            "to measure the contribution of the camera channel by "
            "comparing the score against a normal run with the same model."
        ),
    )
    parser.add_argument(
        "--enable-repair",
        dest="enable_repair",
        action="store_true",
        default=False,
        help=(
            "Enable Turn 3 deictic-repair on Turn 2 failure. When set, "
            "the runner fires the templated repair anchor (named or "
            "deictic per --repair-style) and labels the Turn 3 response. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--subset",
        dest="subset",
        choices=["bank", "contrast"],
        default=None,
        help=(
            "Scenario subset to run. `bank` is the frozen 50-scenario "
            "scenario bank; `contrast` is the separately-tagged "
            "20-scenario distractor-rich subset of controlled minimal "
            "pairs. Defaults to bank."
        ),
    )
    parser.add_argument(
        "--repair-style",
        dest="repair_style",
        choices=["named", "deictic"],
        default=None,
        help=(
            "Turn 3 repair anchor style. Only meaningful with "
            "--enable-repair. `named` (default) uses the explicit "
            "repair line that names both the intended and the wrong "
            "objects. `deictic` uses the deictic-only anchor when "
            "populated and falls back to named for scenarios where a "
            "deictic gesture cannot resolve the reference (e.g. "
            "absent_referent, cross_session_reference, target_context "
            "!= current)."
        ),
    )
    return parser.parse_args(argv)


def _config_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides["model_id"] = args.model
    if args.judge_model is not None:
        overrides["judge_model_id"] = args.judge_model
    if args.judge_family is not None:
        overrides["judge_family"] = args.judge_family
    if getattr(args, "ranking_judge_family", None) is not None:
        overrides["ranking_judge_family"] = args.ranking_judge_family
    if getattr(args, "ranking_judge_model", None) is not None:
        overrides["ranking_judge_model_id"] = args.ranking_judge_model
    if args.trials is not None:
        overrides["trials_per_cell"] = args.trials
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if getattr(args, "no_camera", False):
        overrides["no_camera"] = True
    if getattr(args, "repair_style", None) is not None:
        overrides["repair_style"] = args.repair_style
    if getattr(args, "subset", None) is not None:
        overrides["subset"] = args.subset
    if getattr(args, "enable_repair", False):
        overrides["enable_repair"] = True
    return overrides


def main(argv: list[str] | None = None) -> None:
    """Parse CLI flags and run the benchmark."""
    args = _parse_args(argv)
    config_path = Path(args.config) if getattr(args, "config", None) else DEFAULT_CONFIG_PATH
    base_config = load_runtime_config(config_path)
    overrides = _config_overrides_from_args(args)
    merged = {**base_config, **overrides}
    run(config=merged)


if __name__ == "__main__":
    main()
