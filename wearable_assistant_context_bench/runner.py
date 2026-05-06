"""Runner for the Wearable Assistant Context Bench.

Implements the cross-turn reference-resolution task over the task
set. Each task is a 2-turn conversation; on Turn 2 the runner
labels the candidate's response with the LLM judge. Per-trial
transcripts are written as JSONL alongside ``findings.md`` and
``summary.json``, all containing a reproducibility manifest.

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
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Protocol

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

ResourcePath = Path | Traversable


class JudgeLike(Protocol):
    """Minimal interface required from benchmark judges."""

    @property
    def family(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def label(
        self,
        *,
        response: str,
        task_description: str,
        turn_2_user: str,
        current_answers: list[str],
        prior_answers: list[str],
        clarify_indicators: list[str],
        abstain_indicators: list[str],
        ground_truth_context: str | None = None,
    ) -> JudgeVerdict: ...


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATA_DIR = REPO_ROOT / "data"
PACKAGED_DATA_DIR = files("wearable_assistant_context_bench").joinpath("data")
DATA_DIR: ResourcePath = SOURCE_DATA_DIR if SOURCE_DATA_DIR.is_dir() else PACKAGED_DATA_DIR
DEFAULT_OUTPUT_DIR = Path.cwd() / "runs" / "latest"


def _data_resource(filename: str) -> ResourcePath:
    source_path = SOURCE_DATA_DIR / filename
    if source_path.is_file():
        return source_path
    return PACKAGED_DATA_DIR.joinpath(filename)


TASKS_PATH = _data_resource("tasks.jsonl")
PROMPT_CONDITIONS_PATH = _data_resource("prompt_conditions.json")
DEFAULT_CONFIG_PATH = _data_resource("config.json")

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
    # CIs over the trial outcomes per task/condition cell.
    "trials_per_cell": 1,
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "ranking_condition": DEFAULT_RANKING_CONDITION,
    "no_camera": False,
    # Retained for forward-compat: every task records task_set="main"
    # in the unified 166-task set. The runner loads all tasks
    # regardless of this value.
    "task_set": "main",
}


def _resource_is_file(path: ResourcePath) -> bool:
    return path.is_file()


def load_runtime_config(path: ResourcePath | None = None) -> dict[str, Any]:
    """Load the JSON config file, falling back to the in-memory CONFIG."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not _resource_is_file(path):
        return dict(CONFIG)
    raw = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(CONFIG)
    merged.update(raw)
    return merged


@dataclass
class AnswerSet:
    """Per-task reference-answer lists.

    Carried inline on the :class:`Task` dataclass via the ``reference_answers``
    field rather than loaded from a separate file.
    """

    current_answers: list[str] = field(default_factory=list)
    prior_answers: list[str] = field(default_factory=list)
    clarify_indicators: list[str] = field(default_factory=list)
    abstain_indicators: list[str] = field(default_factory=list)


@dataclass
class Task:
    """One task record loaded from ``tasks.jsonl``.

    JSON line schema (one object per line):
        task_id: str
        task_set: str   # always "main" for the flat pre-release set
        gold_label: str  # current | prior | clarify | abstain
        shift_type: str        # one of the eight shift_type values
        domain: str
        referent_complexity: str
        difficulty: str  # easy | medium | hard
        time_gap_bucket: str | None
        pre_turn_context_scene_description: str | None  # pre-T1 camera state, null when unused
        turn_1_scene_description: str          # camera description at T1
        turn_1_user: str
        turn_2_scene_description: str          # camera description at T2
        turn_2_user: str
        notes: str  # optional
        reference_answers:
            current_answers: list[str]
            prior_answers: list[str]
            clarify_indicators: list[str]
            abstain_indicators: list[str]
    """

    task_id: str
    gold_label: str
    shift_type: str
    domain: str
    referent_complexity: str
    difficulty: str
    turn_1_scene_description: str
    turn_1_user: str
    turn_2_scene_description: str
    turn_2_user: str
    task_set: str = "main"
    pre_turn_context_scene_description: str | None = None
    time_gap_bucket: str | None = None
    notes: str = ""
    reference_answers: AnswerSet = field(default_factory=AnswerSet)


def load_tasks(path: ResourcePath = TASKS_PATH, task_set: str | None = None) -> list[Task]:
    """Load tasks from ``tasks.jsonl``, optionally filtered by task_set.

    Each line is one JSON object. When ``task_set`` is non-None, only
    records whose ``task_set`` field matches are returned.
    """
    tasks: list[Task] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if task_set is not None and entry.get("task_set") != task_set:
                continue
            reference_answers_raw = entry.get("reference_answers") or {}
            reference_answers = AnswerSet(
                current_answers=list(reference_answers_raw.get("current_answers") or []),
                prior_answers=list(reference_answers_raw.get("prior_answers") or []),
                clarify_indicators=list(reference_answers_raw.get("clarify_indicators") or []),
                abstain_indicators=list(reference_answers_raw.get("abstain_indicators") or []),
            )
            tasks.append(
                Task(
                    task_id=entry["task_id"],
                    task_set=entry.get("task_set", "main"),
                    gold_label=entry["gold_label"],
                    shift_type=entry["shift_type"],
                    domain=entry["domain"],
                    referent_complexity=entry["referent_complexity"],
                    difficulty=entry["difficulty"],
                    turn_1_scene_description=entry["turn_1_scene_description"],
                    turn_1_user=entry["turn_1_user"],
                    turn_2_scene_description=entry["turn_2_scene_description"],
                    turn_2_user=entry["turn_2_user"],
                    pre_turn_context_scene_description=entry.get(
                        "pre_turn_context_scene_description"
                    ),
                    time_gap_bucket=entry.get("time_gap_bucket"),
                    notes=entry.get("notes", ""),
                    reference_answers=reference_answers,
                )
            )
    return tasks


def _sha256_of_file(path: ResourcePath) -> str | None:
    try:
        data = path.read_bytes()
    except (OSError, FileNotFoundError):
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
    resolved_judge: JudgeLike,
    judge_resolution_mode: str,
    ranking_judge: JudgeLike | None = None,
) -> dict[str, Any]:
    """Construct the reproducibility manifest dict."""
    warnings: list[str] = []

    def _sha_or_warn(path: ResourcePath, key: str) -> str | None:
        value = _sha256_of_file(path)
        if value is None:
            warnings.append(f"{key} could not be hashed from {path}")
        return value

    judge_prompt_sha = hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    task_set_value = effective_config.get("task_set", "main")

    manifest: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "task_set": task_set_value,
        "camera_injection": not bool(effective_config.get("no_camera", False)),
        "tasks_sha256": _sha_or_warn(TASKS_PATH, "tasks_sha256"),
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
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runner_git_commit": _current_git_commit(),
        "random_seed": None,
    }
    manifest["manifest_warnings"] = warnings
    return manifest


def run(
    adapter: Any | None = None,
    judge: JudgeLike | None = None,
    config: dict[str, Any] | None = None,
    ranking_judge: JudgeLike | None = None,
) -> list[dict]:
    """Run the full benchmark and return the per-trial result list.

    Callers that want real API calls pass no arguments. Tests pass a
    stub `adapter` and `judge` so the loop runs without network.

    Args:
        adapter: Candidate adapter. Defaults to the family-appropriate
            adapter resolved from `model_id`.
        judge: Judge-like object. Defaults to the auto resolution against
            `CONFIG["model_id"]`.
        config: Overrides for CONFIG. Unrecognized keys are ignored.
        ranking_judge: Optional second judge object used for
            cross-candidate ranking comparability. When provided, every
            trial is also labeled by this fixed judge and the result
            dict carries both verdicts. Defaults to the family resolved
            from `ranking_judge_family` / `ranking_judge_model_id`
            config keys; ``None`` if those are unset.

    Returns:
        Per-trial result dicts ready for
        `wearable_assistant_context_bench.aggregation` aggregation.
    """
    effective_config = {**CONFIG, **(config or {})}

    task_set_value = effective_config.get("task_set", "main")
    tasks = load_tasks(TASKS_PATH, task_set=task_set_value)
    conditions = load_prompt_conditions(PROMPT_CONDITIONS_PATH)

    model_config = ModelConfig(
        model_id=effective_config["model_id"],
        temperature=effective_config["temperature"],
    )
    adapter_ = adapter if adapter is not None else _build_adapter(effective_config["model_id"])

    judge_: JudgeLike
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

    ranking_judge_: JudgeLike | None = ranking_judge
    if ranking_judge_ is None and effective_config.get("ranking_judge_family"):
        ranking_judge_ = build_judge(
            family=effective_config["ranking_judge_family"],
            model_id=effective_config.get("ranking_judge_model_id"),
        )

    output_dir = Path(effective_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcripts.jsonl"

    results: list[dict] = []
    with transcript_path.open("w", encoding="utf-8") as transcript_file:
        for task in tasks:
            for condition in conditions:
                for trial in range(effective_config["trials_per_cell"]):
                    result = _run_one_trial(
                        task=task,
                        answers=task.reference_answers,
                        condition=condition,
                        trial=trial,
                        adapter=adapter_,
                        judge=judge_,
                        ranking_judge=ranking_judge_,
                        model_config=model_config,
                        no_camera=bool(effective_config.get("no_camera", False)),
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
        task_policies={s.task_id: s.gold_label for s in tasks},
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


def _run_one_trial(
    *,
    task: Task,
    answers: AnswerSet,
    condition: PromptCondition,
    trial: int,
    adapter: Any,
    judge: JudgeLike,
    model_config: ModelConfig,
    no_camera: bool = False,
    ranking_judge: JudgeLike | None = None,
) -> dict:
    """Run one (task, condition, trial) cell end-to-end.

    When ``no_camera`` is True, the runner strips the ``[Camera: ...]``
    blocks from every user message and skips injecting the
    ``pre_turn_context_scene_description`` standalone message. The candidate sees only the
    user's spoken text. Used for camera channel ablation runs.
    """
    messages: list[dict[str, str]] = []
    # Pre-conversation camera state (only set on cross_session_reference tasks)
    if task.pre_turn_context_scene_description and not no_camera:
        messages.append(
            _build_pre_turn_context_scene_description_message(
                task.pre_turn_context_scene_description
            )
        )
    messages.append(
        _build_message(
            role="user",
            text=task.turn_1_user,
            image=None if no_camera else task.turn_1_scene_description,
        )
    )
    turn_1_response = adapter.query(
        messages=messages, system=condition.system_prompt, config=model_config
    )
    messages.append({"role": "assistant", "content": turn_1_response})
    messages.append(
        _build_message(
            role="user",
            text=task.turn_2_user,
            image=None if no_camera else task.turn_2_scene_description,
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

    task_description = _build_task_description(task)

    ground_truth_context = _build_ground_truth_context(task)

    judge_verdict = judge.label(
        response=turn_2_response,
        task_description=task_description,
        turn_2_user=task.turn_2_user,
        current_answers=answers.current_answers,
        prior_answers=answers.prior_answers,
        clarify_indicators=answers.clarify_indicators,
        abstain_indicators=answers.abstain_indicators,
        ground_truth_context=ground_truth_context,
    )

    turn_2_passed = judge_verdict.selected_label == task.gold_label

    turn_2_ranking_verdict: JudgeVerdict | None = None
    if ranking_judge is not None:
        turn_2_ranking_verdict = ranking_judge.label(
            response=turn_2_response,
            task_description=task_description,
            turn_2_user=task.turn_2_user,
            current_answers=answers.current_answers,
            prior_answers=answers.prior_answers,
            clarify_indicators=answers.clarify_indicators,
            abstain_indicators=answers.abstain_indicators,
            ground_truth_context=ground_truth_context,
        )

    result: dict[str, Any] = {
        "task_id": task.task_id,
        "task_set": task.task_set,
        "condition": condition.name,
        "trial": trial,
        "gold_label": task.gold_label,
        "shift_type": task.shift_type,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "pre_turn_context_scene_description": task.pre_turn_context_scene_description,
        "turn_1_user": task.turn_1_user,
        "turn_1_scene_description": task.turn_1_scene_description,
        "turn_1_response": turn_1_response,
        "turn_2_user": task.turn_2_user,
        "turn_2_scene_description": task.turn_2_scene_description,
        "turn_2_response": turn_2_response,
        "turn_2_code_signals": code_signals,
        "turn_2_judge_label": judge_verdict.selected_label,
        "turn_2_judge_rationale": judge_verdict.rationale,
        "turn_2_passed": turn_2_passed,
    }
    if ranking_judge is not None:
        result["turn_2_ranking_judge_label"] = (
            turn_2_ranking_verdict.selected_label if turn_2_ranking_verdict else None
        )
        result["turn_2_ranking_judge_rationale"] = (
            turn_2_ranking_verdict.rationale if turn_2_ranking_verdict else None
        )
        result["turn_2_ranking_passed"] = (
            (turn_2_ranking_verdict.selected_label == task.gold_label)
            if turn_2_ranking_verdict
            else None
        )
    return result


def _build_task_description(task: Task) -> str:
    """Construct the task description shown to the judge.

    Names neither the gold_label nor the shift_type. Both would
    leak the answer the judge is being asked to produce. The judge
    is pointed at the Turn 2 user message and camera frame as the
    perceptual fields that determine what the assistant should now
    be answering about.
    """
    return (
        f"Turn 1 context:\n{task.turn_1_user}\n\n"
        f"Between Turn 1 and Turn 2 the user's context shifts. "
        f"The Turn 2 user message and camera frame describe what "
        f"the assistant should now be answering about."
    )


def _build_ground_truth_context(task: Task) -> str:
    """Construct the judge-only ground-truth description.

    The candidate model sees perceptual camera descriptions only (no
    object names). The judge sees the perceptual T1/T2 frames plus
    the activity domain so it can determine whether the response
    reflects T2 or T1 context.

    Deliberately omits gold_label, shift_type, and authoring notes.
    Those would either name or category-hint the answer the judge is
    being asked to produce.
    """
    parts: list[str] = [f"Activity domain: {task.domain}."]
    if task.pre_turn_context_scene_description:
        parts.append(f"Pre-conversation camera state: {task.pre_turn_context_scene_description}")
    parts.append(f"Turn 1 camera state: {task.turn_1_scene_description}")
    parts.append(f"Turn 2 camera state: {task.turn_2_scene_description}")
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


def _build_pre_turn_context_scene_description_message(image: str) -> dict[str, str]:
    """Build a standalone camera channel message for `pre_turn_context_scene_description`.

    `pre_turn_context_scene_description` represents what the wearable's camera saw before any
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
        help=(f"trials per (task, condition) cell; default is {CONFIG['trials_per_cell']}"),
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
            "message and skip injecting pre_turn_context_scene_description. Run with this flag "
            "to measure the contribution of the camera channel by "
            "comparing the score against a normal run with the same model."
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
