"""Harbor Terminal-Bench 2.0 command construction and compliance guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

HARBOR_TERMINALBENCH_DATASET = "terminal-bench@2.0"
HARBOR_TERMINALBENCH_CONDITIONS = ("spell", "codex")
HARBOR_TERMINALBENCH_RESULTS_DIR = Path("results") / "harbor-terminal-bench"
HARBOR_SPELL_AGENT_IMPORT_PATH = "harbor_spell_agent:HarborSpellAgent"
HARBOR_CODEX_AGENT_IMPORT_PATH = "harbor_codex_agent:HarborCodexAgent"
HARBOR_CLI_COMMAND = ("uv", "tool", "run", "--from", "harbor", "harbor")
DEFAULT_AGENT_TIMEOUT_SECONDS = 900
DEFAULT_TIMEOUT_SUMMARY_FILENAME = "default-timeout-summary.json"

BANNED_SUBMISSION_FLAGS = frozenset(
    {
        "--timeout-multiplier",
        "--agent-timeout-multiplier",
        "--verifier-timeout-multiplier",
        "--agent-setup-timeout-multiplier",
        "--environment-build-timeout-multiplier",
        "--override-cpus",
        "--override-memory",
        "--override-storage",
        "--override-gpus",
        "--registry-path",
        "--path",
        "--task",
        "--include-task-name",
        "--exclude-task-name",
        "--n-tasks",
        "--max-retries",
        "--retry-include",
        "--retry-exclude",
        "--disable-verification",
    }
)


class HarborTerminalBenchComplianceError(ValueError):
    """Raised when a requested Harbor command is not submission-compliant."""


@dataclass(frozen=True)
class HarborTerminalBenchCommand:
    condition: str
    job_name: str
    model: str
    jobs_dir: Path
    command: list[str]
    submission_compliant: bool
    agent_timeout_multiplier: float


def parse_harbor_terminalbench_conditions(value: str) -> list[str]:
    if value == "all":
        return list(HARBOR_TERMINALBENCH_CONDITIONS)
    if value not in HARBOR_TERMINALBENCH_CONDITIONS:
        raise ValueError(f"Unsupported Harbor Terminal-Bench condition: {value}")
    return [value]


def harbor_terminalbench_output_root(run_id: str) -> Path:
    return HARBOR_TERMINALBENCH_RESULTS_DIR / run_id


def normalize_harbor_codex_model(model: str) -> str:
    if "/" in model:
        return model
    return f"openai/{model}"


def assert_harbor_submission_request_compliant(
    *,
    condition: str,
    dataset: str,
    trials: int,
    task: str | None,
    include_task_names: tuple[str, ...],
    exclude_task_names: tuple[str, ...],
    n_tasks: int | None,
) -> None:
    if dataset != HARBOR_TERMINALBENCH_DATASET:
        raise HarborTerminalBenchComplianceError(
            f"Harbor Terminal-Bench submissions must use {HARBOR_TERMINALBENCH_DATASET}, got {dataset!r}"
        )
    selected_conditions = parse_harbor_terminalbench_conditions(condition)
    if trials < 5:
        raise HarborTerminalBenchComplianceError(
            "Harbor Terminal-Bench submissions require k >= 5. Use --dev-smoke for k=1 smoke runs."
        )
    if task or include_task_names or exclude_task_names or n_tasks is not None:
        raise HarborTerminalBenchComplianceError(
            "Harbor Terminal-Bench submissions must run the full dataset with no task filters."
        )


def assert_harbor_submission_command_compliant(command: list[str]) -> None:
    flags = {part for part in command if part.startswith("--")}
    banned = sorted(flags & BANNED_SUBMISSION_FLAGS)
    if banned:
        raise HarborTerminalBenchComplianceError(
            f"Submission command contains non-compliant Harbor flag(s): {', '.join(banned)}"
        )
    if command[command.index("-d") + 1] != HARBOR_TERMINALBENCH_DATASET:
        raise HarborTerminalBenchComplianceError("Submission command does not use terminal-bench@2.0")
    if int(command[command.index("-k") + 1]) < 5:
        raise HarborTerminalBenchComplianceError("Submission command uses k < 5")


def _uses_expanded_agent_timeout(agent_timeout_multiplier: float | None) -> bool:
    return agent_timeout_multiplier is not None and agent_timeout_multiplier > 1.0


def build_harbor_terminalbench_commands(
    *,
    condition: str,
    run_id: str,
    model: str,
    codex_model: str,
    reasoning_effort: str,
    trials: int,
    jobs_dir: Path | None = None,
    dataset: str = HARBOR_TERMINALBENCH_DATASET,
    dev_smoke: bool = False,
    n_concurrent: int = 4,
    task: str | None = None,
    include_task_names: tuple[str, ...] = (),
    exclude_task_names: tuple[str, ...] = (),
    n_tasks: int | None = None,
    budget: float | None = 5.0,
    agent_timeout_multiplier: float | None = 1.0,
) -> list[HarborTerminalBenchCommand]:
    if (
        agent_timeout_multiplier is not None
        and agent_timeout_multiplier != 1.0
        and agent_timeout_multiplier <= 1.0
    ):
        raise HarborTerminalBenchComplianceError(
            "--agent-timeout-multiplier supports only the default 1.0 or expanded values > 1."
        )
    if _uses_expanded_agent_timeout(agent_timeout_multiplier) and not dev_smoke:
        raise HarborTerminalBenchComplianceError(
            "--agent-timeout-multiplier is only allowed with --dev-smoke and is never submission-compliant."
        )

    if not dev_smoke:
        assert_harbor_submission_request_compliant(
            condition=condition,
            dataset=dataset,
            trials=trials,
            task=task,
            include_task_names=include_task_names,
            exclude_task_names=exclude_task_names,
            n_tasks=n_tasks,
        )

    root = jobs_dir or harbor_terminalbench_output_root(run_id)
    commands: list[HarborTerminalBenchCommand] = []
    for selected in parse_harbor_terminalbench_conditions(condition):
        condition_jobs_dir = root / selected
        job_name = f"{selected}-{run_id}"
        cmd = [
            *HARBOR_CLI_COMMAND,
            "run",
            "-d",
            dataset,
            "-k",
            str(trials),
            "--jobs-dir",
            str(condition_jobs_dir),
            "--job-name",
            job_name,
            "--n-concurrent",
            str(n_concurrent),
        ]
        if selected == "spell":
            selected_model = model
            cmd.extend(
                [
                    "--agent-import-path",
                    HARBOR_SPELL_AGENT_IMPORT_PATH,
                    "-m",
                    selected_model,
                    "--agent-kwarg",
                    f"reasoning_effort={reasoning_effort}",
                ]
            )
            if budget is not None:
                cmd.extend(["--agent-kwarg", f"budget={budget}"])
        else:
            selected_model = normalize_harbor_codex_model(codex_model)
            cmd.extend(
                [
                    "--agent-import-path",
                    HARBOR_CODEX_AGENT_IMPORT_PATH,
                    "-m",
                    selected_model,
                    "--agent-kwarg",
                    f"reasoning_effort={reasoning_effort}",
                ]
            )

        if task:
            cmd.extend(["--task", task])
        for task_name in include_task_names:
            cmd.extend(["--include-task-name", task_name])
        for task_name in exclude_task_names:
            cmd.extend(["--exclude-task-name", task_name])
        if n_tasks is not None:
            cmd.extend(["--n-tasks", str(n_tasks)])
        if _uses_expanded_agent_timeout(agent_timeout_multiplier):
            cmd.extend(["--agent-timeout-multiplier", str(agent_timeout_multiplier)])

        submission_compliant = not dev_smoke
        if submission_compliant:
            assert_harbor_submission_command_compliant(cmd)
        commands.append(
            HarborTerminalBenchCommand(
                condition=selected,
                job_name=job_name,
                model=selected_model,
                jobs_dir=condition_jobs_dir,
                command=cmd,
                submission_compliant=submission_compliant,
                agent_timeout_multiplier=float(agent_timeout_multiplier or 1.0),
            )
        )
    return commands


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _is_resolved(trial: dict[str, Any]) -> bool:
    for key in ("is_resolved", "resolved"):
        value = trial.get(key)
        if isinstance(value, bool):
            return value
    reward = trial.get("reward")
    if isinstance(reward, bool):
        return reward
    if isinstance(reward, (int, float)):
        return reward > 0
    return False


def _trial_identifier(trial: dict[str, Any], fallback: str) -> str:
    for key in ("trial_name", "trial_id", "id", "attempt_id", "name"):
        value = trial.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _task_identifier(trial: dict[str, Any], result_path: Path) -> str | None:
    for key in ("task_id", "task_name", "task"):
        value = trial.get(key)
        if isinstance(value, str) and value:
            return value
    try:
        return result_path.parent.name
    except IndexError:
        return None


def _trial_records_from_result(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if not isinstance(result, dict):
        return []
    for key in ("trials", "results", "attempts", "runs"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [result]


def _agent_execution_window(trial: dict[str, Any]) -> tuple[Any, Any]:
    agent_execution = trial.get("agent_execution")
    if isinstance(agent_execution, dict):
        started = agent_execution.get("started_at") or agent_execution.get("start_time")
        finished = agent_execution.get("finished_at") or agent_execution.get("end_time")
        if started or finished:
            return started, finished
    return (
        trial.get("agent_execution_started_at") or trial.get("started_at") or trial.get("start_time"),
        trial.get("agent_execution_finished_at") or trial.get("finished_at") or trial.get("end_time"),
    )


def _annotate_default_timeout_trial(
    *,
    trial: dict[str, Any],
    result_path: Path,
    fallback_trial_id: str,
    default_timeout_seconds: int,
) -> dict[str, Any]:
    resolved = _is_resolved(trial)
    raw_started, raw_finished = _agent_execution_window(trial)
    started = _parse_datetime(raw_started)
    finished = _parse_datetime(raw_finished)
    warnings: list[str] = []
    wall_seconds: float | None = None
    resolved_within_default_timeout: bool | None

    if started is None or finished is None:
        warnings.append("missing_or_unparsable_agent_execution_timestamps")
        resolved_within_default_timeout = None if resolved else False
    else:
        try:
            wall_seconds = (finished - started).total_seconds()
            resolved_within_default_timeout = resolved and wall_seconds <= default_timeout_seconds
        except TypeError:
            warnings.append("incompatible_agent_execution_timestamp_timezones")
            resolved_within_default_timeout = None if resolved else False

    return {
        "source": str(result_path),
        "task_id": _task_identifier(trial, result_path),
        "trial_id": _trial_identifier(trial, fallback_trial_id),
        "resolved": resolved,
        "reward": trial.get("reward"),
        "agent_execution_wall_seconds": wall_seconds,
        "default_agent_timeout_seconds": default_timeout_seconds,
        "resolved_within_default_timeout": resolved_within_default_timeout,
        "warnings": warnings,
    }


def write_default_timeout_summary(
    *,
    jobs_dir: Path,
    agent_timeout_multiplier: float,
    default_timeout_seconds: int = DEFAULT_AGENT_TIMEOUT_SECONDS,
) -> Path:
    """Write a sidecar summary for expanded Harbor dev-timeout runs."""

    trial_summaries: list[dict[str, Any]] = []
    warnings: list[str] = []
    result_paths = sorted(
        path
        for path in jobs_dir.rglob("result.json")
        if path.name != DEFAULT_TIMEOUT_SUMMARY_FILENAME
    )
    for result_path in result_paths:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{result_path}: could not read JSON: {exc}")
            continue
        for idx, trial in enumerate(_trial_records_from_result(result)):
            trial_summaries.append(
                _annotate_default_timeout_trial(
                    trial=trial,
                    result_path=result_path,
                    fallback_trial_id=f"{result_path.stem}-{idx + 1}",
                    default_timeout_seconds=default_timeout_seconds,
                )
            )

    output_path = jobs_dir / DEFAULT_TIMEOUT_SUMMARY_FILENAME
    output_path.write_text(
        json.dumps(
            {
                "agent_timeout_multiplier": agent_timeout_multiplier,
                "default_agent_timeout_seconds": default_timeout_seconds,
                "trial_count": len(trial_summaries),
                "resolved_within_default_timeout_count": sum(
                    1 for item in trial_summaries if item["resolved_within_default_timeout"] is True
                ),
                "resolved_after_default_timeout_count": sum(
                    1 for item in trial_summaries if item["resolved"] and item["resolved_within_default_timeout"] is False
                ),
                "unknown_default_timeout_count": sum(
                    1 for item in trial_summaries if item["resolved_within_default_timeout"] is None
                ),
                "warnings": warnings,
                "trials": trial_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
