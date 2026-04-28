"""Harbor Terminal-Bench 2.0 command construction and compliance guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HARBOR_TERMINALBENCH_DATASET = "terminal-bench@2.0"
HARBOR_TERMINALBENCH_CONDITIONS = ("spell", "codex")
HARBOR_TERMINALBENCH_RESULTS_DIR = Path("results") / "harbor-terminal-bench"
HARBOR_SPELL_AGENT_IMPORT_PATH = "harbor_spell_agent:HarborSpellAgent"
HARBOR_CODEX_AGENT_IMPORT_PATH = "harbor_codex_agent:HarborCodexAgent"

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
) -> list[HarborTerminalBenchCommand]:
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
            "harbor",
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
            )
        )
    return commands
