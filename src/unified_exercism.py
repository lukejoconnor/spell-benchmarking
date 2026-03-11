"""Unified in-container runner for Exercism."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .datasets import ExercismItem
from .docker_agents import DockerWorkspaceContainer, run_agent_in_container
from .unified_adapters import RunResponse

EXERCISM_CONTAINER_WORKDIR = "/workspace"
EXERCISM_PYTEST_TIMEOUT = 60


@dataclass(slots=True)
class UnifiedExercismOutcome:
    response: RunResponse
    passed: bool
    attempts: int
    test_output: str


def _ensure_git_repo(workspace: Path) -> None:
    if (workspace / ".git").exists():
        return

    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bench@local"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "bench"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=workspace, check=True, capture_output=True)


def _container_stub_path(item: ExercismItem) -> str:
    return f"{EXERCISM_CONTAINER_WORKDIR}/{item.stub_file}"


def _container_test_path(item: ExercismItem) -> str:
    return f"{EXERCISM_CONTAINER_WORKDIR}/{item.test_file}"


def format_unified_exercism_prompt(item: ExercismItem) -> str:
    return (
        "You are solving a Python programming exercise.\n\n"
        "## Instructions\n\n"
        f"{item.instructions}\n\n"
        "## Task\n\n"
        f"The solution file is at {_container_stub_path(item)}. Edit it to implement the solution.\n\n"
        "Current contents:\n"
        f"```python\n{item.stub_content}```\n\n"
        f"Read the test file at {_container_test_path(item)} before starting.\n"
        f"Verify with: python -m pytest {_container_test_path(item)} -v\n\n"
        "Only use standard libraries. Do not modify the test file."
    )


def format_unified_exercism_retry_prompt(item: ExercismItem, test_output: str) -> str:
    return (
        "The tests failed with this output:\n\n"
        f"```\n{test_output}```\n\n"
        "The tests are correct and must not be modified. "
        f"Fix the code in {_container_stub_path(item)} to resolve the errors.\n"
        f"Verify with: python -m pytest {_container_test_path(item)} -v"
    )


def _run_pytest(container: DockerWorkspaceContainer, item: ExercismItem) -> tuple[bool, str]:
    result = container.exec_run(
        "python -m pip install -q pytest >/dev/null 2>&1 && "
        f"python -m pytest {_container_test_path(item)} -v",
        timeout=EXERCISM_PYTEST_TIMEOUT,
        workdir=EXERCISM_CONTAINER_WORKDIR,
    )
    output = (result.stdout + result.stderr).strip()
    if result.timed_out:
        return False, f"TIMEOUT: pytest exceeded {EXERCISM_PYTEST_TIMEOUT} seconds"
    return result.exit_code == 0, output


def run_unified_exercism(
    *,
    condition: str,
    item: ExercismItem,
    workspace: Path,
    model_spec: str | None,
    timeout: int,
    init: str | None,
    agent_file: str | None,
    budget: float,
    depth: int,
    max_tokens: int | None,
    reasoning_effort: str | None,
    web_search: bool,
    trace: bool,
    no_retry: bool,
) -> UnifiedExercismOutcome:
    _ensure_git_repo(workspace)

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    attempts = 0
    test_output = ""
    passed = False
    last_response: RunResponse | None = None

    with DockerWorkspaceContainer(workspace, container_workdir=EXERCISM_CONTAINER_WORKDIR) as container:
        attempts = 1
        first_result = run_agent_in_container(
            container,
            condition=condition,  # type: ignore[arg-type]
            prompt=format_unified_exercism_prompt(item),
            timeout=timeout,
            model_spec=model_spec,
            budget=budget,
            agent_file=agent_file,
            depth=depth,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            trace=trace,
            init=init,
        )

        total_tokens_in += first_result.usage.total_input_tokens if first_result.usage else 0
        total_tokens_out += first_result.usage.output_tokens if first_result.usage else 0
        if first_result.usage and first_result.usage.cost_usd is not None:
            total_cost += first_result.usage.cost_usd

        passed, test_output = _run_pytest(container, item)
        current_result = first_result

        if not passed and not no_retry:
            attempts = 2
            retry_result = run_agent_in_container(
                container,
                condition=condition,  # type: ignore[arg-type]
                prompt=format_unified_exercism_retry_prompt(item, test_output),
                timeout=timeout,
                model_spec=model_spec,
                budget=budget,
                agent_file=agent_file,
                depth=depth,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                web_search=web_search,
                trace=trace,
                init=None,
            )
            total_tokens_in += retry_result.usage.total_input_tokens if retry_result.usage else 0
            total_tokens_out += retry_result.usage.output_tokens if retry_result.usage else 0
            if retry_result.usage and retry_result.usage.cost_usd is not None:
                total_cost += retry_result.usage.cost_usd
            passed, test_output = _run_pytest(container, item)
            current_result = retry_result

        status = "timeout" if current_result.output.timed_out else ("ok" if current_result.output.exit_code == 0 else "error")
        if current_result.raw and not current_result.raw.get("ok", False):
            status = "timeout" if current_result.raw.get("error_type") == "timeout" else "error"

        last_response = RunResponse(
            status=status,
            response_text=current_result.response_text,
            tokens_in=total_tokens_in or None,
            tokens_out=total_tokens_out or None,
            cost_usd=total_cost or None,
            latency_ms=None,
            error_message=current_result.error_message,
            error_type=current_result.error_type,
            metadata={
                "docker": True,
                "unified": True,
                "patch": container.diff(),
                "changed_files": container.changed_files(),
                "raw": current_result.raw,
            },
        )

    assert last_response is not None
    return UnifiedExercismOutcome(
        response=last_response,
        passed=passed,
        attempts=attempts,
        test_output=test_output,
    )
