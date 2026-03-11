"""Unified in-container runner for SWE-bench generation."""

from __future__ import annotations

from pathlib import Path

from .docker_agents import DockerWorkspaceContainer, run_agent_in_container
from .unified_adapters import RunResponse


def run_unified_swebench(
    *,
    condition: str,
    prompt: str,
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
    workspace: Path,
) -> RunResponse:
    with DockerWorkspaceContainer(workspace, container_workdir="/workspace") as container:
        result = run_agent_in_container(
            container,
            condition=condition,  # type: ignore[arg-type]
            prompt=prompt,
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
        patch = container.diff()
        changed_files = container.changed_files()

    status = "timeout" if result.output.timed_out else ("ok" if result.output.exit_code == 0 else "error")
    if result.raw and not result.raw.get("ok", False):
        status = "timeout" if result.raw.get("error_type") == "timeout" else "error"

    return RunResponse(
        status=status,
        response_text=result.response_text,
        tokens_in=result.usage.total_input_tokens if result.usage else None,
        tokens_out=result.usage.output_tokens if result.usage else None,
        cost_usd=result.usage.cost_usd if result.usage else None,
        latency_ms=None,
        error_message=result.error_message,
        error_type=result.error_type,
        metadata={
            "docker": True,
            "unified": True,
            "patch": patch,
            "changed_files": changed_files,
            "raw": result.raw,
        },
    )
