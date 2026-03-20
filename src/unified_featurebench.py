"""Unified in-container runner for FeatureBench."""

from __future__ import annotations

from pathlib import Path

from .docker_agents import DockerWorkspaceContainer, run_agent_in_container
from .featurebench_lib import install_deps_in_container, run_tests_in_container
from .unified_adapters import RunResponse


def run_unified_featurebench(
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
    trace_dir: Path | None,
    workspace: Path,
    item: dict,
    skip_install: bool,
    skip_tests: bool,
) -> tuple[RunResponse, bool | None, list[dict]]:
    test_results: list[dict] = []
    tests_passed: bool | None = None

    with DockerWorkspaceContainer(workspace, container_workdir="/workspace") as container:
        if not skip_install:
            install_deps_in_container(container, item)

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
            trace_dir=trace_dir if condition == "spell" else None,
            init=init,
        )

        base_ref = container.initial_commit or "HEAD"
        patch = container.diff(base_ref)
        changed_files = container.changed_files(base_ref)

        if not skip_tests and patch.strip():
            fail_to_pass = item.get("FAIL_TO_PASS", [])
            if fail_to_pass:
                test_results = run_tests_in_container(container, fail_to_pass, repo=item.get("repo"))
                tests_passed = all(entry["passed"] for entry in test_results)

    status = "timeout" if result.output.timed_out else ("ok" if result.output.exit_code == 0 else "error")
    if result.raw and not result.raw.get("ok", False):
        status = "timeout" if result.raw.get("error_type") == "timeout" else "error"

    return (
        RunResponse(
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
                "patch": patch,
                "changed_files": changed_files,
                "test_results": test_results,
                "trace_dir": result.trace_dir,
                "raw": result.raw,
            },
        ),
        tests_passed,
        test_results,
    )
