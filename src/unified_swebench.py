"""Unified in-container runner for SWE-bench generation."""

from __future__ import annotations

from pathlib import Path

from .docker_agents import run_agent_in_container
from .swebench_container import SWEBenchContainer
from .datasets import SWEBenchItem
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
    trace_dir: Path | None,
    eval_timeout: int,
    item: SWEBenchItem,
    pro_eval_repo: str | None = None,
    pro_dockerhub_username: str = "jefzda",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
) -> RunResponse:
    eval_log_path: str | None = None
    with SWEBenchContainer(
        item,
        pro_eval_repo=pro_eval_repo,
        dockerhub_username=pro_dockerhub_username,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    ) as container:
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
            trace_dir=trace_dir,
            init=init,
        )
        base_ref = container.initial_commit or "HEAD"
        patch = container.diff(base_ref)
        changed_files = container.changed_files(base_ref)
        eval_result = container.run_eval(timeout=eval_timeout)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        log_file = trace_dir / "same-container-eval.log"
        log_file.write_text(eval_result.test_output, encoding="utf-8")
        eval_log_path = str(log_file)

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
            "patch": patch,
            "changed_files": changed_files,
            "trace_dir": result.trace_dir,
            "raw": result.raw,
            "same_container_resolved": eval_result.resolved,
            "same_container_eval": eval_result.raw_report,
            "same_container_eval_log": eval_log_path,
        },
    )
