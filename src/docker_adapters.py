"""Docker-backed adapters built on the shared container-agent layer."""

from __future__ import annotations

from pathlib import Path

from .docker_agents import (
    DockerWorkspaceContainer,
    run_agent_in_container,
)
from .unified_adapters import RunRequest, RunResponse


def _response_from_container_run(
    result,
    *,
    patch: str,
    changed_files: list[str],
    trace_dir: str | None = None,
) -> RunResponse:
    usage = result.usage
    status = "timeout" if result.output.timed_out else ("ok" if result.output.exit_code == 0 else "error")
    if result.raw and not result.raw.get("ok", False):
        status = "timeout" if result.raw.get("error_type") == "timeout" else "error"
    return RunResponse(
        status=status,
        response_text=result.response_text,
        tokens_in=usage.total_input_tokens if usage else None,
        tokens_out=usage.output_tokens if usage else None,
        cost_usd=usage.cost_usd if usage else None,
        latency_ms=None,
        error_message=result.error_message,
        error_type=result.error_type,
        metadata={
            "docker": True,
            "patch": patch,
            "changed_files": changed_files,
            "trace_dir": trace_dir,
            "raw": result.raw,
        },
    )


class _BaseDockerAdapter:
    condition: str

    def __init__(self, *, project_root: Path | None = None, model_spec: str | None = None, budget: float | None = None):
        self.project_root = project_root
        self.model_spec = model_spec
        self.budget = budget

    def _execute(self, req: RunRequest, *, codex_resolver=None) -> RunResponse:
        del codex_resolver
        if not req.workspace:
            raise ValueError(f"{self.__class__.__name__} requires req.workspace")

        with DockerWorkspaceContainer(req.workspace) as container:
            result = run_agent_in_container(
                container,
                condition=self.condition,  # type: ignore[arg-type]
                prompt=req.prompt,
                model_spec=req.model_spec or self.model_spec,
                timeout=req.timeout,
                budget=req.budget if req.budget is not None else self.budget,
                depth=req.depth,
                max_tokens=req.max_tokens,
                reasoning_effort=req.reasoning_effort,
                trace=req.trace,
                trace_dir=req.trace_dir,
                agent_file=req.agent_file,
                web_search=req.web_search,
                init=req.init,
            )
            base_ref = container.initial_commit or "HEAD"
            patch = container.diff(base_ref)
            changed_files = container.changed_files(base_ref)

        return _response_from_container_run(
            result,
            patch=patch,
            changed_files=changed_files,
            trace_dir=result.trace_dir,
        )


class DockerClaudeCodeAdapter(_BaseDockerAdapter):
    condition = "claude_code"

    def __init__(self, model: str, budget: float):
        super().__init__(model_spec=model, budget=budget)

    def run(self, req: RunRequest) -> RunResponse:
        return self._execute(req)


class DockerCodexAdapter(_BaseDockerAdapter):
    condition = "codex"

    def __init__(self, model_spec: str = "codex"):
        super().__init__(model_spec=model_spec)

    def run(self, req: RunRequest) -> RunResponse:
        return self._execute(req)


class DockerSpellAdapter(_BaseDockerAdapter):
    condition = "spell"

    def __init__(self, project_root: Path, model_spec: str):
        super().__init__(project_root=project_root, model_spec=model_spec)

    def run(self, req: RunRequest) -> RunResponse:
        return self._execute(req)


class DockerSweAgentAdapter(_BaseDockerAdapter):
    condition = "swe_agent"

    def __init__(self, model_spec: str, budget: float | None = None):
        super().__init__(model_spec=model_spec, budget=budget)

    def run(self, req: RunRequest) -> RunResponse:
        return self._execute(req)
