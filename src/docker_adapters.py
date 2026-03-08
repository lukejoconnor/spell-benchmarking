"""Docker-based adapters for SWE-bench benchmark execution.

These adapters wrap the agent execution in Docker containers, providing
isolation from the host filesystem. They follow the same RunnerAdapter
protocol as unified_adapters.py.

Container lifecycle:
  1. Create container with workspace bind-mounted at /workspace
  2. Install agent CLI (CC/Codex: NVM + npm inside container; Spell: host-side)
  3. Run agent against the task prompt
  4. Extract git diff as the patch
  5. Stop container

For Spell, the evaluator runs on the host but io/sh commands are bridged
to the container via SPELL_DOCKER_CONTAINER env var.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .docker_swebench import (
    DockerContainer,
    install_claude_code,
    install_codex,
    install_nvm_node,
    parse_claude_code_usage,
    parse_codex_usage,
    run_claude_code,
    run_codex,
    run_spell_docker,
)
from .unified_adapters import (
    DEFAULT_REASONING_EFFORT,
    RunRequest,
    RunResponse,
    build_spell_init,
)

logger = logging.getLogger(__name__)


class DockerClaudeCodeAdapter:
    """Runs Claude Code inside a Docker container."""

    condition = "claude_code"

    def __init__(self, model: str, budget: float):
        self.model = model
        self.budget = budget

    def run(self, req: RunRequest) -> RunResponse:
        model = req.model_spec or self.model
        if model and ":" in model:
            _, model = model.split(":", 1)
        budget = req.budget if req.budget is not None else self.budget

        if not req.workspace:
            raise ValueError("DockerClaudeCodeAdapter requires req.workspace")

        env = {"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]}
        if model:
            env["ANTHROPIC_MODEL"] = model

        start = time.time()
        with DockerContainer(req.workspace) as container:
            # Install NVM + Node + Claude Code
            if not install_nvm_node(container):
                return self._error_response("NVM/Node install failed", start)
            if not install_claude_code(container, env):
                return self._error_response("Claude Code install failed", start)

            # Run agent
            result = run_claude_code(
                container, req.prompt,
                model=model, budget=budget,
                timeout=req.timeout, env=env,
            )

            # Extract patch from container before it's destroyed
            patch = container.diff()
            changed_files = container.changed_files()

        latency_ms = (time.time() - start) * 1000

        if result.timed_out:
            return RunResponse(
                status="timeout",
                response_text="",
                tokens_in=None, tokens_out=None, cost_usd=None,
                latency_ms=latency_ms,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        usage = parse_claude_code_usage(result.stdout)
        tokens_in = None
        tokens_out = None
        cost_usd = None
        if usage:
            tokens_in = usage.total_input_tokens
            tokens_out = usage.output_tokens
            cost_usd = usage.cost_usd

        status = "ok" if result.exit_code == 0 else "error"
        return RunResponse(
            status=status,
            response_text="",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error_message=result.stderr[:2000] if result.exit_code != 0 else None,
            error_type="cli_error" if result.exit_code != 0 else None,
            metadata={
                "docker": True,
                "patch": patch,
                "changed_files": changed_files,
            },
        )

    @staticmethod
    def _error_response(msg: str, start: float) -> RunResponse:
        return RunResponse(
            status="error",
            response_text="",
            tokens_in=None, tokens_out=None, cost_usd=None,
            latency_ms=(time.time() - start) * 1000,
            error_message=msg,
            error_type="install_error",
        )


class DockerCodexAdapter:
    """Runs Codex inside a Docker container."""

    condition = "codex"

    def __init__(self, model_spec: str = "codex"):
        self.model_spec = model_spec

    def run(self, req: RunRequest) -> RunResponse:
        from .unified_adapters import _normalize_codex_model
        model = _normalize_codex_model(req.model_spec, self.model_spec)

        if not req.workspace:
            raise ValueError("DockerCodexAdapter requires req.workspace")

        env = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}

        start = time.time()
        with DockerContainer(req.workspace) as container:
            if not install_nvm_node(container):
                return self._error_response("NVM/Node install failed", start)
            if not install_codex(container, env):
                return self._error_response("Codex install failed", start)

            result = run_codex(
                container, req.prompt,
                model=model, timeout=req.timeout, env=env, web_search=req.web_search,
            )

            patch = container.diff()
            changed_files = container.changed_files()

        latency_ms = (time.time() - start) * 1000

        if result.timed_out:
            return RunResponse(
                status="timeout",
                response_text="",
                tokens_in=None, tokens_out=None, cost_usd=None,
                latency_ms=latency_ms,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        usage = parse_codex_usage(result.stdout)
        tokens_in = usage.total_input_tokens if usage else None
        tokens_out = usage.output_tokens if usage else None
        cost_usd = usage.cost_usd if usage else None

        status = "ok" if result.exit_code == 0 else "error"
        return RunResponse(
            status=status,
            response_text="",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error_message=result.stderr[:2000] if result.exit_code != 0 else None,
            error_type="codex_cli_error" if result.exit_code != 0 else None,
            metadata={
                "docker": True,
                "patch": patch,
                "changed_files": changed_files,
            },
        )

    @staticmethod
    def _error_response(msg: str, start: float) -> RunResponse:
        return RunResponse(
            status="error",
            response_text="",
            tokens_in=None, tokens_out=None, cost_usd=None,
            latency_ms=(time.time() - start) * 1000,
            error_message=msg,
            error_type="install_error",
        )


class DockerSpellAdapter:
    """Runs Spell on the host with io/sh bridged to a Docker container."""

    condition = "spell"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec

    def run(self, req: RunRequest) -> RunResponse:
        if not req.workspace:
            raise ValueError("DockerSpellAdapter requires req.workspace")

        payload = {
            "mode": "spell",
            "model": req.model_spec or self.model_spec,
            "agent": req.agent_file,
            "budget": req.budget,
            "depth": req.depth,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort or DEFAULT_REASONING_EFFORT,
            "trace": req.trace,
        }
        if req.init is not None:
            payload["init"] = req.init
        else:
            payload["prompt"] = req.prompt
        payload = {k: v for k, v in payload.items() if v is not None}

        start = time.time()
        with DockerContainer(req.workspace) as container:
            exec_result, parsed = run_spell_docker(
                container,
                payload=payload,
                project_root=self.project_root,
                timeout=req.timeout,
            )

            patch = container.diff()
            changed_files = container.changed_files()

        latency_ms = (time.time() - start) * 1000

        if exec_result.timed_out:
            return RunResponse(
                status="timeout",
                response_text="",
                tokens_in=None, tokens_out=None, cost_usd=None,
                latency_ms=latency_ms,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        if not parsed:
            return RunResponse(
                status="error",
                response_text="",
                tokens_in=None, tokens_out=None, cost_usd=None,
                latency_ms=latency_ms,
                error_message=f"No JSON response: {exec_result.stderr[:500]}",
                error_type="parse_error",
                metadata={"docker": True},
            )

        usage_total = (parsed.get("usage") or {}).get("total", {})
        status = "ok" if parsed.get("ok") else "error"

        return RunResponse(
            status=status,
            response_text=parsed.get("result"),
            tokens_in=usage_total.get("input_tokens"),
            tokens_out=usage_total.get("output_tokens"),
            cost_usd=usage_total.get("cost"),
            latency_ms=latency_ms,
            error_message=parsed.get("error") if not parsed.get("ok") else None,
            error_type=parsed.get("error_type") if not parsed.get("ok") else None,
            metadata={
                "docker": True,
                "patch": patch,
                "changed_files": changed_files,
                "trace_dir": parsed.get("trace_dir"),
                "raw": parsed,
            },
        )
