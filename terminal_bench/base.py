"""Base class for Terminal-Bench agents that install into Docker containers.

Uses container.exec_run(demux=True) for clean stdout/stderr separation,
enabling reliable JSONL parsing for token/cost tracking.
"""

from __future__ import annotations

import json
import logging
import shlex
import sys
import time
from abc import abstractmethod
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

try:
    from src.docker_agents import (
        ExecOutput,
        ParsedUsage,
        container_exec,
        ensure_uv_exec,
        install_nvm_node_exec,
        iter_dicts,
        parse_jsonl_events,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        ExecOutput,
        ParsedUsage,
        container_exec,
        ensure_uv_exec,
        install_nvm_node_exec,
        iter_dicts,
        parse_jsonl_events,
    )

logger = logging.getLogger(__name__)


class DockerExecAgent(BaseAgent):
    """Base for agents that install into Docker and run via exec_run().

    Subclasses implement:
      - name()              -> str
      - _install_script()   -> str  (shell script to install the agent CLI)
      - _run_command()      -> str  (shell command to run the agent)
      - _parse_usage()      -> ParsedUsage | None
      - _env_vars()         -> dict[str, str]
    """

    def __init__(
        self,
        model_name: str | None = None,
        agent_timeout: int = 600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._agent_timeout = agent_timeout

    def _container_exec(
        self,
        container,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecOutput:
        return container_exec(
            container,
            cmd,
            timeout=timeout,
            env=env,
            workdir=workdir,
        )

    def _detect_workdir(self, container) -> str:
        result = container.exec_run(["pwd"])
        if result.exit_code == 0:
            return result.output.decode().strip()
        return "/root"

    def _install_nvm_node(self, container) -> bool:
        """Install NVM and Node 22 (idempotent). Returns True on success."""
        logger.info("Installing NVM + Node...")
        ok = install_nvm_node_exec(
            lambda command, **kwargs: self._container_exec(container, command, **kwargs),
            env=self._env_vars(),
        )
        if not ok:
            logger.error("NVM/Node install failed")
            return False
        logger.info("Node runtime ready")
        return True

    def _install_agent_cli(self, container) -> bool:
        """Install the agent-specific CLI. Returns True on success."""
        script = self._install_script()
        logger.info(f"Installing {self.name()} CLI...")
        result = self._container_exec(
            container, script, timeout=300, env=self._env_vars(),
        )
        if result.exit_code != 0:
            logger.error(f"{self.name()} install failed:\n{result.stderr[:1000]}")
            return False
        logger.info(f"{self.name()} CLI installed")
        return True

    def _requires_node(self) -> bool:
        """Whether this agent requires Node/NVM bootstrap."""
        return True

    def _requires_uv(self) -> bool:
        """Whether this agent needs uv preinstalled before task work starts."""
        return True

    def _install_uv(self, container) -> bool:
        logger.info("Installing uv...")
        ok = ensure_uv_exec(
            lambda command, **kwargs: self._container_exec(container, command, **kwargs),
            env=self._env_vars(),
        )
        if not ok:
            logger.error("uv install failed")
            return False
        logger.info("uv runtime ready")
        return True

    def _save_logs(
        self,
        logging_dir: Path,
        stdout: str,
        stderr: str,
        usage: ParsedUsage | None,
        latency_s: float,
    ) -> None:
        logging_dir.mkdir(parents=True, exist_ok=True)
        (logging_dir / "stdout.txt").write_text(stdout)
        (logging_dir / "stderr.txt").write_text(stderr)
        result_data = {
            "agent": self.name(),
            "model": self._model_name,
            "latency_s": round(latency_s, 1),
            "input_tokens": usage.input_tokens if usage else 0,
            "cached_input_tokens": usage.cached_input_tokens if usage else 0,
            "cache_write_input_tokens": usage.cache_write_input_tokens if usage else 0,
            "visible_output_tokens": usage.visible_output_tokens if usage else 0,
            "reasoning_output_tokens": usage.reasoning_output_tokens if usage else None,
            "total_input_tokens": usage.total_input_tokens if usage else 0,
            "total_output_tokens": usage.total_output_tokens if usage else 0,
            "cost_usd": usage.cost_usd if usage else None,
            "cost_source": usage.cost_source if usage else None,
        }
        (logging_dir / "result.json").write_text(json.dumps(result_data, indent=2))

    @staticmethod
    @abstractmethod
    def name() -> str: ...

    @abstractmethod
    def _install_script(self) -> str:
        """Shell script to install the agent runtime/CLI."""
        ...

    @abstractmethod
    def _run_command(self, instruction: str, workdir: str) -> str:
        """Shell command to run the agent with the given instruction."""
        ...

    @abstractmethod
    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None) -> ParsedUsage | None:
        """Parse token usage from the agent's structured stdout."""
        ...

    @abstractmethod
    def _env_vars(self) -> dict[str, str]:
        """Env vars needed inside the container (API keys, etc.)."""
        ...

    def _determine_failure_mode(
        self,
        result: ExecOutput,
        usage: ParsedUsage | None,
    ) -> FailureMode:
        """Classify run outcome. Subclasses can inspect parsed output semantics."""
        del usage  # default logic only uses process status
        if result.timed_out:
            return FailureMode.AGENT_TIMEOUT
        if result.exit_code != 0:
            return FailureMode.UNKNOWN_AGENT_ERROR
        return FailureMode.NONE

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        container = session.container
        logger.info(f"{self.name()} agent starting, container={container.id[:12]}")

        workdir = self._detect_workdir(container)
        logger.info(f"Container workdir: {workdir}")

        # Install Node runtime if needed by this agent.
        if self._requires_node() and not self._install_nvm_node(container):
            return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

        if self._requires_uv() and not self._install_uv(container):
            return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

        # Install agent CLI
        if not self._install_agent_cli(container):
            return AgentResult(failure_mode=FailureMode.AGENT_INSTALLATION_FAILED)

        # Run agent
        rendered = self._render_instruction(instruction)
        cmd = self._run_command(rendered, workdir)
        env = self._env_vars()

        start = time.time()
        result = self._container_exec(
            container, cmd,
            timeout=self._agent_timeout,
            env=env,
            workdir=workdir,
        )
        latency_s = time.time() - start

        logger.info(
            f"{self.name()} completed in {latency_s:.1f}s, "
            f"exit={result.exit_code}, timed_out={result.timed_out}"
        )

        # Parse usage
        usage = self._parse_usage(result.stdout, container=container, workdir=workdir)
        if usage:
            cached_str = f" cached={usage.cached_input_tokens:,}" if usage.cached_input_tokens else ""
            reasoning_str = (
                f" reasoning={usage.reasoning_output_tokens:,}"
                if usage.reasoning_output_tokens
                else ""
            )
            logger.info(
                f"Usage: in={usage.input_tokens:,}{cached_str} visible={usage.visible_output_tokens:,}"
                f"{reasoning_str} total_out={usage.total_output_tokens:,}"
                + (f" cost=${usage.cost_usd:.4f}" if usage.cost_usd else "")
            )

        # Save logs
        if logging_dir:
            self._save_logs(logging_dir, result.stdout, result.stderr, usage, latency_s)

        # Failure mode
        failure = self._determine_failure_mode(result, usage)

        return AgentResult(
            total_input_tokens=usage.total_input_tokens if usage else 0,
            total_output_tokens=usage.total_output_tokens if usage else 0,
            failure_mode=failure,
        )
