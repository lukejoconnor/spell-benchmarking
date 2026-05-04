"""mini-swe-agent adapter for Terminal-Bench."""

from __future__ import annotations

import json
import logging
import shlex
import sys
from pathlib import Path

try:
    from .base import DockerExecAgent, ExecOutput, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ExecOutput, ParsedUsage
from terminal_bench.agents.failure_mode import FailureMode

try:
    from src.docker_agents import (
        _extract_swe_agent_response,
        _load_swe_agent_trajectory,
        swe_agent_env_vars,
        swe_agent_install_script,
        swe_agent_parse_usage,
        swe_agent_run_command,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        _extract_swe_agent_response,
        _load_swe_agent_trajectory,
        swe_agent_env_vars,
        swe_agent_install_script,
        swe_agent_parse_usage,
        swe_agent_run_command,
    )

logger = logging.getLogger(__name__)


class _TerminalBenchTrajectoryContainer:
    def __init__(self, agent: "MiniSweAgent", container, workdir: str):
        self._agent = agent
        self._container = container
        self.container_workdir = workdir

    def exec_run(self, cmd: str, *, timeout: int | None = None, workdir: str | None = None):
        return self._agent._container_exec(
            self._container,
            cmd,
            timeout=timeout,
            env=self._agent._env_vars(),
            workdir=workdir,
        )

    def read_text(self, path: str) -> str:
        result = self.exec_run(f"cat {shlex.quote(path)}", timeout=60, workdir="/")
        return result.stdout if result.exit_code == 0 else ""


class MiniSweAgent(DockerExecAgent):
    """Runs mini-swe-agent inside the task container via the shared helpers."""

    def __init__(
        self,
        model_name: str | None = None,
        budget: float = 5.0,
        agent_timeout: int = 600,
        reasoning_effort: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name or "anthropic/claude-opus-4-6",
            agent_timeout=agent_timeout,
            **kwargs,
        )
        self._budget = budget
        self._reasoning_effort = reasoning_effort

    @staticmethod
    def name() -> str:
        return "mini-swe-agent"

    def _requires_node(self) -> bool:
        return False

    def _install_script(self) -> str:
        return swe_agent_install_script()

    def _env_vars(self) -> dict[str, str]:
        return swe_agent_env_vars()

    def _run_command(self, instruction: str, workdir: str) -> str:
        return swe_agent_run_command(
            instruction,
            model=self._model_name or "anthropic/claude-opus-4-6",
            budget=self._budget,
            workdir=workdir,
            reasoning_effort=self._reasoning_effort,
        )

    def _trajectory_container(self, container, workdir: str | None):
        return _TerminalBenchTrajectoryContainer(self, container, workdir or "/")

    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None) -> ParsedUsage | None:
        if container is None:
            return None
        usage = swe_agent_parse_usage(stdout, self._trajectory_container(container, workdir))
        if not usage:
            logger.warning("No mini-swe-agent usage found")
        return usage

    def _extract_response(self, container, stdout: str, workdir: str) -> str | None:
        _path, trajectory = _load_swe_agent_trajectory(self._trajectory_container(container, workdir), stdout)
        if not trajectory:
            return None
        return _extract_swe_agent_response(trajectory) or None

    def _save_trajectory_artifact(self, container, stdout: str, logging_dir: Path, workdir: str) -> None:
        source_path, trajectory = _load_swe_agent_trajectory(self._trajectory_container(container, workdir), stdout)
        if not source_path or not trajectory:
            return
        logging_dir.mkdir(parents=True, exist_ok=True)
        (logging_dir / "trajectory-path.txt").write_text(str(source_path), encoding="utf-8")
        (logging_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")

    def perform_task(self, instruction, session, logging_dir=None):
        container = session.container
        workdir = self._detect_workdir(container)
        result = super().perform_task(instruction, session, logging_dir)
        if logging_dir:
            stdout_file = logging_dir / "stdout.txt"
            stdout = stdout_file.read_text(encoding="utf-8", errors="replace") if stdout_file.exists() else ""
            self._save_trajectory_artifact(container, stdout, logging_dir, workdir)
        return result

    def _determine_failure_mode(self, result: ExecOutput, usage: ParsedUsage | None):
        del usage
        if result.timed_out:
            return FailureMode.AGENT_TIMEOUT
        if result.exit_code != 0:
            return FailureMode.UNKNOWN_AGENT_ERROR
        return FailureMode.NONE
