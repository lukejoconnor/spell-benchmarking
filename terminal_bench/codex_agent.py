"""Codex CLI agent for Terminal-Bench with token tracking.

Installs Codex CLI inside the Docker container and runs with --json
for JSONL output. Parses turn.completed events for token usage.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from .base import DockerExecAgent, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ParsedUsage

try:
    from src.docker_agents import (
        codex_env_vars,
        codex_install_script,
        codex_parse_usage,
        codex_run_command,
        resolve_codex_model,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        codex_env_vars,
        codex_install_script,
        codex_parse_usage,
        codex_run_command,
        resolve_codex_model,
    )

logger = logging.getLogger(__name__)


class CodexAgent(DockerExecAgent):
    """Runs Codex CLI inside the Docker container via exec_run()."""

    def __init__(
        self,
        model_name: str | None = None,
        agent_timeout: int = 600,
        **kwargs,
    ):
        super().__init__(model_name=model_name, agent_timeout=agent_timeout, **kwargs)

    @staticmethod
    def name() -> str:
        return "spell-codex"

    def _resolve_model(self) -> str:
        return resolve_codex_model(self._model_name)

    def _install_script(self) -> str:
        return codex_install_script(version=self.version or "latest")

    def _env_vars(self) -> dict[str, str]:
        return codex_env_vars()

    def _run_command(self, instruction: str, workdir: str) -> str:
        return codex_run_command(
            instruction,
            model=self._resolve_model(),
            workdir=workdir,
        )

    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        """Parse usage from Codex JSONL output.

        Scans all events for the last dict containing token keys.
        Handles turn.completed events and any nested usage structures.
        """
        usage = codex_parse_usage(stdout)
        if not usage:
            logger.warning("No usage data found in Codex output")
        return usage
