"""Claude Code agent for Terminal-Bench with token tracking.

Installs Claude Code inside the Docker container and runs with
--output-format stream-json for JSONL output. Parses the final
result event for token usage and cost.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

try:
    from .base import DockerExecAgent, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ParsedUsage
from terminal_bench.agents.failure_mode import FailureMode

try:
    from src.docker_agents import (
        cc_missing_cli_error,
        cc_env_vars,
        cc_install_script,
        cc_parse_usage,
        cc_run_command,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        cc_missing_cli_error,
        cc_env_vars,
        cc_install_script,
        cc_parse_usage,
        cc_run_command,
    )

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(DockerExecAgent):
    """Runs Claude Code inside the Docker container via exec_run()."""

    ALLOWED_TOOLS = [
        "Bash", "Edit", "Write", "Read", "Glob", "Grep", "LS",
        "WebFetch", "NotebookEdit", "NotebookRead",
        "TodoRead", "TodoWrite", "Agent",
    ]

    def __init__(
        self,
        model_name: str | None = None,
        agent_timeout: int = 600,
        max_budget_usd: float = 5.0,
        reasoning_effort: str | None = None,
        setting_sources: str = "",
        **kwargs,
    ):
        super().__init__(model_name=model_name, agent_timeout=agent_timeout, **kwargs)
        self._max_budget_usd = max_budget_usd
        self._reasoning_effort = reasoning_effort
        self._setting_sources = setting_sources

    @staticmethod
    def name() -> str:
        return "spell-cc"

    def _resolve_model(self) -> str | None:
        if self._model_name:
            return self._model_name.removeprefix("anthropic/")
        return os.environ.get("ANTHROPIC_MODEL")

    def _install_script(self) -> str:
        return cc_install_script(version=self.version or "latest")

    def _requires_uv(self) -> bool:
        return True

    def _env_vars(self) -> dict[str, str]:
        return cc_env_vars(model=self._resolve_model())

    def _run_command(self, instruction: str, workdir: str) -> str:
        del workdir
        return cc_run_command(
            instruction,
            model=self._resolve_model(),
            budget=self._max_budget_usd,
            reasoning_effort=self._reasoning_effort,
            allowed_tools=self.ALLOWED_TOOLS,
            setting_sources=self._setting_sources,
        )

    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None) -> ParsedUsage | None:
        """Parse usage from Claude Code stream-json output.

        The final result event contains:
        {"type": "result", "cost_usd": X, "total_cost_usd": Y,
         "usage": {"input_tokens": N, "output_tokens": M}}
        """
        del container, workdir
        usage = cc_parse_usage(stdout, model=self._resolve_model())
        if not usage:
            logger.warning("No result event found in Claude Code output")
        return usage

    def _determine_failure_mode(self, result, usage):
        del usage
        if result.timed_out:
            return FailureMode.AGENT_TIMEOUT
        if result.exit_code != 0:
            if cc_missing_cli_error(result):
                return FailureMode.AGENT_INSTALLATION_FAILED
            return FailureMode.UNKNOWN_AGENT_ERROR
        return FailureMode.NONE
