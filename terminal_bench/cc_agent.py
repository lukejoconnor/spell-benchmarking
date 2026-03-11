"""Claude Code agent for Terminal-Bench with token tracking.

Installs Claude Code inside the Docker container and runs with
--output-format stream-json for JSONL output. Parses the final
result event for token usage and cost.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path

try:
    from .base import DockerExecAgent, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ParsedUsage

try:
    from src.docker_agents import (
        cc_env_vars,
        cc_install_script,
        cc_parse_usage,
        cc_run_command,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
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
        setting_sources: str = "",
        **kwargs,
    ):
        super().__init__(model_name=model_name, agent_timeout=agent_timeout, **kwargs)
        self._max_budget_usd = max_budget_usd
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

    def _load_claude_json_b64(self) -> str | None:
        # Optional explicit override for nonstandard auth locations.
        if os.environ.get("CLAUDE_JSON_B64"):
            return os.environ["CLAUDE_JSON_B64"]

        claude_json = Path.home() / ".claude.json"
        if not claude_json.exists():
            return None
        return base64.b64encode(claude_json.read_bytes()).decode("ascii")

    def _env_vars(self) -> dict[str, str]:
        claude_json_b64 = self._load_claude_json_b64()
        env = cc_env_vars(model=self._resolve_model())
        if not env.get("ANTHROPIC_API_KEY") and not claude_json_b64:
            logger.warning(
                "No ANTHROPIC_API_KEY or ~/.claude.json found for Claude Code container auth. "
                "Runs may fail unless one is provided."
            )
        return env

    def _run_command(self, instruction: str, workdir: str) -> str:
        del workdir
        return cc_run_command(
            instruction,
            model=self._resolve_model(),
            budget=self._max_budget_usd,
            allowed_tools=self.ALLOWED_TOOLS,
            setting_sources=self._setting_sources,
        )

    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        """Parse usage from Claude Code stream-json output.

        The final result event contains:
        {"type": "result", "cost_usd": X, "total_cost_usd": Y,
         "usage": {"input_tokens": N, "output_tokens": M}}
        """
        usage = cc_parse_usage(stdout)
        if not usage:
            logger.warning("No result event found in Claude Code output")
        return usage
