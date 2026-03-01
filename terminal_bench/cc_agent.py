"""Claude Code agent for Terminal-Bench with token tracking.

Installs Claude Code inside the Docker container and runs with
--output-format stream-json for JSONL output. Parses the final
result event for token usage and cost.
"""

from __future__ import annotations

import logging
import os
import shlex

from base import DockerExecAgent, ParsedUsage, parse_jsonl_events

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
        version = self.version or "latest"
        return f"""
source "$HOME/.nvm/nvm.sh"
npm install -g @anthropic-ai/claude-code@{version}
echo "claude installed: $(claude --version 2>/dev/null || echo unknown)"
"""

    def _env_vars(self) -> dict[str, str]:
        env = {
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
        }
        model = self._resolve_model()
        if model:
            env["ANTHROPIC_MODEL"] = model
        return env

    def _run_command(self, instruction: str, workdir: str) -> str:
        escaped = shlex.quote(instruction)
        parts = ['source "$HOME/.nvm/nvm.sh" &&', "claude"]

        parts.append(f"--setting-sources {shlex.quote(self._setting_sources)}")
        parts.append("--verbose")
        parts.append("--output-format stream-json")
        parts.append(f"--max-budget-usd {self._max_budget_usd}")

        model = self._resolve_model()
        if model:
            parts.append(f"--model {shlex.quote(model)}")

        parts.append(f"--allowedTools {' '.join(self.ALLOWED_TOOLS)}")
        parts.append(f"-p {escaped}")

        return " ".join(parts)

    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        """Parse usage from Claude Code stream-json output.

        The final result event contains:
        {"type": "result", "cost_usd": X, "total_cost_usd": Y,
         "usage": {"input_tokens": N, "output_tokens": M}}
        """
        events = parse_jsonl_events(stdout)

        # Find the last "result" event
        result_event = None
        for event in reversed(events):
            if event.get("type") == "result":
                result_event = event
                break

        if not result_event:
            logger.warning("No result event found in Claude Code output")
            return None

        usage = result_event.get("usage", {})

        def _int(v) -> int:
            return int(v) if isinstance(v, (int, float)) else 0

        input_tokens = _int(usage.get("input_tokens", 0))
        input_tokens += _int(usage.get("cache_creation_input_tokens", 0))
        input_tokens += _int(usage.get("cache_read_input_tokens", 0))
        output_tokens = _int(usage.get("output_tokens", 0))

        cost_usd = result_event.get("total_cost_usd")
        if cost_usd is None:
            cost_usd = result_event.get("cost_usd")
        if cost_usd is not None:
            cost_usd = float(cost_usd)

        return ParsedUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
