"""Claude Code agent for Terminal-Bench with token tracking.

Installs Claude Code inside the Docker container and runs with
--output-format stream-json for JSONL output. Parses the final
result event for token usage and cost.
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
from pathlib import Path

try:
    from .base import DockerExecAgent, ParsedUsage, parse_jsonl_events
except ImportError:
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
if [ -n "${{CLAUDE_JSON_B64:-}}" ]; then
  printf '%s' "$CLAUDE_JSON_B64" | base64 -d > "$HOME/.claude.json"
fi
echo "claude installed: $(claude --version 2>/dev/null || echo unknown)"
"""

    def _load_claude_json_b64(self) -> str | None:
        # Optional explicit override for nonstandard auth locations.
        if os.environ.get("CLAUDE_JSON_B64"):
            return os.environ["CLAUDE_JSON_B64"]

        claude_json = Path.home() / ".claude.json"
        if not claude_json.exists():
            return None
        return base64.b64encode(claude_json.read_bytes()).decode("ascii")

    def _env_vars(self) -> dict[str, str]:
        env = {
            "FORCE_AUTO_BACKGROUND_TASKS": "1",
            "ENABLE_BACKGROUND_TASKS": "1",
        }
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key

        claude_json_b64 = self._load_claude_json_b64()
        if claude_json_b64:
            env["CLAUDE_JSON_B64"] = claude_json_b64

        if not api_key and not claude_json_b64:
            logger.warning(
                "No ANTHROPIC_API_KEY or ~/.claude.json found for Claude Code container auth. "
                "Runs may fail unless one is provided."
            )

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

        # CC's input_tokens is non-cached only; cached tokens are separate
        input_tokens = _int(usage.get("input_tokens", 0))
        cached = (_int(usage.get("cache_creation_input_tokens", 0))
                  + _int(usage.get("cache_read_input_tokens", 0)))
        output_tokens = _int(usage.get("output_tokens", 0))

        cost_usd = result_event.get("total_cost_usd")
        if cost_usd is None:
            cost_usd = result_event.get("cost_usd")
        if cost_usd is not None:
            cost_usd = float(cost_usd)

        return ParsedUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
