"""Codex CLI agent for Terminal-Bench with token tracking.

Installs Codex CLI inside the Docker container and runs with --json
for JSONL output. Parses turn.completed events for token usage.
"""

from __future__ import annotations

import logging
import os
import shlex

from base import DockerExecAgent, ParsedUsage, iter_dicts, parse_jsonl_events

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
        model = self._model_name or "gpt-5-codex"
        # Strip provider prefix (e.g. "openai/gpt-5.3-codex" -> "gpt-5.3-codex")
        if "/" in model:
            model = model.split("/", 1)[1]
        return model

    def _install_script(self) -> str:
        version = self.version or "latest"
        return f"""
source "$HOME/.nvm/nvm.sh"
npm install -g @openai/codex@{version}
mkdir -p "$HOME/.codex"
echo '{{"OPENAI_API_KEY": "'$OPENAI_API_KEY'"}}' > "$HOME/.codex/auth.json"
echo "codex installed: $(codex --version 2>/dev/null || echo unknown)"
"""

    def _env_vars(self) -> dict[str, str]:
        return {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}

    def _run_command(self, instruction: str, workdir: str) -> str:
        model = self._resolve_model()
        escaped = shlex.quote(instruction)
        return (
            f'source "$HOME/.nvm/nvm.sh" && '
            f"codex exec --json --ephemeral --skip-git-repo-check "
            f"--sandbox danger-full-access "
            f"-m {shlex.quote(model)} "
            f"{escaped}"
        )

    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        """Parse usage from Codex JSONL output.

        Scans all events for the last dict containing token keys.
        Handles turn.completed events and any nested usage structures.
        """
        events = parse_jsonl_events(stdout)

        # Find the last dict with usage-like keys
        usage_candidate = None
        for event in events:
            for d in iter_dicts(event):
                if any(k in d for k in ("input_tokens", "output_tokens")):
                    usage_candidate = d

        if not usage_candidate:
            logger.warning("No usage data found in Codex output")
            return None

        def _int(v) -> int:
            return int(v) if isinstance(v, (int, float)) else 0

        # OpenAI: input_tokens is TOTAL (includes cached), cached_input_tokens is subset
        total_in = _int(usage_candidate.get("input_tokens", 0))
        cached = _int(usage_candidate.get("cached_input_tokens", 0))
        output_tokens = _int(usage_candidate.get("output_tokens", 0))

        cost = usage_candidate.get("total_cost_usd") or usage_candidate.get("cost_usd")
        cost_usd = float(cost) if isinstance(cost, (int, float)) else None

        return ParsedUsage(
            input_tokens=total_in - cached,
            cached_input_tokens=cached,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
