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
        extract_codex_thread_id,
        merge_codex_usage,
        parse_codex_rollout,
        parse_jsonl_events,
        resolve_codex_model,
        usage_from_events,
        usage_from_rollout,
        _read_codex_rollout_from_exec,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        codex_env_vars,
        codex_install_script,
        codex_parse_usage,
        codex_run_command,
        extract_codex_thread_id,
        merge_codex_usage,
        parse_codex_rollout,
        parse_jsonl_events,
        resolve_codex_model,
        usage_from_events,
        usage_from_rollout,
        _read_codex_rollout_from_exec,
    )

logger = logging.getLogger(__name__)


class CodexAgent(DockerExecAgent):
    """Runs Codex CLI inside the Docker container via exec_run()."""

    def __init__(
        self,
        model_name: str | None = None,
        agent_timeout: int = 600,
        reasoning_effort: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, agent_timeout=agent_timeout, **kwargs)
        self._reasoning_effort = reasoning_effort

    @staticmethod
    def name() -> str:
        return "spell-codex"

    def _resolve_model(self) -> str:
        return resolve_codex_model(self._model_name)

    def _install_script(self) -> str:
        return codex_install_script(version=self.version)

    def _env_vars(self) -> dict[str, str]:
        return codex_env_vars()

    def _run_command(self, instruction: str, workdir: str) -> str:
        return codex_run_command(
            instruction,
            model=self._resolve_model(),
            workdir=workdir,
            reasoning_effort=self._reasoning_effort,
        )

    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None) -> ParsedUsage | None:
        """Parse usage from Codex JSONL output.

        Scans all events for the last dict containing token keys.
        Handles turn.completed events and any nested usage structures.
        """
        if container is None:
            usage = codex_parse_usage(stdout, model=self._resolve_model())
        else:
            events = parse_jsonl_events(stdout)
            thread_id = extract_codex_thread_id(events)
            rollout_path, rollout_text = _read_codex_rollout_from_exec(
                lambda cmd, **kwargs: self._container_exec(container, cmd, **kwargs),
                thread_id,
            )
            if rollout_path and not rollout_text:
                logger.warning("Codex rollout file found but could not be read: %s", rollout_path)
            rollout_info = parse_codex_rollout(rollout_text) if rollout_text else None
            if thread_id and rollout_path is None:
                logger.warning("No Codex rollout found for thread %s", thread_id)
            elif rollout_path and rollout_info is None:
                logger.warning("Codex rollout had no usable token_count payload: %s", rollout_path)
            stream_usage = usage_from_events(events, model=self._resolve_model())
            rollout_usage = usage_from_rollout(rollout_text, model=self._resolve_model()) if rollout_text else None
            usage = merge_codex_usage(stream_usage, rollout_usage, model=self._resolve_model())
        if not usage:
            logger.warning("No usage data found in Codex output")
        return usage
