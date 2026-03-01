"""Spell agent for Terminal-Bench.

Uses the Spell benchmark API with SPELL_DOCKER_CONTAINER to execute
commands inside the task container via docker exec.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

logger = logging.getLogger(__name__)

# Spell project root (two levels up from benchmarking/terminal_bench/)
SPELL_ROOT = Path(__file__).resolve().parents[2]

# Map litellm model names to Spell benchmark API format (provider:alias)
LITELLM_TO_SPELL_MODEL = {
    "anthropic/claude-opus-4-6-20250826": "anthropic-tc:opus46",
    "anthropic/claude-opus-4-6": "anthropic-tc:opus46",
    "anthropic/claude-opus-4-5-20251101": "anthropic-tc:opus",
    "anthropic/claude-opus-4-5": "anthropic-tc:opus",
    "anthropic/claude-sonnet-4-5-20250929": "anthropic-tc:sonnet",
    "anthropic/claude-sonnet-4-5": "anthropic-tc:sonnet",
    "anthropic/claude-haiku-4-5-20251001": "anthropic-tc:haiku",
    "anthropic/claude-haiku-4-5": "anthropic-tc:haiku",
    "anthropic/claude-sonnet-4-20250514": "anthropic-tc:sonnet",
    "anthropic/claude-opus-4-20250514": "anthropic-tc:opus",
}


def _resolve_spell_model(model_name: str | None) -> str:
    """Convert litellm model name to Spell benchmark API format."""
    if model_name is None:
        return "anthropic-tc:opus46"
    # Already in Spell format (has provider prefix like anthropic-tc:)
    if any(model_name.startswith(p + ":") for p in
           ["anthropic-tc", "anthropic-pf", "openai", "codex-msg", "codex-tc"]):
        return model_name
    # Look up in mapping
    if model_name in LITELLM_TO_SPELL_MODEL:
        return LITELLM_TO_SPELL_MODEL[model_name]
    # Strip provider prefix and try bare model name
    if "/" in model_name:
        bare = model_name.split("/", 1)[1]
        return f"anthropic-tc:{bare}"
    return f"anthropic-tc:{model_name}"


class SpellAgent(BaseAgent):
    """Terminal-Bench agent that runs Spell with io/sh bridged to docker exec."""

    def __init__(
        self,
        model_name: str | None = None,
        budget: float = 5.0,
        depth: int | None = None,
        sh_timeout: int = 120,
        agent_timeout: int = 600,
        agent_file: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = _resolve_spell_model(model_name)
        self._budget = budget
        self._depth = depth
        self._sh_timeout = sh_timeout
        self._agent_timeout = agent_timeout
        self._agent_file = agent_file

    @staticmethod
    def name() -> str:
        return "spell"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        container_id = session.container.id
        logger.info(f"Spell agent starting, container={container_id[:12]}")

        rendered_instruction = self._render_instruction(instruction)

        # Add preamble: io/sh runs in the container via docker exec,
        # but io/spit, io/read-file etc. operate on the host filesystem.
        # The LLM must use io/sh for all file and system operations.
        preamble = (
            "IMPORTANT: You are working inside a remote Docker container. "
            "All file and system operations MUST use (io/sh ...) shell commands. "
            "Do NOT use io/spit, io/slurp, io/read-file, io/write-file, io/ls, "
            "io/exists?, io/mkdir, or any other io/ file functions — they operate "
            "on the host filesystem, not inside the container. "
            "Use shell equivalents via io/sh instead: "
            "echo/cat/ls/mkdir/test/chmod/etc.\n\n"
            "TASK:\n"
        )
        rendered_instruction = preamble + rendered_instruction

        # Build the benchmark API payload
        # Use io-tc agent (has io namespace for shell commands)
        agent_file = self._agent_file or "config/agents/io-tc.agent.edn"
        payload: dict = {
            "mode": "spell",
            "model": self._model_name,
            "prompt": rendered_instruction,
            "budget": self._budget,
            "agent": agent_file,
            "trace": True,
        }
        if self._depth is not None:
            payload["depth"] = self._depth

        # Detect the container's working directory
        workdir_result = session.container.exec_run(["pwd"])
        workdir = workdir_result.output.decode().strip() if workdir_result.exit_code == 0 else "/root"
        logger.info(f"Container working directory: {workdir}")

        # Set up environment: pass SPELL_DOCKER_CONTAINER so io/sh uses docker exec
        env = os.environ.copy()
        env["SPELL_DOCKER_CONTAINER"] = container_id
        env["SPELL_DOCKER_WORKDIR"] = workdir
        # Increase default shell timeout for terminal tasks
        env["SPELL_SH_TIMEOUT"] = str(self._sh_timeout)

        # Run the Spell benchmark API as a subprocess
        clj_cmd = ["clj", "-M", "-m", "spell.benchmark-api", "--request", "-", "--response", "-"]

        start = time.time()
        try:
            proc = subprocess.run(
                clj_cmd,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self._agent_timeout,
                cwd=str(SPELL_ROOT),
                env=env,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Spell agent timed out after {self._agent_timeout}s")
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.AGENT_TIMEOUT,
            )

        latency_s = time.time() - start
        logger.info(f"Spell agent completed in {latency_s:.1f}s, exit={proc.returncode}")

        # Save logs if logging_dir is provided
        if logging_dir:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "stdout.txt").write_text(proc.stdout or "")
            (logging_dir / "stderr.txt").write_text(proc.stderr or "")
            (logging_dir / "payload.json").write_text(json.dumps(payload, indent=2))

        # Parse response
        stdout = (proc.stdout or "").strip()
        if not stdout:
            logger.error(f"No stdout from Spell. stderr: {(proc.stderr or '')[:500]}")
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
            )

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from Spell: {stdout[:500]}")
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
            )

        # Extract usage
        usage = (result.get("usage") or {}).get("total", {})
        tokens_in = usage.get("input_tokens", 0) or 0
        tokens_out = usage.get("output_tokens", 0) or 0

        if logging_dir:
            (logging_dir / "result.json").write_text(json.dumps(result, indent=2))

        if not result.get("ok"):
            logger.warning(f"Spell returned error: {result.get('error', 'unknown')}")

        return AgentResult(
            total_input_tokens=tokens_in,
            total_output_tokens=tokens_out,
            failure_mode=FailureMode.NONE,
        )
