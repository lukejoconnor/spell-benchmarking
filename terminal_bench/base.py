"""Base class for Terminal-Bench agents that install into Docker containers.

Uses container.exec_run(demux=True) for clean stdout/stderr separation,
enabling reliable JSONL parsing for token/cost tracking.
"""

from __future__ import annotations

import json
import logging
import shlex
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

logger = logging.getLogger(__name__)


@dataclass
class ExecOutput:
    """Result from a container exec_run call."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass
class ParsedUsage:
    """Token usage extracted from agent output."""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


# Idempotent NVM + Node 22 install script.
NVM_NODE_INSTALL = r"""
set -e
if [ -d "$HOME/.nvm" ]; then
    source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
    if command -v node >/dev/null 2>&1; then
        echo "node already installed: $(node --version)"
        exit 0
    fi
fi
apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install 22
echo "node installed: $(node --version)"
"""


def parse_jsonl_events(stdout: str) -> list[dict]:
    """Parse JSONL from stdout, skipping non-JSON lines."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                events.append(obj)
        except json.JSONDecodeError:
            continue
    return events


def iter_dicts(v):
    """Recursively yield all dicts nested in a JSON value."""
    if isinstance(v, dict):
        yield v
        for child in v.values():
            yield from iter_dicts(child)
    elif isinstance(v, list):
        for child in v:
            yield from iter_dicts(child)


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
        """Run a shell command in the container via exec_run(demux=True).

        Uses shell `timeout` for time-bounding since docker-py exec_run
        has no native timeout parameter.
        """
        if timeout:
            shell_cmd = f"timeout {timeout}s bash -c {shlex.quote(cmd)}"
        else:
            shell_cmd = cmd

        result = container.exec_run(
            ["bash", "-c", shell_cmd],
            environment=env or {},
            workdir=workdir,
            demux=True,
        )

        stdout_bytes, stderr_bytes = result.output or (b"", b"")
        stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        timed_out = result.exit_code == 124 and timeout is not None

        return ExecOutput(
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def _detect_workdir(self, container) -> str:
        result = container.exec_run(["pwd"])
        if result.exit_code == 0:
            return result.output.decode().strip()
        return "/root"

    def _install_nvm_node(self, container) -> bool:
        """Install NVM and Node 22 (idempotent). Returns True on success."""
        logger.info("Installing NVM + Node...")
        result = self._container_exec(container, NVM_NODE_INSTALL, timeout=300)
        if result.exit_code != 0:
            logger.error(f"NVM/Node install failed:\n{result.stderr[:1000]}")
            return False
        logger.info(result.stdout.strip().splitlines()[-1])
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
            "output_tokens": usage.output_tokens if usage else 0,
            "cost_usd": usage.cost_usd if usage else None,
        }
        (logging_dir / "result.json").write_text(json.dumps(result_data, indent=2))

    @staticmethod
    @abstractmethod
    def name() -> str: ...

    @abstractmethod
    def _install_script(self) -> str:
        """Shell script to install the agent CLI (NVM/Node already available)."""
        ...

    @abstractmethod
    def _run_command(self, instruction: str, workdir: str) -> str:
        """Shell command to run the agent with the given instruction."""
        ...

    @abstractmethod
    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        """Parse token usage from the agent's structured stdout."""
        ...

    @abstractmethod
    def _env_vars(self) -> dict[str, str]:
        """Env vars needed inside the container (API keys, etc.)."""
        ...

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

        # Install NVM + Node
        if not self._install_nvm_node(container):
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
        usage = self._parse_usage(result.stdout)
        if usage:
            logger.info(
                f"Usage: in={usage.input_tokens:,} out={usage.output_tokens:,}"
                + (f" cost=${usage.cost_usd:.4f}" if usage.cost_usd else "")
            )

        # Save logs
        if logging_dir:
            self._save_logs(logging_dir, result.stdout, result.stderr, usage, latency_s)

        # Failure mode
        if result.timed_out:
            failure = FailureMode.AGENT_TIMEOUT
        elif result.exit_code != 0:
            failure = FailureMode.UNKNOWN_AGENT_ERROR
        else:
            failure = FailureMode.NONE

        return AgentResult(
            total_input_tokens=usage.input_tokens if usage else 0,
            total_output_tokens=usage.output_tokens if usage else 0,
            failure_mode=failure,
        )
