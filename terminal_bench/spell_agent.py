"""Spell agent for Terminal-Bench with in-container execution.

Runs Spell's benchmark API inside the task container via DockerExecAgent,
so all benchmarked methods execute in the same container environment.
"""

from __future__ import annotations

import io
import logging
import sys
import tarfile
from pathlib import Path

try:
    from .base import DockerExecAgent, ExecOutput, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ExecOutput, ParsedUsage
from terminal_bench.agents.failure_mode import FailureMode

try:
    from src.docker_agents import (
        SPELL_AGENT_IN_CONTAINER,
        SPELL_BUNDLE_PATHS,
        build_spell_init,
        resolve_spell_model,
        discover_spell_root,
        spell_build_bundle,
        spell_env_vars,
        spell_install_script,
        spell_parse_benchmark_response,
        spell_parse_usage,
        spell_run_command,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.docker_agents import (
        SPELL_AGENT_IN_CONTAINER,
        SPELL_BUNDLE_PATHS,
        build_spell_init,
        resolve_spell_model,
        discover_spell_root,
        spell_build_bundle,
        spell_env_vars,
        spell_install_script,
        spell_parse_benchmark_response,
        spell_parse_usage,
        spell_run_command,
    )

logger = logging.getLogger(__name__)


class SpellAgent(DockerExecAgent):
    """Terminal-Bench agent that runs Spell entirely inside the task container."""

    def __init__(
        self,
        model_name: str | None = None,
        budget: float = 5.0,
        depth: int | None = None,
        sh_timeout: int = 120,
        agent_timeout: int = 600,
        agent_file: str | None = None,
        spell_root: str | None = None,
        reasoning_effort: str = "high",
        trailing: str | None = None,
        trace: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=resolve_spell_model(model_name), agent_timeout=agent_timeout, **kwargs)
        self._budget = budget
        self._depth = depth
        self._sh_timeout = sh_timeout
        self._agent_file = agent_file or SPELL_AGENT_IN_CONTAINER
        self._spell_root = discover_spell_root(spell_root, search_from=Path(__file__).resolve())
        self._spell_bundle: bytes | None = None
        self._reasoning_effort = reasoning_effort
        self._trailing = trailing
        self._trace = trace

    @staticmethod
    def name() -> str:
        return "spell"

    def _requires_node(self) -> bool:
        return False

    def _requires_uv(self) -> bool:
        return False

    def _install_script(self) -> str:
        return spell_install_script()

    def _env_vars(self) -> dict[str, str]:
        return spell_env_vars(self._model_name, sh_timeout=self._sh_timeout)

    def _build_spell_bundle(self) -> bytes:
        return spell_build_bundle(self._spell_root)

    def _install_agent_cli(self, container, logging_dir: Path | None = None) -> bool:
        """Copy Spell runtime into the container, then install Java/Clojure deps."""
        try:
            if self._spell_bundle is None:
                self._spell_bundle = self._build_spell_bundle()
        except Exception as exc:
            logger.error(f"Failed to build Spell bundle: {exc}")
            return False

        prep = self._container_exec(container, "mkdir -p /opt && rm -rf /opt/spell", timeout=60)
        if prep.exit_code != 0:
            logger.error(f"Failed to prepare /opt in container: {prep.stderr[:1000]}")
            return False

        try:
            copied = container.put_archive("/opt", self._spell_bundle)
        except Exception as exc:
            logger.error(f"Failed to copy Spell bundle into container: {exc}")
            return False

        if not copied:
            logger.error("Docker put_archive returned False while copying Spell bundle")
            return False

        logger.info("Installing Spell runtime in container...")
        result = self._container_exec(
            container,
            self._install_script(),
            timeout=600,
            env=self._env_vars(),
        )
        self._write_install_log(logging_dir, result, label="Spell runtime install")
        if result.exit_code != 0:
            logger.error(f"Spell runtime install failed:\n{result.stderr[:1000]}")
            return False
        logger.info("Spell runtime installed")
        return True

    def _run_command(self, instruction: str, workdir: str) -> str:
        del workdir
        init = build_spell_init(instruction, self._trailing)
        return spell_run_command(
            prompt=None,
            init=init,
            model_name=self._model_name,
            agent_file=self._agent_file,
            budget=self._budget,
            depth=self._depth,
            reasoning_effort=self._reasoning_effort,
            trace=self._trace,
        )

    def _extract_response_json(self, stdout: str) -> dict | None:
        return spell_parse_benchmark_response(stdout)

    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None) -> ParsedUsage | None:
        del container, workdir
        usage = spell_parse_usage(stdout)
        if not usage:
            logger.warning("No benchmark-api JSON response found in Spell stdout")
        return usage

    def _extract_traces(self, container, stdout: str, logging_dir: Path) -> None:
        """Copy Spell trace files out of the container before it's torn down."""
        parsed = self._extract_response_json(stdout)
        if not parsed:
            return
        trace_dir = parsed.get("trace_dir")
        if not trace_dir:
            return

        dest = logging_dir / "spell-trace"
        dest.mkdir(parents=True, exist_ok=True)

        # Try to tar-stream the trace directory out of the container.
        try:
            # trace_dir is relative to workdir (usually /app)
            for prefix in ["/app/", "/"]:
                full_path = f"{prefix}{trace_dir}" if not trace_dir.startswith("/") else trace_dir
                try:
                    bits, _ = container.get_archive(full_path)
                    tar_buf = io.BytesIO()
                    for chunk in bits:
                        tar_buf.write(chunk)
                    tar_buf.seek(0)
                    with tarfile.open(fileobj=tar_buf, mode="r") as tar:
                        tar.extractall(path=str(dest))
                    logger.info(f"Extracted Spell traces from {full_path} to {dest}")
                    return
                except Exception:
                    continue
            logger.warning(f"Could not find trace dir '{trace_dir}' in container")
        except Exception as exc:
            logger.warning(f"Failed to extract Spell traces: {exc}")

    def perform_task(self, instruction, session, logging_dir=None):
        """Override to extract traces before the container is torn down."""
        container = session.container
        result = super().perform_task(instruction, session, logging_dir)
        if self._trace and logging_dir:
            # Re-read stdout from saved logs to get trace_dir
            stdout_file = logging_dir / "stdout.txt"
            stdout = stdout_file.read_text() if stdout_file.exists() else ""
            self._extract_traces(container, stdout, logging_dir)
        return result

    def _determine_failure_mode(self, result: ExecOutput, usage: ParsedUsage | None) -> FailureMode:
        del usage
        if result.timed_out:
            return FailureMode.AGENT_TIMEOUT
        if result.exit_code != 0:
            return FailureMode.UNKNOWN_AGENT_ERROR

        parsed = self._extract_response_json(result.stdout)
        if not parsed:
            return FailureMode.UNKNOWN_AGENT_ERROR
        if parsed.get("ok") is True:
            return FailureMode.NONE

        if parsed.get("error_type") == "timeout":
            return FailureMode.AGENT_TIMEOUT
        return FailureMode.UNKNOWN_AGENT_ERROR
