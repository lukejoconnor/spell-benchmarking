"""Harbor installed-agent adapter for Spell on Terminal-Bench 2.0."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from src.docker_agents import (
    spell_build_harbor_bundle,
    spell_env_vars,
    spell_install_script,
    spell_parse_benchmark_response,
    spell_parse_usage,
    spell_run_command,
)

try:  # pragma: no cover - exercised when Harbor imports the adapter.
    from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.paths import EnvironmentPaths
except ModuleNotFoundError:  # pragma: no cover - lets unit tests import without Harbor.

    class NonZeroAgentExitCodeError(RuntimeError):
        pass

    class AgentContext:  # type: ignore[no-redef]
        n_input_tokens: int | None = None
        n_cache_tokens: int | None = None
        n_output_tokens: int | None = None
        cost_usd: float | None = None
        metadata: dict[str, Any] | None = None

    class _FallbackEnvironmentPaths:
        agent_dir = Path("/logs/agent")

    EnvironmentPaths = _FallbackEnvironmentPaths()  # type: ignore[assignment]

    class BaseInstalledAgent:  # type: ignore[no-redef]
        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            version: str | None = None,
            *args: Any,
            **kwargs: Any,
        ):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self._version = version

        def version(self) -> str | None:
            return self._version


class HarborSpellAgent(BaseInstalledAgent):
    """Installed Harbor agent that uploads a Harbor-safe Spell runtime bundle."""

    _OUTPUT_FILENAME = "spell-output.txt"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        budget: float | None = 5.0,
        reasoning_effort: str | None = "medium",
        agent_file: str = "/opt/spell/config/agents/io-tc.agent.edn",
        trailing: str | None = None,
        trace: bool = False,
        sh_timeout: int = 600,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self.budget = float(budget) if budget is not None else None
        self.reasoning_effort = reasoning_effort
        self.agent_file = agent_file
        self.trailing = trailing
        self.trace = (
            trace
            if isinstance(trace, bool)
            else str(trace).lower() in {"1", "true", "yes", "on"}
        )
        self.sh_timeout = int(sh_timeout)

    @staticmethod
    def name() -> str:
        return "spell"

    def version(self) -> str | None:
        return super().version()

    async def install(self, environment: Any) -> None:
        bundle_path = self.logs_dir / "setup" / "spell-harbor-bundle.tar"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_bytes(spell_build_harbor_bundle())

        await self.exec_as_root(
            environment,
            command="mkdir -p /opt /tmp/harbor-spell && rm -rf /opt/spell",
        )
        await environment.upload_file(bundle_path, "/tmp/harbor-spell/spell-bundle.tar")
        await self.exec_as_root(
            environment,
            command="tar -xf /tmp/harbor-spell/spell-bundle.tar -C /opt && rm -rf /tmp/harbor-spell",
        )
        await self.exec_as_agent(
            environment,
            command=spell_install_script(),
            env=spell_env_vars(self.model_name or "", sh_timeout=self.sh_timeout),
            timeout_sec=600,
        )

    async def run(self, instruction: str, environment: Any, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("Model name is required")

        rendered_instruction = (
            self.render_instruction(instruction)
            if hasattr(self, "render_instruction")
            else instruction
        )
        init = None
        if self.trailing is not None:
            from src.docker_agents import build_spell_init

            init = build_spell_init(rendered_instruction, self.trailing)

        output_path = Path(EnvironmentPaths.agent_dir) / self._OUTPUT_FILENAME
        trace_dir = str(Path(EnvironmentPaths.agent_dir) / "spell-trace") if self.trace else None
        command = spell_run_command(
            prompt=rendered_instruction,
            init=init,
            model_name=self.model_name,
            agent_file=self.agent_file,
            budget=self.budget,
            reasoning_effort=self.reasoning_effort,
            trace=self.trace,
            trace_dir=trace_dir,
        )
        env = spell_env_vars(self.model_name, sh_timeout=self.sh_timeout)
        result = await environment.exec(
            command=(
                "set -o pipefail; "
                f"mkdir -p {shlex.quote(str(Path(EnvironmentPaths.agent_dir)))} && "
                f"({command}\n) 2>&1 | tee {shlex.quote(str(output_path))}"
            ),
            env=env,
        )
        self._populate_context_from_text(context, result.stdout)
        if result.return_code != 0:
            raise NonZeroAgentExitCodeError(
                f"Spell exited {result.return_code}; see {self._OUTPUT_FILENAME}"
            )

    def populate_context_post_run(self, context: AgentContext) -> None:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if output_path.exists():
            self._populate_context_from_text(
                context,
                output_path.read_text(encoding="utf-8", errors="replace"),
            )

    def _populate_context_from_text(self, context: AgentContext, stdout: str) -> None:
        parsed = spell_parse_benchmark_response(stdout)
        usage = spell_parse_usage(stdout)
        if usage is not None:
            context.n_input_tokens = usage.total_input_tokens
            context.n_cache_tokens = usage.cached_input_tokens
            context.n_output_tokens = usage.total_output_tokens
            context.cost_usd = usage.cost_usd

        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "agent": "spell",
                "model": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                "agent_file": self.agent_file,
                "trace": self.trace,
            }
        )
        if parsed is not None:
            metadata["spell_ok"] = parsed.get("ok")
            if parsed.get("error"):
                metadata["spell_error"] = parsed.get("error")
            result = parsed.get("result")
            if isinstance(result, str):
                metadata["spell_result_preview"] = result[:1000]
            raw_usage = parsed.get("usage")
            if isinstance(raw_usage, dict):
                metadata["spell_usage"] = raw_usage
        context.metadata = metadata
