"""Harbor installed-agent adapter for Codex with auditable per-call cost."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from src.codex_usage import codex_cost_accounting_from_rollout, normalize_codex_model

try:  # pragma: no cover - exercised by Harbor at runtime.
    from harbor.agents.installed.base import with_prompt_template
    from harbor.agents.installed.codex import Codex as _HarborCodex
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.paths import EnvironmentPaths
    from harbor.utils.trajectory_utils import format_trajectory_json
except ModuleNotFoundError:  # pragma: no cover - keeps unit tests light.
    _HarborCodex = object  # type: ignore[assignment]
    BaseEnvironment = Any  # type: ignore[assignment]

    class AgentContext:  # type: ignore[no-redef]
        n_input_tokens: int | None = None
        n_cache_tokens: int | None = None
        n_output_tokens: int | None = None
        cost_usd: float | None = None
        metadata: dict[str, Any] | None = None

    class _FallbackEnvironmentPaths:
        agent_dir = Path("/logs/agent")

    EnvironmentPaths = _FallbackEnvironmentPaths()  # type: ignore[assignment]

    def format_trajectory_json(value: dict[str, Any]) -> str:  # type: ignore[no-redef]
        return json.dumps(value, indent=2, sort_keys=True)

    def with_prompt_template(fn: Any) -> Any:  # type: ignore[no-redef]
        return fn


class HarborCodexAgent(_HarborCodex):  # type: ignore[misc, valid-type]
    """Submission-grade Codex adapter for Harbor TB2.

    Harbor's built-in Codex adapter records only aggregate token totals and does
    not compute cost. This adapter keeps Harbor's normal trajectory conversion
    shape, explicitly disables Codex web search, and adds per-model-call cost
    accounting so GPT-5.4 long-context pricing is applied at the right boundary.
    """

    _OUTPUT_FILENAME = "codex.txt"

    @staticmethod
    def name() -> str:
        return "spell-codex"

    def _resolved_model(self) -> str:
        model_name = getattr(self, "model_name", None)
        if not model_name:
            raise ValueError("Model name is required")
        return normalize_codex_model(model_name)

    def _codex_web_disabled_flag(self) -> str:
        return "-c " + shlex.quote('web_search="disabled"')

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if _HarborCodex is object:
            raise RuntimeError("Harbor is required to run HarborCodexAgent")

        escaped_instruction = shlex.quote(instruction)
        model = self._resolved_model()

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""

        auth_json_path = self._resolve_auth_json_path()
        env: dict[str, str] = {
            "CODEX_HOME": EnvironmentPaths.agent_dir.as_posix(),
        }
        setup_command = ""

        if auth_json_path:
            self.logger.debug("Codex auth: using auth.json from %s", auth_json_path)
            auth_target = "/tmp/codex-secrets/auth.json"
            await self.exec_as_agent(environment, command="mkdir -p /tmp/codex-secrets", env=env)
            await environment.upload_file(auth_json_path, auth_target)
            if environment.default_user is not None:
                await self.exec_as_root(
                    environment,
                    command=f"chown {environment.default_user} {auth_target}",
                )
            setup_command += (
                "mkdir -p /tmp/codex-secrets\n"
                f'ln -sf {shlex.quote(auth_target)} "$CODEX_HOME/auth.json"\n'
            )
        else:
            self.logger.debug("Codex auth: using OPENAI_API_KEY")
            env["OPENAI_API_KEY"] = self._get_env("OPENAI_API_KEY") or ""

        if openai_base_url := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = openai_base_url

        config_toml_block = ""
        if openai_base_url:
            config_toml_block = (
                '\ncat >>"$CODEX_HOME/config.toml" <<TOML\n'
                'openai_base_url = "${OPENAI_BASE_URL}"\n'
                "TOML"
            )

        if not auth_json_path:
            setup_command += (
                "mkdir -p /tmp/codex-secrets\n"
                "cat >/tmp/codex-secrets/auth.json <<EOF\n"
                '{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF\n'
                'ln -sf /tmp/codex-secrets/auth.json "$CODEX_HOME/auth.json"\n'
            )
        setup_command += config_toml_block

        skills_command = self._build_register_skills_command()
        if skills_command:
            setup_command += f"\n{skills_command}"

        mcp_command = self._build_register_mcp_servers_command()
        if mcp_command:
            setup_command += f"\n{mcp_command}"

        if setup_command.strip():
            await self.exec_as_agent(environment, command=setup_command, env=env)

        try:
            await self.exec_as_agent(
                environment,
                command=(
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                    "codex "
                    f"{self._codex_web_disabled_flag()} "
                    "exec "
                    "--dangerously-bypass-approvals-and-sandbox "
                    "--skip-git-repo-check "
                    f"--model {shlex.quote(model)} "
                    "--json "
                    "--enable unified_exec "
                    f"{cli_flags_arg}"
                    "-- "
                    f"{escaped_instruction} "
                    f"2>&1 </dev/null | tee {EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME}"
                ),
                env=env,
            )
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command='rm -rf /tmp/codex-secrets "$CODEX_HOME/auth.json" "$CODEX_HOME/tmp"',
                    env={"CODEX_HOME": EnvironmentPaths.agent_dir.as_posix()},
                )
            except Exception:
                pass

    def _rollout_text_from_session_dir(self, session_dir: Path) -> str | None:
        session_files = sorted(session_dir.glob("*.jsonl"))
        if not session_files:
            return None
        return session_files[0].read_text(encoding="utf-8", errors="replace")

    def populate_context_post_run(self, context: AgentContext) -> None:
        if _HarborCodex is object:
            return

        session_dir = self._get_session_dir()
        if not session_dir:
            self.logger.debug("No Codex session directory found")
            return

        trajectory = None
        try:
            trajectory = self._convert_events_to_trajectory(session_dir)
        except Exception:
            self.logger.exception("Failed to convert Codex events to trajectory")

        if trajectory is not None:
            trajectory_path = self.logs_dir / "trajectory.json"
            try:
                with open(trajectory_path, "w") as handle:
                    handle.write(format_trajectory_json(trajectory.to_json_dict()))
                self.logger.debug("Wrote Codex trajectory to %s", trajectory_path)
            except OSError as exc:
                self.logger.debug("Failed to write trajectory file %s: %s", trajectory_path, exc)

            if trajectory.final_metrics:
                metrics = trajectory.final_metrics
                context.n_input_tokens = metrics.total_prompt_tokens or 0
                context.n_cache_tokens = metrics.total_cached_tokens or 0
                context.n_output_tokens = metrics.total_completion_tokens or 0

        rollout_text = self._rollout_text_from_session_dir(session_dir)
        accounting = (
            codex_cost_accounting_from_rollout(rollout_text, model=self._resolved_model())
            if rollout_text
            else None
        )
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "agent": self.name(),
                "model": self._resolved_model(),
                "web_search": "disabled",
            }
        )

        if accounting is not None:
            context.cost_usd = accounting.get("cost_usd")
            aggregate = accounting.get("aggregate")
            if isinstance(aggregate, dict):
                context.n_input_tokens = int(aggregate.get("input_tokens") or 0)
                context.n_cache_tokens = int(aggregate.get("cached_input_tokens") or 0)
                context.n_output_tokens = int(aggregate.get("output_tokens") or 0)
            metadata["codex_cost_accounting"] = accounting
        else:
            metadata.setdefault("codex_cost_accounting", {"warnings": ["missing codex rollout token counts"]})

        context.metadata = metadata
