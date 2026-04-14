from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.docker_agents import (
    AgentRunResult,
    CODEX_RUNTIME_CODEX_HOME,
    CODEX_RUNTIME_NPM_CACHE,
    CODEX_RUNTIME_NPM_PREFIX,
    ExecOutput,
    NVM_NODE_INSTALL,
    _read_codex_rollout_from_exec,
    _load_swe_agent_trajectory,
    _normalize_spell_trailing,
    build_spell_init,
    codex_env_vars,
    codex_install_script,
    codex_parse_usage,
    codex_run_command,
    run_agent_in_container,
    resolve_spell_model,
    spell_env_vars,
)


class DockerAgentsTests(unittest.TestCase):
    def test_resolve_spell_model_preserves_fireworks_prefix(self) -> None:
        self.assertEqual(resolve_spell_model("fireworks:glm-5"), "fireworks:glm-5")

    def test_build_spell_init_defaults_to_extend(self) -> None:
        init = build_spell_init("Solve it")

        self.assertIn('(quine prompt "Solve it")', init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_normalizes_shell_escaped_bang_in_trailing(self) -> None:
        init = build_spell_init("Solve it", trailing="'(\\!describe io)")

        self.assertIn("'(!describe io)", init)
        self.assertNotIn("\\!describe", init)

    def test_normalize_spell_trailing_preserves_backslash_bang_inside_strings(self) -> None:
        trailing = "'(do (think \"literal \\\\! stays\") (!describe io))"

        normalized = _normalize_spell_trailing(trailing)

        self.assertEqual(normalized, trailing)

    def test_codex_parse_usage_estimates_cost_and_counts_cached_tokens(self) -> None:
        usage = codex_parse_usage(
            '\n'.join(
                [
                    '{"type":"thread.started","thread_id":"t1"}',
                    '{"type":"turn.completed","usage":{"input_tokens":6804,"cached_input_tokens":5888,"output_tokens":6347}}',
                ]
            ),
            model="gpt-5-codex",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 916)
        self.assertEqual(usage.cached_input_tokens, 5888)
        self.assertEqual(usage.total_input_tokens, 6804)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.065351, places=6)
        self.assertEqual(usage.cost_source, "estimated")
        self.assertIsNone(usage.reasoning_output_tokens)

    def test_codex_env_vars_redirect_runtime_to_tmp(self) -> None:
        env = codex_env_vars()

        self.assertEqual(env["CODEX_HOME"], CODEX_RUNTIME_CODEX_HOME)
        self.assertEqual(env["NPM_CONFIG_PREFIX"], CODEX_RUNTIME_NPM_PREFIX)
        self.assertEqual(env["NPM_CONFIG_CACHE"], CODEX_RUNTIME_NPM_CACHE)
        self.assertEqual(env["npm_config_cache"], CODEX_RUNTIME_NPM_CACHE)
        self.assertNotIn("HOME", env)

    def test_codex_install_script_uses_tmp_runtime_paths(self) -> None:
        script = codex_install_script(version="latest")

        self.assertIn(f'export CODEX_HOME="${{CODEX_HOME:-{CODEX_RUNTIME_CODEX_HOME}}}"', script)
        self.assertIn(f'export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CODEX_RUNTIME_NPM_PREFIX}}}"', script)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", script)
        self.assertIn('unset NPM_CONFIG_PREFIX', script)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', script)
        self.assertIn("npm install -g @openai/codex@latest", script)
        self.assertNotIn("$HOME/.nvm/nvm.sh", script)
        self.assertIn('> "$CODEX_HOME/auth.json"', script)

    def test_codex_run_command_sets_tmp_runtime_before_exec_without_overriding_home(self) -> None:
        cmd = codex_run_command("Return 42.", model="gpt-5-codex")

        self.assertIn(f'export CODEX_HOME="${{CODEX_HOME:-{CODEX_RUNTIME_CODEX_HOME}}}"', cmd)
        self.assertNotIn('export HOME="${HOME:-', cmd)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", cmd)
        self.assertIn('unset NPM_CONFIG_PREFIX', cmd)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', cmd)
        self.assertIn("codex -c 'web_search=\"disabled\"' exec", cmd)

    def test_read_codex_rollout_prefers_tmp_runtime_sessions(self) -> None:
        calls: list[str] = []

        def fake_exec(cmd: str, **kwargs) -> ExecOutput:
            del kwargs
            calls.append(cmd)
            if "find \"$base\"" in cmd:
                return ExecOutput(
                    exit_code=0,
                    stdout=(
                        f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/"
                        "rollout-2026-04-06T10-08-30-thread-1.jsonl\n"
                    ),
                    stderr="",
                )
            if cmd.startswith("cat "):
                return ExecOutput(exit_code=0, stdout='{"type":"session_meta"}\n', stderr="")
            raise AssertionError(f"Unexpected command: {cmd}")

        rollout_path, rollout_text = _read_codex_rollout_from_exec(fake_exec, "thread-1")

        self.assertEqual(
            rollout_path,
            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/rollout-2026-04-06T10-08-30-thread-1.jsonl",
        )
        self.assertEqual(rollout_text, '{"type":"session_meta"}\n')
        self.assertIn(f"{CODEX_RUNTIME_CODEX_HOME}/sessions", calls[0])

    def test_nvm_install_exports_profile_for_installer_shell(self) -> None:
        script = NVM_NODE_INSTALL

        self.assertIn("export PROFILE=/dev/null", script)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", script)
        self.assertIn('unset NPM_CONFIG_PREFIX', script)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', script)
        self.assertNotIn("PROFILE=/dev/null curl", script)

    def test_spell_env_vars_forwards_fireworks_api_key(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test"}, clear=False):
            env = spell_env_vars("fireworks:glm-5")
        self.assertEqual(env["FIREWORKS_API_KEY"], "fw-test")

    def test_load_swe_agent_trajectory_prefers_newest_parseable_file(self) -> None:
        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del cmd, kwargs
                return ExecOutput(
                    exit_code=0,
                    stdout=(
                        "100.0\t/workspace/older.traj.json\n"
                        "200.0\t/workspace/newer.traj.json\n"
                    ),
                    stderr="",
                )

            def read_text(self, path: str) -> str:
                return {
                    "/workspace/older.traj.json": '{"answer":"old"}',
                    "/workspace/newer.traj.json": '{"answer":"new"}',
                }[path]

        path, traj = _load_swe_agent_trajectory(FakeContainer(), "")

        self.assertEqual(path, Path("/workspace/newer.traj.json"))
        self.assertEqual(traj, {"answer": "new"})

    @patch("src.docker_agents._install_agent_runtime", return_value=True)
    def test_codex_run_in_container_attaches_rollout_metadata(self, _install_runtime) -> None:
        rollout_text = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 1_000,
                                    "cached_input_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 350,
                                },
                                "last_token_usage": {
                                    "input_tokens": 1_000,
                                    "cached_input_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 350,
                                },
                                "model_context_window": 258_400,
                            },
                        },
                    }
                ),
            ]
        )

        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del kwargs
                if "codex " in cmd and " exec " in cmd:
                    return ExecOutput(
                        exit_code=0,
                        stdout="\n".join(
                            [
                                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                                json.dumps({"type": "turn.started"}),
                                json.dumps(
                                    {
                                        "type": "turn.completed",
                                        "usage": {
                                            "input_tokens": 1_000,
                                            "cached_input_tokens": 100,
                                            "output_tokens": 200,
                                        },
                                    }
                                ),
                            ]
                        ),
                        stderr="",
                    )
                if "find \"$base\"" in cmd:
                    return ExecOutput(
                        exit_code=0,
                        stdout=(
                            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/"
                            "rollout-2026-04-06T10-08-30-thread-1.jsonl\n"
                        ),
                        stderr="",
                    )
                if cmd.startswith("cat "):
                    return ExecOutput(exit_code=0, stdout=rollout_text, stderr="")
                raise AssertionError(f"Unexpected command: {cmd}")

        result = run_agent_in_container(
            FakeContainer(),
            condition="codex",
            prompt="Return 42.",
            timeout=30,
            model_spec="gpt-5.4",
        )

        self.assertIsInstance(result, AgentRunResult)
        self.assertIsNotNone(result.usage)
        assert result.usage is not None
        self.assertEqual(result.usage.reasoning_output_tokens, 350)
        self.assertEqual(result.metadata["thread_id"], "thread-1")
        self.assertEqual(result.metadata["rollout_status"], "ok")
        self.assertEqual(result.metadata["rollout_token_count"]["total_token_usage"]["reasoning_output_tokens"], 350)
        self.assertEqual(result.metadata["cost_source"], "estimated_with_reasoning")
        self.assertEqual(
            result.raw["rollout_path"],
            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/rollout-2026-04-06T10-08-30-thread-1.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
