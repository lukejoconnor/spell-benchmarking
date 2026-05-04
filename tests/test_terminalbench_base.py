from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = BENCHMARK_ROOT / "terminal_bench" / "base.py"
CC_AGENT_MODULE_PATH = BENCHMARK_ROOT / "terminal_bench" / "cc_agent.py"
SPELL_AGENT_MODULE_PATH = BENCHMARK_ROOT / "terminal_bench" / "spell_agent.py"
SWE_AGENT_MODULE_PATH = BENCHMARK_ROOT / "terminal_bench" / "swe_agent.py"


@dataclass
class _AgentResult:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    failure_mode: object | None = None


class _BaseAgent:
    def __init__(self, **kwargs):
        self.version = kwargs.get("version")

    def _render_instruction(self, instruction: str) -> str:
        return instruction


class _FailureMode:
    AGENT_INSTALLATION_FAILED = "agent_installation_failed"
    AGENT_TIMEOUT = "agent_timeout"
    UNKNOWN_AGENT_ERROR = "unknown_agent_error"
    NONE = "none"


class _TmuxSession:
    pass


class _PwdResult:
    exit_code = 0
    output = b"/workspace\n"


class _FakeContainer:
    id = "fakecontainer123456"

    def exec_run(self, cmd):
        self.last_exec = cmd
        return _PwdResult()


def _load_base_module():
    module_names = [
        "terminal_bench",
        "terminal_bench.agents",
        "terminal_bench.agents.base_agent",
        "terminal_bench.agents.failure_mode",
        "terminal_bench.terminal",
        "terminal_bench.terminal.tmux_session",
    ]
    prior = {name: sys.modules.get(name) for name in module_names}

    terminal_bench_pkg = types.ModuleType("terminal_bench")
    terminal_bench_pkg.__path__ = []  # type: ignore[attr-defined]
    agents_pkg = types.ModuleType("terminal_bench.agents")
    agents_pkg.__path__ = []  # type: ignore[attr-defined]
    terminal_pkg = types.ModuleType("terminal_bench.terminal")
    terminal_pkg.__path__ = []  # type: ignore[attr-defined]
    base_agent_mod = types.ModuleType("terminal_bench.agents.base_agent")
    base_agent_mod.AgentResult = _AgentResult
    base_agent_mod.BaseAgent = _BaseAgent
    failure_mode_mod = types.ModuleType("terminal_bench.agents.failure_mode")
    failure_mode_mod.FailureMode = _FailureMode
    tmux_session_mod = types.ModuleType("terminal_bench.terminal.tmux_session")
    tmux_session_mod.TmuxSession = _TmuxSession

    sys.modules["terminal_bench"] = terminal_bench_pkg
    sys.modules["terminal_bench.agents"] = agents_pkg
    sys.modules["terminal_bench.agents.base_agent"] = base_agent_mod
    sys.modules["terminal_bench.agents.failure_mode"] = failure_mode_mod
    sys.modules["terminal_bench.terminal"] = terminal_pkg
    sys.modules["terminal_bench.terminal.tmux_session"] = tmux_session_mod

    spec = importlib.util.spec_from_file_location("terminal_bench_base_under_test", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name, value in prior.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        raise
    return module, prior


def _load_cc_agent_module():
    base_module, prior = _load_base_module()
    prior["terminal_bench.base"] = sys.modules.get("terminal_bench.base")
    prior["base"] = sys.modules.get("base")
    sys.modules["terminal_bench.base"] = base_module
    sys.modules["base"] = base_module

    spec = importlib.util.spec_from_file_location("terminal_bench_cc_agent_under_test", CC_AGENT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CC_AGENT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _restore_modules(prior)
        raise
    return module, prior


def _load_spell_agent_module():
    base_module, prior = _load_base_module()
    prior["terminal_bench.base"] = sys.modules.get("terminal_bench.base")
    prior["base"] = sys.modules.get("base")
    sys.modules["terminal_bench.base"] = base_module
    sys.modules["base"] = base_module

    spec = importlib.util.spec_from_file_location("terminal_bench_spell_agent_under_test", SPELL_AGENT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SPELL_AGENT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _restore_modules(prior)
        raise
    return module, prior


def _load_swe_agent_module():
    base_module, prior = _load_base_module()
    prior["terminal_bench.base"] = sys.modules.get("terminal_bench.base")
    prior["base"] = sys.modules.get("base")
    sys.modules["terminal_bench.base"] = base_module
    sys.modules["base"] = base_module

    spec = importlib.util.spec_from_file_location("terminal_bench_swe_agent_under_test", SWE_AGENT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SWE_AGENT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        _restore_modules(prior)
        raise
    return module, prior


def _restore_modules(prior: dict[str, object | None]) -> None:
    for name, value in prior.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


class TerminalBenchBaseTests(unittest.TestCase):
    def test_save_logs_records_explicit_visible_reasoning_and_total_output_tokens(self) -> None:
        module, prior = _load_base_module()
        try:
            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _requires_uv(self) -> bool:
                    return False

                def _install_script(self) -> str:
                    return "true"

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "true"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return None

                def _env_vars(self) -> dict[str, str]:
                    return {}

            agent = DummyAgent()
            with tempfile.TemporaryDirectory() as tmp_dir:
                logging_dir = Path(tmp_dir)
                usage = module.ParsedUsage(
                    uncached_input_tokens=39382,
                    cache_write_input_tokens=512,
                    visible_output_tokens=1123,
                    reasoning_output_tokens=17869,
                    cost_usd=0.324007,
                    cost_source="estimated_with_reasoning",
                )
                agent._save_logs(logging_dir, "stdout", "stderr", usage, 12.34)
                result = json.loads((logging_dir / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(result["visible_output_tokens"], 1123)
            self.assertEqual(result["cache_write_input_tokens"], 512)
            self.assertEqual(result["reasoning_output_tokens"], 17869)
            self.assertEqual(result["total_output_tokens"], 18992)
            self.assertNotIn("output_tokens", result)
        finally:
            _restore_modules(prior)

    def test_perform_task_returns_total_output_tokens(self) -> None:
        module, prior = _load_base_module()
        try:
            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _requires_node(self) -> bool:
                    return False

                def _requires_uv(self) -> bool:
                    return False

                def _install_script(self) -> str:
                    return "true"

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "true"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return module.ParsedUsage(
                        uncached_input_tokens=90,
                        cached_input_tokens=10,
                        visible_output_tokens=7,
                        reasoning_output_tokens=11,
                    )

                def _env_vars(self) -> dict[str, str]:
                    return {}

            agent = DummyAgent()
            session = types.SimpleNamespace(container=types.SimpleNamespace(id="container123"))
            with patch.object(agent, "_detect_workdir", return_value="/workspace"), patch.object(
                agent, "_install_agent_cli", return_value=True
            ), patch.object(
                agent,
                "_container_exec",
                return_value=module.ExecOutput(exit_code=0, stdout="{}", stderr=""),
            ):
                result = agent.perform_task("solve it", session)

            self.assertEqual(result.total_input_tokens, 100)
            self.assertEqual(result.total_output_tokens, 18)
        finally:
            _restore_modules(prior)

    def test_perform_task_reports_true_total_output_tokens(self) -> None:
        module, prior = _load_base_module()
        try:
            usage = module.ParsedUsage(
                uncached_input_tokens=39382,
                visible_output_tokens=1123,
                reasoning_output_tokens=17869,
            )

            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _requires_node(self) -> bool:
                    return False

                def _requires_uv(self) -> bool:
                    return False

                def _install_script(self) -> str:
                    return "true"

                def _install_agent_cli(self, container, logging_dir=None) -> bool:
                    del container, logging_dir
                    return True

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "true"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return usage

                def _env_vars(self) -> dict[str, str]:
                    return {}

                def _container_exec(self, container, cmd: str, **kwargs):
                    del container, cmd, kwargs
                    return module.ExecOutput(exit_code=0, stdout="stdout", stderr="")

            agent = DummyAgent()
            session = _TmuxSession()
            session.container = _FakeContainer()

            result = agent.perform_task("instruction", session)

            self.assertEqual(result.total_input_tokens, 39382)
            self.assertEqual(result.total_output_tokens, 18992)
        finally:
            _restore_modules(prior)

    def test_install_agent_cli_writes_install_log(self) -> None:
        module, prior = _load_base_module()
        try:
            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _install_script(self) -> str:
                    return "echo install"

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "true"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return None

                def _env_vars(self) -> dict[str, str]:
                    return {}

            agent = DummyAgent()
            with tempfile.TemporaryDirectory() as tmp_dir:
                with patch.object(
                    agent,
                    "_container_exec",
                    return_value=module.ExecOutput(exit_code=1, stdout="full stdout", stderr="full stderr"),
                ):
                    ok = agent._install_agent_cli(_FakeContainer(), logging_dir=Path(tmp_dir))
                log_text = (Path(tmp_dir) / "install.log").read_text(encoding="utf-8")

            self.assertFalse(ok)
            self.assertIn("== dummy install ==", log_text)
            self.assertIn("exit_code: 1", log_text)
            self.assertIn("full stdout", log_text)
            self.assertIn("full stderr", log_text)
        finally:
            _restore_modules(prior)

    def test_claude_agent_marks_missing_cli_as_install_failure(self) -> None:
        module, prior = _load_cc_agent_module()
        try:
            agent = module.ClaudeCodeAgent()
            result = agent._determine_failure_mode(
                types.SimpleNamespace(
                    timed_out=False,
                    exit_code=1,
                    stdout="",
                    stderr="bootstrap failed\nClaude Code CLI not available after installation\n",
                ),
                usage=None,
            )

            self.assertEqual(result, _FailureMode.AGENT_INSTALLATION_FAILED)
        finally:
            _restore_modules(prior)

    def test_claude_agent_parse_usage_writes_sanitized_ledger_sidecar(self) -> None:
        module, prior = _load_cc_agent_module()
        try:
            agent = module.ClaudeCodeAgent(model_name="anthropic/claude-sonnet-4-6")
            stdout = json.dumps({"type": "system", "session_id": "sess-1"})
            debug_event = {
                "type": "assistant",
                "message": {
                    "id": "msg-1",
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 100_000,
                        "cache_creation_input_tokens": 50_000,
                        "cache_read_input_tokens": 25_000,
                        "output_tokens": 10_000,
                    },
                },
            }
            calls: list[str] = []

            def fake_exec(container, cmd: str, **kwargs):
                del container, kwargs
                calls.append(cmd)
                if "debug.log" in cmd:
                    return types.SimpleNamespace(
                        exit_code=0,
                        stdout=json.dumps(debug_event),
                        stderr="",
                    )
                if "find \"$base\"" in cmd:
                    return types.SimpleNamespace(exit_code=1, stdout="", stderr="")
                if "claude-usage-ledger.json" in cmd:
                    return types.SimpleNamespace(exit_code=0, stdout="", stderr="")
                raise AssertionError(f"Unexpected command: {cmd}")

            with patch.object(agent, "_container_exec", side_effect=fake_exec):
                usage = agent._parse_usage(stdout, container=object())

            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertEqual(usage.total_input_tokens, 175_000)
            self.assertEqual(usage.visible_output_tokens, 10_000)
            self.assertEqual(usage.cost_source, "estimated_debug")
            sidecar_cmd = next(cmd for cmd in calls if "claude-usage-ledger.json" in cmd)
            self.assertIn('"total_input_tokens": 175000', sidecar_cmd)
            self.assertIn('"visible_output_tokens": 10000', sidecar_cmd)
            self.assertIn('"cost_source": "estimated_debug"', sidecar_cmd)
        finally:
            _restore_modules(prior)

    @patch.dict(
        os.environ,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
            "ANTHROPIC_API_KEY": "sk-should-not-forward",
            "CLAUDE_JSON_B64": "legacy-json",
        },
        clear=False,
    )
    def test_claude_agent_env_uses_oauth_only(self) -> None:
        module, prior = _load_cc_agent_module()
        try:
            agent = module.ClaudeCodeAgent(model_name="anthropic/claude-opus-4-7")
            env = agent._env_vars()

            self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "oauth-token")
            self.assertEqual(env["ANTHROPIC_MODEL"], "claude-opus-4-7")
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("CLAUDE_JSON_B64", env)
        finally:
            _restore_modules(prior)

    @patch.dict(os.environ, {}, clear=True)
    def test_claude_agent_env_requires_oauth_token(self) -> None:
        module, prior = _load_cc_agent_module()
        try:
            agent = module.ClaudeCodeAgent()
            with self.assertRaisesRegex(RuntimeError, "CLAUDE_CODE_OAUTH_TOKEN is required"):
                agent._env_vars()
        finally:
            _restore_modules(prior)

    def test_spell_agent_skips_uv_bootstrap(self) -> None:
        module, prior = _load_spell_agent_module()
        try:
            agent = module.SpellAgent(model_name="anthropic-tc:opus46")

            self.assertFalse(agent._requires_uv())
        finally:
            _restore_modules(prior)

    def test_swe_agent_skips_node_and_uses_shared_helpers(self) -> None:
        module, prior = _load_swe_agent_module()
        try:
            agent = module.MiniSweAgent(
                model_name="fireworks_ai/accounts/fireworks/models/glm-5p1",
                budget=1.5,
                agent_timeout=120,
            )

            self.assertFalse(agent._requires_node())
            self.assertFalse(agent._requires_uv())
            self.assertIn("/opt/mini", agent._install_script())
            cmd = agent._run_command("Fix it", "/workspace")
            self.assertIn("pty-run mini", cmd)
            self.assertIn("-m fireworks_ai/accounts/fireworks/models/glm-5p1", cmd)
            self.assertIn("-l 1.5", cmd)
            self.assertNotIn(".mini-swe-agent", cmd)
            self.assertNotIn("/root/.config", cmd)
        finally:
            _restore_modules(prior)

    def test_swe_agent_preinstalls_opt_mini_and_writes_install_log(self) -> None:
        module, prior = _load_swe_agent_module()
        try:
            class FakeContainer(_FakeContainer):
                def __init__(self) -> None:
                    self.archives: list[tuple[str, bytes]] = []

                def put_archive(self, dest_dir: str, archive: bytes) -> bool:
                    self.archives.append((dest_dir, archive))
                    return True

            agent = module.MiniSweAgent(model_name="fireworks_ai/accounts/fireworks/models/glm-5p1")
            container = FakeContainer()
            exec_results = [
                module.ExecOutput(exit_code=0, stdout="", stderr=""),
                module.ExecOutput(exit_code=0, stdout="mini runtime verified", stderr=""),
            ]

            def fake_exec(_container, cmd: str, **kwargs):
                del _container, cmd, kwargs
                return exec_results.pop(0)

            with tempfile.TemporaryDirectory() as tmp_dir:
                with patch.object(agent, "_build_mini_bundle", return_value=b"bundle"), patch.object(
                    agent, "_container_exec", side_effect=fake_exec
                ):
                    ok = agent._install_agent_cli(container, logging_dir=Path(tmp_dir))
                log_text = (Path(tmp_dir) / "install.log").read_text(encoding="utf-8")

            self.assertTrue(ok)
            self.assertEqual(container.archives, [("/opt", b"bundle")])
            self.assertIn("mini-swe-agent runtime verify", log_text)
            self.assertIn("mini runtime verified", log_text)
        finally:
            _restore_modules(prior)

    def test_swe_agent_parse_usage_wraps_terminalbench_container(self) -> None:
        module, prior = _load_swe_agent_module()
        try:
            agent = module.MiniSweAgent(model_name="fireworks_ai/accounts/fireworks/models/glm-5p1")
            seen = {}

            def fake_parse(stdout, container):
                seen["stdout"] = stdout
                seen["container_workdir"] = container.container_workdir
                return module.ParsedUsage(cost_usd=0.25, cost_source="reported")

            with patch.object(module, "swe_agent_parse_usage", side_effect=fake_parse):
                usage = agent._parse_usage("mini stdout", container=_FakeContainer(), workdir="/workspace")

            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertEqual(usage.cost_usd, 0.25)
            self.assertEqual(seen["stdout"], "mini stdout")
            self.assertEqual(seen["container_workdir"], "/workspace")
        finally:
            _restore_modules(prior)

    def test_swe_agent_saves_trajectory_artifact(self) -> None:
        module, prior = _load_swe_agent_module()
        try:
            agent = module.MiniSweAgent(model_name="fireworks_ai/accounts/fireworks/models/glm-5p1")
            with tempfile.TemporaryDirectory() as tmp_dir:
                logging_dir = Path(tmp_dir)
                with patch.object(
                    module,
                    "_load_swe_agent_trajectory",
                    return_value=(Path("/workspace/.mini/run.traj.json"), {"final_response": "done"}),
                ):
                    agent._save_trajectory_artifact(_FakeContainer(), "stdout", logging_dir, "/workspace")

                self.assertEqual(
                    (logging_dir / "trajectory-path.txt").read_text(encoding="utf-8"),
                    "/workspace/.mini/run.traj.json",
                )
                self.assertEqual(
                    json.loads((logging_dir / "trajectory.json").read_text(encoding="utf-8")),
                    {"final_response": "done"},
                )
        finally:
            _restore_modules(prior)

    def test_terminalbench_agents_require_uv_by_default(self) -> None:
        module, prior = _load_base_module()
        try:
            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _requires_node(self) -> bool:
                    return False

                def _install_script(self) -> str:
                    return "true"

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "true"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return None

                def _env_vars(self) -> dict[str, str]:
                    return {}

            self.assertTrue(DummyAgent()._requires_uv())
        finally:
            _restore_modules(prior)

    def test_perform_task_records_touched_tests_and_installs_recovery_wrapper(self) -> None:
        module, prior = _load_base_module()
        try:
            commands: list[str] = []

            class DummyAgent(module.DockerExecAgent):
                @staticmethod
                def name() -> str:
                    return "dummy"

                def _requires_node(self) -> bool:
                    return False

                def _requires_uv(self) -> bool:
                    return False

                def _install_script(self) -> str:
                    return "true"

                def _run_command(self, instruction: str, workdir: str) -> str:
                    del instruction, workdir
                    return "run-agent"

                def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
                    del stdout, container, workdir
                    return None

                def _env_vars(self) -> dict[str, str]:
                    return {}

                def _container_exec(self, container, cmd: str, **kwargs):
                    del container, kwargs
                    commands.append(cmd)
                    if "status --porcelain" in cmd:
                        return module.ExecOutput(exit_code=0, stdout="?? tests/test_created.py\n M src/app.py\n", stderr="")
                    return module.ExecOutput(exit_code=0, stdout="stdout", stderr="")

            agent = DummyAgent()
            session = _TmuxSession()
            session.container = _FakeContainer()

            with tempfile.TemporaryDirectory() as tmp_dir:
                result = agent.perform_task("instruction", session, logging_dir=Path(tmp_dir))
                saved = json.loads((Path(tmp_dir) / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(result.failure_mode, _FailureMode.NONE)
            self.assertEqual(saved["touched_test_files"], ["tests/test_created.py"])
            self.assertTrue(saved["test_patch_collision_recovery_enabled"])
            self.assertTrue(any("/usr/local/bin/patch" in command for command in commands))
        finally:
            _restore_modules(prior)


if __name__ == "__main__":
    unittest.main()
