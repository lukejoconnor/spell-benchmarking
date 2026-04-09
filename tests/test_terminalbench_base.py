from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = BENCHMARK_ROOT / "terminal_bench" / "base.py"


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
                    visible_output_tokens=1123,
                    reasoning_output_tokens=17869,
                    cost_usd=0.324007,
                    cost_source="estimated_with_reasoning",
                )
                agent._save_logs(logging_dir, "stdout", "stderr", usage, 12.34)
                result = json.loads((logging_dir / "result.json").read_text(encoding="utf-8"))

            self.assertEqual(result["visible_output_tokens"], 1123)
            self.assertEqual(result["reasoning_output_tokens"], 17869)
            self.assertEqual(result["total_output_tokens"], 18992)
            self.assertNotIn("output_tokens", result)
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

                def _install_script(self) -> str:
                    return "true"

                def _install_agent_cli(self, container) -> bool:
                    del container
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


if __name__ == "__main__":
    unittest.main()
