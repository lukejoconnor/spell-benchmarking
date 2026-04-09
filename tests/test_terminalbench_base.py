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
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.usage_accounting import UsageBreakdown


def _load_terminalbench_base_module():
    fake_terminal_bench = types.ModuleType("terminal_bench")
    fake_agents = types.ModuleType("terminal_bench.agents")
    fake_terminal = types.ModuleType("terminal_bench.terminal")
    fake_base_agent = types.ModuleType("terminal_bench.agents.base_agent")
    fake_failure_mode = types.ModuleType("terminal_bench.agents.failure_mode")
    fake_tmux_session = types.ModuleType("terminal_bench.terminal.tmux_session")

    class BaseAgent:
        def __init__(self, **kwargs):
            del kwargs

        def _render_instruction(self, instruction: str) -> str:
            return instruction

    @dataclass
    class AgentResult:
        total_input_tokens: int = 0
        total_output_tokens: int = 0
        failure_mode: str | None = None

    class FailureMode:
        NONE = "none"
        AGENT_TIMEOUT = "agent_timeout"
        UNKNOWN_AGENT_ERROR = "unknown_agent_error"
        AGENT_INSTALLATION_FAILED = "agent_installation_failed"

    class TmuxSession:
        def __init__(self, container):
            self.container = container

    fake_base_agent.BaseAgent = BaseAgent
    fake_base_agent.AgentResult = AgentResult
    fake_failure_mode.FailureMode = FailureMode
    fake_tmux_session.TmuxSession = TmuxSession

    sys.modules.setdefault("terminal_bench", fake_terminal_bench)
    sys.modules.setdefault("terminal_bench.agents", fake_agents)
    sys.modules.setdefault("terminal_bench.terminal", fake_terminal)
    sys.modules["terminal_bench.agents.base_agent"] = fake_base_agent
    sys.modules["terminal_bench.agents.failure_mode"] = fake_failure_mode
    sys.modules["terminal_bench.terminal.tmux_session"] = fake_tmux_session

    spec = importlib.util.spec_from_file_location(
        "benchmarking_terminal_bench_base",
        BENCHMARK_ROOT / "terminal_bench" / "base.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_terminalbench_base_module()
DockerExecAgent = _BASE_MODULE.DockerExecAgent
FailureMode = _BASE_MODULE.FailureMode


class _FakeContainer:
    id = "deadbeefcafe"


class _FakeSession:
    def __init__(self) -> None:
        self.container = _FakeContainer()


class _DummyAgent(DockerExecAgent):
    @staticmethod
    def name() -> str:
        return "dummy"

    def _requires_node(self) -> bool:
        return False

    def _install_agent_cli(self, container) -> bool:
        del container
        return True

    def _detect_workdir(self, container) -> str:
        del container
        return "/workdir"

    def _install_script(self) -> str:
        return "true"

    def _run_command(self, instruction: str, workdir: str) -> str:
        return f"run {instruction} in {workdir}"

    def _parse_usage(self, stdout: str, *, container=None, workdir: str | None = None):
        del stdout, container, workdir
        return UsageBreakdown(
            uncached_input_tokens=90,
            cached_input_tokens=10,
            visible_output_tokens=11,
            reasoning_output_tokens=29,
            cost_usd=0.5,
            cost_source="reported",
        )

    def _env_vars(self) -> dict[str, str]:
        return {}

    def _container_exec(self, container, cmd: str, *, timeout=None, env=None, workdir=None):
        del container, cmd, timeout, env, workdir
        return types.SimpleNamespace(
            stdout="{}",
            stderr="",
            exit_code=0,
            timed_out=False,
        )


class TerminalBenchBaseTests(unittest.TestCase):
    def test_save_logs_writes_explicit_output_breakdown_fields(self) -> None:
        agent = _DummyAgent(model_name="demo-model")
        usage = UsageBreakdown(
            uncached_input_tokens=90,
            cached_input_tokens=10,
            visible_output_tokens=11,
            reasoning_output_tokens=29,
            cost_usd=0.5,
            cost_source="reported",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            logging_dir = Path(tmp_dir)
            agent._save_logs(logging_dir, "stdout", "stderr", usage, latency_s=1.2)
            result = json.loads((logging_dir / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(result["visible_output_tokens"], 11)
        self.assertEqual(result["reasoning_output_tokens"], 29)
        self.assertEqual(result["total_output_tokens"], 40)
        self.assertNotIn("output_tokens", result)

    def test_perform_task_returns_total_output_tokens_including_reasoning(self) -> None:
        agent = _DummyAgent(model_name="demo-model")

        result = agent.perform_task("solve it", _FakeSession())

        self.assertEqual(result.total_input_tokens, 100)
        self.assertEqual(result.total_output_tokens, 40)
        self.assertEqual(result.failure_mode, FailureMode.NONE)


if __name__ == "__main__":
    unittest.main()
