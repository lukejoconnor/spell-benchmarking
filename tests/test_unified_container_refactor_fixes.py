import os
import tempfile
import unittest
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import sys
import types

from src.docker_agents import (
    ExecOutput,
    _resolve_agent_for_container,
    container_exec,
    discover_spell_root,
)


def _import_unified_swebench():
    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        def __init__(self, *args, **kwargs):
            pass

    stub.SpellBenchmarkClient = _DummyClient
    prior_client = sys.modules.get("spell_benchmark_client")
    sys.modules["spell_benchmark_client"] = stub
    try:
        module = importlib.import_module("src.unified_swebench")
    finally:
        if prior_client is None:
            sys.modules.pop("spell_benchmark_client", None)
        else:
            sys.modules["spell_benchmark_client"] = prior_client
        sys.modules.pop("src.unified_adapters", None)
    return module


run_unified_swebench = _import_unified_swebench().run_unified_swebench


def _make_spell_root(path: Path) -> None:
    (path / "src" / "spell").mkdir(parents=True, exist_ok=True)
    (path / "config" / "agents").mkdir(parents=True, exist_ok=True)
    (path / "deps.edn").write_text("{}", encoding="utf-8")
    (path / "src" / "spell" / "benchmark_api.clj").write_text("", encoding="utf-8")


class ContainerExecTest(unittest.TestCase):
    def test_container_exec_supports_docker_py_containers(self):
        exec_result = SimpleNamespace(exit_code=0, output=(b"stdout", b"stderr"))
        container = MagicMock()
        container.exec_run.return_value = exec_result

        result = container_exec(
            container,
            "echo hi",
            timeout=5,
            env={"TEST_ENV": "1"},
            workdir="/workspace",
        )

        self.assertEqual(result, ExecOutput(exit_code=0, stdout="stdout", stderr="stderr", timed_out=False))
        container.exec_run.assert_called_once()
        args, kwargs = container.exec_run.call_args
        self.assertEqual(args[0][0:2], ["bash", "-c"])
        self.assertIn("timeout 5s bash -c", args[0][2])
        self.assertEqual(kwargs["environment"], {"TEST_ENV": "1"})
        self.assertEqual(kwargs["workdir"], "/workspace")
        self.assertTrue(kwargs["demux"])


class DiscoverSpellRootTest(unittest.TestCase):
    def test_discover_spell_root_finds_sibling_checkout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sibling_spell = root / "spell"
            _make_spell_root(sibling_spell)

            search_from = root / "spell-benchmarking" / "terminal_bench" / "spell_agent.py"
            search_from.parent.mkdir(parents=True, exist_ok=True)
            search_from.write_text("# stub\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                found = discover_spell_root(search_from=search_from)

            self.assertEqual(found, sibling_spell.resolve())


class ResolveAgentForContainerTest(unittest.TestCase):
    def test_rewrites_agents_from_bundled_spell_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spell_root = Path(tmp_dir) / "spell"
            _make_spell_root(spell_root)
            agent_file = spell_root / "config" / "agents" / "custom.agent.edn"
            agent_file.write_text("{:name custom}", encoding="utf-8")

            resolved = _resolve_agent_for_container(str(agent_file), spell_root=spell_root)

            self.assertEqual(resolved, "/opt/spell/config/agents/custom.agent.edn")

    def test_preserves_external_agent_override_with_same_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spell_root = Path(tmp_dir) / "spell"
            _make_spell_root(spell_root)
            external_root = Path(tmp_dir) / "external" / "config" / "agents"
            external_root.mkdir(parents=True, exist_ok=True)
            agent_file = external_root / "custom.agent.edn"
            agent_file.write_text("{:name external}", encoding="utf-8")

            resolved = _resolve_agent_for_container(str(agent_file), spell_root=spell_root)

            self.assertEqual(resolved, str(agent_file))


class UnifiedSWEBenchRunnerTest(unittest.TestCase):
    @patch("src.unified_swebench.run_agent_in_container")
    @patch("src.unified_swebench.SWEBenchContainer")
    def test_runner_uses_initial_commit_and_records_same_container_eval(self, container_cls, run_agent_mock):
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]
        container.run_eval.return_value = SimpleNamespace(
            resolved=True,
            test_output="eval ok",
            raw_report={"instance-1": {"resolved": True}},
        )

        container_cm = MagicMock()
        container_cm.__enter__.return_value = container
        container_cm.__exit__.return_value = False
        container_cls.return_value = container_cm

        run_agent_mock.return_value = SimpleNamespace(
            output=ExecOutput(exit_code=0, stdout="", stderr="", timed_out=False),
            usage=None,
            response_text="ok",
            error_message=None,
            error_type=None,
            raw={"ok": True, "trace_dir": ".spell-trace"},
            trace_dir="/tmp/host-traces",
        )

        response = run_unified_swebench(
            condition="spell",
            prompt="Fix it",
            model_spec="anthropic-tc:opus46",
            timeout=30,
            init="(do)",
            agent_file="/opt/spell/config/agents/io-tc.agent.edn",
            budget=5.0,
            depth=4,
            max_tokens=None,
            reasoning_effort="high",
            web_search=False,
            trace=True,
            trace_dir=Path("/tmp/host-traces"),
            eval_timeout=90,
            item=SimpleNamespace(
                instance_id="instance-1",
                raw_instance={},
            ),
        )

        self.assertEqual(run_agent_mock.call_args.kwargs["trace_dir"], Path("/tmp/host-traces"))
        container.diff.assert_called_once_with("base-sha")
        container.changed_files.assert_called_once_with("base-sha")
        container.run_eval.assert_called_once_with(timeout=90)
        self.assertEqual(response.metadata["trace_dir"], "/tmp/host-traces")
        self.assertEqual(response.metadata["patch"], "diff --git a/foo b/foo")
        self.assertTrue(response.metadata["same_container_resolved"])


if __name__ == "__main__":
    unittest.main()
