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
    ParsedUsage,
    _resolve_agent_for_container,
    container_exec,
    discover_spell_root,
    run_agent_in_container,
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

    def test_rewrites_spell_relative_agent_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            spell_root = Path(tmp_dir) / "spell"
            _make_spell_root(spell_root)
            agent_file = spell_root / "config" / "agents" / "io-tc.agent.edn"
            agent_file.write_text("{:name io-tc}", encoding="utf-8")

            resolved = _resolve_agent_for_container("config/agents/io-tc.agent.edn", spell_root=spell_root)

            self.assertEqual(resolved, "/opt/spell/config/agents/io-tc.agent.edn")


class UnifiedSWEBenchRunnerTest(unittest.TestCase):
    @patch("src.unified_swebench.run_agent_in_container")
    @patch("src.unified_swebench.SWEBenchContainer")
    def test_runner_uses_initial_commit_and_records_same_container_eval(self, container_cls, run_agent_mock):
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]
        container.image_name = "sweb.eval.x86_64.instance-1:latest"
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
        self.assertEqual(response.metadata["instance_image_name"], "sweb.eval.x86_64.instance-1:latest")
        self.assertTrue(response.metadata["same_container_resolved"])

    @patch("src.unified_swebench.run_agent_in_container")
    @patch("src.unified_swebench.SWEBenchContainer")
    def test_runner_keeps_status_from_swe_agent_raw_trajectory(self, container_cls, run_agent_mock):
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]
        container.image_name = "sweb.eval.x86_64.instance-1:latest"
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
            raw={"trajectory_path": "/workspace/run.traj.json", "trajectory": {"messages": []}},
            trace_dir="/tmp/host-traces/run.traj.json",
            metadata={"cost_source": "estimated_per_call"},
        )

        response = run_unified_swebench(
            condition="swe_agent",
            prompt="Fix it",
            model_spec="fireworks_ai/accounts/fireworks/models/glm-5p1",
            timeout=30,
            init=None,
            agent_file=None,
            budget=5.0,
            depth=4,
            max_tokens=None,
            reasoning_effort=None,
            web_search=False,
            trace=True,
            trace_dir=Path("/tmp/host-traces"),
            eval_timeout=90,
            item=SimpleNamespace(
                instance_id="instance-1",
                raw_instance={},
            ),
        )

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.metadata["cost_source"], "estimated_per_call")
        self.assertTrue(response.metadata["same_container_resolved"])

    @patch("src.unified_swebench.run_agent_in_container")
    @patch("src.unified_swebench.SWEBenchContainer")
    def test_runner_preserves_timeout_with_swe_agent_raw_trajectory(self, container_cls, run_agent_mock):
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]
        container.image_name = "sweb.eval.x86_64.instance-1:latest"
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
            output=ExecOutput(exit_code=124, stdout="", stderr="", timed_out=True),
            usage=None,
            response_text="partial",
            error_message="Timeout after 300s",
            error_type="mini_cli_error",
            raw={"trajectory_path": "/workspace/run.traj.json", "trajectory": {"messages": []}},
            trace_dir="/tmp/host-traces/run.traj.json",
            metadata=None,
        )

        response = run_unified_swebench(
            condition="swe_agent",
            prompt="Fix it",
            model_spec="fireworks_ai/accounts/fireworks/models/glm-5p1",
            timeout=30,
            init=None,
            agent_file=None,
            budget=5.0,
            depth=4,
            max_tokens=None,
            reasoning_effort=None,
            web_search=False,
            trace=True,
            trace_dir=Path("/tmp/host-traces"),
            eval_timeout=90,
            item=SimpleNamespace(
                instance_id="instance-1",
                raw_instance={},
            ),
        )

        self.assertEqual(response.status, "timeout")
        self.assertTrue(response.metadata["same_container_resolved"])


class SweAgentContainerRunnerTest(unittest.TestCase):
    @patch("src.docker_agents.swe_agent_pre_install", return_value=True)
    @patch("src.docker_agents.swe_agent_build_bundle", return_value=b"bundle")
    @patch("src.docker_agents.run_install_script", return_value=True)
    @patch("src.docker_agents.swe_agent_env_vars", return_value={})
    @patch("src.docker_agents.swe_agent_run_command", return_value="mini -t 'prompt' -m 'anthropic/claude-opus-4-6' -y --exit-immediately")
    @patch("src.docker_agents._load_swe_agent_trajectory")
    @patch("src.docker_agents.swe_agent_parse_usage")
    def test_runner_handles_swe_agent_trajectory_and_trace_copy(
        self,
        parse_usage_mock,
        load_traj_mock,
        run_command_mock,
        env_vars_mock,
        install_mock,
        build_bundle_mock,
        pre_install_mock,
    ):
        container = MagicMock()
        container.container_workdir = "/workspace"
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]
        container.copy_to_host.return_value = True
        container.exec_run.return_value = ExecOutput(exit_code=0, stdout="", stderr="")
        load_traj_mock.return_value = (
            Path("/workspace/run.traj.json"),
            {
                "info": {"model_stats": {"instance_cost": 0.25}},
                "trajectory": [{"role": "assistant", "content": "done"}],
            },
        )
        parse_usage_mock.return_value = ParsedUsage(cost_usd=0.25, cost_source="reported")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_agent_in_container(
                container,
                condition="swe_agent",
                prompt="prompt",
                timeout=30,
                model_spec="anthropic/claude-opus-4-6",
                budget=2.0,
                trace=True,
                trace_dir=Path(tmp_dir),
            )

        install_mock.assert_called_once()
        build_bundle_mock.assert_called_once()
        pre_install_mock.assert_called_once_with(container, b"bundle")
        self.assertGreaterEqual(env_vars_mock.call_count, 2)
        run_command_mock.assert_called_once()
        container.exec_run.assert_called_once()
        container.copy_to_host.assert_called_once_with("/workspace/run.traj.json", Path(tmp_dir))
        self.assertEqual(result.response_text, "done")
        self.assertEqual(result.usage.cost_usd, 0.25)
        self.assertEqual(result.metadata["cost_source"], "reported")
        self.assertEqual(result.trace_dir, str(Path(tmp_dir) / "run.traj.json"))
        self.assertEqual(result.raw["trajectory_path"], "/workspace/run.traj.json")


if __name__ == "__main__":
    unittest.main()
