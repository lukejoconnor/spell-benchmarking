import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.docker_agents import ExecOutput, container_exec, discover_spell_root
from src.unified_swebench import run_unified_swebench


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


class UnifiedSWEBenchRunnerTest(unittest.TestCase):
    @patch("src.unified_swebench.run_agent_in_container")
    @patch("src.unified_swebench.DockerWorkspaceContainer")
    def test_runner_uses_initial_commit_and_persists_traces(self, container_cls, run_agent_mock):
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]

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
            workspace=Path("/tmp/workspace"),
        )

        self.assertEqual(run_agent_mock.call_args.kwargs["trace_dir"], Path("/tmp/host-traces"))
        container.diff.assert_called_once_with("base-sha")
        container.changed_files.assert_called_once_with("base-sha")
        self.assertEqual(response.metadata["trace_dir"], "/tmp/host-traces")
        self.assertEqual(response.metadata["patch"], "diff --git a/foo b/foo")


if __name__ == "__main__":
    unittest.main()
