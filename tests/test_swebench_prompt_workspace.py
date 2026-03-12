from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


def _import_bench_with_stubbed_client():
    existing = sys.modules.get("bench")
    if existing is not None:
        return existing

    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise NotImplementedError

    stub.SpellBenchmarkClient = _DummyClient
    prior_client = sys.modules.get("spell_benchmark_client")
    sys.modules["spell_benchmark_client"] = stub
    try:
        return importlib.import_module("bench")
    finally:
        if prior_client is None:
            sys.modules.pop("spell_benchmark_client", None)
        else:
            sys.modules["spell_benchmark_client"] = prior_client
        sys.modules.pop("src.unified_adapters", None)


class SwebenchPromptWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = _import_bench_with_stubbed_client()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("bench", None)
        sys.modules.pop("src.unified_adapters", None)
        stubbed_client = sys.modules.get("spell_benchmark_client")
        if stubbed_client is not None and getattr(stubbed_client, "__file__", None) is None:
            sys.modules.pop("spell_benchmark_client", None)

    def test_local_runs_use_host_workspace(self):
        workspace = Path("/tmp/host-worktree")

        prompt_workspace = self.bench._swebench_prompt_workspace(
            workspace,
            docker=False,
            unified=False,
        )

        self.assertEqual(prompt_workspace, workspace)

    def test_container_runs_use_container_workspace(self):
        workspace = Path("/tmp/host-worktree")

        docker_workspace = self.bench._swebench_prompt_workspace(
            workspace,
            docker=True,
            unified=False,
        )
        unified_workspace = self.bench._swebench_prompt_workspace(
            workspace,
            docker=False,
            unified=True,
        )

        self.assertEqual(docker_workspace, Path("/workspace"))
        self.assertEqual(unified_workspace, Path("/workspace"))


if __name__ == "__main__":
    unittest.main()
