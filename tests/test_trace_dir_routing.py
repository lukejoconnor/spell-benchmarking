import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import importlib

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


bench = _import_bench_with_stubbed_client()


class TraceDirRoutingTest(unittest.TestCase):
    def test_run_id_includes_entropy_suffix(self):
        run_id_1 = bench._run_id("swebench")
        run_id_2 = bench._run_id("swebench")
        self.assertRegex(run_id_1, r"^swebench_\d{8}_\d{6}_[0-9a-f]{8}$")
        self.assertRegex(run_id_2, r"^swebench_\d{8}_\d{6}_[0-9a-f]{8}$")
        self.assertNotEqual(run_id_1, run_id_2)

    def test_trace_dir_for_request_returns_none_without_base(self):
        result = bench._trace_dir_for_request(None, "spell", "item-1")
        self.assertIsNone(result)

    def test_trace_dir_for_condition_namespaces_by_condition_and_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_condition(base, "codex", name="baseline/v1")
            self.assertEqual(result, base / "codex" / "baseline_v1")
            assert result is not None
            self.assertTrue(result.exists())

    def test_trace_dir_for_request_namespaces_by_condition_and_item(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_request(base, "spell", "instance/repo:item")
            self.assertEqual(result, base / "spell" / "instance_repo_item")
            assert result is not None
            self.assertTrue(result.exists())

    def test_trace_dir_for_request_includes_name_segment_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_request(base, "spell", "instance/repo:item", name="variant/v1")
            self.assertEqual(result, base / "spell" / "variant_v1" / "instance_repo_item")
            assert result is not None
            self.assertTrue(result.exists())

    def test_trace_dir_for_request_adds_attempt_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_request(base, "spell", "exercism-item", attempt=2)
            self.assertEqual(result, base / "spell" / "exercism-item" / "attempt-2")
            assert result is not None
            self.assertTrue(result.exists())

    def test_swebench_prompt_workspace_uses_pro_app_path(self):
        pro_item = SimpleNamespace(raw_instance={"dockerhub_tag": "repo.tag"})
        standard_item = SimpleNamespace(raw_instance={})

        self.assertEqual(bench._swebench_prompt_workspace(pro_item), Path("/app"))
        self.assertEqual(bench._swebench_prompt_workspace(standard_item), Path("/testbed"))


if __name__ == "__main__":
    unittest.main()
