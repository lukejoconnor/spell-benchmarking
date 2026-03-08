import tempfile
import unittest
from pathlib import Path
import sys
import types

stub = types.ModuleType("spell_benchmark_client")


class _DummyClient:  # pragma: no cover - import wiring only
    pass


stub.SpellBenchmarkClient = _DummyClient
sys.modules.setdefault("spell_benchmark_client", stub)

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
import bench


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

    def test_trace_dir_for_request_namespaces_by_condition_and_item(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_request(base, "spell", "instance/repo:item")
            self.assertEqual(result, base / "spell" / "instance_repo_item")
            assert result is not None
            self.assertTrue(result.exists())

    def test_trace_dir_for_request_adds_attempt_suffix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            result = bench._trace_dir_for_request(base, "spell", "exercism-item", attempt=2)
            self.assertEqual(result, base / "spell" / "exercism-item" / "attempt-2")
            assert result is not None
            self.assertTrue(result.exists())

    def test_swebench_worktree_tag_includes_variant_and_run_id(self):
        tag = bench._swebench_worktree_tag(
            "spell",
            "swebench_20260304_120000_deadbeef",
            variant="fix-loop/v1",
        )
        self.assertEqual(tag, "spell.fix-loop_v1.swebench_20260304_120000_deadbeef")


if __name__ == "__main__":
    unittest.main()
