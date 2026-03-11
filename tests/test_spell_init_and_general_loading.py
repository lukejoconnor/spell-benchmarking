import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

def _import_bench_with_stubbed_client():
    stub = type(sys)("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        pass

    stub.SpellBenchmarkClient = _DummyClient
    sys.modules.setdefault("spell_benchmark_client", stub)

    import bench as bench_module

    return bench_module


bench = _import_bench_with_stubbed_client()
from src.datasets import BenchmarkItem
from src.unified_adapters import build_spell_init


class SpellInitRegressionTest(unittest.TestCase):
    def test_build_spell_init_defaults_to_extend(self):
        init = build_spell_init("Solve it")

        self.assertIn("(quine prompt \"Solve it\")", init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_accepts_custom_trailing(self):
        init = build_spell_init(
            "Solve it",
            preamble='(quine sys-reminder "Use tools")',
            trailing="'(do (!describe io) (!extend))",
        )

        self.assertIn('(quine sys-reminder "Use tools")', init)
        self.assertIn("'(do (!describe io) (!extend))", init)
        self.assertNotIn("'(!extend))))", init)

    def test_general_spell_init_does_not_inject_old_preambles(self):
        requests = []
        loaded_item = BenchmarkItem(
            id="omni_math_1",
            dataset="omni_math",
            context="",
            question="Compute 2+2.",
            answer="4",
        )

        def fake_run_requests_parallel(reqs, adapter, max_parallel, progress_callback=None):
            requests.extend(reqs)
            return [
                bench.RunResponse(
                    status="ok",
                    response_text="4",
                    tokens_in=None,
                    tokens_out=None,
                    cost_usd=None,
                    latency_ms=1.0,
                    error_message=None,
                    error_type=None,
                    metadata={},
                )
                for _ in reqs
            ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.jsonl"
            with patch.dict(bench.DATASETS, {"omni_math": lambda: [loaded_item]}, clear=False), \
                 patch.object(
                     bench,
                     "_resolve_spell_agent_or_die",
                     return_value=SimpleNamespace(
                         agent_file="config/agents/cli.agent.edn",
                         spell_mode="tc",
                         source="test",
                         policy_key="general",
                     ),
                 ), \
                 patch.object(bench, "_build_adapter", return_value=object()), \
                 patch.object(bench, "_print_summary"), \
                 patch.object(bench, "append_jsonl"), \
                 patch.object(bench, "_evaluate_general", return_value=("4", True)), \
                 patch.object(bench, "run_requests_parallel", side_effect=fake_run_requests_parallel):
                bench.run_general.callback(
                    dataset="omni_math",
                    condition="spell",
                    model="anthropic-tc:opus46",
                    claude_model=None,
                    codex_model="codex",
                    n=None,
                    items=None,
                    parallel=1,
                    budget=1.0,
                    agent=None,
                    agent_msg=None,
                    agent_tc=None,
                    depth=None,
                    max_tokens=None,
                    timeout=5,
                    thinking=False,
                    thinking_budget=10000,
                    reasoning_effort="high",
                    verbosity=None,
                    responses_api=False,
                    suffix_grammar=False,
                    web_search=False,
                    trace=False,
                    dry_run=False,
                    run_id="test-general-spell",
                    output=str(output_path),
                )

        self.assertEqual(len(requests), 1)
        self.assertIn("Compute 2+2.", requests[0].init)
        self.assertNotIn("Solve the problem using a combination", requests[0].init)
        self.assertIn("'(!extend)", requests[0].init)


class GeneralLoadingRegressionTest(unittest.TestCase):
    def test_project_euler_items_filter_uses_normalized_ids(self):
        loaded_item = BenchmarkItem(
            id="project_euler_0001",
            dataset="project_euler",
            context="",
            question="Q",
            answer="A",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.jsonl"
            with patch("bench.load_project_euler", return_value=[loaded_item]) as load_mock, \
                 patch("bench.click.echo") as echo_mock:
                bench.run_general.callback(
                    dataset="project_euler",
                    condition="one_shot",
                    model=None,
                    claude_model=None,
                    codex_model="codex",
                    n=None,
                    items="project_euler_1",
                    parallel=1,
                    budget=1.0,
                    agent=None,
                    agent_msg=None,
                    agent_tc=None,
                    depth=None,
                    max_tokens=None,
                    timeout=5,
                    thinking=False,
                    thinking_budget=10000,
                    reasoning_effort="high",
                    verbosity=None,
                    responses_api=False,
                    suffix_grammar=False,
                    web_search=False,
                    trace=False,
                    dry_run=True,
                    run_id="test-general",
                    output=str(output_path),
                )

        load_mock.assert_called_once_with(max_items=None, problem_numbers=[1])
        echoed_lines = [call.args[0] for call in echo_mock.call_args_list if call.args]
        self.assertIn("Loaded 1 items", echoed_lines)


if __name__ == "__main__":
    unittest.main()
