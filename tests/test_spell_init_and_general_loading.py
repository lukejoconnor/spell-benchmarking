import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types

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


bench = _import_bench_with_stubbed_client()
from src.datasets import BenchmarkItem
from src.provenance import benchmark_git_versions
from src.unified_adapters import RunResponse, build_spell_init


def teardown_module(_module) -> None:
    sys.modules.pop("bench", None)
    sys.modules.pop("src.unified_adapters", None)
    stubbed_client = sys.modules.get("spell_benchmark_client")
    if stubbed_client is not None and getattr(stubbed_client, "__file__", None) is None:
        sys.modules.pop("spell_benchmark_client", None)


class SpellInitRegressionTest(unittest.TestCase):
    def test_build_spell_init_defaults_to_extend(self):
        init = build_spell_init("Solve it")

        self.assertIn("(quine prompt \"Solve it\")", init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_accepts_custom_trailing(self):
        init = build_spell_init(
            "Solve it",
            trailing='\'(do (quine sys-reminder "Use tools") (!describe io) (!extend))',
        )

        self.assertIn('(quine sys-reminder "Use tools")', init)
        self.assertIn("'(do (quine sys-reminder \"Use tools\") (!describe io) (!extend))", init)
        self.assertNotIn("'(!extend))))", init)

    def test_fix_loop_variant_uses_bound_prompt_symbol(self):
        init = bench._make_spell_init(
            "Repository: demo/repo\n\n## Issue to Fix\n\nHandle newline-sensitive input.",
            "'(patterns/fix-loop prompt)",
        )

        self.assertIn("(quine prompt \"Repository: demo/repo\\n\\n## Issue to Fix\\n\\nHandle newline-sensitive input.\")", init)
        self.assertIn("'(patterns/fix-loop prompt)", init)


class GeneralLoadingRegressionTest(unittest.TestCase):
    def test_mk_record_preserves_reasoning_effort(self):
        record = bench._mk_record(
            run_id="run-1",
            benchmark="general",
            dataset="gsm8k",
            split=None,
            item_id="item-1",
            condition="spell",
            model_spec="codex-tc:gpt-5.3-codex",
            agent_file=None,
            response=RunResponse(
                status="ok",
                response_text="done",
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.12,
                latency_ms=50.0,
            ),
            response_text="done",
            patch=None,
            answer_extracted="42",
            correct=True,
            reasoning_effort="high",
        )

        self.assertEqual(record.reasoning_effort, "high")

    def test_mk_record_drops_unsupported_reasoning_effort(self):
        record = bench._mk_record(
            run_id="run-1",
            benchmark="general",
            dataset="gsm8k",
            split=None,
            item_id="item-1",
            condition="spell",
            model_spec="anthropic-tc:opus46",
            agent_file=None,
            response=RunResponse(
                status="ok",
                response_text="done",
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.12,
                latency_ms=50.0,
            ),
            response_text="done",
            patch=None,
            answer_extracted="42",
            correct=True,
            reasoning_effort="high",
        )

        self.assertIsNone(record.reasoning_effort)

    def test_mk_record_includes_git_versions_metadata(self):
        with patch.object(
            bench,
            "benchmark_git_versions",
            return_value={
                "spell": {"commit": "spell-sha", "branch": "main", "dirty": False},
                "benchmarking": {"commit": "bench-sha", "branch": "HEAD", "dirty": True},
            },
        ):
            record = bench._mk_record(
                run_id="run-1",
                benchmark="general",
                dataset="gsm8k",
                split=None,
                item_id="item-1",
                condition="spell",
                model_spec="codex-tc:gpt-5.3-codex",
                agent_file=None,
                response=RunResponse(
                    status="ok",
                    response_text="done",
                    tokens_in=10,
                    tokens_out=5,
                    cost_usd=0.12,
                    latency_ms=50.0,
                    metadata={"trace_dir": "/tmp/trace"},
                ),
                response_text="done",
                patch=None,
                answer_extracted="42",
                correct=True,
                metadata={"source": "pilot"},
            )

        self.assertEqual(record.metadata["source"], "pilot")
        self.assertEqual(record.metadata["trace_dir"], "/tmp/trace")
        self.assertEqual(
            record.metadata["versions"],
            {
                "spell": {"commit": "spell-sha", "branch": "main", "dirty": False},
                "benchmarking": {"commit": "bench-sha", "branch": "HEAD", "dirty": True},
            },
        )

    def test_benchmark_git_versions_dedupes_standalone_checkout(self):
        benchmark_git_versions.cache_clear()
        try:
            with patch("src.provenance.resolve_spell_root", return_value=BENCHMARK_ROOT), patch(
                "src.provenance.subprocess.run"
            ) as run_mock:
                run_mock.side_effect = [
                    types.SimpleNamespace(stdout="bench-sha\n"),
                    types.SimpleNamespace(stdout="main\n"),
                    types.SimpleNamespace(stdout=" M bench.py\n"),
                ]

                versions = benchmark_git_versions()

            self.assertEqual(
                versions,
                {
                    "benchmarking": {"commit": "bench-sha", "branch": "main", "dirty": True},
                },
            )
            self.assertEqual(run_mock.call_count, 3)
        finally:
            benchmark_git_versions.cache_clear()

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
            with patch.object(bench, "load_project_euler", return_value=[loaded_item]) as load_mock, \
                 patch.object(bench.click, "echo") as echo_mock:
                bench.run_general.callback(
                    dataset="project_euler",
                    condition="one_shot",
                    model=None,
                    claude_model=None,
                    codex_model="codex",
                    swe_agent_model=None,
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
