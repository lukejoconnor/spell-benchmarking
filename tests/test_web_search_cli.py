from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


def _import_bench_with_stubbed_client():
    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        pass

    stub.SpellBenchmarkClient = _DummyClient
    sys.modules.setdefault("spell_benchmark_client", stub)

    import bench as bench_module  # local import after stubbing

    return bench_module


class WebSearchCliRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = _import_bench_with_stubbed_client()
        from click.testing import CliRunner

        cls.runner = CliRunner()

    def _ok_response(self):
        return self.bench.RunResponse(
            status="ok",
            response_text="42",
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            latency_ms=1.0,
            error_message=None,
            error_type=None,
            metadata={},
        )

    def test_general_gaia_defaults_web_search_on_for_all_conditions(self):
        requests = []
        gaia_item = self.bench.BenchmarkItem(
            id="gaia_1",
            dataset="gaia",
            context="",
            question="Who won?",
            answer="A",
            metadata={},
        )

        def fake_run_requests_parallel(reqs, adapter, max_parallel, progress_callback=None):
            requests.extend(reqs)
            return [self._ok_response() for _ in reqs]

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.dict(self.bench.DATASETS, {"gaia": lambda: [gaia_item]}, clear=False), \
             patch.object(
                 self.bench,
                 "_resolve_spell_agent_or_die",
                 return_value=SimpleNamespace(
                     agent_file="config/agents/cli.agent.edn",
                     spell_mode="tc",
                     source="test",
                     policy_key="general",
                 ),
             ), \
             patch.object(self.bench, "_build_adapter", return_value=object()), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_evaluate_general", return_value=("A", True)), \
             patch.object(self.bench, "run_requests_parallel", side_effect=fake_run_requests_parallel):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "general",
                    "--dataset",
                    "gaia",
                    "--condition",
                    "all",
                    "--model",
                    "anthropic-tc:opus46",
                    "--claude-model",
                    "sonnet",
                    "--codex-model",
                    "codex",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(req.web_search is True for req in requests))

    def test_general_gaia_no_web_search_turns_off_requests_and_prompt(self):
        requests = []
        gaia_item = self.bench.BenchmarkItem(
            id="gaia_1",
            dataset="gaia",
            context="",
            question="Who won?",
            answer="A",
            metadata={},
        )

        def fake_run_requests_parallel(reqs, adapter, max_parallel, progress_callback=None):
            requests.extend(reqs)
            return [self._ok_response() for _ in reqs]

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.dict(self.bench.DATASETS, {"gaia": lambda: [gaia_item]}, clear=False), \
             patch.object(
                 self.bench,
                 "_resolve_spell_agent_or_die",
                 return_value=SimpleNamespace(
                     agent_file="config/agents/cli.agent.edn",
                     spell_mode="tc",
                     source="test",
                     policy_key="general",
                 ),
             ), \
             patch.object(self.bench, "_build_adapter", return_value=object()), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_evaluate_general", return_value=("A", True)), \
             patch.object(self.bench, "run_requests_parallel", side_effect=fake_run_requests_parallel):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "general",
                    "--dataset",
                    "gaia",
                    "--condition",
                    "all",
                    "--model",
                    "anthropic-tc:opus46",
                    "--claude-model",
                    "sonnet",
                    "--codex-model",
                    "codex",
                    "--no-web-search",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(req.web_search is False for req in requests))
        self.assertTrue(all("Use web search" not in req.prompt for req in requests))
        self.assertTrue(all("non-web tools" in req.prompt for req in requests))

    def test_featurebench_web_search_flag_reaches_run_request(self):
        seen = []

        class DummyWorktree:
            def __init__(self, path: Path):
                self.path = path

            def diff(self):
                return ""

            def changed_files(self):
                return []

            def remove(self):
                return None

        class DummyAdapter:
            def run(self, req):
                seen.append(req)
                return self_outer._ok_response()

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(
                 self.bench,
                 "load_tasks",
                 return_value=[{"instance_id": "fb-1", "problem_statement": "do it", "repo": "demo"}],
             ), \
             patch.object(self.bench, "RepoCache", return_value=object()), \
             patch.object(
                 self.bench,
                 "setup_workspace",
                 side_effect=lambda cache, item, tag: DummyWorktree(Path(tempfile.mkdtemp(prefix="fb-test-"))),
             ), \
             patch.object(self.bench, "install_deps"), \
             patch.object(self.bench, "mask_functions"), \
             patch.object(self.bench, "featurebench_format_prompt", return_value="featurebench prompt"), \
             patch.object(self.bench, "_build_adapter", return_value=DummyAdapter()), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "append_jsonl"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "featurebench",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--web-search",
                    "--workspace-root",
                    tmp_dir,
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0].web_search, True)

    def test_scienceagentbench_web_search_flag_reaches_run_request(self):
        seen = []
        item = self.bench.ScienceAgentBenchItem(
            instance_id="sab-1",
            domain="chemistry",
            subtask_categories=["x"],
            task_inst="Write code",
            dataset_folder_tree="dataset/",
            dataset_preview="",
            domain_knowledge="",
            gold_program_name="solution.py",
            output_fname="outputs/result.csv",
            eval_script_name="eval.py",
            github_name="demo/repo",
        )

        class DummyAdapter:
            def run(self, req):
                seen.append(req)
                return self_outer._ok_response()

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_scienceagentbench", return_value=[item]), \
             patch.object(self.bench, "_build_adapter", return_value=DummyAdapter()), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "append_jsonl"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "scienceagentbench",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--web-search",
                    "--workspace-root",
                    tmp_dir,
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0].web_search, True)

    def test_exercism_dry_run_reports_web_search_flag(self):
        exercism_item = self.bench.ExercismItem(
            slug="two-fer",
            name="Two Fer",
            difficulty=1,
            instructions="Return a string",
            stub_file="two_fer.py",
            stub_content="def two_fer():\n    pass\n",
            test_file="test_two_fer.py",
            test_content="def test_placeholder():\n    assert True\n",
            extra_files=[],
        )

        with patch.object(self.bench, "load_exercism", return_value=[exercism_item]):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "exercism",
                    "--condition",
                    "codex",
                    "--dry-run",
                    "--web-search",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"web_search": true', result.output)


if __name__ == "__main__":
    unittest.main()
