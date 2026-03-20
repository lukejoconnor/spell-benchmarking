from __future__ import annotations

import importlib
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


class WebSearchCliRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = _import_bench_with_stubbed_client()
        from click.testing import CliRunner

        cls.runner = CliRunner()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("bench", None)
        sys.modules.pop("src.unified_adapters", None)
        stubbed_client = sys.modules.get("spell_benchmark_client")
        if stubbed_client is not None and getattr(stubbed_client, "__file__", None) is None:
            sys.modules.pop("spell_benchmark_client", None)

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

    def _web_search_by_condition(self, requests):
        return {req.condition: req.web_search for req in requests}

    def test_resolve_web_search_default_treats_gaia_subsets_as_gaia(self):
        self.assertTrue(self.bench._resolve_web_search_default({"gaia_l1"}, None))
        self.assertTrue(self.bench._resolve_web_search_default({"gaia_l2"}, None))
        self.assertTrue(self.bench._resolve_web_search_default({"gaia_l3"}, None))
        self.assertTrue(self.bench._resolve_web_search_default({"gaia", "math"}, None))
        self.assertFalse(self.bench._resolve_web_search_default({"math"}, None))

    def test_general_gaia_rejects_unsupported_conditions(self):
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

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("GAIA currently supports only web-enabled conditions", result.output)
        self.assertEqual(requests, [])

    def test_general_gaia_subset_defaults_web_search_on(self):
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
             patch.dict(self.bench.DATASETS, {"gaia_l1": lambda: [gaia_item]}, clear=False), \
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
                    "gaia_l1",
                    "--condition",
                    "claude_code",
                    "--claude-model",
                    "sonnet",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0].web_search, True)
        self.assertIn("Use web search", requests[0].prompt)
        self.assertIn("Return ONLY the final answer", requests[0].prompt)

    def test_general_gaia_no_web_search_is_rejected(self):
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
                    "--codex-model",
                    "codex",
                    "--condition",
                    "codex",
                    "--no-web-search",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("GAIA requires web search", result.output)
        self.assertEqual(requests, [])

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

        def fake_run_unified_featurebench(**kwargs):
            seen.append(kwargs["web_search"])
            return self._ok_response(), None, []

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
             patch.object(self.bench, "mask_functions"), \
             patch.object(self.bench, "featurebench_format_prompt", return_value="featurebench prompt"), \
             patch.object(self.bench, "run_unified_featurebench", side_effect=fake_run_unified_featurebench), \
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
        self.assertIs(seen[0], True)

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

    def test_swebench_pro_dry_run_succeeds_without_unified_flag(self):
        pro_item = SimpleNamespace(instance_id="django__django-1")

        with patch.object(self.bench, "load_swebench_pro", return_value=[pro_item]):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "pro",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"dataset": "pro"', result.output)
        self.assertIn('"clean_eval": false', result.output)

    def test_swebench_pro_dry_run_skips_eager_eval_repo_bootstrap(self):
        pro_item = SimpleNamespace(instance_id="django__django-1")

        with patch.object(self.bench, "load_swebench_pro", return_value=[pro_item]), \
             patch.object(self.bench, "ensure_scale_pro_eval_repo") as mock_ensure:
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "pro",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_ensure.assert_not_called()

    def test_swebench_pro_run_eagerly_bootstraps_eval_repo_before_generation(self):
        pro_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={"dockerhub_tag": "django.1"},
        )
        events = []

        def fake_ensure(repo_path):
            events.append(("ensure", repo_path))
            return Path(repo_path)

        def fake_run_unified_swebench(**_kwargs):
            events.append(("run", None))
            return self.bench.RunResponse(
                status="ok",
                response_text="done",
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=None,
                error_message=None,
                error_type=None,
                metadata={
                    "docker": True,
                    "patch": "",
                    "changed_files": [],
                },
            )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_pro", return_value=[pro_item]), \
             patch.object(self.bench, "ensure_scale_pro_eval_repo", side_effect=fake_ensure), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "pro",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--n",
                    "1",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(events[0], ("ensure", self.bench.SCALE_PRO_EVAL_REPO_DEFAULT))
        self.assertEqual(events[1][0], "run")
        self.assertIn("Pro eval repo ready", result.output)

    def test_swebench_keep_images_skips_final_image_cleanup(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )

        def fake_run_unified_swebench(**kwargs):
            return self.bench.RunResponse(
                status="ok",
                response_text="done",
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=None,
                error_message=None,
                error_type=None,
                metadata={
                    "docker": True,
                    "patch": "",
                    "changed_files": [],
                    "instance_image_name": "sweb.eval.x86_64.django__django-1:latest",
                },
            )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "_cleanup_swebench_instance_images") as cleanup_mock, \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "lite",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--n",
                    "1",
                    "--keep-images",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        cleanup_mock.assert_not_called()

    def test_swebench_multi_condition_defers_image_cleanup_until_after_generation(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )
        events = []

        def fake_run_unified_swebench(**kwargs):
            events.append(("run", kwargs["condition"]))
            return self.bench.RunResponse(
                status="ok",
                response_text="done",
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=None,
                error_message=None,
                error_type=None,
                metadata={
                    "docker": True,
                    "patch": "",
                    "changed_files": [],
                    "instance_image_name": "sweb.eval.x86_64.django__django-1:latest",
                },
            )

        def fake_cleanup(image_names):
            events.append(("cleanup", sorted(image_names)))

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "_resolve_spell_agent_or_die", return_value=SimpleNamespace(
                 agent_file="config/agents/cli.agent.edn",
                 spell_mode="tc",
                 source="test",
                 policy_key="swebench",
             )), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "_cleanup_swebench_instance_images", side_effect=fake_cleanup), \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "lite",
                    "--condition",
                    "both",
                    "--model",
                    "anthropic-tc:claude-opus-4-6",
                    "--claude-model",
                    "claude-opus-4-6",
                    "--n",
                    "1",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            events,
            [
                ("run", "spell"),
                ("run", "claude_code"),
                ("cleanup", ["sweb.eval.x86_64.django__django-1:latest"]),
            ],
        )

    def test_swebench_clean_eval_defers_image_cleanup_until_after_eval(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )
        events = []

        def fake_run_unified_swebench(**kwargs):
            events.append(("run", kwargs["condition"]))
            return self.bench.RunResponse(
                status="ok",
                response_text="done",
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=None,
                error_message=None,
                error_type=None,
                metadata={
                    "docker": True,
                    "patch": "",
                    "changed_files": [],
                    "instance_image_name": "sweb.eval.x86_64.django__django-1:latest",
                },
            )

        def fake_build_eval_plan(**_kwargs):
            return "princeton-nlp/SWE-bench_Lite", ["codex"], [{"condition": "codex"}]

        def fake_run_eval_plan(plan):
            events.append(("eval", [step["condition"] for step in plan]))

        def fake_cleanup(image_names):
            events.append(("cleanup", sorted(image_names)))

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "_build_swebench_eval_plan", side_effect=fake_build_eval_plan), \
             patch.object(self.bench, "_run_swebench_eval_plan", side_effect=fake_run_eval_plan), \
             patch.object(self.bench, "_cleanup_swebench_instance_images", side_effect=fake_cleanup), \
             patch.object(self.bench, "append_jsonl"), \
             patch.object(self.bench, "_print_summary"), \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "lite",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--n",
                    "1",
                    "--clean-eval",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            events,
            [
                ("run", "codex"),
                ("eval", ["codex"]),
                ("cleanup", ["sweb.eval.x86_64.django__django-1:latest"]),
            ],
        )

    def test_software_benchmark_help_omits_legacy_container_flags(self):
        swebench_help = self.runner.invoke(self.bench.cli, ["swebench", "--help"])
        featurebench_help = self.runner.invoke(self.bench.cli, ["featurebench", "--help"])
        exercism_help = self.runner.invoke(self.bench.cli, ["exercism", "--help"])

        self.assertEqual(swebench_help.exit_code, 0, swebench_help.output)
        self.assertEqual(featurebench_help.exit_code, 0, featurebench_help.output)
        self.assertEqual(exercism_help.exit_code, 0, exercism_help.output)
        self.assertNotIn("--docker", swebench_help.output)
        self.assertNotIn("--unified", swebench_help.output)
        self.assertNotIn("--workspace-root", swebench_help.output)
        self.assertNotIn("--keep-workspaces", swebench_help.output)
        self.assertIn("--clean-eval / --no-clean-eval", swebench_help.output)
        self.assertIn("--keep-images / --no-keep-images", swebench_help.output)
        self.assertNotIn("--unified", featurebench_help.output)
        self.assertNotIn("--unified", exercism_help.output)


if __name__ == "__main__":
    unittest.main()
