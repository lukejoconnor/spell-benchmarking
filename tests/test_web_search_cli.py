from __future__ import annotations

import importlib
import json
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

    def _ok_response(self, **overrides):
        payload = {
            "status": "ok",
            "response_text": "42",
            "tokens_in": None,
            "tokens_out": None,
            "cached_input_tokens": None,
            "cost_usd": None,
            "latency_ms": 1.0,
            "error_message": None,
            "error_type": None,
            "metadata": {},
        }
        payload.update(overrides)
        return self.bench.RunResponse(
            **payload,
        )

    def _web_search_by_condition(self, requests):
        return {req.condition: req.web_search for req in requests}

    def test_swebench_paper_compliant_prompt_omits_hints_and_adds_instruction(self):
        item = self.bench.SWEBenchItem(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix a bug.",
            hints_text="Maintainer hint with an attached patch.",
            test_patch="",
            patch="",
            metadata={},
            raw_instance={},
        )

        default_prompt = self.bench._format_swebench_prompt(item, workspace="/testbed")
        paper_prompt = self.bench._format_swebench_prompt(
            item,
            workspace="/testbed",
            paper_compliant=True,
        )

        self.assertIn("## Hints", default_prompt)
        self.assertIn("Maintainer hint", default_prompt)
        self.assertNotIn("## Hints", paper_prompt)
        self.assertNotIn("Maintainer hint", paper_prompt)
        self.assertIn("Paper-compliance rule: Do not use web browsing", paper_prompt)
        self.assertIn("Use only the repository files in /testbed", paper_prompt)

    def test_swebench_paper_compliant_rejects_web_search(self):
        result = self.runner.invoke(
            self.bench.cli,
            [
                "swebench",
                "--dataset",
                "lite",
                "--condition",
                "codex",
                "--paper-compliant",
                "--web-search",
                "--dry-run",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--paper-compliant cannot be combined with --web-search", result.output)

    def test_swebench_paper_compliant_rejects_custom_spell_agents(self):
        result = self.runner.invoke(
            self.bench.cli,
            [
                "swebench",
                "--dataset",
                "lite",
                "--condition",
                "spell",
                "--model",
                "openai-tc:gpt-5.4",
                "--paper-compliant",
                "--agent",
                "config/agents/cli.agent.edn",
                "--dry-run",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--paper-compliant does not support custom Spell agent overrides", result.output)

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
             patch.object(self.bench, "_evaluate_general", return_value=("A", True, None)), \
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
             patch.object(self.bench, "_evaluate_general", return_value=("A", True, None)), \
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
             patch.object(self.bench, "_evaluate_general", return_value=("A", True, None)), \
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

    def test_featurebench_record_preserves_cached_input_tokens(self):
        written_records = []

        class DummyWorktree:
            def __init__(self, path: Path):
                self.path = path

            def diff(self):
                return ""

            def changed_files(self):
                return []

            def remove(self):
                return None

        def fake_run_unified_featurebench(**_kwargs):
            return (
                self._ok_response(
                    response_text="done",
                    tokens_in=120,
                    tokens_out=30,
                    cached_input_tokens=17,
                    metadata={"patch": "diff --git", "changed_files": ["demo.py"]},
                ),
                True,
                [{"name": "pytest", "passed": True}],
            )

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

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
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
             patch.object(self.bench, "_print_summary"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "featurebench",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--workspace-root",
                    tmp_dir,
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        self.assertEqual(written_records[0].cached_input_tokens, 17)

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

    def test_scienceagentbench_record_preserves_cached_input_tokens(self):
        written_records = []
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
                return self_outer._ok_response(
                    response_text="```python\nprint('hello')\n```",
                    tokens_in=80,
                    tokens_out=20,
                    cached_input_tokens=13,
                )

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_scienceagentbench", return_value=[item]), \
             patch.object(self.bench, "_build_adapter", return_value=DummyAdapter()), \
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
             patch.object(self.bench, "_print_summary"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "scienceagentbench",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--workspace-root",
                    tmp_dir,
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        self.assertEqual(written_records[0].cached_input_tokens, 13)

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

    def test_exercism_record_preserves_cached_input_tokens(self):
        written_records = []
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
        outcome = SimpleNamespace(
            response=self._ok_response(
                response_text="done",
                tokens_in=60,
                tokens_out=10,
                cached_input_tokens=19,
            ),
            passed=False,
            test_output="AssertionError",
            attempts=1,
        )

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_exercism", return_value=[exercism_item]), \
             patch.object(self.bench, "run_unified_exercism", return_value=outcome), \
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
             patch.object(self.bench, "_print_summary"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "exercism",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--difficulty",
                    "1",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        self.assertEqual(written_records[0].cached_input_tokens, 19)

    def test_exercism_one_shot_record_aggregates_cached_input_tokens(self):
        written_records = []
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
        responses = iter(
            [
                self._ok_response(
                    response_text=(
                        "```python\n"
                        "def two_fer(name='you'):\n"
                        "    return f'One for {name}, one for me.'\n"
                        "```"
                    ),
                    tokens_in=40,
                    tokens_out=10,
                    cached_input_tokens=5,
                    cost_usd=0.1,
                    latency_ms=10.0,
                ),
                self._ok_response(
                    response_text=(
                        "```python\n"
                        "def two_fer(name='you'):\n"
                        "    return f'One for {name}, one for me.'\n"
                        "```"
                    ),
                    tokens_in=30,
                    tokens_out=12,
                    cached_input_tokens=7,
                    cost_usd=0.2,
                    latency_ms=20.0,
                ),
            ]
        )

        class DummyAdapter:
            def run(self, req):
                return next(responses)

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_exercism", return_value=[exercism_item]), \
             patch.object(self.bench, "_build_adapter", return_value=DummyAdapter()), \
             patch.object(
                 self.bench,
                 "_run_exercism_pytest",
                 side_effect=[(False, "first failure"), (True, "all good")],
             ), \
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
             patch.object(self.bench, "_print_summary"):
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "exercism",
                    "--condition",
                    "one_shot",
                    "--model",
                    "openai-tc:gpt-5.4",
                    "--difficulty",
                    "1",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        self.assertEqual(written_records[0].cached_input_tokens, 12)

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

    def test_swebench_record_preserves_cached_input_tokens(self):
        written_records = []
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )

        def fake_run_unified_swebench(**_kwargs):
            return self._ok_response(
                response_text="done",
                tokens_in=150,
                tokens_out=40,
                cached_input_tokens=23,
                metadata={
                    "docker": True,
                    "patch": "diff --git",
                    "changed_files": ["demo.py"],
                },
            )

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
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
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        self.assertEqual(written_records[0].cached_input_tokens, 23)
        self.assertIs(written_records[0].metadata["paper_compliant"], False)
        self.assertIs(written_records[0].metadata["hints_included"], False)
        self.assertIs(written_records[0].metadata["paper_compliance_instruction"], False)
        self.assertIs(written_records[0].metadata["web_search"], False)

    def test_swebench_prewarm_runs_before_generation(self):
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

        def fake_prewarm(items, **kwargs):
            events.append(("prewarm", [item.instance_id for item in items], kwargs))
            return SimpleNamespace(unique_env_count=1, built_env_count=1)

        def fake_run_unified_swebench(**kwargs):
            events.append(("run", kwargs["condition"]))
            return self._ok_response(metadata={"docker": True, "patch": "", "changed_files": []})

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "prewarm_swebench_env_images", side_effect=fake_prewarm), \
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
                    "lite",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--n",
                    "1",
                    "--prewarm-envs",
                    "--env-build-workers",
                    "3",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            events,
            [
                (
                    "prewarm",
                    ["django__django-1"],
                    {
                        "instance_image_tag": "latest",
                        "env_image_tag": "latest",
                        "max_workers": 3,
                    },
                ),
                ("run", "codex"),
            ],
        )
        self.assertIn("Prewarm summary: 1 unique env image, 1 built this run", result.output)

    def test_swebench_prewarm_only_skips_generation(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(
                 self.bench,
                 "prewarm_swebench_env_images",
                 return_value=SimpleNamespace(unique_env_count=1, built_env_count=0),
             ) as prewarm_mock, \
             patch.object(self.bench, "run_unified_swebench") as run_mock, \
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
                    "--prewarm-envs",
                    "--prewarm-only",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        prewarm_mock.assert_called_once()
        run_mock.assert_not_called()
        self.assertIn("Prewarm summary: 1 unique env image, 0 built this run", result.output)

    def test_swebench_env_image_cache_wires_to_prewarm_and_generation(self):
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

        def fake_prewarm(items, **kwargs):
            events.append(("prewarm", kwargs))
            return SimpleNamespace(unique_env_count=1, built_env_count=1)

        def fake_run_unified_swebench(**kwargs):
            events.append(("run", kwargs))
            return self._ok_response(metadata={"docker": True, "patch": "", "changed_files": []})

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "prewarm_swebench_env_images", side_effect=fake_prewarm), \
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
                    "lite",
                    "--condition",
                    "codex",
                    "--codex-model",
                    "codex",
                    "--n",
                    "1",
                    "--prewarm-envs",
                    "--env-image-cache-repository",
                    "us-central1-docker.pkg.dev/project/cache/sweb-env",
                    "--env-image-cache-read-only",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][0], "prewarm")
        self.assertEqual(events[0][1]["instance_image_tag"], "latest")
        self.assertEqual(events[0][1]["env_image_tag"], "latest")
        self.assertEqual(events[0][1]["max_workers"], 1)
        self.assertEqual(
            events[0][1]["env_image_cache"].repository,
            "us-central1-docker.pkg.dev/project/cache/sweb-env",
        )
        self.assertFalse(events[0][1]["env_image_cache"].write_enabled)
        self.assertEqual(events[1][0], "run")
        self.assertEqual(
            events[1][1]["env_image_cache"].repository,
            "us-central1-docker.pkg.dev/project/cache/sweb-env",
        )
        self.assertFalse(events[1][1]["env_image_cache"].write_enabled)

    def test_swebench_env_image_cache_rejects_non_docker_backend(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]):
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
                    "--backend",
                    "singularity",
                    "--env-image-cache-repository",
                    "us-central1-docker.pkg.dev/project/cache/sweb-env",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("only supported with the docker backend", result.output)

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

    def test_swebench_container_setup_error_surfaces_log_tail_in_record(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )
        written_records = []
        error = self.bench.SWEBenchContainerSetupError(
            stage="build_container",
            detail="docker build exited with code 1",
            log_path="/tmp/prepare_image.log",
            log_tail="pip install failed",
        )

        def fake_append_jsonl(_path, record, **_kwargs):
            written_records.append(record)

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "run_unified_swebench", side_effect=error), \
             patch.object(self.bench, "append_jsonl", side_effect=fake_append_jsonl), \
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
                    "--web-search",
                    "--n",
                    "1",
                    "--output",
                    str(Path(tmp_dir) / "out.jsonl"),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(written_records), 1)
        record = written_records[0]
        self.assertEqual(record.error_type, "container_error")
        self.assertIn("docker build exited with code 1", record.error_message)
        self.assertIn("pip install failed", record.error_message)
        self.assertEqual(record.metadata["container_error_stage"], "build_container")
        self.assertEqual(record.metadata["container_error_log_path"], "/tmp/prepare_image.log")
        self.assertEqual(record.metadata["container_error_log_tail"], "pip install failed")
        self.assertIs(record.metadata["web_search"], True)

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

    def test_swebench_resume_skips_completed_items_and_rebuilds_predictions(self):
        lite_items = [
            SimpleNamespace(
                instance_id="django__django-1",
                repo="django/django",
                base_commit="abc123",
                problem_statement="Fix it",
                hints_text="",
                metadata={},
                raw_instance={},
            ),
            SimpleNamespace(
                instance_id="django__django-2",
                repo="django/django",
                base_commit="def456",
                problem_statement="Fix it again",
                hints_text="",
                metadata={},
                raw_instance={},
            ),
        ]
        run_calls = []
        prediction_batches = []

        def fake_run_unified_swebench(**kwargs):
            run_calls.append(kwargs["item"].instance_id)
            return self._ok_response(
                response_text="done",
                metadata={
                    "docker": True,
                    "patch": f"patch-for-{kwargs['item'].instance_id}",
                    "changed_files": ["demo.py"],
                },
            )

        def fake_write_predictions(results, _path, model_name=None):
            prediction_batches.append((results, model_name))

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=lite_items), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "write_predictions_jsonl", side_effect=fake_write_predictions), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            output_path = Path(tmp_dir) / "out.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "run_id": "resume-run",
                        "item_id": "django__django-1",
                        "dataset": "lite",
                        "condition": "codex",
                        "name": None,
                        "patch": "existing-patch",
                    }
                ) + "\n",
                encoding="utf-8",
            )
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
                    "--run-id",
                    "resume-run",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_calls, ["django__django-2"])
        self.assertIn("Resume: 1/2 items already recorded; running 1 pending", result.output)
        self.assertEqual(len(prediction_batches), 1)
        predictions, model_name = prediction_batches[0]
        self.assertEqual(model_name, "codex")
        self.assertEqual([prediction.instance_id for prediction in predictions], ["django__django-1", "django__django-2"])
        self.assertEqual([prediction.patch for prediction in predictions], ["existing-patch", "patch-for-django__django-2"])

    def test_swebench_paper_compliant_resume_rejects_non_compliant_rows(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(self.bench, "run_unified_swebench") as run_mock, \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            output_path = Path(tmp_dir) / "out.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "run_id": "resume-run",
                        "item_id": "django__django-1",
                        "dataset": "lite",
                        "condition": "codex",
                        "name": None,
                        "patch": "existing-patch",
                        "metadata": {
                            "paper_compliant": False,
                            "hints_included": True,
                            "paper_compliance_instruction": False,
                            "web_search": False,
                        },
                    }
                ) + "\n",
                encoding="utf-8",
            )
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
                    "--run-id",
                    "resume-run",
                    "--paper-compliant",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Cannot resume a paper-compliant SWE-bench run", result.output)
        run_mock.assert_not_called()

    def test_swebench_resume_is_scoped_by_condition(self):
        lite_item = SimpleNamespace(
            instance_id="django__django-1",
            repo="django/django",
            base_commit="abc123",
            problem_statement="Fix it",
            hints_text="",
            metadata={},
            raw_instance={},
        )
        run_calls = []

        def fake_run_unified_swebench(**kwargs):
            run_calls.append(kwargs["condition"])
            return self._ok_response(
                response_text="done",
                metadata={
                    "docker": True,
                    "patch": "spell-patch",
                    "changed_files": ["demo.py"],
                },
            )

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=[lite_item]), \
             patch.object(
                 self.bench,
                 "_resolve_spell_agent_or_die",
                 return_value=SimpleNamespace(
                     agent_file="config/agents/cli.agent.edn",
                     spell_mode="tc",
                     source="test",
                     policy_key="swebench",
                 ),
             ), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "write_predictions_jsonl"), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            output_path = Path(tmp_dir) / "out.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "run_id": "resume-run",
                        "item_id": "django__django-1",
                        "dataset": "lite",
                        "condition": "codex",
                        "name": None,
                        "patch": "codex-patch",
                    }
                ) + "\n",
                encoding="utf-8",
            )
            result = self.runner.invoke(
                self.bench.cli,
                [
                    "swebench",
                    "--dataset",
                    "lite",
                    "--condition",
                    "spell",
                    "--model",
                    "openai-tc:gpt-5.4",
                    "--run-id",
                    "resume-run",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_calls, ["spell"])
        self.assertNotIn("Resume:", result.output)

    def test_swebench_resume_is_scoped_by_dataset(self):
        lite_items = [
            SimpleNamespace(
                instance_id="django__django-1",
                repo="django/django",
                base_commit="abc123",
                problem_statement="Fix it",
                hints_text="",
                metadata={},
                raw_instance={},
            ),
            SimpleNamespace(
                instance_id="django__django-2",
                repo="django/django",
                base_commit="def456",
                problem_statement="Fix it again",
                hints_text="",
                metadata={},
                raw_instance={},
            ),
        ]
        run_calls = []
        prediction_batches = []

        def fake_run_unified_swebench(**kwargs):
            run_calls.append(kwargs["item"].instance_id)
            return self._ok_response(
                response_text="done",
                metadata={
                    "docker": True,
                    "patch": f"patch-for-{kwargs['item'].instance_id}",
                    "changed_files": ["demo.py"],
                },
            )

        def fake_write_predictions(results, _path, model_name=None):
            prediction_batches.append((results, model_name))

        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench, "load_swebench_lite", return_value=lite_items), \
             patch.object(self.bench, "run_unified_swebench", side_effect=fake_run_unified_swebench), \
             patch.object(self.bench, "write_predictions_jsonl", side_effect=fake_write_predictions), \
             patch.object(self.bench, "_resolve_trace_dir", return_value=None):
            output_path = Path(tmp_dir) / "out.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "run_id": "resume-run",
                        "item_id": "django__django-1",
                        "dataset": "mini",
                        "condition": "codex",
                        "name": None,
                        "patch": "mini-patch",
                    }
                ) + "\n",
                encoding="utf-8",
            )
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
                    "--run-id",
                    "resume-run",
                    "--output",
                    str(output_path),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(run_calls, ["django__django-1", "django__django-2"])
        self.assertNotIn("Resume:", result.output)
        self.assertEqual(len(prediction_batches), 1)
        predictions, model_name = prediction_batches[0]
        self.assertEqual(model_name, "codex")
        self.assertEqual([prediction.instance_id for prediction in predictions], ["django__django-1", "django__django-2"])
        self.assertEqual([prediction.patch for prediction in predictions], ["patch-for-django__django-1", "patch-for-django__django-2"])

    def test_print_summary_tolerates_truncated_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp_dir, \
             patch.object(self.bench.click, "echo") as echo_mock:
            output_path = Path(tmp_dir) / "out.jsonl"
            output_path.write_text(
                json.dumps(
                    {
                        "run_id": "resume-run",
                        "item_id": "django__django-1",
                        "condition": "codex",
                        "name": None,
                        "correct": True,
                        "error_message": None,
                    }
                ) + "\n" + '{"run_id": "resume-run"',
                encoding="utf-8",
            )

            self.bench._print_summary(output_path, "resume-run", "codex")

        echoed = "\n".join(call.args[0] for call in echo_mock.call_args_list)
        self.assertIn("Summary (codex): total=1 errors=0", echoed)
        self.assertIn("accuracy=100.0%", echoed)

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
        self.assertIn("--prewarm-envs / --no-prewarm-envs", swebench_help.output)
        self.assertIn("--env-build-workers INTEGER", swebench_help.output)
        self.assertIn("--env-image-cache-repository TEXT", swebench_help.output)
        self.assertIn("--env-image-cache-read-only / --no-env-image-cache-read-only", swebench_help.output)
        self.assertIn("--prewarm-only", swebench_help.output)
        self.assertIn("--paper-compliant / --no-paper-compliant", swebench_help.output)
        self.assertNotIn("--unified", featurebench_help.output)
        self.assertNotIn("--unified", exercism_help.output)


if __name__ == "__main__":
    unittest.main()
