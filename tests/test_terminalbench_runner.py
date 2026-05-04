from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))
for parent in BENCHMARK_ROOT.parents:
    if (parent / "spell_benchmark_client.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src.terminalbench_runner import (
    TERMINALBENCH_DATASET,
    TERMINALBENCH_HEAD_DATASET,
    TERMINALBENCH_HEAD_REGISTRY,
    TERMINALBENCH_OLD_CORE_DATASET,
    build_terminalbench_command,
    parse_terminalbench_conditions,
    prepare_terminalbench_env,
    resolve_terminalbench_dataset,
    terminalbench_condition_run_id,
    terminalbench_records_from_run,
    terminalbench_run_dir,
)


class TerminalBenchRunnerTests(unittest.TestCase):
    def test_parse_terminalbench_conditions_expands_all_and_dedupes(self) -> None:
        self.assertEqual(
            parse_terminalbench_conditions(["all"]),
            ["spell", "claude_code", "codex"],
        )
        self.assertEqual(
            parse_terminalbench_conditions(["spell", "codex", "spell"]),
            ["spell", "codex"],
        )
        self.assertEqual(
            parse_terminalbench_conditions(["swe_agent"]),
            ["swe_agent"],
        )

    def test_terminalbench_dataset_aliases_keep_old_core_default_and_head_aliases(self) -> None:
        self.assertEqual(TERMINALBENCH_DATASET, TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset(None), TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("old-core"), TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("core-0.1.1"), TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("1.1"), TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("2.0"), TERMINALBENCH_HEAD_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("head"), TERMINALBENCH_HEAD_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("latest"), TERMINALBENCH_HEAD_DATASET)

    def test_terminalbench_dataset_resolution_preserves_raw_passthrough(self) -> None:
        self.assertEqual(resolve_terminalbench_dataset("terminal-bench-core==0.1.1"), TERMINALBENCH_OLD_CORE_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("terminal-bench-core"), "terminal-bench-core")
        self.assertEqual(resolve_terminalbench_dataset("terminal-bench-core==head"), TERMINALBENCH_HEAD_DATASET)
        self.assertEqual(resolve_terminalbench_dataset("custom-dataset==2026.04"), "custom-dataset==2026.04")

    @patch.dict("os.environ", {}, clear=True)
    def test_prepare_terminalbench_env_enables_buildkit_by_default(self) -> None:
        env = prepare_terminalbench_env(BENCHMARK_ROOT)

        self.assertEqual(env["DOCKER_BUILDKIT"], "1")

    @patch.dict("os.environ", {"DOCKER_BUILDKIT": "0"}, clear=True)
    def test_prepare_terminalbench_env_respects_explicit_buildkit_override(self) -> None:
        env = prepare_terminalbench_env(BENCHMARK_ROOT)

        self.assertEqual(env["DOCKER_BUILDKIT"], "0")

    def test_build_terminalbench_command_threads_spell_kwargs(self) -> None:
        cmd = build_terminalbench_command(
            condition="spell",
            run_id="spell_demo",
            model="codex-tc:gpt-5.3-codex",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world", "csv-to-parquet"),
            n_tasks=2,
            n_concurrent=1,
            budget=7.5,
            cleanup=False,
            reasoning_effort="medium",
            test_timeout_sec=120,
            agent_file="config/agents/io-tc.agent.edn",
            trailing="'(!describe io)",
            trace=False,
        )

        rendered = " ".join(cmd)
        self.assertIn("--agent-import-path spell_agent:SpellAgent", rendered)
        self.assertIn("--no-cleanup", rendered)
        self.assertIn("--global-agent-timeout-sec 3600", rendered)
        self.assertIn("--global-test-timeout-sec 120", rendered)
        self.assertIn("-t hello-world", rendered)
        self.assertIn("-t csv-to-parquet", rendered)
        self.assertIn("--n-tasks 2", rendered)
        self.assertIn("-k budget=7.5", rendered)
        self.assertIn("-k reasoning_effort=medium", rendered)
        self.assertIn("-k agent_file=config/agents/io-tc.agent.edn", rendered)
        self.assertIn("-k trailing='(!describe io)", rendered)
        self.assertIn("-k trace=false", rendered)

    def test_build_terminalbench_command_resolves_dataset_alias(self) -> None:
        cmd = build_terminalbench_command(
            condition="codex",
            run_id="codex_demo",
            model="openai/gpt-5.3-codex",
            dataset="2.0",
            task_ids=(),
            n_tasks=1,
            n_concurrent=1,
            budget=7.5,
            cleanup=True,
            reasoning_effort="high",
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        self.assertEqual(cmd[cmd.index("--dataset") + 1], TERMINALBENCH_HEAD_DATASET)
        self.assertEqual(cmd[cmd.index("--local-registry-path") + 1], str(TERMINALBENCH_HEAD_REGISTRY))

    def test_build_terminalbench_command_preserves_raw_head_passthrough(self) -> None:
        cmd = build_terminalbench_command(
            condition="codex",
            run_id="codex_demo",
            model="openai/gpt-5.3-codex",
            dataset="terminal-bench-core==head",
            task_ids=(),
            n_tasks=1,
            n_concurrent=1,
            budget=7.5,
            cleanup=True,
            reasoning_effort="high",
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        self.assertEqual(cmd[cmd.index("--dataset") + 1], TERMINALBENCH_HEAD_DATASET)
        self.assertNotIn("--local-registry-path", cmd)

    def test_build_terminalbench_command_threads_claude_timeout_and_effort(self) -> None:
        claude_cmd = build_terminalbench_command(
            condition="claude_code",
            run_id="claude_demo",
            model="anthropic/claude-opus-4-6",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world",),
            n_tasks=1,
            n_concurrent=1,
            budget=7.5,
            cleanup=False,
            reasoning_effort="medium",
            agent_timeout_sec=2400,
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        rendered = " ".join(claude_cmd)
        self.assertIn("--global-agent-timeout-sec 2400", rendered)
        self.assertIn("-k agent_timeout=2400", rendered)
        self.assertIn("-k reasoning_effort=medium", rendered)

    def test_build_terminalbench_command_threads_codex_effort(self) -> None:
        codex_cmd = build_terminalbench_command(
            condition="codex",
            run_id="codex_demo",
            model="openai/gpt-5.3-codex",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world",),
            n_tasks=1,
            n_concurrent=1,
            budget=7.5,
            cleanup=False,
            reasoning_effort="medium",
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        self.assertIn("-k reasoning_effort=medium", " ".join(codex_cmd))

    def test_build_terminalbench_command_threads_swe_agent_model_without_reasoning_effort(self) -> None:
        cmd = build_terminalbench_command(
            condition="swe_agent",
            run_id="swe_agent_demo",
            model="fireworks_ai/accounts/fireworks/models/glm-5p1",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world",),
            n_tasks=1,
            n_concurrent=1,
            budget=2.5,
            cleanup=False,
            reasoning_effort="high",
            agent_timeout_sec=1800,
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        rendered = " ".join(cmd)
        self.assertIn("--agent-import-path swe_agent:MiniSweAgent", rendered)
        self.assertIn("--model fireworks_ai/accounts/fireworks/models/glm-5p1", rendered)
        self.assertIn("-k budget=2.5", rendered)
        self.assertIn("-k agent_timeout=1800", rendered)
        self.assertNotIn("reasoning_effort", rendered)

    def test_build_terminalbench_command_threads_deepseek_swe_agent_reasoning_effort(self) -> None:
        cmd = build_terminalbench_command(
            condition="swe_agent",
            run_id="swe_agent_demo",
            model="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world",),
            n_tasks=1,
            n_concurrent=1,
            budget=2.5,
            cleanup=False,
            reasoning_effort="high",
            agent_timeout_sec=1800,
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        rendered = " ".join(cmd)
        self.assertIn("--model fireworks_ai/accounts/fireworks/models/deepseek-v4-pro", rendered)
        self.assertIn("-k reasoning_effort=high", rendered)

    def test_terminalbench_dry_run_threads_test_timeout_cli_option(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--condition",
                "spell",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--run-id",
                "terminalbench_demo",
                "--test-timeout-sec",
                "120",
                "--dry-run",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--global-agent-timeout-sec", result.output)
        self.assertIn("--global-test-timeout-sec", result.output)
        self.assertIn("120", result.output)

    def test_terminalbench_dry_run_omits_test_timeout_by_default(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--condition",
                "spell",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--run-id",
                "terminalbench_demo",
                "--dry-run",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--global-agent-timeout-sec", result.output)
        self.assertNotIn("--global-test-timeout-sec", result.output)

    def test_terminalbench_dry_run_threads_swe_agent_model(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--condition",
                "swe_agent",
                "--swe-agent-model",
                "fireworks_ai/accounts/fireworks/models/glm-5p1",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--run-id",
                "terminalbench_demo",
                "--dry-run",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"conditions": [\n    "swe_agent"\n  ]', result.output)
        self.assertIn("swe_agent:MiniSweAgent", result.output)
        self.assertIn("fireworks_ai/accounts/fireworks/models/glm-5p1", result.output)
        self.assertIn("swe_agent_terminalbench_demo", result.output)
        self.assertNotIn("reasoning_effort=high", result.output)

    def test_terminalbench_dry_run_threads_deepseek_swe_agent_reasoning_effort(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--condition",
                "swe_agent",
                "--swe-agent-model",
                "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--run-id",
                "terminalbench_demo",
                "--dry-run",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("fireworks_ai/accounts/fireworks/models/deepseek-v4-pro", result.output)
        self.assertIn("reasoning_effort=high", result.output)

    def test_terminalbench_swe_agent_requires_model(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--condition",
                "swe_agent",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--dry-run",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--swe-agent-model is required", result.output)

    def test_terminalbench_default_all_does_not_include_swe_agent(self) -> None:
        import bench

        result = CliRunner().invoke(
            bench.cli,
            [
                "terminalbench",
                "--dataset",
                "old-core",
                "-t",
                "hello-world",
                "--run-id",
                "terminalbench_demo",
                "--dry-run",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"conditions": [\n    "spell",\n    "claude_code",\n    "codex"\n  ]', result.output)
        self.assertNotIn("swe_agent:MiniSweAgent", result.output)

    def test_build_terminalbench_command_includes_supported_anthropic_spell_effort(self) -> None:
        cmd = build_terminalbench_command(
            condition="spell",
            run_id="spell_demo",
            model="anthropic-tc:opus46",
            dataset="terminal-bench-core==0.1.1",
            task_ids=("hello-world",),
            n_tasks=1,
            n_concurrent=1,
            budget=7.5,
            cleanup=False,
            reasoning_effort="medium",
            test_timeout_sec=None,
            agent_file=None,
            trailing=None,
            trace=False,
        )

        self.assertIn("-k reasoning_effort=medium", " ".join(cmd))

    def test_terminalbench_run_id_sanitizes_name(self) -> None:
        self.assertEqual(
            terminalbench_condition_run_id("terminalbench_demo", "spell", name="io/tc"),
            "spell_io_tc_terminalbench_demo",
        )
        self.assertEqual(
            terminalbench_run_dir("terminalbench_demo", "spell", name="io/tc"),
            Path("results") / "terminal-bench" / "spell_io_tc_terminalbench_demo",
        )

    def test_terminalbench_records_copy_spell_traces_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "spell_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.spell_demo"
            agent_log_dir = trial_dir / "agent-logs"
            trace_source = agent_log_dir / "spell-trace" / "trace-1"
            trace_source.mkdir(parents=True, exist_ok=True)
            (trace_source / "tree.txt").write_text("trace", encoding="utf-8")
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "cost_usd": 0.42,
                        "total_input_tokens": 100,
                        "visible_output_tokens": 25,
                        "reasoning_output_tokens": 75,
                        "total_output_tokens": 100,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.spell_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                                "total_input_tokens": 100,
                                "total_output_tokens": 25,
                                "agent_started_at": "2026-03-27T12:00:00+00:00",
                                "agent_ended_at": "2026-03-27T12:00:02+00:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            host_trace_root = root / "traces" / "spell"
            with patch(
                "src.terminalbench_runner.benchmark_git_versions",
                return_value={
                    "spell": {"commit": "spell-sha", "branch": "main", "dirty": False},
                    "benchmarking": {"commit": "bench-sha", "branch": "pr-25", "dirty": False},
                },
            ):
                records = terminalbench_records_from_run(
                    run_dir=run_dir,
                    run_id="terminalbench_demo",
                    condition="spell",
                    dataset="terminal-bench-core==0.1.1",
                    model_spec="codex-tc:gpt-5.3-codex",
                    name="demo",
                    agent_file="/tmp/agent.edn",
                    reasoning_effort="medium",
                    host_trace_root=host_trace_root,
                )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "resolved")
            self.assertTrue(record.correct)
            self.assertEqual(record.cost_usd, 0.42)
            self.assertIsNone(record.cached_input_tokens)
            self.assertEqual(record.latency_ms, 2000.0)
            self.assertEqual(record.reasoning_effort, "medium")
            self.assertEqual(record.visible_output_tokens, 25)
            self.assertEqual(record.reasoning_output_tokens, 75)
            self.assertEqual(
                record.metadata["versions"],
                {
                    "spell": {"commit": "spell-sha", "branch": "main", "dirty": False},
                    "benchmarking": {"commit": "bench-sha", "branch": "pr-25", "dirty": False},
                },
            )
            copied_trace = Path(record.metadata["trace_dir"])
            self.assertTrue((copied_trace / "trace-1" / "tree.txt").exists())

    def test_terminalbench_records_store_resolved_dataset_and_alias_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "results" / "codex_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.codex_demo"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.codex_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="codex",
                dataset="2.0",
                model_spec="openai/gpt-5.3-codex",
                name=None,
                agent_file=None,
                requested_dataset="2.0",
            )

            self.assertEqual(records[0].dataset, TERMINALBENCH_HEAD_DATASET)
            self.assertEqual(records[0].metadata["requested_dataset"], "2.0")
            self.assertEqual(records[0].metadata["resolved_dataset"], TERMINALBENCH_HEAD_DATASET)

    def test_terminalbench_records_preserve_swe_agent_litellm_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "results" / "swe_agent_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.swe_agent_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps({"cost_usd": 0.12, "cost_source": "reported"}),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.swe_agent_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="swe_agent",
                dataset="old-core",
                model_spec="fireworks_ai/accounts/fireworks/models/glm-5p1",
                name=None,
                agent_file=None,
                requested_dataset="old-core",
                reasoning_effort="high",
            )

            self.assertEqual(records[0].condition, "swe_agent")
            self.assertEqual(records[0].provider, "fireworks_ai")
            self.assertEqual(records[0].model, "fireworks_ai/accounts/fireworks/models/glm-5p1")
            self.assertIsNone(records[0].reasoning_effort)
            self.assertEqual(records[0].cost_usd, 0.12)
            self.assertEqual(records[0].metadata["cost_source"], "reported")

    def test_terminalbench_records_deepseek_swe_agent_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "results" / "swe_agent_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.swe_agent_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True)
            (agent_log_dir / "usage.json").write_text(
                json.dumps({"cost_usd": 0.12, "cost_source": "reported"}),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.swe_agent_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="swe_agent",
                dataset="old-core",
                model_spec="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro",
                name=None,
                agent_file=None,
                requested_dataset="old-core",
                reasoning_effort="high",
            )

            self.assertEqual(records[0].reasoning_effort, "high")

    def test_terminalbench_records_extract_recovery_caveat_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "results" / "codex_demo"
            trial_dir = run_dir / "swe-bench-fsspec" / "swe-bench-fsspec.1-of-1.codex_demo"
            agent_log_dir = trial_dir / "agent-logs"
            sessions_dir = trial_dir / "sessions"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            sessions_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "touched_test_files": ["fsspec/implementations/tests/test_dirfs.py"],
                        "test_patch_collision_recovery_enabled": True,
                    }
                ),
                encoding="utf-8",
            )
            (sessions_dir / "tests.log").write_text(
                "SPELL_TB_TEST_PATCH_COLLISION_RECOVERED path=fsspec/implementations/tests/test_dirfs.py\n",
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "swe-bench-fsspec.1-of-1.codex_demo",
                                "task_id": "swe-bench-fsspec",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="codex",
                dataset="2.0",
                model_spec="openai/gpt-5.4",
                name=None,
                agent_file=None,
            )

            metadata = records[0].metadata
            self.assertTrue(metadata["test_patch_collision_recovered"])
            self.assertEqual(metadata["touched_test_files"], ["fsspec/implementations/tests/test_dirfs.py"])
            self.assertEqual(metadata["test_patch_collision_paths"], ["fsspec/implementations/tests/test_dirfs.py"])
            self.assertIn("official test patch collided", metadata["score_caveat"])

    def test_terminalbench_records_recover_spell_visible_and_reasoning_from_agent_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "spell_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.spell_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "output_tokens": 1123,
                        "reasoning_output_tokens": 17869,
                        "total_output_tokens": 18992,
                        "cost_usd": 0.324007,
                        "cost_source": "estimated_with_reasoning",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.spell_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                                "total_input_tokens": 39382,
                                "total_output_tokens": 1123,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="spell",
                dataset="terminal-bench-core==0.1.1",
                model_spec="anthropic-tc:opus46",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "resolved")
            self.assertTrue(record.correct)
            self.assertEqual(record.visible_output_tokens, 1123)
            self.assertEqual(record.reasoning_output_tokens, 17869)
            self.assertEqual(record.cost_usd, 0.324007)
            self.assertEqual(record.metadata["tb_agent_result"]["total_output_tokens"], 18992)

    def test_terminalbench_records_recover_spell_visible_and_reasoning_tokens_from_legacy_agent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "spell_demo"
            trial_dir = run_dir / "conda-env-conflict-resolution" / (
                "conda-env-conflict-resolution.1-of-1.spell_demo"
            )
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "cost_usd": 0.324007,
                        "total_input_tokens": 39_382,
                        "output_tokens": 1_123,
                        "reasoning_output_tokens": 17_869,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "conda-env-conflict-resolution.1-of-1.spell_demo",
                                "task_id": "conda-env-conflict-resolution",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                                "total_input_tokens": 39_382,
                                "total_output_tokens": 1_123,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="spell",
                dataset="terminal-bench-core==0.1.1",
                model_spec="codex-tc:gpt-5.4",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.visible_output_tokens, 1_123)
            self.assertEqual(record.reasoning_output_tokens, 17_869)
            self.assertAlmostEqual(record.cost_usd or 0.0, 0.324007, places=6)

    def test_terminalbench_failures_become_unscored_error_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "spell_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.spell_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.spell_demo",
                                "task_id": "hello-world",
                                "is_resolved": False,
                                "failure_mode": "agent_timeout",
                                "parser_results": {"check": "failed"},
                                "total_input_tokens": 100,
                                "total_output_tokens": 25,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="spell",
                dataset="terminal-bench-core==0.1.1",
                model_spec="anthropic-tc:opus46",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "timeout")
            self.assertIsNone(record.correct)
            self.assertEqual(record.error_type, "timeout")
            self.assertEqual(record.error_message, "Terminal-Bench failure: agent_timeout")

    def test_terminalbench_records_use_codex_agent_usage_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "codex_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.codex_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 1_000_000,
                        "cached_input_tokens": 250_000,
                        "visible_output_tokens": 100_000,
                        "total_output_tokens": 100_000,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.codex_demo",
                                "task_id": "hello-world",
                                "is_resolved": False,
                                "failure_mode": "unset",
                                "parser_results": {"check": "failed"},
                                "total_input_tokens": 1_000_000,
                                "total_output_tokens": 100_000,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="codex",
                dataset="terminal-bench-core==0.1.1",
                model_spec="openai/gpt-5.3-codex",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "unresolved")
            self.assertAlmostEqual(record.cost_usd or 0.0, 2.75625, places=6)
            self.assertEqual(record.cached_input_tokens, 250_000)
            self.assertEqual(record.metadata["resolved_model"], "gpt-5.3-codex")
            self.assertEqual(record.metadata["cost_source"], "estimated")

    def test_terminalbench_records_preserve_claude_cache_write_and_reprice_underreported_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "claude_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.claude_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 175_000,
                        "cached_input_tokens": 25_000,
                        "cache_write_input_tokens": 50_000,
                        "visible_output_tokens": 10_000,
                        "total_input_tokens": 175_000,
                        "total_output_tokens": 10_000,
                        "cost_usd": 0.4575,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.claude_demo",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                                "total_input_tokens": 175_000,
                                "total_output_tokens": 10_000,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="claude_code",
                dataset="terminal-bench-core==0.1.1",
                model_spec="anthropic/claude-sonnet-4-6",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "resolved")
            self.assertEqual(record.cached_input_tokens, 25_000)
            self.assertEqual(record.cache_write_input_tokens, 50_000)
            self.assertAlmostEqual(record.cost_usd or 0.0, 0.645, places=6)
            self.assertEqual(record.metadata["cost_source"], "estimated")

    def test_terminalbench_records_skip_cost_estimate_when_shared_pricing_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "codex_demo"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.codex_demo"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 1_000_000,
                        "cached_input_tokens": 250_000,
                        "output_tokens": 100_000,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.codex_demo",
                                "task_id": "hello-world",
                                "is_resolved": False,
                                "failure_mode": "unset",
                                "parser_results": {"check": "failed"},
                                "total_input_tokens": 1_000_000,
                                "total_output_tokens": 100_000,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.usage_accounting.load_pricing_table", side_effect=FileNotFoundError):
                records = terminalbench_records_from_run(
                    run_dir=run_dir,
                    run_id="terminalbench_demo",
                    condition="codex",
                    dataset="terminal-bench-core==0.1.1",
                    model_spec="openai/gpt-5.3-codex",
                    name=None,
                    agent_file=None,
                )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "unresolved")
            self.assertIsNone(record.cost_usd)
            self.assertEqual(record.cached_input_tokens, 250_000)
            self.assertNotIn("cost_source", record.metadata)

    def test_terminalbench_records_preserve_codex_agent_reasoning_cost_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "codex_reasoning"
            trial_dir = run_dir / "hello-world" / "hello-world.1-of-1.codex_reasoning"
            agent_log_dir = trial_dir / "agent-logs"
            agent_log_dir.mkdir(parents=True, exist_ok=True)
            (agent_log_dir / "result.json").write_text(
                json.dumps(
                    {
                        "input_tokens": 900,
                        "cached_input_tokens": 100,
                        "visible_output_tokens": 200,
                        "reasoning_output_tokens": 350,
                        "total_input_tokens": 1_000,
                        "total_output_tokens": 550,
                        "cost_usd": 0.009775,
                        "cost_source": "estimated_with_reasoning",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "trial_name": "hello-world.1-of-1.codex_reasoning",
                                "task_id": "hello-world",
                                "is_resolved": True,
                                "failure_mode": "none",
                                "parser_results": {"check": "passed"},
                                "total_input_tokens": 1_000,
                                "total_output_tokens": 200,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_reasoning",
                condition="codex",
                dataset="terminal-bench-core==0.1.1",
                model_spec="openai/gpt-5.4",
                name=None,
                agent_file=None,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "resolved")
            self.assertAlmostEqual(record.cost_usd or 0.0, 0.009775, places=8)
            self.assertEqual(record.visible_output_tokens, 200)
            self.assertEqual(record.reasoning_output_tokens, 350)
            self.assertEqual(record.metadata["cost_source"], "estimated_with_reasoning")
            self.assertEqual(record.metadata["reasoning_output_tokens"], 350)


if __name__ == "__main__":
    unittest.main()
