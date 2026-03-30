from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.terminalbench_runner import (
    build_terminalbench_command,
    parse_terminalbench_conditions,
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
            agent_file="config/agents/io-acm-tc.agent.edn",
            trailing="'(!describe io)",
            trace=False,
        )

        rendered = " ".join(cmd)
        self.assertIn("--agent-import-path spell_agent:SpellAgent", rendered)
        self.assertIn("--no-cleanup", rendered)
        self.assertIn("-t hello-world", rendered)
        self.assertIn("-t csv-to-parquet", rendered)
        self.assertIn("--n-tasks 2", rendered)
        self.assertIn("-k budget=7.5", rendered)
        self.assertIn("-k reasoning_effort=medium", rendered)
        self.assertIn("-k agent_file=config/agents/io-acm-tc.agent.edn", rendered)
        self.assertIn("-k trailing='(!describe io)", rendered)
        self.assertIn("-k trace=false", rendered)

    def test_terminalbench_run_id_sanitizes_name(self) -> None:
        self.assertEqual(
            terminalbench_condition_run_id("terminalbench_demo", "spell", name="io/acm"),
            "spell_io_acm_terminalbench_demo",
        )
        self.assertEqual(
            terminalbench_run_dir("terminalbench_demo", "spell", name="io/acm"),
            Path("results") / "terminal-bench" / "spell_io_acm_terminalbench_demo",
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
                json.dumps({"cost_usd": 0.42}),
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
            records = terminalbench_records_from_run(
                run_dir=run_dir,
                run_id="terminalbench_demo",
                condition="spell",
                dataset="terminal-bench-core==0.1.1",
                model_spec="anthropic-tc:opus46",
                name="demo",
                agent_file="/tmp/agent.edn",
                host_trace_root=host_trace_root,
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.status, "resolved")
            self.assertTrue(record.correct)
            self.assertEqual(record.cost_usd, 0.42)
            self.assertEqual(record.latency_ms, 2000.0)
            copied_trace = Path(record.metadata["trace_dir"])
            self.assertTrue((copied_trace / "trace-1" / "tree.txt").exists())

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
            self.assertAlmostEqual(record.cost_usd or 0.0, 3.15, places=6)
            self.assertEqual(record.metadata["resolved_model"], "gpt-5.3-codex")
            self.assertEqual(record.metadata["cost_source"], "estimated")


if __name__ == "__main__":
    unittest.main()
