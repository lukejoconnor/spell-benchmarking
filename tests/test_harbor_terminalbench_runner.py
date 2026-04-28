from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.harbor_terminalbench_runner import (
    DEFAULT_TIMEOUT_SUMMARY_FILENAME,
    HARBOR_CODEX_AGENT_IMPORT_PATH,
    HARBOR_CLI_COMMAND,
    HARBOR_TERMINALBENCH_DATASET,
    HarborTerminalBenchComplianceError,
    assert_harbor_submission_command_compliant,
    build_harbor_terminalbench_commands,
    write_default_timeout_summary,
)
from scripts.prepare_harbor_tb2_submission import main as prepare_submission_main
from scripts.prepare_harbor_tb2_submission import validate_config


class HarborTerminalBenchRunnerTests(unittest.TestCase):
    def test_submission_spell_command_is_harbor_tb2_and_has_no_banned_overrides(self) -> None:
        commands = build_harbor_terminalbench_commands(
            condition="spell",
            run_id="demo",
            model="openai-tc:gpt-5.4",
            codex_model="gpt-5.4",
            reasoning_effort="medium",
            trials=5,
            jobs_dir=Path("jobs"),
        )

        self.assertEqual(len(commands), 1)
        cmd = commands[0].command
        self.assertEqual(tuple(cmd[: len(HARBOR_CLI_COMMAND)]), HARBOR_CLI_COMMAND)
        self.assertEqual(cmd[len(HARBOR_CLI_COMMAND) : len(HARBOR_CLI_COMMAND) + 3], ["run", "-d", HARBOR_TERMINALBENCH_DATASET])
        self.assertIn("--agent-import-path", cmd)
        self.assertIn("harbor_spell_agent:HarborSpellAgent", cmd)
        self.assertIn("--agent-kwarg", cmd)
        self.assertIn("reasoning_effort=medium", cmd)
        self.assertNotIn("--timeout-multiplier", cmd)
        self.assertNotIn("--agent-timeout-multiplier", cmd)
        self.assertNotIn("--registry-path", cmd)
        self.assertNotIn("--task", cmd)
        self.assertTrue(commands[0].submission_compliant)

    def test_submission_codex_uses_custom_adapter(self) -> None:
        commands = build_harbor_terminalbench_commands(
            condition="codex",
            run_id="demo",
            model="openai-tc:gpt-5.4",
            codex_model="gpt-5.4",
            reasoning_effort="medium",
            trials=5,
            jobs_dir=Path("jobs"),
        )

        cmd = commands[0].command
        self.assertNotIn("-a", cmd)
        self.assertIn("--agent-import-path", cmd)
        self.assertIn(HARBOR_CODEX_AGENT_IMPORT_PATH, cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "openai/gpt-5.4")
        self.assertIn("reasoning_effort=medium", cmd)
        self.assertTrue(commands[0].submission_compliant)

    def test_dev_codex_also_uses_custom_adapter_with_medium_effort(self) -> None:
        commands = build_harbor_terminalbench_commands(
            condition="codex",
            run_id="demo",
            model="openai-tc:gpt-5.4",
            codex_model="gpt-5.4",
            reasoning_effort="medium",
            trials=5,
            jobs_dir=Path("jobs"),
            dev_smoke=True,
        )

        cmd = commands[0].command
        self.assertNotIn("-a", cmd)
        self.assertIn(HARBOR_CODEX_AGENT_IMPORT_PATH, cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "openai/gpt-5.4")
        self.assertIn("reasoning_effort=medium", cmd)

    def test_submission_rejects_k_less_than_five(self) -> None:
        with self.assertRaisesRegex(HarborTerminalBenchComplianceError, "k >= 5"):
            build_harbor_terminalbench_commands(
                condition="spell",
                run_id="demo",
                model="openai-tc:gpt-5.4",
                codex_model="gpt-5.4",
                reasoning_effort="medium",
                trials=1,
            )

    def test_submission_rejects_task_filters(self) -> None:
        with self.assertRaisesRegex(HarborTerminalBenchComplianceError, "full dataset"):
            build_harbor_terminalbench_commands(
                condition="spell",
                run_id="demo",
                model="openai-tc:gpt-5.4",
                codex_model="gpt-5.4",
                reasoning_effort="medium",
                trials=5,
                n_tasks=1,
            )

    def test_dev_smoke_allows_k_one_and_task_limit(self) -> None:
        commands = build_harbor_terminalbench_commands(
            condition="all",
            run_id="demo",
            model="openai-tc:gpt-5.4",
            codex_model="gpt-5.4",
            reasoning_effort="medium",
            trials=1,
            jobs_dir=Path("jobs"),
            dev_smoke=True,
            n_tasks=1,
        )

        self.assertEqual([item.condition for item in commands], ["spell", "codex"])
        self.assertTrue(all(not item.submission_compliant for item in commands))
        self.assertTrue(all("--n-tasks" in item.command for item in commands))

    def test_submission_rejects_agent_timeout_multiplier(self) -> None:
        with self.assertRaisesRegex(HarborTerminalBenchComplianceError, "agent-timeout-multiplier"):
            build_harbor_terminalbench_commands(
                condition="spell",
                run_id="demo",
                model="openai-tc:gpt-5.4",
                codex_model="gpt-5.4",
                reasoning_effort="medium",
                trials=5,
                agent_timeout_multiplier=2.0,
            )

    def test_dev_smoke_includes_agent_timeout_multiplier(self) -> None:
        commands = build_harbor_terminalbench_commands(
            condition="spell",
            run_id="demo",
            model="openai-tc:gpt-5.4",
            codex_model="gpt-5.4",
            reasoning_effort="medium",
            trials=1,
            jobs_dir=Path("jobs"),
            dev_smoke=True,
            agent_timeout_multiplier=2.5,
        )

        cmd = commands[0].command
        self.assertFalse(commands[0].submission_compliant)
        self.assertEqual(commands[0].agent_timeout_multiplier, 2.5)
        self.assertIn("--agent-timeout-multiplier", cmd)
        self.assertEqual(cmd[cmd.index("--agent-timeout-multiplier") + 1], "2.5")

    def test_command_guard_rejects_banned_submission_flag(self) -> None:
        cmd = [
            *HARBOR_CLI_COMMAND,
            "run",
            "-d",
            HARBOR_TERMINALBENCH_DATASET,
            "-k",
            "5",
            "--timeout-multiplier",
            "2",
        ]
        with self.assertRaisesRegex(HarborTerminalBenchComplianceError, "non-compliant"):
            assert_harbor_submission_command_compliant(cmd)

    def test_default_timeout_summary_annotates_synthetic_harbor_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = Path(tmp) / "jobs"
            jobs_dir.mkdir()
            (jobs_dir / "result.json").write_text(
                json.dumps({"stats": {"n_total_trials": 4}}),
                encoding="utf-8",
            )
            result_dir = jobs_dir / "tasks" / "task-a"
            result_dir.mkdir(parents=True)
            (result_dir / "result.json").write_text(
                json.dumps(
                    {
                        "trials": [
                            {
                                "task_id": "within",
                                "trial_id": "trial-1",
                                "verifier_result": {"rewards": {"reward": 1.0}},
                                "agent_execution": {
                                    "started_at": "2026-04-28T00:00:00+00:00",
                                    "finished_at": "2026-04-28T00:15:00+00:00",
                                },
                            },
                            {
                                "task_id": "after",
                                "trial_id": "trial-2",
                                "is_resolved": True,
                                "reward": 1.0,
                                "agent_execution": {
                                    "started_at": "2026-04-28T00:00:00+00:00",
                                    "finished_at": "2026-04-28T00:15:01+00:00",
                                },
                            },
                            {
                                "task_id": "unresolved",
                                "trial_id": "trial-3",
                                "is_resolved": False,
                                "reward": 0.0,
                                "agent_execution": {
                                    "started_at": "2026-04-28T00:00:00Z",
                                    "finished_at": "2026-04-28T00:01:00Z",
                                },
                            },
                            {
                                "task_id": "missing-timestamps",
                                "trial_id": "trial-4",
                                "is_resolved": True,
                                "reward": 1.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            summary_path = write_default_timeout_summary(
                jobs_dir=jobs_dir,
                agent_timeout_multiplier=2.0,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary_path.name, DEFAULT_TIMEOUT_SUMMARY_FILENAME)
        self.assertEqual(summary["agent_timeout_multiplier"], 2.0)
        self.assertEqual(summary["trial_count"], 4)
        by_task = {item["task_id"]: item for item in summary["trials"]}
        self.assertEqual(by_task["within"]["reward"], 1.0)
        self.assertTrue(by_task["within"]["resolved_within_default_timeout"])
        self.assertEqual(by_task["within"]["agent_execution_wall_seconds"], 900.0)
        self.assertFalse(by_task["after"]["resolved_within_default_timeout"])
        self.assertFalse(by_task["unresolved"]["resolved_within_default_timeout"])
        self.assertIsNone(by_task["missing-timestamps"]["resolved_within_default_timeout"])
        self.assertIn("missing_or_unparsable_agent_execution_timestamps", by_task["missing-timestamps"]["warnings"])

    def test_submission_config_validator_rejects_timeout_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"dataset":"terminal-bench@2.0","n_attempts":5,"timeout_multiplier":2.0}',
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("timeout" in error for error in errors))

    def test_submission_config_validator_accepts_real_harbor_dataset_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "terminal-bench", "version": "2.0"}],
                        "n_attempts": 5,
                        "timeout_multiplier": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertEqual(errors, [])

    def test_submission_config_validator_rejects_real_harbor_wrong_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "terminal-bench-core", "version": "head"}],
                        "n_attempts": 5,
                    }
                ),
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("terminal-bench-core@head" in error for error in errors))

    def test_submission_config_validator_rejects_k_one_and_task_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "terminal-bench",
                                "version": "2.0",
                                "task_names": ["hello-world"],
                            }
                        ],
                        "n_attempts": 1,
                    }
                ),
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("n_attempts" in error for error in errors))
        self.assertTrue(any("task_names" in error for error in errors))

    def test_submission_config_validator_rejects_timeout_override_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "terminal-bench", "version": "2.0"}],
                        "n_attempts": 5,
                        "override_timeout_sec": 7200,
                    }
                ),
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("override_timeout_sec" in error for error in errors))

    def test_submission_config_validator_rejects_agent_setup_timeout_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "terminal-bench", "version": "2.0"}],
                        "n_attempts": 5,
                        "agents": [{"override_setup_timeout_sec": 3600}],
                    }
                ),
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("override_setup_timeout_sec" in error for error in errors))

    def test_prepare_submission_copies_job_dirs_directly_under_agent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "metadata.yaml"
            metadata.write_text("agent_name: Spell\n", encoding="utf-8")
            job_dir = root / "spell-job"
            job_dir.mkdir()
            (job_dir / "config.json").write_text(
                json.dumps(
                    {
                        "datasets": [{"name": "terminal-bench", "version": "2.0"}],
                        "n_attempts": 5,
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "result.json").write_text("{}", encoding="utf-8")
            output_root = root / "submissions" / "terminal-bench" / "2.0"

            with patch(
                "sys.argv",
                [
                    "prepare_harbor_tb2_submission.py",
                    "--agent",
                    "spell__gpt-5.4",
                    "--job-dir",
                    str(job_dir),
                    "--metadata",
                    str(metadata),
                    "--output-root",
                    str(output_root),
                ],
            ):
                exit_code = prepare_submission_main()

            target = output_root / "spell__gpt-5.4"
            self.assertEqual(exit_code, 0)
            self.assertTrue((target / "metadata.yaml").is_file())
            self.assertTrue((target / "spell-job" / "config.json").is_file())
            self.assertFalse((target / "jobs").exists())


if __name__ == "__main__":
    unittest.main()
