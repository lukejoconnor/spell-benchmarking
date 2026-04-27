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
    HARBOR_TERMINALBENCH_DATASET,
    HarborTerminalBenchComplianceError,
    assert_harbor_submission_command_compliant,
    build_harbor_terminalbench_commands,
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
        self.assertEqual(cmd[:4], ["harbor", "run", "-d", HARBOR_TERMINALBENCH_DATASET])
        self.assertIn("--agent-import-path", cmd)
        self.assertIn("harbor_spell_agent:HarborSpellAgent", cmd)
        self.assertIn("--agent-kwarg", cmd)
        self.assertIn("reasoning_effort=medium", cmd)
        self.assertNotIn("--timeout-multiplier", cmd)
        self.assertNotIn("--agent-timeout-multiplier", cmd)
        self.assertNotIn("--registry-path", cmd)
        self.assertNotIn("--task", cmd)
        self.assertTrue(commands[0].submission_compliant)

    def test_submission_rejects_builtin_codex_until_web_disabled_adapter_exists(self) -> None:
        with self.assertRaisesRegex(HarborTerminalBenchComplianceError, "Stage 2 adapter"):
            build_harbor_terminalbench_commands(
                condition="codex",
                run_id="demo",
                model="openai-tc:gpt-5.4",
                codex_model="gpt-5.4",
                reasoning_effort="medium",
                trials=5,
                jobs_dir=Path("jobs"),
            )

    def test_dev_codex_uses_harbor_builtin_with_medium_effort(self) -> None:
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
        self.assertIn("-a", cmd)
        self.assertEqual(cmd[cmd.index("-a") + 1], "codex")
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

    def test_command_guard_rejects_banned_submission_flag(self) -> None:
        cmd = [
            "harbor",
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
