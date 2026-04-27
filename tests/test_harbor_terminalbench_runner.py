from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.harbor_terminalbench_runner import (
    HARBOR_TERMINALBENCH_DATASET,
    HarborTerminalBenchComplianceError,
    assert_harbor_submission_command_compliant,
    build_harbor_terminalbench_commands,
)
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

    def test_submission_codex_uses_harbor_builtin_with_medium_effort(self) -> None:
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
                '{"dataset":"terminal-bench@2.0","timeout_multiplier":2.0}',
                encoding="utf-8",
            )

            errors = validate_config(path)

        self.assertTrue(any("timeout" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
