from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.docker_agents import ExecOutput, ParsedUsage
from src.unified_exercism import run_unified_exercism


def _item():
    return SimpleNamespace(
        slug="hello-world",
        name="Hello World",
        difficulty=1,
        instructions="Return a greeting.",
        stub_file="hello_world.py",
        stub_content="def hello():\n    return ''\n",
        test_file="hello_world_test.py",
        test_content="def test_hello():\n    assert True\n",
    )


class UnifiedExercismTests(unittest.TestCase):
    @patch("src.unified_exercism._ensure_git_repo")
    @patch("src.unified_exercism._run_pytest")
    @patch("src.unified_exercism.run_agent_in_container")
    @patch("src.unified_exercism.DockerWorkspaceContainer")
    def test_retry_preserves_unknown_token_counts(
        self,
        container_cls,
        run_agent_mock,
        run_pytest_mock,
        ensure_git_repo_mock,
    ) -> None:
        del ensure_git_repo_mock
        container = MagicMock()
        container.initial_commit = "base-sha"
        container.diff.return_value = "diff --git a/foo b/foo"
        container.changed_files.return_value = ["foo.py"]

        container_cm = MagicMock()
        container_cm.__enter__.return_value = container
        container_cm.__exit__.return_value = False
        container_cls.return_value = container_cm

        run_agent_mock.side_effect = [
            SimpleNamespace(
                output=ExecOutput(exit_code=0, stdout="", stderr="", timed_out=False),
                usage=ParsedUsage(cost_usd=0.25),
                response_text="first",
                error_message=None,
                error_type=None,
                raw={"ok": True},
            ),
            SimpleNamespace(
                output=ExecOutput(exit_code=0, stdout="", stderr="", timed_out=False),
                usage=ParsedUsage(cost_usd=0.5),
                response_text="second",
                error_message=None,
                error_type=None,
                raw={"ok": True},
            ),
        ]
        run_pytest_mock.side_effect = [
            (False, "first failure"),
            (True, "passed"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            outcome = run_unified_exercism(
                condition="swe_agent",
                item=_item(),
                workspace=Path(tmp_dir),
                model_spec="anthropic/claude-opus-4-6",
                timeout=30,
                init=None,
                agent_file=None,
                budget=2.0,
                depth=4,
                max_tokens=None,
                reasoning_effort="high",
                web_search=False,
                trace=True,
                trace_dir=Path(tmp_dir) / "traces",
                no_retry=False,
            )

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.attempts, 2)
        self.assertIsNone(outcome.response.tokens_in)
        self.assertIsNone(outcome.response.tokens_out)
        self.assertEqual(outcome.response.cost_usd, 0.75)


if __name__ == "__main__":
    unittest.main()
