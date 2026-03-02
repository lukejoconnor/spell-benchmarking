import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.unified_adapters import (
    CODEX_NO_CUSTOM_SCRIPT,
    PROJECT_ROOT,
    ClaudeCodeAdapter,
    CodexAdapter,
    RunRequest,
)


def _request(condition: str, model_spec: str | None = None) -> RunRequest:
    return RunRequest(
        benchmark="general",
        dataset="gsm8k",
        item_id="item-1",
        condition=condition,
        prompt="Return 42.",
        model_spec=model_spec,
        timeout=5,
    )


class UnifiedAdapterCommandDefaultsTest(unittest.TestCase):
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_disables_custom_setting_sources(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": "ok",
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens": 4,
                    },
                    "total_cost_usd": 0.01,
                }
            ),
            stderr="",
        )

        adapter = ClaudeCodeAdapter(model="sonnet", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:sonnet"))

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:3], ["claude", "--setting-sources", ""])

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_uses_no_custom_wrapper(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        adapter = CodexAdapter(project_root=PROJECT_ROOT, model_spec="codex")
        response = adapter.run(_request("codex", "codex"))

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[0], str(CODEX_NO_CUSTOM_SCRIPT))
        self.assertIn("exec", cmd)
        self.assertTrue(response.metadata.get("no_custom_config"))


if __name__ == "__main__":
    unittest.main()
