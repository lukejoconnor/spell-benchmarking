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
    SpellAdapter,
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

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_applies_default_format_for_nonsoftware(self, client_run_mock):
        client_run_mock.return_value = SimpleNamespace(
            ok=True,
            result="42",
            usage={"total": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0}},
            latency_ms=1.0,
            trace_dir=None,
            raw={},
            error=None,
            error_type=None,
        )
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")
        req = _request("spell", "anthropic-tc:opus46")

        response = adapter.run(req)
        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertEqual(payload.get("format"), {"required": ["result"]})

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_skips_default_format_for_software(self, client_run_mock):
        client_run_mock.return_value = SimpleNamespace(
            ok=True,
            result="ok",
            usage={"total": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0}},
            latency_ms=1.0,
            trace_dir=None,
            raw={},
            error=None,
            error_type=None,
        )
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")
        req = RunRequest(
            benchmark="swebench",
            dataset="mini",
            item_id="item-1",
            condition="spell",
            prompt="Fix bug",
            model_spec="anthropic-tc:opus46",
            timeout=5,
        )

        response = adapter.run(req)
        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertNotIn("format", payload)

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_honors_explicit_format_override(self, client_run_mock):
        client_run_mock.return_value = SimpleNamespace(
            ok=True,
            result="42",
            usage={"total": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0}},
            latency_ms=1.0,
            trace_dir=None,
            raw={},
            error=None,
            error_type=None,
        )
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")
        req = RunRequest(
            benchmark="general",
            dataset="gsm8k",
            item_id="item-1",
            condition="spell",
            prompt="Return 42.",
            model_spec="anthropic-tc:opus46",
            timeout=5,
            format_spec={"required": [":final_answer"]},
        )

        response = adapter.run(req)
        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertEqual(payload.get("format"), {"required": [":final_answer"]})


if __name__ == "__main__":
    unittest.main()
