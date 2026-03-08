import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.unified_adapters import (
    CODEX_NO_CUSTOM_SCRIPT,
    PROJECT_ROOT,
    ClaudeCodeAdapter,
    CodexAdapter,
    OneShotAdapter,
    RunRequest,
    SpellAdapter,
)


def _request(condition: str, model_spec: str | None = None, trace_dir: Path | None = None) -> RunRequest:
    return RunRequest(
        benchmark="general",
        dataset="gsm8k",
        item_id="item-1",
        condition=condition,
        prompt="Return 42.",
        model_spec=model_spec,
        timeout=5,
        trace_dir=trace_dir,
    )


class UnifiedAdapterCommandDefaultsTest(unittest.TestCase):
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_disables_custom_setting_sources(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "assistant", "message": "working"}),
                    json.dumps(
                        {
                            "type": "result",
                            "result": "ok",
                            "usage": {
                                "input_tokens": 10,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "output_tokens": 4,
                            },
                            "total_cost_usd": 0.01,
                            "num_turns": 2,
                            "session_id": "sess-1",
                        }
                    ),
                ]
            ),
            stderr="",
        )

        adapter = ClaudeCodeAdapter(model="sonnet", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:sonnet"))

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:3], ["claude", "--setting-sources", ""])
        self.assertIn("stream-json", cmd)
        self.assertEqual(response.metadata.get("num_turns"), 2)
        self.assertEqual(response.metadata.get("session_id"), "sess-1")

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

    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_writes_trace_file(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "result": "ok",
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 1,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 4,
                    },
                    "total_cost_usd": 0.01,
                    "num_turns": 1,
                    "session_id": "sess-2",
                }
            ),
            stderr="stderr text",
        )
        adapter = ClaudeCodeAdapter(model="sonnet", budget=1.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = adapter.run(_request("claude_code", "claude_code:sonnet", trace_dir=Path(tmp_dir)))

            trace_file = response.metadata.get("trace_file")
            self.assertTrue(trace_file)
            trace_path = Path(trace_file)
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["condition"], "claude_code")
            self.assertEqual(payload["item_id"], "item-1")
            self.assertEqual(payload["events"][-1]["type"], "result")

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_writes_trace_file(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps(
                        {
                            "type": "assistant_event",
                            "item": {"type": "assistant_message", "role": "assistant", "content": "done"},
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                        }
                    ),
                    "not-json-line",
                ]
            ),
            stderr="",
        )
        adapter = CodexAdapter(project_root=PROJECT_ROOT, model_spec="codex")
        with tempfile.TemporaryDirectory() as tmp_dir:
            response = adapter.run(_request("codex", "codex", trace_dir=Path(tmp_dir)))

            trace_file = response.metadata.get("trace_file")
            self.assertTrue(trace_file)
            trace_path = Path(trace_file)
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["condition"], "codex")
            self.assertEqual(payload["item_id"], "item-1")
            self.assertEqual(payload["non_json_lines"], ["not-json-line"])

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

    def test_one_shot_adapter_injects_spell_repo_for_benchmark_api(self):
        adapter = OneShotAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")

        self.assertEqual(adapter.client.clj_cmd[0:4], ["clj", "-Sdeps", adapter.client.clj_cmd[2], "-M"])
        self.assertIn(str(PROJECT_ROOT), adapter.client.clj_cmd[2])
        self.assertEqual(adapter.client.clj_cmd[-2:], ["-m", "spell.benchmark-api"])


if __name__ == "__main__":
    unittest.main()
