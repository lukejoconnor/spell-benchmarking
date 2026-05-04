import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.unified_adapters import (
    CODEX_NO_CUSTOM_SCRIPT,
    PROJECT_ROOT,
    ClaudeCodeAdapter,
    CodexAdapter,
    OneShotAdapter,
    RunRequest,
    SpellAdapter,
    SweAgentAdapter,
)
from src.docker_agents import codex_run_command


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
    @patch.dict(
        os.environ,
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "oauth-token",
            "ANTHROPIC_API_KEY": "sk-should-not-forward",
            "CLAUDE_JSON_B64": "legacy-json",
        },
        clear=False,
    )
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
        req = _request("claude_code", "claude_code:sonnet")
        req.reasoning_effort = "high"
        response = adapter.run(req)

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:3], ["claude", "--setting-sources", ""])
        self.assertIn("stream-json", cmd)
        self.assertIn("--effort", cmd)
        self.assertIn("high", cmd)
        self.assertEqual(response.metadata.get("num_turns"), 2)
        self.assertEqual(response.metadata.get("session_id"), "sess-1")
        self.assertEqual(response.tokens_in, 10)
        self.assertIsNone(response.cached_input_tokens)
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "oauth-token")
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDE_JSON_B64", env)
        self.assertNotEqual(env.get("HOME"), os.environ.get("HOME"))
        self.assertTrue(Path(env["HOME"]).name.startswith("spell-claude-code-home-"))

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_requires_oauth_token(self, run_mock):
        adapter = ClaudeCodeAdapter(model="sonnet", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:sonnet"))

        self.assertEqual(response.status, "error")
        self.assertEqual(response.error_type, "auth_error")
        self.assertIn("CLAUDE_CODE_OAUTH_TOKEN is required", response.error_message or "")
        run_mock.assert_not_called()

    @patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}, clear=False)
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_preserves_partial_stdout_usage_without_result_event(self, run_mock):
        assistant_event = {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "usage": {
                    "input_tokens": 100_000,
                    "cache_creation_input_tokens": 50_000,
                    "cache_read_input_tokens": 25_000,
                    "output_tokens": 10_000,
                },
            },
        }
        run_mock.return_value = SimpleNamespace(
            returncode=1,
            stdout="\n".join([json.dumps(assistant_event), json.dumps(assistant_event)]),
            stderr="",
        )

        adapter = ClaudeCodeAdapter(model="claude-sonnet-4-6", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:claude-sonnet-4-6"))

        self.assertEqual(response.status, "error")
        self.assertIn("No result event found", response.error_message or "")
        self.assertEqual(response.tokens_in, 175_000)
        self.assertEqual(response.tokens_out, 10_000)
        self.assertEqual(response.cached_input_tokens, 75_000)
        self.assertIsNotNone(response.usage)
        assert response.usage is not None
        self.assertEqual(response.usage.cached_input_tokens, 25_000)
        self.assertEqual(response.usage.cache_write_input_tokens, 50_000)
        self.assertAlmostEqual(response.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(response.metadata.get("cost_source"), "estimated_partial")

    @patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}, clear=False)
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_prefers_debug_usage_without_result_event(self, run_mock):
        partial_event = {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "usage": {
                    "input_tokens": 100_000,
                    "cache_creation_input_tokens": 50_000,
                    "cache_read_input_tokens": 25_000,
                    "output_tokens": 42,
                },
            },
        }
        debug_event = {
            "type": "assistant",
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "usage": {
                    "input_tokens": 100_000,
                    "cache_creation_input_tokens": 50_000,
                    "cache_read_input_tokens": 25_000,
                    "output_tokens": 10_000,
                },
            },
        }

        def fake_run(cmd, **_kwargs):
            debug_path = Path(cmd[cmd.index("--debug-file") + 1])
            debug_path.write_text(json.dumps(debug_event), encoding="utf-8")
            return SimpleNamespace(returncode=1, stdout=json.dumps(partial_event), stderr="")

        run_mock.side_effect = fake_run

        adapter = ClaudeCodeAdapter(model="claude-sonnet-4-6", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:claude-sonnet-4-6"))

        self.assertEqual(response.status, "error")
        self.assertEqual(response.tokens_in, 175_000)
        self.assertEqual(response.tokens_out, 10_000)
        self.assertAlmostEqual(response.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(response.metadata.get("cost_source"), "estimated_debug")
        self.assertEqual(response.metadata.get("claude_usage_ledger_sources"), ["debug"])

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_uses_no_custom_wrapper(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        adapter = CodexAdapter(project_root=PROJECT_ROOT, model_spec="codex")
        response = adapter.run(_request("codex", "codex"))

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[0], str(CODEX_NO_CUSTOM_SCRIPT))
        self.assertIn("exec", cmd)
        self.assertNotIn("--ephemeral", cmd)
        disabled_idx = cmd.index('web_search="disabled"')
        exec_idx = cmd.index("exec")
        self.assertLess(disabled_idx, exec_idx)
        self.assertNotIn("--search", cmd)
        self.assertTrue(response.metadata.get("no_custom_config"))

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_passes_search_flag_before_exec_when_requested(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        adapter = CodexAdapter(project_root=PROJECT_ROOT, model_spec="codex")
        response = adapter.run(
            RunRequest(
                benchmark="general",
                dataset="gsm8k",
                item_id="item-1",
                condition="codex",
                prompt="Return 42.",
                model_spec="codex",
                timeout=5,
                web_search=True,
            )
        )

        self.assertEqual(response.status, "ok")
        cmd = run_mock.call_args.args[0]
        self.assertLess(cmd.index("--search"), cmd.index("exec"))

    @patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}, clear=False)
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
                            "type": "turn.completed",
                            "usage": {"input_tokens": 7, "cached_input_tokens": 4, "output_tokens": 2},
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
            self.assertEqual(response.tokens_in, 7)
            self.assertEqual(response.cached_input_tokens, 4)
            self.assertEqual(response.metadata.get("cached_input_tokens"), 4)
            self.assertEqual(response.metadata.get("cost_source"), "estimated")

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_recovers_reasoning_tokens_from_rollout(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps({"type": "turn.started"}),
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {
                                "input_tokens": 1_000,
                                "cached_input_tokens": 100,
                                "output_tokens": 200,
                                "cost_usd": 0.01,
                            },
                        }
                    ),
                    "non-json-line",
                ]
            ),
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sessions_root = Path(tmp_dir) / "sessions"
            rollout_dir = sessions_root / "2026" / "04" / "06"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            rollout_path = rollout_dir / "rollout-2026-04-06T10-08-30-thread-1.jsonl"
            rollout_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {
                                    "type": "token_count",
                                    "info": {
                                        "total_token_usage": {
                                            "input_tokens": 1_000,
                                            "cached_input_tokens": 100,
                                            "output_tokens": 200,
                                            "reasoning_output_tokens": 50,
                                        },
                                        "last_token_usage": {
                                            "input_tokens": 1_000,
                                            "cached_input_tokens": 100,
                                            "output_tokens": 200,
                                            "reasoning_output_tokens": 50,
                                        },
                                        "model_context_window": 258_400,
                                    },
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            adapter = CodexAdapter(project_root=PROJECT_ROOT, model_spec="gpt-5.4", sessions_root=sessions_root)
            response = adapter.run(_request("codex", "gpt-5.4"))

            self.assertEqual(response.status, "ok")
            self.assertAlmostEqual(response.cost_usd or 0.0, 0.01, places=8)
            self.assertEqual(response.metadata["reasoning_output_tokens"], 50)
            self.assertEqual(response.metadata["rollout_status"], "ok")
            self.assertEqual(response.metadata["cost_source"], "reported")
            self.assertEqual(response.metadata["raw"]["events"][0]["type"], "thread.started")
            self.assertEqual(response.metadata["raw"]["non_json_lines"], ["non-json-line"])
            self.assertEqual(response.metadata["raw"]["rollout_token_count"]["total_token_usage"]["reasoning_output_tokens"], 50)
            self.assertEqual(response.metadata["raw"]["rollout_path"], str(rollout_path))

    @patch("src.unified_adapters.subprocess.run")
    def test_codex_adapter_handles_missing_rollout(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 1_000, "cached_input_tokens": 100, "output_tokens": 200},
                        }
                    ),
                ]
            ),
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = CodexAdapter(
                project_root=PROJECT_ROOT,
                model_spec="gpt-5.4",
                sessions_root=Path(tmp_dir) / "sessions",
            )
            response = adapter.run(_request("codex", "gpt-5.4"))

            self.assertEqual(response.status, "ok")
            self.assertAlmostEqual(response.cost_usd or 0.0, 0.005275, places=8)
            self.assertEqual(response.metadata["rollout_status"], "missing")
            self.assertEqual(response.metadata["cost_source"], "estimated")

    @patch("src.unified_adapters.subprocess.run")
    def test_swe_agent_adapter_reads_trajectory_and_writes_trace(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        adapter = SweAgentAdapter(model="anthropic/claude-opus-4-6", budget=2.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            trajectory = {
                "info": {"model_stats": {"instance_cost": 0.42}},
                "trajectory": [
                    {"role": "assistant", "content": "final answer"},
                ],
            }
            (workspace / "run.traj.json").write_text(json.dumps(trajectory), encoding="utf-8")

            response = adapter.run(
                RunRequest(
                    benchmark="general",
                    dataset="gsm8k",
                    item_id="item-1",
                    condition="swe_agent",
                    prompt="Return 42.",
                    model_spec="anthropic/claude-opus-4-6",
                    timeout=5,
                    workspace=workspace,
                    trace_dir=Path(tmp_dir) / "traces",
                )
            )

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.response_text, "final answer")
            self.assertEqual(response.cost_usd, 0.42)
            self.assertIn("pty-run mini", run_mock.call_args.args[0])
            self.assertIn("-t 'Return 42.'", run_mock.call_args.args[0])
            trace_file = response.metadata.get("trace_file")
            self.assertTrue(trace_file)
            trace_path = Path(trace_file)
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["condition"], "swe_agent")
            self.assertEqual(payload["item_id"], "item-1")
            self.assertEqual(payload["trajectory"]["info"]["model_stats"]["instance_cost"], 0.42)

    @patch("src.unified_adapters.subprocess.run")
    def test_swe_agent_adapter_threads_deepseek_reasoning_effort_and_estimates_cost(self, run_mock):
        run_mock.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        adapter = SweAgentAdapter(model="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro", budget=2.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            trajectory = {
                "info": {
                    "model_stats": {"instance_cost": 0.0},
                    "config": {
                        "model": {
                            "model_name": "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro"
                        }
                    },
                },
                "messages": [
                    {
                        "extra": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 100,
                                    "prompt_tokens_details": {"cached_tokens": 250},
                                }
                            }
                        }
                    }
                ],
            }
            (workspace / "run.traj.json").write_text(json.dumps(trajectory), encoding="utf-8")

            response = adapter.run(
                RunRequest(
                    benchmark="general",
                    dataset="gsm8k",
                    item_id="item-1",
                    condition="swe_agent",
                    prompt="Return 42.",
                    model_spec="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro",
                    timeout=5,
                    workspace=workspace,
                    reasoning_effort="high",
                )
            )

            self.assertEqual(response.status, "ok")
            self.assertIn("model.model_kwargs.reasoning_effort=high", run_mock.call_args.args[0])
            self.assertEqual(response.metadata["cost_source"], "estimated")
            self.assertAlmostEqual(response.cost_usd or 0.0, 0.00168925, places=9)

    @patch("src.unified_adapters.subprocess.run")
    def test_swe_agent_adapter_recovers_timeout_trajectory_cost(self, run_mock):
        run_mock.side_effect = subprocess.TimeoutExpired(
            cmd="mini",
            timeout=5,
            output="partial stdout",
            stderr="partial stderr",
        )
        adapter = SweAgentAdapter(model="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro", budget=2.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            trajectory = {
                "info": {
                    "model_stats": {"instance_cost": 0.0},
                    "config": {
                        "model": {
                            "model_name": "fireworks_ai/accounts/fireworks/models/deepseek-v4-pro"
                        }
                    },
                },
                "messages": [
                    {
                        "extra": {
                            "response": {
                                "usage": {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 100,
                                    "prompt_tokens_details": {"cached_tokens": 250},
                                }
                            }
                        }
                    }
                ],
            }
            (workspace / "run.traj.json").write_text(json.dumps(trajectory), encoding="utf-8")

            response = adapter.run(
                RunRequest(
                    benchmark="general",
                    dataset="gsm8k",
                    item_id="item-1",
                    condition="swe_agent",
                    prompt="Return 42.",
                    model_spec="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro",
                    timeout=5,
                    workspace=workspace,
                    reasoning_effort="high",
                )
            )

            self.assertEqual(response.status, "timeout")
            self.assertEqual(response.error_type, "timeout")
            self.assertEqual(response.metadata["cost_source"], "estimated")
            self.assertEqual(response.metadata["stdout_tail"], "partial stdout")
            self.assertEqual(response.metadata["stderr_tail"], "partial stderr")
            self.assertAlmostEqual(response.cost_usd or 0.0, 0.00168925, places=9)

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
        self.assertEqual(response.tokens_in, 1)
        self.assertIsNone(response.cached_input_tokens)

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_reports_total_and_cached_input_tokens(self, client_run_mock):
        client_run_mock.return_value = SimpleNamespace(
            ok=True,
            result="42",
            usage={
                "total": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 2,
                    "output_tokens": 3,
                    "cost": 0.01,
                }
            },
            latency_ms=1.0,
            trace_dir=None,
            raw={},
            error=None,
            error_type=None,
        )
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")

        response = adapter.run(_request("spell", "anthropic-tc:opus46"))

        self.assertEqual(response.tokens_in, 13)
        self.assertEqual(response.cached_input_tokens, 3)

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_resolves_fireworks_reasoning_effort(self, client_run_mock):
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
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="fireworks:qwen3p6-plus")

        response = adapter.run(_request("spell", "fireworks:qwen3p6-plus"))

        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertEqual(payload["reasoning_effort"], "32000")

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_resolves_fireworks_tc_reasoning_effort(self, client_run_mock):
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
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="fireworks-tc:qwen3p6-plus")

        response = adapter.run(_request("spell", "fireworks-tc:qwen3p6-plus"))

        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertEqual(payload["reasoning_effort"], "32000")

    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_one_shot_adapter_drops_unsupported_fireworks_reasoning_effort(self, client_run_mock):
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
        adapter = OneShotAdapter(project_root=PROJECT_ROOT, model_spec="fireworks:kimi-k2p6")

        response = adapter.run(_request("one_shot", "fireworks:kimi-k2p6"))

        self.assertEqual(response.status, "ok")
        payload = client_run_mock.call_args.args[0]
        self.assertNotIn("reasoning_effort", payload)

    @patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}, clear=False)
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_reports_total_and_cached_input_tokens(self, run_mock):
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
                }
            ),
            stderr="",
        )

        adapter = ClaudeCodeAdapter(model="sonnet", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:sonnet"))

        self.assertEqual(response.tokens_in, 13)
        self.assertEqual(response.cached_input_tokens, 3)
        self.assertEqual(response.metadata["cost_source"], "reported")

    @patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token"}, clear=False)
    @patch("src.unified_adapters.subprocess.run")
    def test_claude_adapter_reprices_underreported_cache_write_cost(self, run_mock):
        run_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "result",
                    "result": "ok",
                    "usage": {
                        "input_tokens": 100_000,
                        "cache_creation_input_tokens": 50_000,
                        "cache_read_input_tokens": 25_000,
                        "output_tokens": 10_000,
                    },
                    "total_cost_usd": 0.4575,
                }
            ),
            stderr="",
        )

        adapter = ClaudeCodeAdapter(model="claude-sonnet-4-6", budget=1.0)
        response = adapter.run(_request("claude_code", "claude_code:claude-sonnet-4-6"))

        self.assertEqual(response.status, "ok")
        self.assertEqual(response.tokens_in, 175_000)
        self.assertEqual(response.cached_input_tokens, 75_000)
        self.assertAlmostEqual(response.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(response.metadata["cost_source"], "estimated")

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

    @patch("src.unified_adapters.tempfile.TemporaryDirectory")
    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_spell_adapter_cleans_up_temp_cwd_when_workspace_missing(
        self, client_run_mock, tempdir_mock
    ):
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
        tempdir_cm = MagicMock()
        tempdir_cm.__enter__.return_value = "/tmp/spell-bench-test"
        tempdir_mock.return_value = tempdir_cm
        adapter = SpellAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")

        response = adapter.run(_request("spell", "anthropic-tc:opus46"))

        self.assertEqual(response.status, "ok")
        self.assertEqual(client_run_mock.call_args.kwargs["cwd"], Path("/tmp/spell-bench-test"))
        tempdir_cm.__exit__.assert_called_once()

    @patch("src.unified_adapters.tempfile.TemporaryDirectory")
    @patch("src.unified_adapters.SpellBenchmarkClient.run")
    def test_one_shot_adapter_cleans_up_temp_cwd_when_workspace_missing(
        self, client_run_mock, tempdir_mock
    ):
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
        tempdir_cm = MagicMock()
        tempdir_cm.__enter__.return_value = "/tmp/spell-bench-test"
        tempdir_mock.return_value = tempdir_cm
        adapter = OneShotAdapter(project_root=PROJECT_ROOT, model_spec="anthropic-tc:opus46")

        response = adapter.run(_request("one_shot", "anthropic-tc:opus46"))

        self.assertEqual(response.status, "ok")
        self.assertEqual(client_run_mock.call_args.kwargs["cwd"], Path("/tmp/spell-bench-test"))
        tempdir_cm.__exit__.assert_called_once()

    def test_codex_run_command_includes_search_flag_before_exec_when_requested(self):
        container = MagicMock()
        env = {"OPENAI_API_KEY": "test"}

        container.exec_run(
            codex_run_command(
                "Return 42.",
                model="gpt-5-codex",
                web_search=True,
            ),
            timeout=30,
            env=env,
        )

        cmd = container.exec_run.call_args.args[0]
        self.assertIn("codex --search exec", cmd)

    def test_run_codex_disables_search_when_not_requested(self):
        container = MagicMock()
        env = {"OPENAI_API_KEY": "test"}

        container.exec_run(
            codex_run_command(
                "Return 42.",
                model="gpt-5-codex",
                web_search=False,
            ),
            timeout=30,
            env=env,
        )

        cmd = container.exec_run.call_args.args[0]
        self.assertIn("codex -c 'web_search=\"disabled\"' exec", cmd)
        self.assertNotIn("codex --search exec", cmd)
        self.assertIn('export CODEX_HOME="${CODEX_HOME:-/tmp/bench-agent-runtime/codex/home}"', cmd)
        self.assertNotIn('export HOME="${HOME:-', cmd)


if __name__ == "__main__":
    unittest.main()
