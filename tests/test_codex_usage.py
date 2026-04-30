from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.codex_usage import (
    codex_cost_accounting_from_rollout,
    estimate_cost,
    merge_codex_usage,
    parse_codex_rollout,
    usage_from_payload,
    usage_from_rollout,
)


class CodexUsageTests(unittest.TestCase):
    def test_estimate_cost_does_not_add_reasoning_to_total_output_tokens(self) -> None:
        cost = estimate_cost(
            "gpt-5.4",
            1_000,
            200,
            cached_input_tokens=100,
            reasoning_output_tokens=50,
        )

        self.assertAlmostEqual(cost or 0.0, 0.005275, places=8)

    def test_estimate_cost_without_reasoning_matches_legacy(self) -> None:
        cost = estimate_cost(
            "gpt-5-codex",
            42_069,
            512,
            cached_input_tokens=33_408,
        )

        self.assertAlmostEqual(cost or 0.0, 0.02012225, places=8)

    def test_estimate_cost_uses_base_pricing_above_former_long_context_threshold(self) -> None:
        cost = estimate_cost(
            "gpt-5.4",
            272_001,
            100_000,
            cached_input_tokens=20_000,
            cache_write_input_tokens=10_000,
            reasoning_output_tokens=25_000,
        )

        self.assertAlmostEqual(cost or 0.0, 2.1412525, places=8)

    def test_usage_from_payload_keeps_input_tokens_as_total_when_cached_subset_present(self) -> None:
        usage = usage_from_payload(
            {
                "input_tokens": 42_069,
                "cached_input_tokens": 33_408,
                "output_tokens": 512,
            },
            model="gpt-5-codex",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 42_069)
        self.assertEqual(usage.cached_input_tokens, 33_408)
        self.assertEqual(usage.total_input_tokens, 42_069)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.02012225, places=8)
        self.assertEqual(usage.cost_source, "estimated")

    def test_usage_from_payload_uses_total_input_tokens_field_without_subtracting_cached_tokens(self) -> None:
        usage = usage_from_payload(
            {
                "total_input_tokens": 42_069,
                "cached_input_tokens": 33_408,
                "output_tokens": 512,
            },
            model="gpt-5-codex",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 42_069)
        self.assertEqual(usage.cached_input_tokens, 33_408)
        self.assertEqual(usage.total_input_tokens, 42_069)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.02012225, places=8)
        self.assertEqual(usage.cost_source, "estimated")

    def test_usage_from_payload_propagates_reasoning_output_tokens(self) -> None:
        usage = usage_from_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
                "reasoning_output_tokens": 50,
            },
            model="gpt-5.4",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.reasoning_output_tokens, 50)
        self.assertEqual(usage.visible_output_tokens, 150)
        self.assertEqual(usage.total_output_tokens, 200)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.005275, places=8)
        self.assertEqual(usage.cost_source, "estimated")

    def test_parse_codex_rollout_picks_latest_non_null_info(self) -> None:
        rollout = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "sess-1"}}),
                json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}}),
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
                                    "reasoning_output_tokens": 300,
                                },
                                "last_token_usage": {
                                    "input_tokens": 400,
                                    "cached_input_tokens": 25,
                                    "output_tokens": 50,
                                    "reasoning_output_tokens": 75,
                                },
                                "model_context_window": 258_400,
                            },
                        },
                    }
                ),
            ]
        )

        parsed = parse_codex_rollout(rollout)

        self.assertEqual(parsed["session_id"], "sess-1")
        self.assertEqual(parsed["model_context_window"], 258_400)
        self.assertEqual(parsed["total_token_usage"]["reasoning_output_tokens"], 300)

    def test_parse_codex_rollout_returns_none_when_only_null_info(self) -> None:
        rollout = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "sess-1"}}),
                json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}}),
            ]
        )

        self.assertIsNone(parse_codex_rollout(rollout))

    def test_usage_from_rollout_round_trip(self) -> None:
        rollout = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "sess-1"}}),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 20_707,
                                    "cached_input_tokens": 10_624,
                                    "output_tokens": 259,
                                    "reasoning_output_tokens": 183,
                                    "total_tokens": 20_966,
                                },
                                "last_token_usage": {
                                    "input_tokens": 20_707,
                                    "cached_input_tokens": 10_624,
                                    "output_tokens": 259,
                                    "reasoning_output_tokens": 183,
                                    "total_tokens": 20_966,
                                },
                                "model_context_window": 258_400,
                            },
                        },
                    }
                ),
            ]
        )

        usage = usage_from_rollout(rollout, model="gpt-5.4")

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 20_707)
        self.assertEqual(usage.cached_input_tokens, 10_624)
        self.assertEqual(usage.total_output_tokens, 259)
        self.assertEqual(usage.output_tokens, 76)
        self.assertEqual(usage.reasoning_output_tokens, 183)
        self.assertEqual(usage.cost_source, "estimated")

    def test_cost_accounting_uses_base_pricing_for_each_call(self) -> None:
        rollout = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "sess-1"}}),
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
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 273_001,
                                    "cached_input_tokens": 20_100,
                                    "output_tokens": 100_200,
                                    "reasoning_output_tokens": 25_050,
                                },
                                "last_token_usage": {
                                    "input_tokens": 272_001,
                                    "cached_input_tokens": 20_000,
                                    "output_tokens": 100_000,
                                    "reasoning_output_tokens": 25_000,
                                },
                            },
                        },
                    }
                ),
            ]
        )

        accounting = codex_cost_accounting_from_rollout(rollout, model="gpt-5.4")

        self.assertIsNotNone(accounting)
        assert accounting is not None
        self.assertNotIn("long_context_pricing_applied", accounting["calls"][0])
        self.assertAlmostEqual(accounting["cost_usd"] or 0.0, 2.1402775, places=8)
        self.assertEqual(accounting["aggregate"]["output_tokens"], 100_200)

    def test_cost_accounting_falls_back_to_cumulative_deltas(self) -> None:
        rollout = "\n".join(
            [
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
                                },
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 1_400,
                                    "cached_input_tokens": 125,
                                    "output_tokens": 250,
                                },
                            },
                        },
                    }
                ),
            ]
        )

        accounting = codex_cost_accounting_from_rollout(rollout, model="gpt-5.4")

        self.assertIsNotNone(accounting)
        assert accounting is not None
        self.assertEqual([call["source"] for call in accounting["calls"]], ["delta_total_token_usage", "delta_total_token_usage"])
        self.assertEqual(accounting["calls"][1]["input_tokens"], 400)
        self.assertTrue(accounting["warnings"])

    def test_merge_codex_usage_uses_rollout_estimate_when_stream_lacks_reported_cost(self) -> None:
        stream = usage_from_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
            },
            model="gpt-5.4",
        )
        rollout = usage_from_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
                "reasoning_output_tokens": 50,
            },
            model="gpt-5.4",
        )

        merged = merge_codex_usage(stream, rollout, model="gpt-5.4")

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.reasoning_output_tokens, 50)
        self.assertEqual(merged.cost_source, "estimated_with_reasoning")

    def test_merge_codex_usage_preserves_reported_stream_cost(self) -> None:
        stream = usage_from_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
                "cost_usd": 0.01,
            },
            model="gpt-5.4",
        )
        rollout = usage_from_payload(
            {
                "input_tokens": 1_000,
                "cached_input_tokens": 100,
                "output_tokens": 200,
                "reasoning_output_tokens": 50,
            },
            model="gpt-5.4",
        )

        merged = merge_codex_usage(stream, rollout, model="gpt-5.4")

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(merged.reasoning_output_tokens, 50)
        self.assertAlmostEqual(merged.cost_usd or 0.0, 0.01, places=8)
        self.assertEqual(merged.cost_source, "reported")


if __name__ == "__main__":
    unittest.main()
