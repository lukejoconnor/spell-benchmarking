from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.usage_accounting import (
    claude_code_usage_from_events,
    claude_code_usage_from_result,
    claude_code_usage_from_text,
    estimate_cost,
    usage_from_totals,
)


class UsageAccountingTests(unittest.TestCase):
    def test_usage_from_totals_splits_reasoning_from_visible_outputs(self) -> None:
        usage = usage_from_totals(
            total_input_tokens=125,
            total_output_tokens=45,
            cached_input_tokens=20,
            cache_write_input_tokens=5,
            reasoning_output_tokens=12,
        )

        assert usage is not None
        self.assertEqual(usage.uncached_input_tokens, 100)
        self.assertEqual(usage.cached_input_tokens, 20)
        self.assertEqual(usage.cache_write_input_tokens, 5)
        self.assertEqual(usage.visible_output_tokens, 33)
        self.assertEqual(usage.reasoning_output_tokens, 12)
        self.assertEqual(usage.total_output_tokens, 45)

    def test_claude_code_usage_from_result_prefers_estimated_cost_when_reported_undercounts(self) -> None:
        usage = claude_code_usage_from_result(
            {
                "type": "result",
                "usage": {
                    "input_tokens": 100_000,
                    "cache_creation_input_tokens": 50_000,
                    "cache_read_input_tokens": 25_000,
                    "output_tokens": 10_000,
                },
                "total_cost_usd": 0.4575,
            },
            model="claude-sonnet-4-6",
        )

        assert usage is not None
        self.assertEqual(usage.uncached_input_tokens, 100_000)
        self.assertEqual(usage.cached_input_tokens, 25_000)
        self.assertEqual(usage.cache_write_input_tokens, 50_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated")

    def test_claude_code_usage_from_result_normalizes_provider_qualified_model(self) -> None:
        usage = claude_code_usage_from_result(
            {
                "type": "result",
                "usage": {
                    "input_tokens": 100_000,
                    "cache_creation_input_tokens": 50_000,
                    "cache_read_input_tokens": 25_000,
                    "output_tokens": 10_000,
                },
                "total_cost_usd": 0.4575,
            },
            model="anthropic/claude-sonnet-4-6",
        )

        assert usage is not None
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated")

    def test_claude_code_usage_from_events_prefers_result_event(self) -> None:
        usage = claude_code_usage_from_events(
            [
                {
                    "type": "assistant",
                    "message": {
                        "id": "msg-1",
                        "role": "assistant",
                        "usage": {
                            "input_tokens": 1,
                            "cache_creation_input_tokens": 2,
                            "cache_read_input_tokens": 3,
                            "output_tokens": 4,
                        },
                    },
                },
                {
                    "type": "result",
                    "usage": {
                        "input_tokens": 100_000,
                        "cache_creation_input_tokens": 50_000,
                        "cache_read_input_tokens": 25_000,
                        "output_tokens": 10_000,
                    },
                    "total_cost_usd": 0.4575,
                },
            ],
            model="claude-sonnet-4-6",
        )

        assert usage is not None
        self.assertEqual(usage.total_input_tokens, 175_000)
        self.assertEqual(usage.visible_output_tokens, 10_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated")

    def test_claude_code_usage_from_events_falls_back_to_assistant_message_usage(self) -> None:
        usage = claude_code_usage_from_events(
            [
                {
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
            ],
            model="claude-sonnet-4-6",
        )

        assert usage is not None
        self.assertEqual(usage.uncached_input_tokens, 100_000)
        self.assertEqual(usage.cached_input_tokens, 25_000)
        self.assertEqual(usage.cache_write_input_tokens, 50_000)
        self.assertEqual(usage.visible_output_tokens, 10_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated_partial")

    def test_claude_code_usage_from_events_dedupes_repeated_assistant_message_id(self) -> None:
        event = {
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

        usage = claude_code_usage_from_events([event, event], model="claude-sonnet-4-6")

        assert usage is not None
        self.assertEqual(usage.total_input_tokens, 175_000)
        self.assertEqual(usage.visible_output_tokens, 10_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated_partial")

    def test_claude_code_usage_from_text_uses_latest_assistant_record_by_message_id(self) -> None:
        first = {
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
        latest = {
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

        usage = claude_code_usage_from_text(
            "\n".join([json.dumps(first), json.dumps(latest)]),
            model="claude-sonnet-4-6",
            assistant_cost_source="estimated_ledger",
        )

        assert usage is not None
        self.assertEqual(usage.total_input_tokens, 175_000)
        self.assertEqual(usage.visible_output_tokens, 10_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated_ledger")

    def test_estimate_cost_stays_on_standard_pricing_at_the_long_context_threshold(self) -> None:
        usage = usage_from_totals(
            total_input_tokens=272_000,
            total_output_tokens=100_000,
            cached_input_tokens=20_000,
            cache_write_input_tokens=10_000,
            reasoning_output_tokens=25_000,
        )

        self.assertIsNotNone(usage)
        self.assertAlmostEqual(estimate_cost("gpt-5.4", usage) or 0.0, 2.14125, places=8)

    def test_estimate_cost_switches_to_long_context_pricing_above_threshold(self) -> None:
        usage = usage_from_totals(
            total_input_tokens=272_001,
            total_output_tokens=100_000,
            cached_input_tokens=20_000,
            cache_write_input_tokens=10_000,
            reasoning_output_tokens=25_000,
        )

        self.assertIsNotNone(usage)
        self.assertAlmostEqual(estimate_cost("gpt-5.4", usage) or 0.0, 3.532505, places=8)


if __name__ == "__main__":
    unittest.main()
