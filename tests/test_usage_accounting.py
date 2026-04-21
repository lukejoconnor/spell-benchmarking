from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.usage_accounting import claude_code_usage_from_result, usage_from_totals


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


if __name__ == "__main__":
    unittest.main()
