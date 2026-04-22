from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.usage_accounting import estimate_cost, usage_from_totals


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
