from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.codex_usage import usage_from_payload


class CodexUsageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
