from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.unified_results import append_jsonl, load_jsonl, summarize_records


class UnifiedResultsTest(unittest.TestCase):
    def test_terminalbench_errors_stay_in_accuracy_denominator(self):
        rows = [
            {"benchmark": "terminalbench", "condition": "spell", "correct": True},
            {"benchmark": "terminalbench", "condition": "spell", "correct": None, "error_type": "test_timeout"},
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["by_condition"]["spell"]["accuracy"], 0.5)
        self.assertEqual(summary["errors"], 1)

    def test_swebench_uses_official_outcome_even_after_generation_timeout(self):
        rows = [
            {"benchmark": "swebench", "condition": "spell", "correct": None,
             "status": "timeout", "metadata": {"same_container_resolved": True}},
            {"benchmark": "swebench", "condition": "spell", "correct": None,
             "metadata": {"same_container_resolved": False}},
        ]
        summary = summarize_records(rows)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["scored"], 2)

    def test_unscored_outputs_are_reported_explicitly(self):
        summary = summarize_records([{"condition": "spell", "correct": None}])
        self.assertIsNone(summary["accuracy"])
        self.assertEqual(summary["unscored"], 1)

    def test_general_timeout_with_recovered_correct_text_is_unsuccessful(self):
        summary = summarize_records([{"benchmark": "general", "condition": "spell",
                                      "correct": True, "status": "timeout"}])
        self.assertEqual(summary["accuracy"], 0)
        self.assertEqual(summary["errors"], 1)

    def test_load_jsonl_tolerates_truncated_tail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            path.write_text(
                json.dumps({"item_id": "one"}) + "\n"
                + json.dumps({"item_id": "two"}) + "\n"
                + '{"item_id": "broken"',
                encoding="utf-8",
            )

            rows = load_jsonl(path, tolerate_invalid=True)

        self.assertEqual([row["item_id"] for row in rows], ["one", "two"])

    def test_append_jsonl_separates_after_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "records.jsonl"
            path.write_text('{"item_id": "broken"', encoding="utf-8")

            append_jsonl(path, {"item_id": "two"})

            rows = load_jsonl(path, tolerate_invalid=True)
            text = path.read_text(encoding="utf-8")

        self.assertEqual([row["item_id"] for row in rows], ["two"])
        self.assertTrue(text.startswith('{"item_id": "broken"\n'))


if __name__ == "__main__":
    unittest.main()
