from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.unified_results import append_jsonl, load_jsonl


class UnifiedResultsTest(unittest.TestCase):
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
