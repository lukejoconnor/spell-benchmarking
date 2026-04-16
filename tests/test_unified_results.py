from __future__ import annotations

import sys
from unittest.mock import patch
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.unified_results import append_jsonl, filter_jsonl_rows, load_jsonl


def test_load_jsonl_repairs_truncated_tail(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"run_id":"run-1","condition":"spell","name":null,"item_id":"item-1"}\n'
        '{"run_id":"run-1","condition":"spell","name":null,"item_id":',
        encoding="utf-8",
    )

    rows = load_jsonl(path, repair_truncated_tail=True)

    assert rows == [
        {
            "run_id": "run-1",
            "condition": "spell",
            "name": None,
            "item_id": "item-1",
        }
    ]
    assert path.read_text(encoding="utf-8") == (
        '{"run_id":"run-1","condition":"spell","name":null,"item_id":"item-1"}\n'
    )


def test_append_jsonl_separates_existing_content_and_fsyncs(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    path.write_text('{"run_id":"run-1"}', encoding="utf-8")
    calls: list[int] = []

    with patch("src.unified_results.os.fsync", side_effect=lambda fd: calls.append(fd)):
        append_jsonl(path, {"run_id": "run-2", "condition": "spell"})

    assert path.read_text(encoding="utf-8") == (
        '{"run_id":"run-1"}\n{"run_id": "run-2", "condition": "spell"}\n'
    )
    assert calls


def test_filter_jsonl_rows_filters_on_run_condition_and_name() -> None:
    rows = [
        {"run_id": "run-1", "condition": "spell", "name": None, "item_id": "one"},
        {"run_id": "run-1", "condition": "codex", "name": None, "item_id": "two"},
        {"run_id": "run-2", "condition": "spell", "name": "baseline/v1", "item_id": "three"},
    ]

    assert filter_jsonl_rows(rows, run_id="run-1", condition="spell", name=None) == [
        {"run_id": "run-1", "condition": "spell", "name": None, "item_id": "one"}
    ]
    assert filter_jsonl_rows(rows, run_id="run-2", condition="spell", name="baseline/v1") == [
        {"run_id": "run-2", "condition": "spell", "name": "baseline/v1", "item_id": "three"}
    ]
