"""Unified record schema and JSONL helpers for benchmark runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class UnifiedRunRecord:
    run_id: str
    timestamp: str
    benchmark: str
    dataset: str
    split: str | None
    item_id: str
    condition: str
    provider: str | None
    model: str | None
    agent_file: str | None
    status: str
    response_text: str | None
    patch: str | None
    answer_extracted: str | None
    correct: bool | None
    uncached_input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    visible_output_tokens: int | None
    reasoning_output_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    error_type: str | None
    error_message: str | None
    metadata: dict[str, Any]
    name: str | None = None
    reasoning_effort: str | None = None

    @property
    def tokens_in(self) -> int | None:
        values = (
            self.uncached_input_tokens,
            self.cached_input_tokens,
            self.cache_write_input_tokens,
        )
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    @property
    def tokens_out(self) -> int | None:
        return self.visible_output_tokens


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path | str, record: UnifiedRunRecord | dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) if isinstance(record, UnifiedRunRecord) else record
    needs_separator = output_path.exists() and output_path.stat().st_size > 0
    if needs_separator:
        with output_path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_separator = f.read(1) != b"\n"
    with output_path.open("a", encoding="utf-8") as f:
        if needs_separator:
            f.write("\n")
        f.write(json.dumps(payload, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path | str, *, repair_truncated_tail: bool = False) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    text = input_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    prefix_end = 0
    for idx, line in enumerate(lines):
        line_start = prefix_end
        prefix_end += len(line)
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            remainder = "".join(lines[idx + 1 :]).strip()
            if remainder:
                raise
            if repair_truncated_tail:
                input_path.write_text(text[:line_start], encoding="utf-8")
            break
    return rows


def filter_jsonl_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    condition: str | None = None,
    name: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (run_id is None or row.get("run_id") == run_id)
        and (condition is None or row.get("condition") == condition)
        and (name is None or row.get("name") == name)
    ]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    errors = sum(1 for r in records if r.get("error_message"))
    correct_values = [r.get("correct") for r in records if r.get("correct") is not None]
    accuracy = (sum(1 for v in correct_values if v) / len(correct_values)) if correct_values else None
    costs = [r.get("cost_usd") for r in records if r.get("cost_usd") is not None]
    latencies = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]

    by_condition: dict[str, dict[str, Any]] = {}
    for r in records:
        condition = r.get("condition", "unknown")
        name = r.get("name")
        bucket_key = f"{condition}/{name}" if name else condition
        bucket = by_condition.setdefault(bucket_key, {"total": 0, "errors": 0, "correct": 0, "scored": 0, "cost_usd": 0.0})
        bucket["total"] += 1
        if r.get("error_message"):
            bucket["errors"] += 1
        if r.get("correct") is not None:
            bucket["scored"] += 1
            if r.get("correct"):
                bucket["correct"] += 1
        if r.get("cost_usd") is not None:
            bucket["cost_usd"] += float(r["cost_usd"])

    for bucket in by_condition.values():
        bucket["accuracy"] = (bucket["correct"] / bucket["scored"]) if bucket["scored"] else None

    return {
        "total": total,
        "errors": errors,
        "accuracy": accuracy,
        "total_cost_usd": sum(costs) if costs else None,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "by_condition": by_condition,
    }


def parse_model_spec(model_spec: str | None) -> tuple[str | None, str | None]:
    if not model_spec:
        return None, None
    if ":" not in model_spec:
        return None, model_spec
    provider, model = model_spec.split(":", 1)
    return provider, model
