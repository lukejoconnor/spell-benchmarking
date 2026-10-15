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


def _needs_record_separator(output_path: Path) -> bool:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False

    with output_path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def append_jsonl(
    path: Path | str,
    record: UnifiedRunRecord | dict[str, Any],
    *,
    flush: bool = False,
    fsync: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) if isinstance(record, UnifiedRunRecord) else record
    prefix = "\n" if _needs_record_separator(output_path) else ""
    with output_path.open("a", encoding="utf-8") as f:
        f.write(prefix + json.dumps(payload, default=str) + "\n")
        if flush or fsync:
            f.flush()
        if fsync:
            os.fsync(f.fileno())


def load_jsonl(path: Path | str, *, tolerate_invalid: bool = False) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if not tolerate_invalid:
                        raise
    return rows


def record_outcome(record: dict[str, Any]) -> bool | None:
    """Read the benchmark's score; an execution error is an unsuccessful item."""
    if record.get("benchmark") == "swebench":
        resolved = (record.get("metadata") or {}).get("same_container_resolved")
        if resolved is not None:
            return bool(resolved)
    if record.get("error_message") or record.get("error_type") or record.get("status") in {
        "error", "timeout", "container_error", "agent_timeout", "test_timeout",
    }:
        return False
    if record.get("correct") is not None:
        return bool(record["correct"])
    return None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    errors = sum(1 for r in records if r.get("error_message") or r.get("error_type") or r.get("status") in {"error", "timeout"})
    outcomes = [record_outcome(r) for r in records]
    scored = sum(value is not None for value in outcomes)
    accuracy = (sum(value is True for value in outcomes) / total) if scored else None
    costs = [r.get("cost_usd") for r in records if r.get("cost_usd") is not None]
    latencies = [r.get("latency_ms") for r in records if r.get("latency_ms") is not None]

    by_condition: dict[str, dict[str, Any]] = {}
    for r, outcome in zip(records, outcomes):
        condition = r.get("condition", "unknown")
        name = r.get("name")
        bucket_key = f"{condition}/{name}" if name else condition
        bucket = by_condition.setdefault(bucket_key, {"total": 0, "errors": 0, "correct": 0, "scored": 0, "cost_usd": 0.0})
        bucket["total"] += 1
        if r.get("error_message") or r.get("error_type") or r.get("status") in {"error", "timeout"}:
            bucket["errors"] += 1
        if outcome is not None:
            bucket["scored"] += 1
            if outcome:
                bucket["correct"] += 1
        if r.get("cost_usd") is not None:
            bucket["cost_usd"] += float(r["cost_usd"])

    for bucket in by_condition.values():
        bucket["accuracy"] = (bucket["correct"] / bucket["total"]) if bucket["scored"] else None
        bucket["unscored"] = bucket["total"] - bucket["scored"]

    return {
        "total": total,
        "errors": errors,
        "scored": scored,
        "unscored": total - scored,
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
