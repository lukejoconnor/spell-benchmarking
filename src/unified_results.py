"""Unified record schema and JSONL helpers for benchmark runs."""

from __future__ import annotations

import json
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
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    latency_ms: float | None
    error_type: str | None
    error_message: str | None
    metadata: dict[str, Any]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path | str, record: UnifiedRunRecord | dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record) if isinstance(record, UnifiedRunRecord) else record
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        bucket = by_condition.setdefault(condition, {"total": 0, "errors": 0, "correct": 0, "scored": 0, "cost_usd": 0.0})
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
