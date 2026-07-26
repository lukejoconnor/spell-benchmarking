#!/usr/bin/env python3
"""Paired GPT-5.6 Sol versus GPT-5.4 analysis for the Reviewer 3 pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


LABELS = ("GPT-5.4", "GPT-5.6 Sol")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_two_sided_sign_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, successes)
        for successes in range(min(first_only, second_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def latency_ms(record: dict) -> float | None:
    if record.get("latency_ms") is not None:
        return record["latency_ms"]
    raw = (record.get("metadata") or {}).get("raw") or {}
    return raw.get("latency_ms")


def benchmark_comparison(
    records_by_label: dict[str, list[dict]],
    *,
    success,
    excluded_items: set[str],
) -> dict:
    indexed = {
        label: {
            record["item_id"]: record
            for record in records
            if record["item_id"] not in excluded_items
        }
        for label, records in records_by_label.items()
    }
    item_sets = {label: set(records) for label, records in indexed.items()}
    if len({frozenset(items) for items in item_sets.values()}) != 1:
        raise ValueError(f"Compared item sets differ: {item_sets}")
    items = sorted(next(iter(item_sets.values())))

    outcomes = {
        label: {item: bool(success(indexed[label][item])) for item in items}
        for label in LABELS
    }
    first_only = [
        item for item in items if outcomes[LABELS[0]][item] and not outcomes[LABELS[1]][item]
    ]
    second_only = [
        item for item in items if outcomes[LABELS[1]][item] and not outcomes[LABELS[0]][item]
    ]
    both = [
        item for item in items if outcomes[LABELS[0]][item] and outcomes[LABELS[1]][item]
    ]
    neither = [
        item
        for item in items
        if not outcomes[LABELS[0]][item] and not outcomes[LABELS[1]][item]
    ]

    per_model = {}
    error_items_by_label: dict[str, set[str]] = {}
    for label in LABELS:
        records = [indexed[label][item] for item in items]
        errors = [
            record
            for record in records
            if record.get("status") in {"error", "timeout"}
            or record.get("error_type") is not None
        ]
        error_items_by_label[label] = {record["item_id"] for record in errors}
        costs = [
            record["cost_usd"] for record in records if record.get("cost_usd") is not None
        ]
        latencies = [
            value
            for record in records
            if (value := latency_ms(record)) is not None
        ]
        per_model[label] = {
            "resolved": sum(outcomes[label].values()),
            "errors": [
                {
                    "item_id": record["item_id"],
                    "status": record.get("status"),
                    "error_type": record.get("error_type"),
                    "error_message": record.get("error_message"),
                    "cost_usd": record.get("cost_usd"),
                    "latency_seconds": (
                        latency_ms(record) / 1000
                        if latency_ms(record) is not None
                        else None
                    ),
                    "input_tokens": (
                        (record.get("uncached_input_tokens") or 0)
                        + (record.get("cached_input_tokens") or 0)
                    ),
                    "output_tokens": (
                        (record.get("visible_output_tokens") or 0)
                        + (record.get("reasoning_output_tokens") or 0)
                    ),
                }
                for record in errors
            ],
            "known_cost_usd": sum(costs),
            "unknown_cost_rows": len(records) - len(costs),
            "max_item_cost_usd": max(costs),
            "median_latency_seconds": statistics.median(latencies) / 1000,
            "mean_latency_seconds": statistics.mean(latencies) / 1000,
        }

    clean_items = sorted(set(items) - set.union(*error_items_by_label.values()))
    clean_first_only = [
        item
        for item in clean_items
        if outcomes[LABELS[0]][item] and not outcomes[LABELS[1]][item]
    ]
    clean_second_only = [
        item
        for item in clean_items
        if outcomes[LABELS[1]][item] and not outcomes[LABELS[0]][item]
    ]

    return {
        "n": len(items),
        "both_resolved": len(both),
        "neither_resolved": len(neither),
        "gpt54_only": first_only,
        "gpt56_only": second_only,
        "exact_sign_test_p": exact_two_sided_sign_p(len(first_only), len(second_only)),
        "rows_without_errors_in_either_model": {
            "n": len(clean_items),
            "gpt54_resolved": sum(outcomes[LABELS[0]][item] for item in clean_items),
            "gpt56_resolved": sum(outcomes[LABELS[1]][item] for item in clean_items),
            "gpt54_only": clean_first_only,
            "gpt56_only": clean_second_only,
            "exact_sign_test_p": exact_two_sided_sign_p(
                len(clean_first_only), len(clean_second_only)
            ),
        },
        "models": per_model,
    }


def source_files(summary: dict, field: str) -> dict[str, Path]:
    selected = {
        model["label"]: Path(model[field])
        for model in summary["models"]
        if model["label"] in LABELS
    }
    if set(selected) != set(LABELS):
        raise ValueError(f"Missing model source files: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-summary", type=Path, required=True)
    parser.add_argument("--swe-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    terminal_summary = json.loads(args.terminal_summary.read_text())
    swe_summary = json.loads(args.swe_summary.read_text())

    terminal_files = source_files(terminal_summary, "original_source_file")
    swe_files = source_files(swe_summary, "source_file")

    terminal = benchmark_comparison(
        {label: load_jsonl(path) for label, path in terminal_files.items()},
        success=lambda record: record.get("correct") is True,
        excluded_items=set(),
    )
    swe = benchmark_comparison(
        {label: load_jsonl(path) for label, path in swe_files.items()},
        success=lambda record: (record.get("metadata") or {}).get(
            "same_container_resolved"
        )
        is True,
        excluded_items=set(swe_summary["shared_infrastructure_failures"]),
    )

    output = {
        "comparison": "GPT-5.6 Sol versus GPT-5.4, both medium effort",
        "terminal_bench": terminal,
        "swe_bench_lite": swe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
