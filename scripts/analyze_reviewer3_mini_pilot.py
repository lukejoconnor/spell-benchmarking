#!/usr/bin/env python3
"""Build an exact-item Spell vs mini-swe-agent pilot comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path], benchmark: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                item = row["item_id"]
                resolved = (
                    row.get("metadata", {}).get("same_container_resolved") is True
                    if benchmark == "swebench"
                    else row.get("correct") is True
                )
                candidate = {
                    "resolved": resolved,
                    "error": row.get("error_type"),
                    "cost_usd": row.get("cost_usd"),
                    "latency_ms": row.get("latency_ms"),
                    "source": str(path),
                }
                previous = rows.get(item)
                if previous is None or (previous["error"] and not candidate["error"]):
                    rows[item] = candidate
    return rows


def read_spell_swe(path: Path) -> dict[str, bool]:
    with path.open(newline="") as handle:
        return {
            row["item"]: row["spell_med"] == "1"
            for row in csv.DictReader(handle, delimiter="\t")
        }


def read_spell_tb(path: Path) -> dict[str, bool]:
    with path.open(newline="") as handle:
        return {
            row["item"]: row["resolved"] == "1"
            for row in csv.DictReader(handle, delimiter="\t")
            if row["condition_id"] == "gpt_default_med"
        }


def exact_mcnemar_p(spell_only: int, mini_only: int) -> float:
    discordant = spell_only + mini_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(spell_only, mini_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def summarize(
    benchmark: str,
    mini: dict[str, dict[str, Any]],
    spell: dict[str, bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing = sorted(set(mini) - set(spell))
    if missing:
        raise ValueError(f"{benchmark}: missing Spell outcomes for {missing}")
    errors = sorted(item for item, row in mini.items() if row["error"])
    if errors:
        raise ValueError(f"{benchmark}: unresolved infrastructure errors for {errors}")

    paired = []
    for item, mini_row in sorted(mini.items()):
        paired.append(
            {
                "benchmark": benchmark,
                "item": item,
                "spell_resolved": spell[item],
                "mini_resolved": mini_row["resolved"],
                "mini_cost_usd": mini_row["cost_usd"],
                "mini_latency_ms": mini_row["latency_ms"],
                "mini_source": mini_row["source"],
            }
        )

    both = sum(row["spell_resolved"] and row["mini_resolved"] for row in paired)
    spell_only = sum(row["spell_resolved"] and not row["mini_resolved"] for row in paired)
    mini_only = sum(not row["spell_resolved"] and row["mini_resolved"] for row in paired)
    neither = len(paired) - both - spell_only - mini_only
    costs = [row["mini_cost_usd"] for row in paired if row["mini_cost_usd"] is not None]
    latencies = [row["mini_latency_ms"] / 1000 for row in paired if row["mini_latency_ms"] is not None]
    summary = {
        "benchmark": benchmark,
        "n": len(paired),
        "spell_resolved": both + spell_only,
        "mini_resolved": both + mini_only,
        "both": both,
        "spell_only": spell_only,
        "mini_only": mini_only,
        "neither": neither,
        "mcnemar_exact_p": exact_mcnemar_p(spell_only, mini_only),
        "mini_cost_usd": sum(costs),
        "mini_cost_per_item_usd": sum(costs) / len(paired),
        "mini_median_latency_s": statistics.median(latencies) if latencies else None,
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-swe", type=Path, nargs="+", required=True)
    parser.add_argument("--mini-tb", type=Path, nargs="+", required=True)
    parser.add_argument("--spell-swe", type=Path, required=True)
    parser.add_argument("--spell-tb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [
        summarize("SWE-bench Lite", read_jsonl(args.mini_swe, "swebench"), read_spell_swe(args.spell_swe)),
        summarize("Terminal-Bench 1.1", read_jsonl(args.mini_tb, "terminalbench"), read_spell_tb(args.spell_tb)),
    ]
    summaries = [comparison[0] for comparison in comparisons]
    paired = [row for comparison in comparisons for row in comparison[1]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "paired_items.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
