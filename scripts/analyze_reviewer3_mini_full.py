#!/usr/bin/env python3
"""Build exact-item full-run Spell vs mini-swe-agent comparisons."""

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


def exact_mcnemar_p(spell_only: int, mini_only: int) -> float:
    discordant = spell_only + mini_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(spell_only, mini_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def summarize(
    benchmark: str,
    mini: dict[str, dict[str, Any]],
    spell: dict[str, dict[str, Any]],
    expected_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(mini) != expected_n:
        raise ValueError(f"{benchmark}: expected {expected_n} mini rows, found {len(mini)}")
    if len(spell) != expected_n:
        raise ValueError(f"{benchmark}: expected {expected_n} Spell rows, found {len(spell)}")
    mini_only_items = sorted(set(mini) - set(spell))
    spell_only_items = sorted(set(spell) - set(mini))
    if mini_only_items or spell_only_items:
        raise ValueError(
            f"{benchmark}: item mismatch; mini-only={mini_only_items}, Spell-only={spell_only_items}"
        )
    paired = []
    for item in sorted(mini):
        mini_row = mini[item]
        spell_row = spell[item]
        paired.append(
            {
                "benchmark": benchmark,
                "item": item,
                "spell_resolved": spell_row["resolved"],
                "mini_resolved": mini_row["resolved"],
                "spell_cost_usd": spell_row["cost_usd"],
                "mini_cost_usd": mini_row["cost_usd"],
                "spell_latency_ms": spell_row["latency_ms"],
                "mini_latency_ms": mini_row["latency_ms"],
                "spell_source": spell_row["source"],
                "mini_source": mini_row["source"],
            }
        )

    both = sum(row["spell_resolved"] and row["mini_resolved"] for row in paired)
    spell_only = sum(row["spell_resolved"] and not row["mini_resolved"] for row in paired)
    mini_only = sum(not row["spell_resolved"] and row["mini_resolved"] for row in paired)
    neither = len(paired) - both - spell_only - mini_only
    spell_costs = [
        float(row["spell_cost_usd"]) for row in paired if row["spell_cost_usd"] is not None
    ]
    mini_costs = [
        float(row["mini_cost_usd"]) for row in paired if row["mini_cost_usd"] is not None
    ]
    spell_cost = sum(spell_costs)
    mini_cost = sum(mini_costs)
    mini_latencies = [
        float(row["mini_latency_ms"]) / 1000
        for row in paired
        if row["mini_latency_ms"] is not None
    ]
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
        "spell_error_items": sorted(item for item, row in spell.items() if row["error"]),
        "mini_error_items": sorted(item for item, row in mini.items() if row["error"]),
        "spell_cost_usd": spell_cost,
        "spell_cost_n": len(spell_costs),
        "mini_cost_usd": mini_cost,
        "mini_cost_n": len(mini_costs),
        "mini_minus_spell_cost_usd": mini_cost - spell_cost,
        "mini_cost_ratio_vs_spell": mini_cost / spell_cost,
        "spell_cost_per_item_usd": spell_cost / len(paired),
        "mini_cost_per_item_usd": mini_cost / len(paired),
        "spell_cost_per_resolved_usd": spell_cost / (both + spell_only),
        "mini_cost_per_resolved_usd": mini_cost / (both + mini_only),
        "mini_median_latency_s": statistics.median(mini_latencies) if mini_latencies else None,
    }
    return summary, paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-swe", type=Path, nargs="+", required=True)
    parser.add_argument("--mini-tb", type=Path, nargs="+", required=True)
    parser.add_argument("--spell-swe", type=Path, nargs="+", required=True)
    parser.add_argument("--spell-tb", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [
        summarize(
            "SWE-bench Lite",
            read_jsonl(args.mini_swe, "swebench"),
            read_jsonl(args.spell_swe, "swebench"),
            300,
        ),
        summarize(
            "Terminal-Bench 1.1",
            read_jsonl(args.mini_tb, "terminalbench"),
            read_jsonl(args.spell_tb, "terminalbench"),
            80,
        ),
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
