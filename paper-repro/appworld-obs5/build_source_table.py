#!/usr/bin/env python3
"""Update Figure 5 AppWorld rows from raw GCP Terminal-Bench JSONLs.

This script is intentionally scoped to AppWorld so it can be run while other
agents refresh LongBench or other Figure 5 rows in parallel. It validates the
57-item `appworld-dev` set, computes paired Spell-vs-Codex statistics from the
raw unified JSONL records, and rewrites only the AppWorld rows in the paper's
Figure 5 source tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parents[1]
SPELL_ROOT = BENCHMARK_ROOT.parent
NOTEBOOK_ROOT = SPELL_ROOT / "notebook"
FIGURE_DIR = NOTEBOOK_ROOT / "writeup" / "paper" / "figures"
DEFAULT_BASE_TABLE = FIGURE_DIR / "obs5_benchmark_summary_data.tsv"
DEFAULT_TOKEN_TABLE = FIGURE_DIR / "obs5_benchmark_summary_data_with_tokens.tsv"
DEFAULT_PER_ITEM = SCRIPT_DIR / "appworld_obs5_per_item.tsv"
ITEMS_FILE = SCRIPT_DIR / "appworld-dev-items.txt"
TOKEN_FIELDS = (
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "total_input_tokens",
    "visible_output_tokens",
    "reasoning_output_tokens",
    "output_tokens",
    "total_tokens",
)

DEFAULT_SPELL_JSONL = (
    BENCHMARK_ROOT
    / "results"
    / "gcp"
    / "appworld-full-spell"
    / "terminal-bench"
    / "terminalbench_20260501_093549_a6450522.jsonl"
)
DEFAULT_CODEX_JSONL = (
    BENCHMARK_ROOT
    / "results"
    / "gcp"
    / "appworld-full-codex"
    / "terminal-bench"
    / "terminalbench_20260501_093737_4b6649d8.jsonl"
)


def load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record["item_id"]
            if item_id in records:
                raise RuntimeError(f"duplicate item_id {item_id!r} in {path}")
            records[item_id] = record
    return records


def load_items(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def load_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cost(records: dict[str, dict]) -> float:
    return sum(float(record.get("cost_usd") or 0.0) for record in records.values())


def number(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def usage(record: dict) -> dict[str, int]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    agent = metadata.get("tb_agent_result") if isinstance(metadata.get("tb_agent_result"), dict) else {}
    total_input = (
        number(record.get("total_input_tokens"))
        or number(record.get("tokens_in"))
        or number(agent.get("total_input_tokens"))
        or number(agent.get("input_tokens"))
    )
    cached = number(record.get("cached_input_tokens")) or number(agent.get("cached_input_tokens"))
    cache_write = number(record.get("cache_write_input_tokens")) or number(agent.get("cache_write_input_tokens"))
    uncached = number(record.get("uncached_input_tokens"))
    if uncached == 0 and total_input:
        uncached = max(total_input - cached - cache_write, 0)

    reasoning = (
        number(record.get("reasoning_output_tokens"))
        or number(metadata.get("reasoning_output_tokens"))
        or number(agent.get("reasoning_output_tokens"))
    )
    visible = (
        number(record.get("visible_output_tokens"))
        or number(record.get("tokens_out"))
        or number(agent.get("visible_output_tokens"))
        or number(agent.get("output_tokens"))
    )
    total_output = number(record.get("total_output_tokens")) or number(agent.get("total_output_tokens"))
    if visible == 0 and total_output:
        visible = max(total_output - reasoning, 0)

    total_input = uncached + cached + cache_write
    output = visible + reasoning
    return {
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "total_input_tokens": total_input,
        "visible_output_tokens": visible,
        "reasoning_output_tokens": reasoning,
        "output_tokens": output,
        "total_tokens": total_input + output,
    }


def usage_totals(records: dict[str, dict]) -> dict[str, float]:
    totals = {field: 0 for field in TOKEN_FIELDS}
    for record in records.values():
        row_usage = usage(record)
        for field in TOKEN_FIELDS:
            totals[field] += row_usage[field]
    n_items = len(records)
    return {
        **totals,
        **{f"{field}_per_item": totals[field] / n_items for field in TOKEN_FIELDS},
    }


def passed(records: dict[str, dict]) -> dict[str, bool]:
    return {item_id: record.get("correct") is True for item_id, record in records.items()}


def sign_test_p_value(spell: dict[str, bool], codex: dict[str, bool]) -> float:
    spell_only = sum(spell[item] and not codex[item] for item in spell)
    codex_only = sum((not spell[item]) and codex[item] for item in spell)
    n = spell_only + codex_only
    if n == 0:
        return 1.0
    k = min(spell_only, codex_only)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def rel_source(path: Path) -> str:
    try:
        return str(path.relative_to(SPELL_ROOT))
    except ValueError:
        return str(path)


def validate_items(expected: list[str], spell: dict[str, dict], codex: dict[str, dict]) -> None:
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise RuntimeError("appworld-dev-items.txt contains duplicate items")
    for label, records in (("Spell", spell), ("Codex", codex)):
        actual = set(records)
        if actual != expected_set:
            missing = sorted(expected_set - actual)
            extra = sorted(actual - expected_set)
            raise RuntimeError(f"{label} item mismatch: missing={missing} extra={extra}")


def update_appworld_rows(
    table_path: Path,
    *,
    spell: dict[str, dict],
    codex: dict[str, dict],
    note: str,
) -> None:
    rows, fieldnames = load_tsv(table_path)
    spell_outcomes = passed(spell)
    codex_outcomes = passed(codex)
    p_value = sign_test_p_value(spell_outcomes, codex_outcomes)
    star = significance(p_value)
    values = {
        "Spell": {
            "n_items": len(spell),
            "n_pass": sum(spell_outcomes.values()),
            "cost_usd": f"{cost(spell):.2f}",
            "source": "benchmarking/results/gcp/appworld-full-spell/terminal-bench/*.jsonl",
            **usage_totals(spell),
        },
        "Codex CLI": {
            "n_items": len(codex),
            "n_pass": sum(codex_outcomes.values()),
            "cost_usd": f"{cost(codex):.2f}",
            "source": "benchmarking/results/gcp/appworld-full-codex/terminal-bench/*.jsonl",
            **usage_totals(codex),
        },
    }

    changed = 0
    for row in rows:
        if row.get("benchmark") != "AppWorld dev":
            continue
        system = row["system"]
        for key, value in values[system].items():
            if key in fieldnames:
                row[key] = value
        row["p_value"] = repr(p_value)
        row["significance"] = star
        row["note"] = note
        changed += 1
    if changed == 0:
        for system in ("Spell", "Codex CLI"):
            row = {field: "" for field in fieldnames}
            row["benchmark"] = "AppWorld dev"
            row["system"] = system
            for key, value in values[system].items():
                if key in fieldnames:
                    row[key] = value
            row["p_value"] = repr(p_value)
            row["significance"] = star
            row["note"] = note
            rows.append(row)
        changed = 2
    if changed != 2:
        raise RuntimeError(f"expected 2 AppWorld rows in {table_path}, updated {changed}")
    write_tsv(table_path, rows, fieldnames)


def write_per_item(path: Path, spell: dict[str, dict], codex: dict[str, dict], items: list[str]) -> None:
    fieldnames = [
        "item_id",
        "spell_correct",
        "codex_correct",
        "outcome",
        "spell_cost_usd",
        "codex_cost_usd",
        "spell_latency_s",
        "codex_latency_s",
        "spell_error_type",
        "codex_error_type",
    ]
    rows: list[dict[str, object]] = []
    spell_outcomes = passed(spell)
    codex_outcomes = passed(codex)
    for item in items:
        s = spell[item]
        c = codex[item]
        if spell_outcomes[item] and codex_outcomes[item]:
            outcome = "both_pass"
        elif spell_outcomes[item] and not codex_outcomes[item]:
            outcome = "spell_only"
        elif (not spell_outcomes[item]) and codex_outcomes[item]:
            outcome = "codex_only"
        else:
            outcome = "both_fail"
        rows.append(
            {
                "item_id": item,
                "spell_correct": spell_outcomes[item],
                "codex_correct": codex_outcomes[item],
                "outcome": outcome,
                "spell_cost_usd": f"{float(s.get('cost_usd') or 0.0):.6f}",
                "codex_cost_usd": f"{float(c.get('cost_usd') or 0.0):.6f}",
                "spell_latency_s": f"{float(s.get('latency_ms') or 0.0) / 1000.0:.3f}",
                "codex_latency_s": f"{float(c.get('latency_ms') or 0.0) / 1000.0:.3f}",
                "spell_error_type": s.get("error_type") or "",
                "codex_error_type": c.get("error_type") or "",
            }
        )
    write_tsv(path, rows, fieldnames)


def summarize(spell: dict[str, dict], codex: dict[str, dict]) -> str:
    spell_outcomes = passed(spell)
    codex_outcomes = passed(codex)
    spell_only = sum(spell_outcomes[item] and not codex_outcomes[item] for item in spell)
    codex_only = sum((not spell_outcomes[item]) and codex_outcomes[item] for item in spell)
    both_pass = sum(spell_outcomes[item] and codex_outcomes[item] for item in spell)
    both_fail = sum((not spell_outcomes[item]) and (not codex_outcomes[item]) for item in spell)
    p_value = sign_test_p_value(spell_outcomes, codex_outcomes)
    return (
        f"Spell {sum(spell_outcomes.values())}/{len(spell)} (${cost(spell):.2f}); "
        f"Codex {sum(codex_outcomes.values())}/{len(codex)} (${cost(codex):.2f}); "
        f"H2H spell_only={spell_only} codex_only={codex_only} both_pass={both_pass} both_fail={both_fail}; "
        f"paired p={p_value!r} {significance(p_value)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spell-jsonl", type=Path, default=DEFAULT_SPELL_JSONL)
    parser.add_argument("--codex-jsonl", type=Path, default=DEFAULT_CODEX_JSONL)
    parser.add_argument("--base-table", type=Path, default=DEFAULT_BASE_TABLE)
    parser.add_argument("--token-table", type=Path, default=DEFAULT_TOKEN_TABLE)
    parser.add_argument("--per-item", type=Path, default=DEFAULT_PER_ITEM)
    parser.add_argument("--check", action="store_true", help="validate and summarize without writing outputs")
    args = parser.parse_args()

    items = load_items(ITEMS_FILE)
    spell = load_jsonl(args.spell_jsonl)
    codex = load_jsonl(args.codex_jsonl)
    validate_items(items, spell, codex)

    note = "GPT-5.4 medium paper/paper full dev-set rerun; no benchmark-harness exclusions applied."
    print(summarize(spell, codex))
    if args.check:
        return

    update_appworld_rows(args.base_table, spell=spell, codex=codex, note=note)
    update_appworld_rows(args.token_table, spell=spell, codex=codex, note=note)
    write_per_item(args.per_item, spell, codex, items)
    print(f"wrote {args.base_table}")
    print(f"wrote {args.token_table}")
    print(f"wrote {args.per_item}")


if __name__ == "__main__":
    main()
