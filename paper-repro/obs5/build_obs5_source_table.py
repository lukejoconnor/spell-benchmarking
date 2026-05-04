#!/usr/bin/env python3
"""Build the Figure 5 benchmark-summary source tables from raw artifacts.

The script is intentionally rooted in the benchmark artifacts rather than in
notebook summary tables. It writes the table consumed by the paper Figure 5
renderer plus the non-token companion table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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

TOKEN_FIELDNAMES = tuple(
    name for field in TOKEN_FIELDS for name in (field, f"{field}_per_item")
)

FIELDNAMES = (
    "benchmark",
    "n_items",
    "system",
    "n_pass",
    "cost_usd",
    "p_value",
    "significance",
    "source",
    "note",
    *TOKEN_FIELDNAMES,
)

BASE_FIELDNAMES = (
    "benchmark",
    "n_items",
    "system",
    "n_pass",
    "cost_usd",
    "p_value",
    "significance",
    "source",
    "note",
)


@dataclass(frozen=True)
class ArtifactSpec:
    benchmark: str
    system: str
    patterns: tuple[str, ...]
    scorer: str
    source: str
    note: str


def default_spell_root() -> Path:
    # .../spell/benchmarking/paper-repro/obs5/build_obs5_source_table.py
    return Path(__file__).resolve().parents[3]


def terminal_or_general_passed(record: dict) -> bool:
    return record.get("correct") is True


def swe_passed(record: dict) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and "same_container_resolved" in metadata:
        return metadata.get("same_container_resolved") is True
    return record.get("same_container_resolved") is True or record.get("correct") is True


SCORERS: dict[str, Callable[[dict], bool]] = {
    "correct": terminal_or_general_passed,
    "swe": swe_passed,
}


SPECS = (
    ArtifactSpec(
        benchmark="Terminal-Bench 1.1",
        system="Spell",
        patterns=("benchmarking/results/gcp/paper-fig3-tb-spell-gpt54-medium/terminal-bench/*.jsonl",),
        scorer="correct",
        source="benchmarking/results/gcp/paper-fig3-tb-spell-gpt54-medium/terminal-bench/*.jsonl",
        note="GPT-5.4 medium paper-branch replication; paired p-value recomputed from matching item IDs.",
    ),
    ArtifactSpec(
        benchmark="Terminal-Bench 1.1",
        system="Codex CLI",
        patterns=("results/paper-fig3-tb-codex-2026-04-30/paper-fig3-tb-codex-gpt54-medium/terminal-bench/*.jsonl",),
        scorer="correct",
        source="results/paper-fig3-tb-codex-2026-04-30/paper-fig3-tb-codex-gpt54-medium/terminal-bench/*.jsonl",
        note="GPT-5.4 medium paper-branch replication; paired p-value recomputed from matching item IDs.",
    ),
    ArtifactSpec(
        benchmark="SWE-bench Lite",
        system="Spell",
        patterns=("benchmarking/results/gcp/sbl-spell-medium-s*/unified/*.jsonl",),
        scorer="swe",
        source="benchmarking/results/gcp/sbl-spell-medium-s*/unified/*.jsonl",
        note="GPT-5.4 medium paper-compliant sweep; no issue hints; no benchmark-harness exclusions applied.",
    ),
    ArtifactSpec(
        benchmark="SWE-bench Lite",
        system="Codex CLI",
        patterns=("benchmarking/results/gcp/sbl-codex-medium-s*/unified/*.jsonl",),
        scorer="swe",
        source="benchmarking/results/gcp/sbl-codex-medium-s*/unified/*.jsonl",
        note="GPT-5.4 medium paper-compliant sweep; no issue hints; no benchmark-harness exclusions applied.",
    ),
    ArtifactSpec(
        benchmark="LongBench v2",
        system="Spell",
        patterns=(
            "benchmarking/results/gcp/lbp54m-simple-20260501-pilot-spell/unified/*.jsonl",
            "benchmarking/results/gcp/lbp54m-simple-20260501-rest-spell/unified/*.jsonl",
        ),
        scorer="correct",
        source="benchmarking/results/gcp/lbp54m-simple-20260501-{pilot,rest}-spell/unified/*.jsonl",
        note="GPT-5.4 medium paper/paper rerun with plain `'(!extend)` trailing; no benchmark-harness exclusions applied.",
    ),
    ArtifactSpec(
        benchmark="LongBench v2",
        system="Codex CLI",
        patterns=(
            "benchmarking/results/gcp/lbp54m-20260501-pilot-codex/unified/*.jsonl",
            "benchmarking/results/gcp/lbp54m-20260501-rest-codex/unified/*.jsonl",
        ),
        scorer="correct",
        source="benchmarking/results/gcp/lbp54m-20260501-{pilot,rest}-codex/unified/*.jsonl",
        note="GPT-5.4 medium paper/paper rerun; no benchmark-harness exclusions applied.",
    ),
    ArtifactSpec(
        benchmark="AppWorld dev",
        system="Spell",
        patterns=("benchmarking/results/gcp/appworld-full-spell/terminal-bench/terminalbench_20260501_200559_*.jsonl",),
        scorer="correct",
        source="benchmarking/results/gcp/appworld-full-spell/terminal-bench/terminalbench_20260501_200559_*.jsonl",
        note="GPT-5.4 medium paper/paper full dev-set rerun with plain `'(!extend)` trailing; filename-pinned to the simple-trailing rerun (older `:coding` artifact coexists in the same directory); no benchmark-harness exclusions applied.",
    ),
    ArtifactSpec(
        benchmark="AppWorld dev",
        system="Codex CLI",
        patterns=("benchmarking/results/gcp/appworld-full-codex/terminal-bench/*.jsonl",),
        scorer="correct",
        source="benchmarking/results/gcp/appworld-full-codex/terminal-bench/*.jsonl",
        note="GPT-5.4 medium paper/paper full dev-set rerun; no benchmark-harness exclusions applied.",
    ),
)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def match_paths(spell_root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(str(spell_root / pattern))))
    if not paths:
        raise FileNotFoundError(f"no raw artifacts matched: {', '.join(patterns)}")
    return paths


def load_records(spell_root: Path, patterns: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for path in match_paths(spell_root, patterns):
        records.extend(read_jsonl(path))
    return records


def as_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def usage(record: dict) -> dict[str, int]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    agent = metadata.get("tb_agent_result") if isinstance(metadata.get("tb_agent_result"), dict) else {}
    raw = metadata.get("raw") if isinstance(metadata.get("raw"), dict) else {}
    raw_usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    raw_total = raw_usage.get("total") if isinstance(raw_usage.get("total"), dict) else {}

    cached = (
        as_int(record.get("cached_input_tokens"))
        or as_int(agent.get("cached_input_tokens"))
        or as_int(raw_total.get("cached_input_tokens"))
        or as_int(raw_total.get("cache_read_input_tokens"))
    )
    cache_write = (
        as_int(record.get("cache_write_input_tokens"))
        or as_int(agent.get("cache_write_input_tokens"))
        or as_int(raw_total.get("cache_write_input_tokens"))
        or as_int(raw_total.get("cache_creation_input_tokens"))
    )
    uncached = (
        as_int(record.get("uncached_input_tokens"))
        or as_int(agent.get("uncached_input_tokens"))
        or as_int(raw_total.get("uncached_input_tokens"))
    )
    total_input = (
        as_int(record.get("total_input_tokens"))
        or as_int(record.get("tokens_in"))
        or as_int(agent.get("total_input_tokens"))
        or as_int(agent.get("input_tokens"))
        or as_int(raw_total.get("input_tokens"))
    )
    if not uncached and total_input:
        uncached = max(total_input - cached - cache_write, 0)
    total_input = uncached + cached + cache_write

    reasoning = (
        as_int(record.get("reasoning_output_tokens"))
        or as_int(metadata.get("reasoning_output_tokens"))
        or as_int(agent.get("reasoning_output_tokens"))
        or as_int(raw_total.get("reasoning_output_tokens"))
        or as_int(raw_total.get("reasoning_tokens"))
    )
    visible = (
        as_int(record.get("visible_output_tokens"))
        or as_int(record.get("tokens_out"))
        or as_int(agent.get("visible_output_tokens"))
        or as_int(agent.get("output_tokens"))
        or as_int(raw_total.get("visible_output_tokens"))
    )
    total_output = (
        as_int(record.get("total_output_tokens"))
        or as_int(agent.get("total_output_tokens"))
        or as_int(raw_total.get("output_tokens"))
    )
    if not visible and total_output:
        visible = max(total_output - reasoning, 0)

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


def sign_test_p_value(spell: dict[str, bool], codex: dict[str, bool]) -> float:
    keys = set(spell) & set(codex)
    wins = sum(spell[key] and not codex[key] for key in keys)
    losses = sum((not spell[key]) and codex[key] for key in keys)
    n_discordant = wins + losses
    if n_discordant == 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2**n_discordant)
    return min(1.0, p)


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def aggregate(spec: ArtifactSpec, records: list[dict]) -> tuple[dict[str, object], dict[str, bool]]:
    scorer = SCORERS[spec.scorer]
    outcomes = {record["item_id"]: scorer(record) for record in records}
    if len(outcomes) != len(records):
        raise RuntimeError(f"{spec.benchmark}/{spec.system} has duplicate item IDs")

    totals = {field: 0 for field in TOKEN_FIELDS}
    for record in records:
        per_record = usage(record)
        for field in TOKEN_FIELDS:
            totals[field] += per_record[field]

    n_items = len(records)
    row: dict[str, object] = {
        "benchmark": spec.benchmark,
        "n_items": n_items,
        "system": spec.system,
        "n_pass": sum(outcomes.values()),
        "cost_usd": f"{sum(float(record.get('cost_usd') or 0.0) for record in records):.2f}",
        "p_value": "",
        "significance": "",
        "source": spec.source,
        "note": spec.note,
    }
    for field in TOKEN_FIELDS:
        row[field] = totals[field]
        row[f"{field}_per_item"] = totals[field] / n_items if n_items else 0.0
    return row, outcomes


def build_rows(spell_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outcomes_by_benchmark: dict[str, dict[str, dict[str, bool]]] = {}
    for spec in SPECS:
        records = load_records(spell_root, spec.patterns)
        row, outcomes = aggregate(spec, records)
        rows.append(row)
        outcomes_by_benchmark.setdefault(spec.benchmark, {})[spec.system] = outcomes

    for idx, row in enumerate(rows):
        systems = outcomes_by_benchmark[row["benchmark"]]
        if set(systems) != {"Spell", "Codex CLI"}:
            raise RuntimeError(f"missing paired systems for {row['benchmark']}: {sorted(systems)}")
        if row["system"] == "Spell":
            p_value = sign_test_p_value(systems["Spell"], systems["Codex CLI"])
            p_value_s = repr(p_value)
            sig = significance(p_value)
            row["p_value"] = p_value_s
            row["significance"] = sig
            rows[idx + 1]["p_value"] = p_value_s
            rows[idx + 1]["significance"] = sig
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spell-root", type=Path, default=default_spell_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for obs5_benchmark_summary_data_with_tokens.tsv",
    )
    parser.add_argument(
        "--base-output",
        type=Path,
        default=None,
        help="Path for obs5_benchmark_summary_data.tsv",
    )
    parser.add_argument("--check", action="store_true", help="Fail if outputs would change")
    args = parser.parse_args()

    spell_root = args.spell_root.resolve()
    output = args.output or spell_root / "notebook/writeup/paper/figures/obs5_benchmark_summary_data_with_tokens.tsv"
    base_output = args.base_output or spell_root / "notebook/writeup/paper/figures/obs5_benchmark_summary_data.tsv"
    rows = build_rows(spell_root)

    import io

    def render(fieldnames: Iterable[str]) -> str:
        handle = io.StringIO()
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return handle.getvalue()

    rendered = render(FIELDNAMES)
    rendered_base = render(BASE_FIELDNAMES)
    if args.check:
        diffs = []
        if output.exists() and output.read_text() != rendered:
            diffs.append(str(output))
        if base_output.exists() and base_output.read_text() != rendered_base:
            diffs.append(str(base_output))
        if diffs:
            raise SystemExit("outputs are stale: " + ", ".join(diffs))
        print("Obs 5 source tables are up to date")
        return

    write_tsv(output, rows, FIELDNAMES)
    write_tsv(base_output, rows, BASE_FIELDNAMES)
    print(f"wrote {output}")
    print(f"wrote {base_output}")


if __name__ == "__main__":
    main()
