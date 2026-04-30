#!/usr/bin/env python3
"""Token-level pruning accounting for paper Spell traces.

This script uses Spell's Clojure trace walker to identify the exact source
forms removed by `prune`/`rethink`, then tokenizes the removed forms and each
turn's full traced completion source with `o200k_base`. In this analysis,
completion means prefix plus response, not just the response suffix.

Default run specs cover the six GPT-5.4 paper Spell rows:
SWE-bench Lite and Terminal-Bench 1.1, each at low/medium/high effort.

Run from the benchmarking repo with:

    uv run --with tiktoken python scripts/analyze_spell_pruning_tokens.py

Set SPELL_ROOT if the sibling Spell checkout is not `..`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import tiktoken
except ModuleNotFoundError as exc:
    raise SystemExit("tiktoken is required; run with `uv run --with tiktoken ...`") from exc


BENCH_ROOT = Path(__file__).resolve().parents[1]
RESULTS_GCP = BENCH_ROOT / "results" / "gcp"
TRACES_GCP = BENCH_ROOT / "traces" / "gcp"
DEFAULT_OUTPUT_DIR = BENCH_ROOT / "paper-repro" / "pruning-token-analysis"
TOKENIZER = tiktoken.get_encoding("o200k_base")
EFFORT_ORDER = {"low": 0, "med": 1, "medium": 1, "high": 2}
BENCHMARK_ORDER = {"swe-lite": 0, "terminal-bench": 1}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    benchmark: str
    label: str
    effort: str
    vm_prefixes: tuple[str, ...]
    result_subdir: str
    jsonl_glob: str
    trace_has_timestamp_child: bool


RUN_SPECS = (
    RunSpec(
        "swe_lite_low",
        "swe-lite",
        "SWE-bench Lite",
        "low",
        ("sbl-spell-low-s",),
        "unified",
        "*.jsonl",
        False,
    ),
    RunSpec(
        "swe_lite_med",
        "swe-lite",
        "SWE-bench Lite",
        "med",
        ("sbl-spell-medium-s",),
        "unified",
        "*.jsonl",
        False,
    ),
    RunSpec(
        "swe_lite_high",
        "swe-lite",
        "SWE-bench Lite",
        "high",
        ("sbl-spell-high-s",),
        "unified",
        "*.jsonl",
        False,
    ),
    RunSpec(
        "terminal_bench_low",
        "terminal-bench",
        "Terminal-Bench 1.1",
        "low",
        ("paper-fig3-tb-spell-gpt54-low",),
        "terminal-bench",
        "terminalbench_*.jsonl",
        True,
    ),
    RunSpec(
        "terminal_bench_med",
        "terminal-bench",
        "Terminal-Bench 1.1",
        "med",
        ("paper-fig3-tb-spell-gpt54-medium",),
        "terminal-bench",
        "terminalbench_*.jsonl",
        True,
    ),
    RunSpec(
        "terminal_bench_high",
        "terminal-bench",
        "Terminal-Bench 1.1",
        "high",
        ("paper-fig3-tb-spell-gpt54-high",),
        "terminal-bench",
        "terminalbench_*.jsonl",
        True,
    ),
)


CLOJURE_EXTRACTOR = r"""
(ns pruning-token-extract
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [spell.trace-tool :as tt]))

(defn completion-source [dir node]
  (if-let [file (:file node)]
    (let [source-file (io/file dir file)]
      (if (.exists source-file)
        (slurp source-file)
        (pr-str (:program node))))
    (pr-str (:program node))))

(defn trace-turns [dir trace]
  (->> (:nodes trace)
       (filter :program)
       (sort-by :id)
       (mapv (fn [node]
               {:node_id (:id node)
                :completion (completion-source dir node)
                :response (or (:response node) "")
                :file (:file node)}))))

(defn row [dir]
  (let [{:keys [trace]} (tt/load-trace dir)
        details (vec (tt/trace-pruning-accounting trace))]
    {:trace_dir dir
     :turns (trace-turns dir trace)
     :details (mapv (fn [detail]
                      {:node_id (:node-id detail)
                       :trigger (name (:trigger detail))
                       :kind (name (:kind detail))
                       :target_texts (mapv #(pr-str (:form %)) (:targets detail))
                       :chars_pruned (:chars-pruned detail)})
                    details)}))

(defn -main [dirs-file out-file]
  (with-open [r (io/reader dirs-file)
              w (io/writer out-file)]
    (doseq [dir (line-seq r)]
      (when (seq dir)
        (.write w
                (str (json/write-str
                      (try
                        (row dir)
                        (catch Throwable t
                          {:trace_dir dir
                           :error (.getMessage t)})))
                     "\n"))))))
"""


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, disallowed_special=()))


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def vm_matches(vm_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(vm_name == prefix or vm_name.startswith(prefix) for prefix in prefixes)


def effort_matches(record: dict, effort: str) -> bool:
    value = record.get("reasoning_effort")
    if value is None:
        return True
    if effort == "med":
        return value in {"med", "medium"}
    return value == effort


def collect_records(spec: RunSpec) -> dict[str, dict]:
    chosen: dict[str, tuple[str, dict]] = {}
    for vm_dir in sorted(RESULTS_GCP.iterdir()):
        if not vm_dir.is_dir() or not vm_matches(vm_dir.name, spec.vm_prefixes):
            continue
        result_dir = vm_dir / spec.result_subdir
        if not result_dir.exists():
            continue
        for jsonl_path in sorted(result_dir.glob(spec.jsonl_glob)):
            for record in load_jsonl(jsonl_path):
                if record.get("condition") != "spell" or not effort_matches(record, spec.effort):
                    continue
                item = record.get("item_id")
                if not item:
                    continue
                enriched = dict(record)
                enriched["_vm_name"] = vm_dir.name
                enriched["_jsonl_path"] = str(jsonl_path)
                timestamp = str(record.get("timestamp") or record.get("run_id") or "")
                previous = chosen.get(item)
                if previous is None or timestamp >= previous[0]:
                    chosen[item] = (timestamp, enriched)
    return {item: record for item, (_, record) in chosen.items()}


def locate_trace_dir(spec: RunSpec, record: dict) -> Path | None:
    vm = record.get("_vm_name")
    run_id = record.get("run_id")
    name = record.get("name")
    item = record.get("item_id")
    if not all([vm, run_id, name, item]):
        return None
    base = TRACES_GCP / vm / run_id / "spell" / name / item
    if not base.exists():
        return None
    if not spec.trace_has_timestamp_child:
        return base if (base / "trace.edn").exists() else None
    children = sorted(path for path in base.iterdir() if path.is_dir() and (path / "trace.edn").exists())
    return children[-1] if children else None


def collect_trace_dirs() -> tuple[list[dict], list[str]]:
    trace_rows = []
    warnings = []
    for spec in RUN_SPECS:
        records = collect_records(spec)
        for item, record in sorted(records.items()):
            trace_dir = locate_trace_dir(spec, record)
            if trace_dir is None:
                warnings.append(f"{spec.run_id}: missing trace for {item}")
                continue
            trace_rows.append(
                {
                    "run_id": spec.run_id,
                    "benchmark": spec.benchmark,
                    "label": spec.label,
                    "effort": spec.effort,
                    "item": item,
                    "trace_dir": str(trace_dir),
                    "jsonl_path": record.get("_jsonl_path", ""),
                }
            )
    return trace_rows, warnings


def run_clojure_extractor(spell_root: Path, trace_rows: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="spell-pruning-tokens-") as tmp:
        tmp_dir = Path(tmp)
        script = tmp_dir / "extract.clj"
        dirs_file = tmp_dir / "trace-dirs.txt"
        out_file = tmp_dir / "trace-rows.jsonl"
        script.write_text(CLOJURE_EXTRACTOR)
        dirs_file.write_text("\n".join(row["trace_dir"] for row in trace_rows) + "\n")
        cmd = [
            "clj",
            "-M",
            "-i",
            str(script),
            "-e",
            f'(pruning-token-extract/-main "{dirs_file}" "{out_file}")',
        ]
        subprocess.run(cmd, cwd=spell_root, check=True)
        return [json.loads(line) for line in out_file.read_text().splitlines() if line.strip()]


def previous_node_map(turns: list[dict]) -> dict[int, int]:
    ids = [int(turn["node_id"]) for turn in turns]
    return {node_id: ids[index - 1] for index, node_id in enumerate(ids) if index > 0}


def trace_token_metrics(meta: dict, extracted: dict) -> dict:
    turns = sorted(extracted["turns"], key=lambda turn: int(turn["node_id"]))
    previous_by_node = previous_node_map(turns)
    emitted_by_node: dict[int, int] = {}
    peek_tokens = 0
    explicit_tokens = 0
    kind_tokens: dict[str, int] = {}

    for detail in extracted["details"]:
        target_tokens = sum(count_tokens(text) for text in detail["target_texts"])
        kind_tokens[detail["kind"]] = kind_tokens.get(detail["kind"], 0) + target_tokens
        node_id = int(detail["node_id"])
        emit_node_id = node_id
        if detail["trigger"] == "peek":
            emit_node_id = previous_by_node.get(node_id, node_id)
            peek_tokens += target_tokens
        else:
            explicit_tokens += target_tokens
        emitted_by_node[emit_node_id] = emitted_by_node.get(emit_node_id, 0) + target_tokens

    cumulative = 0
    cumulative_saved = 0
    completion_tokens = 0
    response_tokens = 0
    for turn in turns:
        completion_tokens += count_tokens(turn["completion"])
        response_tokens += count_tokens(turn["response"])
        cumulative_saved += cumulative
        cumulative += emitted_by_node.get(int(turn["node_id"]), 0)

    total_pruned = peek_tokens + explicit_tokens
    turn_count = len(turns)
    return {
        **meta,
        "turns": turn_count,
        "completion_tokens": completion_tokens,
        "response_tokens": response_tokens,
        "tokens_pruned_once": total_pruned,
        "peek_tokens_pruned_once": peek_tokens,
        "explicit_tokens_pruned_once": explicit_tokens,
        "prune_tokens_pruned_once": kind_tokens.get("prune", 0),
        "rethink_tokens_pruned_once": kind_tokens.get("rethink", 0),
        "cumulative_tokens_pruned": cumulative_saved,
        "mean_tokens_pruned_per_turn": total_pruned / turn_count if turn_count else 0.0,
        "mean_completion_tokens": completion_tokens / turn_count if turn_count else 0.0,
        "mean_response_tokens": response_tokens / turn_count if turn_count else 0.0,
        "mean_cumulative_tokens_pruned": cumulative_saved / turn_count if turn_count else 0.0,
    }


def summarize(trace_metrics: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in trace_metrics:
        groups.setdefault((row["benchmark"], row["effort"]), []).append(row)

    out = []
    for (benchmark, effort), rows in sorted(
        groups.items(),
        key=lambda item: (BENCHMARK_ORDER[item[0][0]], EFFORT_ORDER[item[0][1]]),
    ):
        turns = sum(int(row["turns"]) for row in rows)
        completion = sum(int(row["completion_tokens"]) for row in rows)
        response = sum(int(row["response_tokens"]) for row in rows)
        pruned = sum(int(row["tokens_pruned_once"]) for row in rows)
        cumulative = sum(int(row["cumulative_tokens_pruned"]) for row in rows)
        out.append(
            {
                "benchmark": benchmark,
                "effort": effort,
                "n_traces": len(rows),
                "turns": turns,
                "mean_turn_count": turns / len(rows) if rows else 0.0,
                "tokens_pruned_once": pruned,
                "completion_tokens": completion,
                "response_tokens": response,
                "cumulative_tokens_pruned": cumulative,
                "mean_tokens_pruned_per_turn": pruned / turns if turns else 0.0,
                "mean_completion_tokens": completion / turns if turns else 0.0,
                "mean_response_tokens": response / turns if turns else 0.0,
                "mean_cumulative_tokens_pruned": cumulative / turns if turns else 0.0,
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spell-root", type=Path, default=Path(os.environ.get("SPELL_ROOT", BENCH_ROOT.parent)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    trace_rows, warnings = collect_trace_dirs()
    if not trace_rows:
        raise SystemExit("No matching trace directories found")
    extracted_rows = run_clojure_extractor(args.spell_root.resolve(), trace_rows)
    if len(extracted_rows) != len(trace_rows):
        raise RuntimeError(f"extractor returned {len(extracted_rows)} rows for {len(trace_rows)} traces")

    metrics = []
    for meta, extracted in zip(trace_rows, extracted_rows):
        if extracted.get("error"):
            warnings.append(f"{meta['run_id']}: {meta['item']}: {extracted['error']}")
            continue
        metrics.append(trace_token_metrics(meta, extracted))
    summary_rows = summarize(metrics)

    trace_fields = [
        "benchmark",
        "label",
        "effort",
        "run_id",
        "item",
        "turns",
        "completion_tokens",
        "response_tokens",
        "tokens_pruned_once",
        "peek_tokens_pruned_once",
        "explicit_tokens_pruned_once",
        "prune_tokens_pruned_once",
        "rethink_tokens_pruned_once",
        "cumulative_tokens_pruned",
        "mean_tokens_pruned_per_turn",
        "mean_completion_tokens",
        "mean_response_tokens",
        "mean_cumulative_tokens_pruned",
        "trace_dir",
        "jsonl_path",
    ]
    summary_fields = [
        "benchmark",
        "effort",
        "n_traces",
        "turns",
        "mean_turn_count",
        "tokens_pruned_once",
        "completion_tokens",
        "response_tokens",
        "cumulative_tokens_pruned",
        "mean_tokens_pruned_per_turn",
        "mean_completion_tokens",
        "mean_response_tokens",
        "mean_cumulative_tokens_pruned",
    ]
    write_tsv(args.output_dir / "pruning_trace_metrics.tsv", metrics, trace_fields)
    write_tsv(args.output_dir / "pruning_summary.tsv", summary_rows, summary_fields)
    if warnings:
        (args.output_dir / "warnings.txt").write_text("\n".join(warnings) + "\n")

    print(f"Wrote {args.output_dir / 'pruning_trace_metrics.tsv'}")
    print(f"Wrote {args.output_dir / 'pruning_summary.tsv'}")
    if warnings:
        print(f"Wrote {args.output_dir / 'warnings.txt'} ({len(warnings)} warnings)")
    print()
    writer = csv.DictWriter(sys.stdout, fieldnames=summary_fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
