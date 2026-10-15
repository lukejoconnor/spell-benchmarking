#!/usr/bin/env python3
"""Count emitted Spell feature use and tool batching from fresh trace roots."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.spell_trace_features import FEATURE_PATTERNS, analyze_trace_roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_roots", nargs="+", type=Path, help="Selected run directories or trace.edn files")
    parser.add_argument("--label", default="Spell", help="Benchmark/condition label")
    parser.add_argument("--output-dir", type=Path, default=Path("results/feature-utilization"))
    args = parser.parse_args()
    try:
        result = analyze_trace_roots(args.trace_roots, args.label)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "feature_usage.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "feature_usage.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "feature", "uses", "traces_using", "traces"])
        for name, counts in result["features"].items():
            writer.writerow([args.label, name, counts["uses"], counts["traces_using"], counts["traces"]])
    with (args.output_dir / "trace_metrics.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["trace", "turns", *FEATURE_PATTERNS])
        for trace in result["traces"]:
            writer.writerow([trace["trace"], trace["turns"], *trace["counts"].values()])
    print(f"Analyzed {result['trace_count']} traces / {result['turns']} turns; skipped {len(result['skipped'])} traces.")
    if not result["trace_count"]:
        raise SystemExit("No eligible model-response traces; inspect feature_usage.json.")


if __name__ == "__main__":
    main()
