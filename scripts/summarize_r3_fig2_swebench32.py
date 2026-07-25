#!/usr/bin/env python3
"""Summarize the seeded Reviewer 3 SWE-bench Lite model pilot."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided 95% Wilson score interval as proportions."""
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = (p + z2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)
        / denominator
    )
    return [centre - half_width, centre + half_width]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected_items = set(manifest["items"])
    expected_models = {entry["model"].split(":", 1)[1]: entry for entry in manifest["models"]}

    records_by_model: dict[str, list[dict]] = {}
    source_files: dict[str, str] = {}
    for path in sorted(args.results_root.glob("r3s32-*/unified/*.jsonl")):
        records = load_jsonl(path)
        if not records:
            continue
        model = records[0]["model"]
        if model not in expected_models:
            continue
        if model in records_by_model:
            raise ValueError(f"Multiple result files found for {model}")
        item_ids = [record["item_id"] for record in records]
        if len(item_ids) != manifest["n"] or set(item_ids) != expected_items:
            raise ValueError(f"{model} does not contain the exact seeded manifest")
        if len(item_ids) != len(set(item_ids)):
            raise ValueError(f"{model} contains duplicate item IDs")
        records_by_model[model] = records
        source_files[model] = str(path)

    missing = set(expected_models) - set(records_by_model)
    if missing:
        raise ValueError(f"Missing models: {sorted(missing)}")

    errors_by_item: dict[str, list[str | None]] = defaultdict(list)
    for records in records_by_model.values():
        for record in records:
            errors_by_item[record["item_id"]].append(record.get("error_type"))
    shared_infrastructure_failures = sorted(
        item_id
        for item_id, error_types in errors_by_item.items()
        if len(error_types) == len(expected_models)
        and all(error_type == "container_error" for error_type in error_types)
    )
    excluded = set(shared_infrastructure_failures)

    summaries = []
    for model_entry in manifest["models"]:
        model = model_entry["model"].split(":", 1)[1]
        records = records_by_model[model]
        evaluable = [record for record in records if record["item_id"] not in excluded]
        resolved = sum(record["metadata"].get("same_container_resolved") is True for record in evaluable)
        terminal_errors = [
            record
            for record in evaluable
            if record.get("status") in {"error", "timeout"} or record.get("error_type") is not None
        ]
        error_counts = Counter(record.get("error_type") or record["status"] for record in terminal_errors)
        known_costs = [record["cost_usd"] for record in records if record.get("cost_usd") is not None]
        total = len(evaluable)
        if resolved + len(terminal_errors) > total:
            raise ValueError(f"Resolved/error overlap for {model}")

        summaries.append(
            {
                **model_entry,
                "source_file": source_files[model],
                "sampled": len(records),
                "evaluable": total,
                "resolved": resolved,
                "resolved_rate": resolved / total,
                "resolved_wilson_95": wilson_interval(resolved, total),
                "terminal_errors": len(terminal_errors),
                "terminal_error_rate": len(terminal_errors) / total,
                "terminal_error_wilson_95": wilson_interval(len(terminal_errors), total),
                "wrong_without_terminal_error": total - resolved - len(terminal_errors),
                "error_types": dict(sorted(error_counts.items())),
                "known_cost_usd": sum(known_costs),
                "records_with_unknown_cost": len(records) - len(known_costs),
            }
        )

    output = {
        "dataset": manifest["dataset"],
        "seed": manifest["seed"],
        "selection": manifest["selection"],
        "sampled_items": manifest["n"],
        "evaluable_items": manifest["n"] - len(excluded),
        "shared_infrastructure_failures": shared_infrastructure_failures,
        "error_definition": (
            "Terminal model/run errors after excluding shared infrastructure failures; "
            "includes timeouts and non-null error_type records."
        ),
        "models": summaries,
        "known_cost_usd": sum(summary["known_cost_usd"] for summary in summaries),
        "records_with_unknown_cost": sum(
            summary["records_with_unknown_cost"] for summary in summaries
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
