#!/usr/bin/env python3
"""Merge the Reviewer 3 Terminal-Bench subset with credit-failure retries."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


RETRY_NUMBER = re.compile(r"credit-retry(\d+)$")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided 95% Wilson score interval as proportions."""
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    centre = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) + z_squared / (4 * total)) / total
        )
        / denominator
    )
    return [centre - half_width, centre + half_width]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def result_files(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("r3tb32-*/terminal-bench/*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retry-manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    retry_manifest = json.loads(args.retry_manifest.read_text())
    expected_items = set(manifest["items"])
    expected_models = {
        entry["model"].split(":", 1)[1]: entry for entry in manifest["models"]
    }
    expected_retries = {
        model.split(":", 1)[1]: set(items)
        for model, items in retry_manifest["models"].items()
    }

    originals: dict[str, tuple[Path, list[dict]]] = {}
    retry_candidates: dict[tuple[str, str], tuple[int, Path, dict]] = {}

    for path in result_files(args.results_root):
        records = load_jsonl(path)
        if not records:
            continue
        model = records[0]["model"]
        if model not in expected_models:
            continue
        if any(record["model"] != model for record in records):
            raise ValueError(f"Mixed models in {path}")

        name = records[0].get("name") or ""
        if any((record.get("name") or "") != name for record in records):
            raise ValueError(f"Mixed names in {path}")

        retry_match = RETRY_NUMBER.search(name)
        item_ids = [record["item_id"] for record in records]
        if retry_match:
            retry_number = int(retry_match.group(1))
            if retry_number < 3:
                continue
            for record in records:
                key = (model, record["item_id"])
                previous = retry_candidates.get(key)
                if previous is None or retry_number > previous[0]:
                    retry_candidates[key] = (retry_number, path, record)
            continue

        if len(item_ids) == manifest["n"] and set(item_ids) == expected_items:
            if len(item_ids) != len(set(item_ids)):
                raise ValueError(f"Duplicate original item IDs in {path}")
            if model in originals:
                raise ValueError(f"Multiple exact original runs found for {model}")
            originals[model] = (path, records)

    missing_originals = set(expected_models) - set(originals)
    if missing_originals:
        raise ValueError(f"Missing exact original runs: {sorted(missing_originals)}")

    summaries = []
    for model_entry in manifest["models"]:
        model = model_entry["model"].split(":", 1)[1]
        original_path, original_records = originals[model]
        canonical = {record["item_id"]: record for record in original_records}
        replacement_sources: set[str] = set()

        for item_id in expected_retries.get(model, set()):
            candidate = retry_candidates.get((model, item_id))
            if candidate is None:
                raise ValueError(f"Missing valid retry candidate for {model}/{item_id}")
            _, replacement_path, replacement = candidate
            canonical[item_id] = replacement
            replacement_sources.add(str(replacement_path))

        if set(canonical) != expected_items or len(canonical) != manifest["n"]:
            raise ValueError(f"Canonical item set mismatch for {model}")

        records = [canonical[item_id] for item_id in manifest["items"]]
        resolved = sum(record.get("correct") is True for record in records)
        terminal_errors = [
            record
            for record in records
            if record.get("status") in {"error", "timeout"}
            or record.get("error_type") is not None
        ]
        error_counts = Counter(
            record.get("error_type") or record["status"] for record in terminal_errors
        )
        known_costs = [
            record["cost_usd"] for record in records if record.get("cost_usd") is not None
        ]
        total = len(records)

        summaries.append(
            {
                **model_entry,
                "original_source_file": str(original_path),
                "replacement_source_files": sorted(replacement_sources),
                "sampled": total,
                "resolved": resolved,
                "resolved_rate": resolved / total,
                "resolved_wilson_95": wilson_interval(resolved, total),
                "terminal_errors": len(terminal_errors),
                "terminal_error_rate": len(terminal_errors) / total,
                "terminal_error_wilson_95": wilson_interval(
                    len(terminal_errors), total
                ),
                "wrong_without_terminal_error": sum(
                    record.get("correct") is not True
                    and record not in terminal_errors
                    for record in records
                ),
                "error_types": dict(sorted(error_counts.items())),
                "replacement_count": len(expected_retries.get(model, set())),
                "known_cost_usd": sum(known_costs),
                "records_with_unknown_cost": total - len(known_costs),
            }
        )

    output = {
        "dataset": manifest["dataset"],
        "selection": manifest["selection"],
        "sampled_items": manifest["n"],
        "retry_manifest": str(args.retry_manifest),
        "retry_policy": (
            "Replace only the 88 original Fireworks HTTP 412 credit-failure attempts; "
            "when a task has multiple retry rounds, retain the highest retry number."
        ),
        "error_definition": (
            "Terminal model/run errors; includes timeout statuses and records with a "
            "non-null error_type. Errors remain in the 32-task denominator."
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
