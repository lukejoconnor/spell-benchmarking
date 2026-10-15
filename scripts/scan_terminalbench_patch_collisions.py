#!/usr/bin/env python3
"""Identify Terminal-Bench artifacts with evaluator test-patch collisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.terminalbench_patch_recovery import (  # noqa: E402
    artifacts_to_jsonable,
    scan_terminalbench_patch_collision_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="results/terminal-bench",
        help="Terminal-Bench results root to scan",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "table"),
        default="jsonl",
        help="Output format",
    )
    args = parser.parse_args()

    records = artifacts_to_jsonable(scan_terminalbench_patch_collision_artifacts(Path(args.root)))
    if args.format == "table":
        print("status\ttask_id\ttrial_name\trun_dir")
        for record in records:
            print(
                "\t".join(
                    [
                        str(record["status"]),
                        str(record["task_id"]),
                        str(record["trial_name"]),
                        str(record["run_dir"]),
                    ]
                )
            )
    else:
        for record in records:
            print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
