#!/usr/bin/env python3
"""Stage and sanity-check Harbor Terminal-Bench 2.0 submission artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.harbor_terminalbench_runner import HARBOR_TERMINALBENCH_DATASET

BANNED_CONFIG_KEYS = {
    "agent_timeout_multiplier",
    "verifier_timeout_multiplier",
    "agent_setup_timeout_multiplier",
    "environment_build_timeout_multiplier",
    "override_timeout_sec",
    "override_setup_timeout_sec",
    "max_timeout_sec",
    "override_cpus",
    "override_memory",
    "override_memory_mb",
    "override_storage",
    "override_storage_mb",
    "override_gpus",
    "registry_path",
    "path",
    "task",
    "task_names",
    "include_task_name",
    "include_task_names",
    "exclude_task_name",
    "exclude_task_names",
    "n_tasks",
    "mounts_json",
    "disable_verification",
}


def _walk_config(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.append((path, child))
            found.extend(_walk_config(child, path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(_walk_config(child, f"{prefix}[{idx}]"))
    return found


def _dataset_ref(dataset: Any) -> str | None:
    if isinstance(dataset, str):
        return dataset
    if not isinstance(dataset, dict):
        return None

    ref = dataset.get("ref")
    if isinstance(ref, str) and ref:
        return ref

    name = dataset.get("name")
    version = dataset.get("version")
    if isinstance(name, str) and isinstance(version, str):
        return f"{name}@{version}"
    if isinstance(name, str) and "@" in name:
        return name
    return None


def _dataset_refs(config: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    top_level = _dataset_ref(config.get("dataset"))
    if top_level:
        refs.append(top_level)

    datasets = config.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            ref = _dataset_ref(dataset)
            if ref:
                refs.append(ref)
    return refs


def validate_config(config_path: Path) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    refs = _dataset_refs(config)
    if not refs:
        errors.append(f"{config_path}: missing Harbor dataset reference")
    for ref in refs:
        if ref != HARBOR_TERMINALBENCH_DATASET:
            errors.append(
                f"{config_path}: dataset is {ref!r}, expected {HARBOR_TERMINALBENCH_DATASET!r}"
            )

    attempts = config.get("n_attempts")
    if attempts is None:
        errors.append(f"{config_path}: missing n_attempts; cannot verify k >= 5")
    elif not isinstance(attempts, int) or attempts < 5:
        errors.append(f"{config_path}: n_attempts={attempts!r}, expected integer >= 5")

    for path, value in _walk_config(config):
        key = path.rsplit(".", 1)[-1]
        if key == "timeout_multiplier" and value not in (None, 1, 1.0):
            errors.append(f"{config_path}: {path}={value!r} modifies task timeouts")
        if key in BANNED_CONFIG_KEYS and value not in (None, "", [], {}):
            errors.append(f"{config_path}: {path}={value!r} is not submission-compliant")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="Submission agent slug, e.g. spell__gpt-5.4")
    parser.add_argument("--job-dir", action="append", required=True, type=Path, help="Harbor job directory to copy")
    parser.add_argument("--metadata", required=True, type=Path, help="metadata.yaml to copy into the submission")
    parser.add_argument("--output-root", default=Path("submissions") / "terminal-bench" / "2.0", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.metadata.is_file():
        raise SystemExit(f"metadata file not found: {args.metadata}")

    errors: list[str] = []
    for job_dir in args.job_dir:
        config_path = job_dir / "config.json"
        if not config_path.is_file():
            errors.append(f"{job_dir}: missing config.json")
            continue
        errors.extend(validate_config(config_path))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2

    target = args.output_root / args.agent
    print(f"Staging Harbor TB2 submission under {target}")
    if args.dry_run:
        return 0

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metadata, target / "metadata.yaml")
    for job_dir in args.job_dir:
        shutil.copytree(job_dir, target / job_dir.name, dirs_exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
