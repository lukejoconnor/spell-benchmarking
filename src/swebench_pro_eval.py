"""Helpers for SWE-bench Pro evaluation via ScaleAI's pipeline."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import PurePosixPath, Path
from typing import Any

SCALE_PRO_EVAL_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
TRACE_ARTIFACT_BASENAMES = {"trace.edn", "tree.txt"}


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _is_trace_artifact_path(path: str) -> bool:
    norm = path.strip().lstrip("./")
    if not norm or norm == "/dev/null":
        return False
    posix_path = PurePosixPath(norm)
    base = posix_path.name
    if base in TRACE_ARTIFACT_BASENAMES or base.endswith(".spl"):
        return True
    if norm.startswith("traces/") or "/traces/" in norm:
        return True
    return False


def strip_trace_artifact_sections(patch: str) -> tuple[str, list[str]]:
    """Drop trace artifact file sections from a unified diff patch."""
    if not patch:
        return patch, []

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    removed_paths: list[str] = []

    for section in sections:
        if not section:
            continue
        if not section.startswith("diff --git "):
            kept.append(section)
            continue

        match = re.match(r"^diff --git a/([^\n]+) b/([^\n]+)", section)
        if not match:
            kept.append(section)
            continue

        src_path, dst_path = match.group(1), match.group(2)
        if _is_trace_artifact_path(src_path) or _is_trace_artifact_path(dst_path):
            removed_paths.append(dst_path if dst_path != "/dev/null" else src_path)
            continue

        kept.append(section)

    return "".join(kept), removed_paths


def convert_predictions_jsonl_to_scale_json(
    predictions_path: Path | str,
    output_path: Path | str,
    *,
    default_prefix: str = "",
    strip_trace_artifacts: bool = True,
) -> dict[str, Any]:
    """Convert standard SWE-bench JSONL predictions to ScaleAI's JSON format."""
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted: list[dict[str, str]] = []
    removed_trace_sections = 0
    removed_trace_paths: list[str] = []

    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {predictions_path}") from exc

            instance_id = row.get("instance_id")
            if not instance_id:
                raise ValueError(f"Missing instance_id on line {line_no} in {predictions_path}")

            patch = row.get("model_patch")
            if patch is None:
                patch = row.get("patch", "")
            if patch is None:
                patch = ""
            if not isinstance(patch, str):
                patch = str(patch)

            patch = strip_binary_hunks(patch)

            if strip_trace_artifacts:
                patch, removed_paths = strip_trace_artifact_sections(patch)
                removed_trace_sections += len(removed_paths)
                removed_trace_paths.extend(removed_paths)

            converted_row: dict[str, str] = {
                "instance_id": instance_id,
                "patch": patch,
            }
            prefix = row.get("prefix", default_prefix)
            if prefix:
                converted_row["prefix"] = str(prefix)
            converted.append(converted_row)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)

    return {
        "count": len(converted),
        "removed_trace_sections": removed_trace_sections,
        "removed_trace_paths": sorted(set(removed_trace_paths)),
    }


def ensure_scale_pro_eval_repo(
    repo_path: Path | str,
    *,
    repo_url: str = SCALE_PRO_EVAL_REPO_URL,
) -> Path:
    """Ensure the ScaleAI SWE-bench Pro eval repo exists locally."""
    repo_path = Path(repo_path)
    script_path = repo_path / "swe_bench_pro_eval.py"
    scripts_dir = repo_path / "run_scripts"
    if script_path.exists() and scripts_dir.exists():
        return repo_path

    if repo_path.exists() and any(repo_path.iterdir()):
        raise RuntimeError(
            f"{repo_path} exists but is missing SWE-bench Pro eval assets "
            "(expected swe_bench_pro_eval.py and run_scripts/)."
        )

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to clone SWE-bench Pro eval repo from {repo_url}: {exc.stderr or exc.stdout}"
        ) from exc
    return repo_path


def write_raw_sample_jsonl(
    raw_sample_path: Path | str,
    *,
    dataset_name: str,
    split: str,
) -> Path:
    """Materialize raw sample JSONL from a HuggingFace dataset."""
    from datasets import load_dataset  # Lazy import to keep base import cheap

    raw_sample_path = Path(raw_sample_path)
    raw_sample_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_name, split=split)
    with raw_sample_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return raw_sample_path


def load_raw_samples(raw_sample_path: Path | str) -> dict[str, dict[str, Any]]:
    """Load raw sample metadata keyed by instance_id from CSV or JSONL."""
    raw_sample_path = Path(raw_sample_path)
    if not raw_sample_path.exists():
        raise FileNotFoundError(f"Raw sample file not found: {raw_sample_path}")

    by_instance: dict[str, dict[str, Any]] = {}
    if raw_sample_path.suffix.lower() == ".jsonl":
        with raw_sample_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                instance_id = row.get("instance_id")
                if not instance_id:
                    raise ValueError(
                        f"Missing instance_id in raw sample JSONL line {line_no}: {raw_sample_path}"
                    )
                by_instance[str(instance_id)] = dict(row)
        return by_instance

    with raw_sample_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "instance_id" not in (reader.fieldnames or []):
            raise ValueError(f"CSV raw sample missing instance_id column: {raw_sample_path}")
        for row in reader:
            instance_id = (row.get("instance_id") or "").strip()
            if instance_id:
                by_instance[instance_id] = dict(row)
    return by_instance


def dockerhub_image_for_instance(
    raw_samples: dict[str, dict[str, Any]],
    instance_id: str,
    *,
    dockerhub_username: str,
) -> str:
    """Construct DockerHub image URI for a specific Pro instance."""
    sample = raw_samples.get(instance_id)
    if sample is None:
        raise KeyError(f"instance_id {instance_id} not present in raw sample metadata")
    tag = (sample.get("dockerhub_tag") or "").strip()
    if not tag:
        raise KeyError(f"Raw sample for {instance_id} is missing dockerhub_tag")
    return f"{dockerhub_username}/sweap-images:{tag}"


def check_docker_manifest_access(image_uri: str, *, env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Check whether a Docker image manifest is accessible."""
    try:
        proc = subprocess.run(
            ["docker", "manifest", "inspect", image_uri],
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        return False, "docker command not found"

    if proc.returncode == 0:
        return True, ""
    msg = (proc.stderr or proc.stdout or "").strip()
    return False, msg or f"docker manifest inspect failed with exit code {proc.returncode}"
