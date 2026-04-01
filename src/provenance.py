"""Helpers for recording benchmark run provenance."""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def resolve_spell_root() -> Path:
    env_spell_root = os.environ.get("SPELL_ROOT")
    if env_spell_root:
        return Path(env_spell_root).expanduser().resolve()

    nested_candidate = BENCHMARK_ROOT.parent
    if (nested_candidate / "config" / "agents").exists():
        return nested_candidate

    sibling_candidate = BENCHMARK_ROOT.parent / "spell"
    if (sibling_candidate / "config" / "agents").exists():
        return sibling_candidate

    return BENCHMARK_ROOT


def _git_version_for_repo(path: Path) -> dict[str, Any] | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return None

    return {"commit": sha, "branch": branch, "dirty": dirty}


@functools.cache
def benchmark_git_versions() -> dict[str, Any]:
    """Capture git revision info for benchmark-related repos when available."""
    versions: dict[str, Any] = {}
    seen_paths: set[Path] = set()

    for label, path in (("benchmarking", BENCHMARK_ROOT), ("spell", resolve_spell_root())):
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        versions[label] = _git_version_for_repo(resolved_path)

    return versions
