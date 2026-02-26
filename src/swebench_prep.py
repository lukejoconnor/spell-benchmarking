"""Workspace management for SWE-bench evaluation.

Thin wrapper around workspace.RepoCache for SWE-bench items.
"""

from pathlib import Path

from .datasets import SWEBenchItem
from .workspace import RepoCache, Worktree


def setup_swebench_workspace(cache: RepoCache, item: SWEBenchItem, tag: str = "") -> Worktree:
    """Create a worktree for a SWE-bench item at its base commit.

    Args:
        cache: Shared RepoCache for clone reuse.
        item: SWE-bench item with repo and base_commit.
        tag: Suffix to disambiguate worktrees (e.g. condition name).
    """
    name = item.instance_id
    if tag:
        name = f"{name}.{tag}"
    return cache.worktree(item.repo, item.base_commit, name)


def get_default_workspace_root() -> Path:
    return Path("/tmp/swe_ws")
