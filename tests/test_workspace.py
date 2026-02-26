from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

from src.workspace import RepoCache


def _run_in_parallel(*targets) -> None:
    barrier = threading.Barrier(len(targets))
    errors: list[BaseException] = []

    def wrapped(target) -> None:
        try:
            barrier.wait(timeout=2)
            target()
        except BaseException as exc:  # pragma: no cover - error passthrough helper
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    if errors:
        raise errors[0]


def _make_fake_git_runner():
    lock = threading.Lock()
    active_by_repo: dict[str, int] = defaultdict(int)
    max_by_repo: dict[str, int] = defaultdict(int)

    def fake_run(cmd, cwd=None, capture_output=True, timeout=None):  # noqa: ANN001
        if cmd[:3] == ["git", "worktree", "add"]:
            repo_key = str(cwd)
            dest = Path(cmd[-2])
            with lock:
                active_by_repo[repo_key] += 1
                max_by_repo[repo_key] = max(max_by_repo[repo_key], active_by_repo[repo_key])
            dest.mkdir(parents=True, exist_ok=True)
            time.sleep(0.05)
            with lock:
                active_by_repo[repo_key] -= 1
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        if cmd[:3] == ["git", "worktree", "remove"]:
            repo_key = str(cwd)
            dest = Path(cmd[-1])
            with lock:
                active_by_repo[repo_key] += 1
                max_by_repo[repo_key] = max(max_by_repo[repo_key], active_by_repo[repo_key])
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(0.05)
            with lock:
                active_by_repo[repo_key] -= 1
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    return fake_run, max_by_repo


def test_parallel_worktree_adds_same_repo_are_serialized(tmp_path: Path, monkeypatch) -> None:
    cache = RepoCache(tmp_path)
    repo = "org/repo"
    slug = repo.replace("/", "__")
    repo_path = tmp_path / "repos" / slug
    repo_path.mkdir(parents=True, exist_ok=True)

    fake_run, max_by_repo = _make_fake_git_runner()
    monkeypatch.setattr("src.workspace.subprocess.run", fake_run)
    monkeypatch.setattr(RepoCache, "ensure", lambda self, repo_name: repo_path)

    _run_in_parallel(
        lambda: cache.worktree(repo, "commit-a", "wt-a"),
        lambda: cache.worktree(repo, "commit-b", "wt-b"),
    )

    assert max_by_repo[str(repo_path)] == 1


def test_parallel_worktree_add_and_remove_same_repo_are_serialized(tmp_path: Path, monkeypatch) -> None:
    cache = RepoCache(tmp_path)
    repo = "org/repo"
    slug = repo.replace("/", "__")
    repo_path = tmp_path / "repos" / slug
    repo_path.mkdir(parents=True, exist_ok=True)

    fake_run, max_by_repo = _make_fake_git_runner()
    monkeypatch.setattr("src.workspace.subprocess.run", fake_run)
    monkeypatch.setattr(RepoCache, "ensure", lambda self, repo_name: repo_path)

    wt = cache.worktree(repo, "commit-a", "wt-a")
    assert wt.path.exists()

    _run_in_parallel(
        wt.remove,
        lambda: cache.worktree(repo, "commit-b", "wt-b"),
    )

    assert max_by_repo[str(repo_path)] == 1
