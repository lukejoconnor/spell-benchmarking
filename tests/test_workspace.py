from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

from src.workspace import RepoCache, Worktree


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


def test_worktree_add_prunes_before_add(tmp_path: Path, monkeypatch) -> None:
    cache = RepoCache(tmp_path)
    repo = "org/repo"
    slug = repo.replace("/", "__")
    repo_path = tmp_path / "repos" / slug
    repo_path.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, capture_output=True, timeout=None):  # noqa: ANN001
        calls.append(cmd)
        if cmd[:3] == ["git", "worktree", "add"]:
            dest = Path(cmd[-2])
            dest.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("src.workspace.subprocess.run", fake_run)
    monkeypatch.setattr(RepoCache, "ensure", lambda self, repo_name: repo_path)

    cache.worktree(repo, "commit-a", "wt-a")

    prune_idx = next(i for i, cmd in enumerate(calls) if cmd[:3] == ["git", "worktree", "prune"])
    add_idx = next(i for i, cmd in enumerate(calls) if cmd[:3] == ["git", "worktree", "add"])
    assert prune_idx < add_idx


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


def test_worktree_remove_prunes_registry(tmp_path: Path, monkeypatch) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)
    wt_path = tmp_path / "repo-wt"
    wt_path.mkdir(parents=True, exist_ok=True)
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, capture_output=True, timeout=None):  # noqa: ANN001
        calls.append(cmd)
        if cmd[:3] == ["git", "worktree", "remove"]:
            shutil.rmtree(wt_path, ignore_errors=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("src.workspace.subprocess.run", fake_run)

    Worktree(wt_path, repo_path).remove()

    assert any(cmd[:3] == ["git", "worktree", "remove"] for cmd in calls)
    assert any(cmd[:3] == ["git", "worktree", "prune"] for cmd in calls)


def test_worktree_diff_and_changed_files_include_new_files(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bench@local"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "bench"], cwd=repo_path, check=True, capture_output=True)

    (repo_path / "base.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo_path, check=True, capture_output=True)

    (repo_path / "new_file.py").write_text("print('hello')\n", encoding="utf-8")

    wt = Worktree(repo_path, repo_path)
    diff = wt.diff()
    changed = wt.changed_files()

    assert "new_file.py" in diff
    assert "new_file.py" in changed

    # diff()/changed_files() should not leave staged changes behind.
    cached = subprocess.run(
        ["git", "diff", "--name-only", "--cached", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert cached.stdout.strip() == ""
