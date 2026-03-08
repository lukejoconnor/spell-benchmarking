"""Git workspace management via shared repo clones and worktrees.

Repos are cloned once under <root>/repos/ and reused across tasks.
Each task gets a lightweight git worktree checked out at the target commit.
Worktrees are removed after use.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


class RepoCache:
    """Thread-safe cache of cloned git repositories.

    Usage:
        cache = RepoCache(workspace_root)
        wt = cache.worktree("django/django", commit="abc123", name="django__django-16379")
        # ... use wt.path ...
        wt.remove()
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._repo_dir = self.root / "repos"
        self._locks: dict[str, threading.RLock] = {}
        self._meta_lock = threading.Lock()

    def _lock_for(self, slug: str) -> threading.RLock:
        with self._meta_lock:
            if slug not in self._locks:
                self._locks[slug] = threading.RLock()
            return self._locks[slug]

    def ensure(self, repo: str) -> Path:
        """Ensure repo is cloned and up to date. Returns path to clone."""
        slug = repo.replace("/", "__")
        repo_path = self._repo_dir / slug

        with self._lock_for(slug):
            if repo_path.exists():
                # Validate and fetch
                ok = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=repo_path, capture_output=True, timeout=30,
                ).returncode == 0
                if not ok:
                    shutil.rmtree(repo_path)
                else:
                    subprocess.run(
                        ["git", "fetch", "--quiet", "origin"],
                        cwd=repo_path, capture_output=True, timeout=600,
                    )
                    return repo_path

            repo_path.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(repo_path)],
                capture_output=True, timeout=1800,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"Clone of {repo} failed: {stderr}")

        return repo_path

    def worktree(self, repo: str, commit: str, name: str) -> Worktree:
        """Create a worktree checked out at commit.

        Args:
            repo: GitHub repo path (e.g. "django/django")
            commit: Git commit to check out
            name: Directory name under self.root for this worktree
        """
        slug = repo.replace("/", "__")
        repo_lock = self._lock_for(slug)
        with repo_lock:
            repo_path = self.ensure(repo)
            dest = self.root / name

            if dest.exists():
                Worktree(dest, repo_path, repo_lock).remove()

            result = subprocess.run(
                ["git", "worktree", "add", "--quiet", "--detach", str(dest), commit],
                cwd=repo_path, capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"Worktree add failed for {name} at {commit}: {stderr}")

            return Worktree(dest, repo_path, repo_lock)


class Worktree:
    """A git worktree with convenience methods for patch extraction."""

    def __init__(self, path: Path, repo_path: Path, mutation_lock: threading.RLock | None = None):
        self.path = path
        self.repo_path = repo_path
        self._mutation_lock = mutation_lock

    def diff(self, base: str = "HEAD") -> str:
        """Return git diff as a string, including newly created files.

        Args:
            base: Commit to diff against (default HEAD). Use the original
                  checkout commit to capture changes that were committed
                  inside the worktree (e.g. by fix-loop).
        """
        if not self.path.exists():
            return ""
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=self.path, capture_output=True, timeout=60,
        )
        if add_result.returncode != 0:
            return ""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", base],
                cwd=self.path, capture_output=True, timeout=60,
            )
        finally:
            subprocess.run(
                ["git", "reset", "-q"],
                cwd=self.path, capture_output=True, timeout=60,
            )
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace")

    def changed_files(self, base: str = "HEAD") -> list[str]:
        """Return list of modified file paths, including newly created files.

        Args:
            base: Commit to diff against (default HEAD).
        """
        if not self.path.exists():
            return []
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=self.path, capture_output=True, timeout=60,
        )
        if add_result.returncode != 0:
            return []
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached", base],
                cwd=self.path, capture_output=True, timeout=60,
            )
        finally:
            subprocess.run(
                ["git", "reset", "-q"],
                cwd=self.path, capture_output=True, timeout=60,
            )
        text = result.stdout.decode("utf-8", errors="replace")
        return [f.strip() for f in text.splitlines() if f.strip()]

    def remove(self) -> None:
        """Remove this worktree."""
        if self._mutation_lock is not None:
            with self._mutation_lock:
                self._remove_unlocked()
            return
        self._remove_unlocked()

    def _remove_unlocked(self) -> None:
        """Remove this worktree without acquiring a repo mutation lock."""
        if not self.path.exists():
            return
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=self.repo_path, capture_output=True, timeout=60,
        )
        # Fallback if worktree remove fails (e.g. repo_path gone)
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
