"""Workspace management for SWE-bench evaluation.

Handles cloning repos, checking out commits, and extracting patches.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

from .datasets import SWEBenchItem


@dataclass
class SWEBenchWorkspace:
    """Manages a repo workspace for SWE-bench evaluation."""

    item: SWEBenchItem
    workspace_root: Path
    _repo_path: Path = None

    def __post_init__(self):
        self.workspace_root = Path(self.workspace_root)
        # Use instance_id as directory name (e.g., "django__django-16379")
        self._repo_path = self.workspace_root / self.item.instance_id

    @property
    def repo_path(self) -> Path:
        """Path to the cloned repository."""
        return self._repo_path

    def setup(self) -> Path:
        """Clone repo (if needed) and checkout base commit.

        Reuses existing clones — just resets and checks out the base commit.
        Returns the path to the workspace directory.
        """
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        if self._repo_path.exists():
            # Reuse existing clone: discard changes, checkout base commit
            subprocess.run(
                ['git', 'checkout', '.'],
                cwd=self._repo_path,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                ['git', 'clean', '-fd'],
                cwd=self._repo_path,
                capture_output=True,
                timeout=60,
            )
        else:
            # Clone the repository
            repo_url = f"https://github.com/{self.item.repo}.git"
            result = subprocess.run(
                ['git', 'clone', '--quiet', repo_url, str(self._repo_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for large repos
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone {repo_url}: {result.stderr}")

        # Checkout the base commit
        result = subprocess.run(
            ['git', 'checkout', '--quiet', self.item.base_commit],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout {self.item.base_commit}: {result.stderr}"
            )

        return self._repo_path

    def extract_patch(self) -> str:
        """Extract current changes as a git diff patch.

        Returns the diff suitable for SWE-bench evaluation format.
        """
        if not self._repo_path.exists():
            raise RuntimeError("Workspace not set up. Call setup() first.")

        # Get diff of all changes (staged and unstaged)
        result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get diff: {result.stderr}")

        return result.stdout

    def get_changed_files(self) -> list[str]:
        """Get list of files that have been modified."""
        if not self._repo_path.exists():
            raise RuntimeError("Workspace not set up. Call setup() first.")

        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD'],
            cwd=self._repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get changed files: {result.stderr}")

        return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

    def cleanup(self):
        """Reset all changes and restore to base commit."""
        if not self._repo_path.exists():
            return

        subprocess.run(
            ['git', 'checkout', '.'],
            cwd=self._repo_path,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ['git', 'clean', '-fd'],
            cwd=self._repo_path,
            capture_output=True,
            timeout=60,
        )

    def remove(self):
        """Completely remove the workspace directory."""
        if self._repo_path.exists():
            shutil.rmtree(self._repo_path)


def get_default_workspace_root() -> Path:
    """Get default workspace root directory."""
    # Use /tmp for workspaces by default (can be configured)
    return Path('/tmp/swebench_workspaces')
