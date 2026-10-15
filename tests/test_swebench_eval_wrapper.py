from __future__ import annotations

import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.swebench_eval_wrapper import _patch_repo_setup_commands


def test_patch_repo_setup_commands_adds_missing_branch_fallback() -> None:
    commands = [
        "git clone -o origin --branch 1.7 --single-branch https://github.com/sympy/sympy /testbed",
        "chmod -R 777 /testbed",
        "cd /testbed",
        "conda activate testbed",
    ]

    patched = _patch_repo_setup_commands(
        commands,
        repo="sympy/sympy",
        repo_directory="/testbed",
        branch="1.7",
    )

    assert patched[0].startswith("if git ls-remote --exit-code --heads https://github.com/sympy/sympy 1.7")
    assert "git clone -o origin --branch 1.7 --single-branch https://github.com/sympy/sympy /testbed" in patched[0]
    assert "git clone -o origin https://github.com/sympy/sympy /testbed" in patched[0]
    assert "missing remote branch 1.7 for sympy/sympy" in patched[0]
    assert patched[4] == "python -m pip install 'pip<24' -qq"


def test_patch_repo_setup_commands_leaves_unpinned_clone_unchanged() -> None:
    commands = [
        "git clone -o origin  --single-branch https://github.com/django/django /testbed",
        "cd /testbed",
        "conda activate testbed",
    ]

    patched = _patch_repo_setup_commands(
        commands,
        repo="django/django",
        repo_directory="/testbed",
        branch="",
    )

    assert patched[0] == commands[0]
    assert patched[3] == "python -m pip install 'pip<24' -qq"


def test_patch_repo_setup_commands_removes_orphaned_matplotlib_vcs_versioning() -> None:
    commands = [
        "git clone -o origin  --single-branch https://github.com/matplotlib/matplotlib /testbed",
        "cd /testbed",
        "conda activate testbed",
        "python -m pip install -e .",
    ]

    patched = _patch_repo_setup_commands(
        commands,
        repo="matplotlib/matplotlib",
        repo_directory="/testbed",
        branch="",
    )

    assert patched[3] == "python -m pip install 'pip<24' -qq"
    assert patched[4] == "python -m pip uninstall -y vcs-versioning"
    assert patched[5] == "python -m pip install -e ."
