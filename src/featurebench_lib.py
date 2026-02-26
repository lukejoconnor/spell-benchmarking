"""FeatureBench helpers shared by the unified benchmark CLI."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

from .workspace import RepoCache, Worktree


def mask_functions(workspace: Path, problem_statement: str) -> None:
    """Stub matching interface functions in-place to mirror FeatureBench Docker state."""
    pattern = r"Path:\s*`([^`]+)`\s*\n```python\n(.*?)```"
    matches = re.findall(pattern, problem_statement, re.DOTALL)
    if not matches:
        return

    for container_path, interface_code in matches:
        rel_path = container_path
        for prefix in [str(workspace) + "/", "/testbed/"]:
            if rel_path.startswith(prefix):
                rel_path = rel_path[len(prefix):]
                break

        file_path = workspace / rel_path
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(interface_code.strip() + "\n", encoding="utf-8")
            continue

        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = file_path.read_text(encoding="latin-1")

        try:
            iface_tree = ast.parse(interface_code)
            source_tree = ast.parse(source)
        except SyntaxError:
            continue

        names = set()
        for node in ast.walk(iface_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        if not names:
            continue

        lines = source.splitlines()
        changed = False
        for node in ast.walk(source_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
                body_start = node.body[0].lineno - 1
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                    body_start = node.body[0].end_lineno
                body_end = node.end_lineno

                for i in range(body_start, body_end):
                    lines[i] = ""

                def_line = lines[node.lineno - 1]
                indent = " " * (len(def_line) - len(def_line.lstrip()) + 4)
                lines[body_start] = f"{indent}pass"
                changed = True

        if changed:
            file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tasks(split: str = "lite", task_indices: list[int] | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("LiberCoders/FeatureBench", split=split)
    if task_indices:
        return [ds[i] for i in task_indices]
    return list(ds)


def setup_workspace(cache: RepoCache, item: dict, tag: str = "") -> Worktree:
    """Create a worktree for this task instance at the target commit.

    Args:
        cache: Shared RepoCache for clone reuse.
        tag: Suffix to disambiguate worktrees (e.g. condition name).
    """
    name = item["instance_id"].replace("/", "__")
    if tag:
        name = f"{name}.{tag}"
    return cache.worktree(item["repo"], item["base_commit"], name)


def install_deps(workspace: Path, item: dict) -> None:
    settings = json.loads(item.get("repo_settings", "{}"))
    install_cmd = settings.get("install", "")
    if not install_cmd:
        return

    subprocess.run(
        ["bash", "-c", install_cmd],
        cwd=workspace,
        capture_output=True,
        timeout=600,
    )


def detect_test_deps(repo: str) -> list[str]:
    deps = {
        "mwaskom/seaborn": ["pytest", "numpy", "pandas", "matplotlib", "scipy"],
        "pandas-dev/pandas": ["pytest", "hypothesis", "pyarrow"],
        "sphinx-doc/sphinx": ["pytest"],
    }
    return deps.get(repo, ["pytest"])


def run_tests(workspace: Path, fail_to_pass: list[str], repo: str | None = None) -> list[dict]:
    results = []
    extra_deps = detect_test_deps(repo) if repo else ["pytest"]
    uv_prefix = ["uv", "run"]
    for dep in extra_deps:
        uv_prefix.extend(["--with", dep])

    for test_path in fail_to_pass:
        cmd = uv_prefix + ["pytest", test_path, "-x", "-q", "--tb=short"]
        proc = subprocess.run(cmd, cwd=workspace, capture_output=True, timeout=300)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        results.append(
            {
                "test": test_path,
                "passed": proc.returncode == 0,
                "output": (stdout + stderr)[-2000:],
            }
        )
    return results


def format_prompt(item: dict, workspace: Path) -> str:
    text = item["problem_statement"]
    text = text.replace("/testbed/", f"{workspace}/")
    text = text.replace("/testbed", str(workspace))
    return text
