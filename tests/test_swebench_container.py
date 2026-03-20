from __future__ import annotations

from pathlib import Path

from src.swebench_container import (
    SWEBenchContainer,
    _build_pro_eval_entryscript,
    _env_exports_from_dockerfiles,
    _parse_string_list,
)
from src.datasets import SWEBenchItem


def test_parse_string_list_accepts_json_python_and_native_lists() -> None:
    assert _parse_string_list('["a", "b"]') == ["a", "b"]
    assert _parse_string_list("['a', 'b']") == ["a", "b"]
    assert _parse_string_list(["a", "b"]) == ["a", "b"]
    assert _parse_string_list("") == []


def test_env_exports_from_dockerfiles_collects_env_lines(tmp_path: Path) -> None:
    base = tmp_path / "base.Dockerfile"
    instance = tmp_path / "instance.Dockerfile"
    base.write_text("FROM python:3.11\nENV FOO=bar\nENV LEGACY one\n", encoding="utf-8")
    instance.write_text('ENV BAR=baz\nENV MESSAGE hello world\nRUN echo hi\n', encoding="utf-8")

    exports = _env_exports_from_dockerfiles(base, instance)

    assert "export FOO=bar" in exports
    assert "export LEGACY=one" in exports
    assert "export BAR=baz" in exports
    assert 'export MESSAGE="hello world"' in exports


def test_build_pro_eval_entryscript_keeps_full_before_repo_set_cmd() -> None:
    entryscript = _build_pro_eval_entryscript(
        container_workdir="/app",
        env_exports="export FOO=bar",
        before_repo_set_cmd="git reset --hard\ngit clean -fd\ngit checkout -- tests",
        selected_tests=["tests/a.py", "tests/b.py"],
    )

    assert "export FOO=bar\nset -e\ncd /app\n" in entryscript
    assert "git reset --hard\ngit clean -fd\ngit checkout -- tests\n" in entryscript
    assert "bash /tmp/run_script.sh tests/a.py,tests/b.py > /tmp/stdout.log 2> /tmp/stderr.log" in entryscript
    assert entryscript.endswith(
        "python /tmp/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json\n"
    )


def test_swebench_container_detects_pro_rows() -> None:
    pro_item = SWEBenchItem(
        instance_id="id",
        repo="org/repo",
        base_commit="abc123",
        problem_statement="Fix it",
        hints_text="",
        test_patch="",
        patch="",
        raw_instance={"dockerhub_tag": "repo.tag"},
    )
    standard_item = SWEBenchItem(
        instance_id="id2",
        repo="org/repo",
        base_commit="def456",
        problem_statement="Fix it",
        hints_text="",
        test_patch="",
        patch="",
        raw_instance={},
    )

    assert SWEBenchContainer(pro_item).is_pro is True
    assert SWEBenchContainer(standard_item).is_pro is False
