from __future__ import annotations

from pathlib import Path

from src.swebench_container import SWEBenchContainer, _env_exports_from_dockerfiles, _parse_string_list
from src.datasets import SWEBenchItem


def test_parse_string_list_accepts_json_python_and_native_lists() -> None:
    assert _parse_string_list('["a", "b"]') == ["a", "b"]
    assert _parse_string_list("['a', 'b']") == ["a", "b"]
    assert _parse_string_list(["a", "b"]) == ["a", "b"]
    assert _parse_string_list("") == []


def test_env_exports_from_dockerfiles_collects_env_lines(tmp_path: Path) -> None:
    base = tmp_path / "base.Dockerfile"
    instance = tmp_path / "instance.Dockerfile"
    base.write_text("FROM python:3.11\nENV FOO=bar\n", encoding="utf-8")
    instance.write_text("ENV BAR=baz\nRUN echo hi\n", encoding="utf-8")

    exports = _env_exports_from_dockerfiles(base, instance)

    assert "export FOO=bar" in exports
    assert "export BAR=baz" in exports


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
