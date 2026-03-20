from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.swebench_container import (
    SWEBenchContainer,
    _build_pro_eval_entryscript,
    _env_exports_from_dockerfiles,
    _parse_string_list,
)
from src.datasets import SWEBenchItem


def _make_item(*, instance_id: str = "id", raw_instance: dict | None = None) -> SWEBenchItem:
    return SWEBenchItem(
        instance_id=instance_id,
        repo="org/repo",
        base_commit="abc123",
        problem_statement="Fix it",
        hints_text="",
        test_patch="",
        patch="",
        raw_instance=raw_instance or {},
    )


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
    pro_item = _make_item(raw_instance={"dockerhub_tag": "repo.tag"})
    standard_item = _make_item(instance_id="id2")

    assert SWEBenchContainer(pro_item).is_pro is True
    assert SWEBenchContainer(standard_item).is_pro is False


def test_resolve_image_name_standard() -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(instance_image_key="sweb.eval.x86_64.id:latest")

    assert container._resolve_image_name() == "sweb.eval.x86_64.id:latest"


def test_resolve_image_name_pro() -> None:
    pro_item = _make_item(raw_instance={"dockerhub_tag": "repo.tag"})
    container = SWEBenchContainer(pro_item, dockerhub_username="testuser")

    with patch("src.swebench_container.dockerhub_image_for_instance", return_value="testuser/repo:tag"):
        assert container._resolve_image_name() == "testuser/repo:tag"


def test_exit_calls_cleanup_when_enabled() -> None:
    container = SWEBenchContainer(_make_item(), cleanup=True)

    with patch.object(container, "stop") as stop_mock, patch.object(
        container, "cleanup_image"
    ) as cleanup_mock:
        container.__exit__(None, None, None)

    stop_mock.assert_called_once_with()
    cleanup_mock.assert_called_once_with()


def test_exit_skips_cleanup_when_disabled() -> None:
    container = SWEBenchContainer(_make_item(), cleanup=False)

    with patch.object(container, "stop") as stop_mock, patch.object(
        container, "cleanup_image"
    ) as cleanup_mock:
        container.__exit__(None, None, None)

    stop_mock.assert_called_once_with()
    cleanup_mock.assert_not_called()


def test_stop_removes_stopped_container() -> None:
    container = SWEBenchContainer(_make_item())
    container.container_id = "container-123"

    with patch(
        "src.swebench_container.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stderr=""),
    ) as run_mock:
        container.stop()

    assert [call.args[0] for call in run_mock.call_args_list] == [
        ["docker", "stop", "container-123"],
        ["docker", "rm", "container-123"],
    ]
    assert container.container_id is None
