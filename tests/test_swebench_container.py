from __future__ import annotations

import sys
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from swebench.harness.docker_build import BuildImageError

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.swebench_container import (
    SWEBenchContainer,
    SWEBenchContainerSetupError,
    SWEBenchEnvImageCacheConfig,
    _build_pro_eval_entryscript,
    _env_exports_from_dockerfiles,
    _instance_cache_key,
    _normalize_build_image_error,
    _parse_string_list,
    prewarm_swebench_env_images,
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


def _docker_completed(returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stderr=stderr)


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


def test_exit_only_stops_container() -> None:
    container = SWEBenchContainer(_make_item())

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


def test_stop_falls_back_to_kill_after_timeout() -> None:
    container = SWEBenchContainer(_make_item())
    container.container_id = "container-123"

    with patch(
        "src.swebench_container.subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(["docker", "stop", "container-123"], timeout=120),
            SimpleNamespace(returncode=0, stderr=""),
            SimpleNamespace(returncode=0, stderr=""),
        ],
    ) as run_mock:
        container.stop()

    assert [call.args[0] for call in run_mock.call_args_list] == [
        ["docker", "stop", "container-123"],
        ["docker", "kill", "container-123"],
        ["docker", "rm", "container-123"],
    ]
    assert container.container_id is None


def test_stop_timeout_surfaces_failed_cleanup_and_preserves_container_id() -> None:
    container = SWEBenchContainer(_make_item())
    container.container_id = "container-123"

    with patch(
        "src.swebench_container.subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(["docker", "stop", "container-123"], timeout=120),
            SimpleNamespace(returncode=1, stderr="docker daemon unhealthy"),
            SimpleNamespace(returncode=1, stderr="docker daemon unhealthy"),
        ],
    ) as run_mock, pytest.raises(
        RuntimeError,
        match="Failed to clean up timed-out SWE-bench container",
    ):
        container.stop()

    assert [call.args[0] for call in run_mock.call_args_list] == [
        ["docker", "stop", "container-123"],
        ["docker", "kill", "container-123"],
        ["docker", "rm", "container-123"],
    ]
    assert container.container_id == "container-123"


def test_stop_still_attempts_rm_when_stop_errors() -> None:
    container = SWEBenchContainer(_make_item())
    container.container_id = "container-123"

    with patch(
        "src.swebench_container.subprocess.run",
        side_effect=[
            RuntimeError("docker stop failed"),
            SimpleNamespace(returncode=0, stderr=""),
        ],
    ) as run_mock:
        container.stop()

    assert [call.args[0] for call in run_mock.call_args_list] == [
        ["docker", "stop", "container-123"],
        ["docker", "rm", "container-123"],
    ]
    assert container.container_id is None


def test_normalize_build_image_error_includes_log_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "prepare_image.log"
    log_path.write_text("step 1 ok\nstep 2 failed: wheel build exploded\n", encoding="utf-8")
    exc = BuildImageError(
        "sweb.eval.x86_64.id:latest",
        "docker build exited with code 1",
        SimpleNamespace(log_file=log_path),
    )

    normalized = _normalize_build_image_error(exc, stage="build_container")

    assert isinstance(normalized, SWEBenchContainerSetupError)
    assert normalized.stage == "build_container"
    assert normalized.log_path == str(log_path)
    assert normalized.log_tail == "step 1 ok\nstep 2 failed: wheel build exploded"
    assert "docker build exited with code 1" in str(normalized)
    assert "Build log tail" in str(normalized)


def test_build_with_auto_env_surfaces_failed_env_build_log(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    env_build_dir = tmp_path / "env-build"
    env_build_dir.mkdir()
    log_path = env_build_dir / "build_image.log"
    log_path.write_text("step 1 ok\nstep 2 failed: gcc missing\n", encoding="utf-8")

    with (
        patch(
            "src.swebench_container.build_container",
            side_effect=BuildImageError(
                "sweb.env.shared:latest",
                "env image not found",
                SimpleNamespace(log_file="prepare.log"),
            ),
        ),
        patch(
            "src.swebench_container.build_env_images",
            return_value=(
                [],
                [
                    (
                        "sweb.env.shared:latest",
                        {"setup_env.sh": "echo hi"},
                        "FROM python:3.11",
                        "linux/amd64",
                        object(),
                        env_build_dir,
                    )
                ],
            ),
        ),
    ):
        with pytest.raises(SWEBenchContainerSetupError) as excinfo:
            container._build_with_auto_env(tmp_path / "run")

    assert excinfo.value.stage == "build_env_images"
    assert excinfo.value.log_path == str(log_path)
    assert excinfo.value.log_tail == "step 1 ok\nstep 2 failed: gcc missing"


def test_build_with_auto_env_serializes_shared_env_rebuilds(tmp_path: Path) -> None:
    container_a = SWEBenchContainer(_make_item(instance_id="item-a"))
    container_b = SWEBenchContainer(_make_item(instance_id="item-b"))
    shared_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    for container in (container_a, container_b):
        container._test_spec = shared_spec
        container._docker_client = object()
        container._logger = object()

    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    attempt_counts: dict[int, int] = {}
    build_env_call_count = 0
    build_env_kwargs: list[dict[str, object]] = []

    def fake_build_container(test_spec, docker_client, run_id, logger, *_args):
        del test_spec, docker_client, logger
        thread_id = threading.get_ident()
        with state_lock:
            attempt = attempt_counts.get(thread_id, 0) + 1
            attempt_counts[thread_id] = attempt
            env_ready = build_env_call_count > 0
        if not env_ready:
            if attempt == 1:
                barrier.wait(timeout=2)
            raise BuildImageError(
                "sweb.env.shared:latest",
                "env image not found",
                SimpleNamespace(log_file="build.log"),
            )
        return SimpleNamespace(id=f"container-{run_id}")

    def fake_build_env_images(docker_client, specs, *_args, **_kwargs):
        del docker_client, specs
        nonlocal build_env_call_count
        with state_lock:
            build_env_call_count += 1
            build_env_kwargs.append(dict(_kwargs))
        return [], []

    with (
        patch("src.swebench_container.build_container", side_effect=fake_build_container),
        patch("src.swebench_container.build_env_images", side_effect=fake_build_env_images),
        patch("src.swebench_container.setup_logger", return_value=object()),
        patch("src.swebench_container.close_logger"),
    ):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(container_a._build_with_auto_env, tmp_path / "a"),
                pool.submit(container_b._build_with_auto_env, tmp_path / "b"),
            ]
            results = [future.result() for future in futures]

    assert build_env_call_count == 1
    assert build_env_kwargs == [{"force_rebuild": False, "max_workers": 1}]
    assert all(result.id.startswith("container-") for result in results)


def test_prewarm_swebench_env_images_dedupes_specs_and_uses_requested_workers() -> None:
    item_a = _make_item(instance_id="item-a", raw_instance={"instance_id": "item-a"})
    item_b = _make_item(instance_id="item-b", raw_instance={"instance_id": "item-b"})
    pro_item = _make_item(instance_id="item-pro", raw_instance={"dockerhub_tag": "repo.tag"})
    docker_client = SimpleNamespace(close=lambda: None)

    def fake_make_test_spec(raw_instance, **_kwargs):
        key = "sweb.env.shared:latest"
        if raw_instance["instance_id"] == "item-b":
            key = "sweb.env.unique:latest"
        return SimpleNamespace(env_image_key=key, instance_image_key=f"{key}.instance")

    with (
        patch("src.swebench_container.make_test_spec", side_effect=fake_make_test_spec),
        patch("src.swebench_container._maybe_patch_swebench_builder") as patch_builder_mock,
        patch("src.swebench_container.docker.from_env", return_value=docker_client),
        patch(
            "src.swebench_container.build_env_images",
            return_value=([("shared",), ("unique",)], []),
        ) as build_mock,
    ):
        result = prewarm_swebench_env_images(
            [item_a, item_b, pro_item],
            instance_image_tag="instance-tag",
            env_image_tag="env-tag",
            max_workers=3,
        )

    _, specs = build_mock.call_args.args[:2]
    assert [spec.env_image_key for spec in specs] == [
        "sweb.env.shared:latest",
        "sweb.env.unique:latest",
    ]
    patch_builder_mock.assert_called_once_with()
    assert build_mock.call_args.kwargs == {"force_rebuild": False, "max_workers": 3}
    assert result.unique_env_count == 2
    assert result.built_env_count == 2


def test_prewarm_swebench_env_images_uses_remote_cache_for_hits_and_pushes_misses(tmp_path: Path) -> None:
    item_a = _make_item(instance_id="item-a", raw_instance={"instance_id": "item-a"})
    item_b = _make_item(instance_id="item-b", raw_instance={"instance_id": "item-b"})
    cache_config = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-env",
    )
    docker_client = SimpleNamespace(close=lambda: None)

    def fake_make_test_spec(raw_instance, **_kwargs):
        key = "sweb.env.shared:latest"
        if raw_instance["instance_id"] == "item-b":
            key = "sweb.env.unique:latest"
        return SimpleNamespace(env_image_key=key, instance_image_key=f"{key}.instance")

    shared_remote_ref = cache_config.remote_ref_for("sweb.env.shared:latest")
    unique_remote_ref = cache_config.remote_ref_for("sweb.env.unique:latest")
    docker_calls: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == shared_remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == [shared_remote_ref, "sweb.env.shared:latest"]:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == shared_remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == unique_remote_ref:
            return _docker_completed(returncode=1, stderr="manifest unknown")
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.env.unique:latest", unique_remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == unique_remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == unique_remote_ref:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch("src.swebench_container.make_test_spec", side_effect=fake_make_test_spec),
        patch("src.swebench_container._maybe_patch_swebench_builder"),
        patch("src.swebench_container.docker.from_env", return_value=docker_client),
        patch(
            "src.swebench_container.build_env_images",
            return_value=([("sweb.env.unique:latest", {}, "", "", object(), tmp_path / "env-build")], []),
        ) as build_mock,
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = prewarm_swebench_env_images(
            [item_a, item_b],
            instance_image_tag="instance-tag",
            env_image_tag="env-tag",
            max_workers=3,
            env_image_cache=cache_config,
        )

    assert result.unique_env_count == 2
    assert result.built_env_count == 1
    build_mock.assert_called_once()
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", shared_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", shared_remote_ref, "sweb.env.shared:latest"] for call in docker_calls)
    assert any(call == ["docker", "rmi", shared_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", unique_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", "sweb.env.unique:latest", unique_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "push", unique_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", unique_remote_ref] for call in docker_calls)


def test_instance_cache_key_changes_when_repo_setup_changes() -> None:
    shared = {
        "env_image_key": "sweb.env.shared:latest",
        "instance_image_key": "sweb.eval.shared:latest",
        "env_script_list": ["setup env"],
        "docker_specs": {"base_image": "python:3.11"},
    }
    spec_a = SimpleNamespace(repo_script_list=["clone branch"], **shared)
    spec_b = SimpleNamespace(repo_script_list=["clone branch", "pip install 'pip<24'"], **shared)

    assert _instance_cache_key(spec_a) != _instance_cache_key(spec_b)


def test_build_with_auto_env_uses_cached_instance_image_without_local_rebuild(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-images",
    )

    remote_ref = container.env_image_cache.remote_ref_for(
        _instance_cache_key(container._test_spec),
        kind="instance",
    )
    docker_calls: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == [remote_ref, "sweb.eval.shared:latest"]:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == remote_ref:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch(
            "src.swebench_container.build_container",
            return_value=SimpleNamespace(id="container-from-cache"),
        ) as build_mock,
        patch("src.swebench_container.build_env_images") as build_env_mock,
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id == "container-from-cache"
    build_mock.assert_called_once()
    build_env_mock.assert_not_called()
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", remote_ref, "sweb.eval.shared:latest"] for call in docker_calls)
    assert any(call == ["docker", "rmi", remote_ref] for call in docker_calls)
    assert not any(call[:2] == ["docker", "push"] for call in docker_calls)


def test_build_with_auto_env_pushes_instance_cache_after_local_build(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-images",
    )

    remote_ref = container.env_image_cache.remote_ref_for(
        _instance_cache_key(container._test_spec),
        kind="instance",
    )
    docker_calls: list[list[str]] = []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == remote_ref:
            return _docker_completed(returncode=1, stderr="manifest unknown")
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.eval.shared:latest", remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == remote_ref:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch(
            "src.swebench_container.build_container",
            return_value=SimpleNamespace(id="container-local-build"),
        ) as build_mock,
        patch("src.swebench_container.build_env_images") as build_env_mock,
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id == "container-local-build"
    build_mock.assert_called_once()
    build_env_mock.assert_not_called()
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", "sweb.eval.shared:latest", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "push", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", remote_ref] for call in docker_calls)


def test_build_with_auto_env_removes_bad_cached_instance_image_before_local_rebuild(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
        repo_script_list=["setup repo"],
        env_script_list=["setup env"],
        docker_specs={"base_image": "python:3.11"},
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-images",
    )

    remote_ref = container.env_image_cache.remote_ref_for(
        _instance_cache_key(container._test_spec),
        kind="instance",
    )
    docker_calls: list[list[str]] = []
    build_container_calls = 0

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == [remote_ref, "sweb.eval.shared:latest"]:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.eval.shared:latest", remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] in {remote_ref, "sweb.eval.shared:latest"}:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    def fake_build_container(test_spec, docker_client, run_id, logger, *_args):
        del test_spec, docker_client, logger
        nonlocal build_container_calls
        build_container_calls += 1
        if build_container_calls == 1:
            raise BuildImageError(
                "sweb.eval.shared:latest",
                "cached instance image is stale",
                SimpleNamespace(log_file="build.log"),
            )
        return SimpleNamespace(id=f"container-{run_id}")

    with (
        patch("src.swebench_container.build_container", side_effect=fake_build_container) as build_mock,
        patch("src.swebench_container.build_env_images") as build_env_mock,
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id.startswith("container-")
    assert build_mock.call_count == 2
    build_env_mock.assert_not_called()
    assert build_container_calls == 2
    assert any(call == ["docker", "rmi", "sweb.eval.shared:latest"] for call in docker_calls)


def test_build_with_auto_env_falls_back_to_local_env_build_and_pushes_cache(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-images",
    )

    env_remote_ref = container.env_image_cache.remote_ref_for("sweb.env.shared:latest", kind="env")
    instance_remote_ref = container.env_image_cache.remote_ref_for(
        _instance_cache_key(container._test_spec),
        kind="instance",
    )
    docker_calls: list[list[str]] = []
    build_container_calls = 0

    def fake_build_container(test_spec, docker_client, run_id, logger, *_args):
        del test_spec, docker_client, logger
        nonlocal build_container_calls
        build_container_calls += 1
        if build_container_calls < 3:
            raise BuildImageError(
                "sweb.env.shared:latest",
                "env image not found",
                SimpleNamespace(log_file="build.log"),
            )
        return SimpleNamespace(id=f"container-{run_id}")

    def fake_build_env_images(docker_client, specs, *_args, **_kwargs):
        del docker_client, specs
        return [("sweb.env.shared:latest", {}, "", "", object(), tmp_path / "env-build")], []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == instance_remote_ref:
            return _docker_completed(returncode=1, stderr="manifest unknown")
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == env_remote_ref:
            return _docker_completed(returncode=1, stderr="manifest unknown")
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.env.shared:latest", env_remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == env_remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.eval.shared:latest", instance_remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == instance_remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] in {env_remote_ref, instance_remote_ref}:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch("src.swebench_container.build_container", side_effect=fake_build_container),
        patch("src.swebench_container.build_env_images", side_effect=fake_build_env_images) as build_mock,
        patch("src.swebench_container.setup_logger", return_value=object()),
        patch("src.swebench_container.close_logger"),
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id.startswith("container-")
    build_mock.assert_called_once()
    assert build_container_calls == 3
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", instance_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", env_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", "sweb.env.shared:latest", env_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "push", env_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", env_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", "sweb.eval.shared:latest", instance_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "push", instance_remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", instance_remote_ref] for call in docker_calls)


def test_build_with_auto_env_rebuilds_locally_after_stale_cache_hit(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-env",
    )

    remote_ref = container.env_image_cache.remote_ref_for("sweb.env.shared:latest")
    docker_calls: list[list[str]] = []
    build_container_calls = 0

    def fake_build_container(test_spec, docker_client, run_id, logger, *_args):
        del test_spec, docker_client, logger
        nonlocal build_container_calls
        build_container_calls += 1
        if build_container_calls in (1, 2):
            raise BuildImageError(
                "sweb.env.shared:latest",
                "env image not found",
                SimpleNamespace(log_file="build.log"),
            )
        if build_container_calls == 3:
            raise BuildImageError(
                "sweb.env.shared:latest",
                "cached env image is stale",
                SimpleNamespace(log_file="build.log"),
            )
        return SimpleNamespace(id=f"container-{run_id}")

    def fake_build_env_images(docker_client, specs, *_args, **_kwargs):
        del docker_client, specs
        return [("sweb.env.shared:latest", {}, "", "", object(), tmp_path / "env-build")], []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == [remote_ref, "sweb.env.shared:latest"]:
            return _docker_completed()
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.env.shared:latest", remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == remote_ref:
            return _docker_completed()
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == remote_ref:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch("src.swebench_container.build_container", side_effect=fake_build_container),
        patch("src.swebench_container.build_env_images", side_effect=fake_build_env_images) as build_mock,
        patch("src.swebench_container.setup_logger", return_value=object()),
        patch("src.swebench_container.close_logger"),
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id.startswith("container-")
    build_mock.assert_called_once()
    assert build_container_calls == 4
    assert any(call == ["docker", "pull", "--platform", "linux/amd64", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "tag", remote_ref, "sweb.env.shared:latest"] for call in docker_calls)
    assert any(call == ["docker", "push", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", remote_ref] for call in docker_calls)


def test_build_with_auto_env_ignores_cache_push_failures(tmp_path: Path) -> None:
    container = SWEBenchContainer(_make_item())
    container._test_spec = SimpleNamespace(
        env_image_key="sweb.env.shared:latest",
        instance_image_key="sweb.eval.shared:latest",
    )
    container._docker_client = object()
    container._logger = object()
    container.env_image_cache = SWEBenchEnvImageCacheConfig(
        repository="us-central1-docker.pkg.dev/project/cache/sweb-env",
    )

    remote_ref = container.env_image_cache.remote_ref_for("sweb.env.shared:latest")
    docker_calls: list[list[str]] = []
    build_container_calls = 0

    def fake_build_container(test_spec, docker_client, run_id, logger, *_args):
        del test_spec, docker_client, logger
        nonlocal build_container_calls
        build_container_calls += 1
        if build_container_calls < 3:
            raise BuildImageError(
                "sweb.env.shared:latest",
                "env image not found",
                SimpleNamespace(log_file="build.log"),
            )
        return SimpleNamespace(id=f"container-{run_id}")

    def fake_build_env_images(docker_client, specs, *_args, **_kwargs):
        del docker_client, specs
        return [("sweb.env.shared:latest", {}, "", "", object(), tmp_path / "env-build")], []

    def fake_subprocess_run(cmd, **_kwargs):
        docker_calls.append(list(cmd))
        if cmd[:2] == ["docker", "pull"] and cmd[-1] == remote_ref:
            return _docker_completed(returncode=1, stderr="manifest unknown")
        if cmd[:2] == ["docker", "tag"] and cmd[-2:] == ["sweb.env.shared:latest", remote_ref]:
            return _docker_completed()
        if cmd[:2] == ["docker", "push"] and cmd[-1] == remote_ref:
            return _docker_completed(returncode=1, stderr="denied")
        if cmd[:2] == ["docker", "rmi"] and cmd[-1] == remote_ref:
            return _docker_completed()
        raise AssertionError(f"unexpected docker command: {cmd}")

    with (
        patch("src.swebench_container.build_container", side_effect=fake_build_container),
        patch("src.swebench_container.build_env_images", side_effect=fake_build_env_images),
        patch("src.swebench_container.setup_logger", return_value=object()),
        patch("src.swebench_container.close_logger"),
        patch("src.swebench_container.subprocess.run", side_effect=fake_subprocess_run),
    ):
        result = container._build_with_auto_env(tmp_path / "run")

    assert result.id.startswith("container-")
    assert build_container_calls == 3
    assert any(call == ["docker", "push", remote_ref] for call in docker_calls)
    assert any(call == ["docker", "rmi", remote_ref] for call in docker_calls)


def test_prewarm_swebench_env_images_surfaces_failed_build_logs(tmp_path: Path) -> None:
    item = _make_item(instance_id="item-a", raw_instance={"instance_id": "item-a"})
    env_build_dir = tmp_path / "env-build"
    env_build_dir.mkdir()
    log_path = env_build_dir / "build_image.log"
    log_path.write_text("step 1 ok\nstep 2 failed: gcc missing\n", encoding="utf-8")
    docker_client = SimpleNamespace(close=lambda: None)

    with (
        patch(
            "src.swebench_container.make_test_spec",
            return_value=SimpleNamespace(
                env_image_key="sweb.env.shared:latest",
                instance_image_key="sweb.eval.shared:latest",
            ),
        ),
        patch("src.swebench_container.docker.from_env", return_value=docker_client),
        patch(
            "src.swebench_container.build_env_images",
            return_value=(
                [],
                [
                    (
                        "sweb.env.shared:latest",
                        {"setup_env.sh": "echo hi"},
                        "FROM python:3.11",
                        "linux/amd64",
                        object(),
                        env_build_dir,
                    )
                ],
            ),
        ),
    ):
        with pytest.raises(SWEBenchContainerSetupError) as excinfo:
            prewarm_swebench_env_images([item], max_workers=2)

    assert excinfo.value.stage == "build_env_images"
    assert excinfo.value.log_path == str(log_path)
    assert excinfo.value.log_tail == "step 1 ok\nstep 2 failed: gcc missing"
