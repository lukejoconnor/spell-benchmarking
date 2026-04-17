"""Official-image SWE-bench container runner."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import shlex
import subprocess
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
from swebench.harness.constants import (
    DOCKER_PATCH,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
)
from swebench.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    setup_logger,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from .datasets import SWEBenchItem
from .swebench_eval_wrapper import _maybe_patch_swebench_builder
from .swebench_pro_eval import dockerhub_image_for_instance, ensure_scale_pro_eval_repo

logger = logging.getLogger(__name__)

PRO_WORKDIR = "/app"
_SETUP_LOG_TAIL_CHARS = 4000
_DOCKER_STOP_TIMEOUT_SECONDS = 120
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 30
_ENV_BUILD_LOCKS: dict[str, threading.Lock] = {}
_ENV_BUILD_LOCKS_GUARD = threading.Lock()


@dataclass(slots=True)
class SameContainerEvalResult:
    resolved: bool | None
    test_output: str
    raw_report: dict[str, Any] | None


@dataclass(slots=True)
class SWEBenchEnvPrewarmResult:
    unique_env_count: int
    built_env_count: int


@dataclass(slots=True, frozen=True)
class SWEBenchEnvImageCacheConfig:
    repository: str
    write_enabled: bool = True

    def remote_ref_for(self, image_key: str, *, kind: str = "env") -> str:
        return f"{self.repository}:{_image_cache_tag(image_key, kind=kind)}"


@dataclass(slots=True)
class SWEBenchContainerSetupError(Exception):
    stage: str
    detail: str
    log_path: str | None = None
    log_tail: str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.detail)

    def __str__(self) -> str:
        parts = [f"SWE-bench container setup failed during {self.stage}: {self.detail}"]
        if self.log_path:
            parts.append(f"Build log: {self.log_path}")
        if self.log_tail:
            parts.append(f"Build log tail:\n{self.log_tail}")
        return "\n\n".join(parts)

    def metadata(self) -> dict[str, str]:
        metadata = {"container_error_stage": self.stage}
        if self.log_path:
            metadata["container_error_log_path"] = self.log_path
        if self.log_tail:
            metadata["container_error_log_tail"] = self.log_tail
        return metadata


def _archive_bytes_for_files(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _parse_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    raise ValueError(f"Expected list-like value, got {type(value).__name__}")


def _shell_quote_legacy_env_value(value: str) -> str:
    if not any(char.isspace() for char in value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _docker_env_line_to_export(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("ENV "):
        return None

    payload = stripped[4:].strip()
    if not payload:
        return None

    first_token = payload.split(None, 1)[0]
    if "=" not in first_token:
        parts = payload.split(None, 1)
        if len(parts) != 2:
            return None
        key, value = parts
        return f"export {key}={_shell_quote_legacy_env_value(value)}"

    return f"export {payload}"


def _env_exports_from_dockerfiles(*dockerfiles: Path) -> str:
    exports: list[str] = []
    for dockerfile in dockerfiles:
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            export_line = _docker_env_line_to_export(line)
            if export_line:
                exports.append(export_line)
    return "\n".join(exports)


def _build_pro_eval_entryscript(
    *,
    container_workdir: str,
    env_exports: str,
    before_repo_set_cmd: str,
    selected_tests: list[str],
) -> str:
    selected_arg = ",".join(selected_tests)
    entry_lines = []
    if env_exports:
        entry_lines.append(env_exports)
    entry_lines.extend(
        [
            "set -e",
            f"cd {container_workdir}",
        ]
    )
    if before_repo_set_cmd:
        entry_lines.append(before_repo_set_cmd)
    run_cmd = "bash /tmp/run_script.sh"
    if selected_arg:
        run_cmd = f"{run_cmd} {shlex.quote(selected_arg)}"
    entry_lines.extend(
        [
            f"{run_cmd} > /tmp/stdout.log 2> /tmp/stderr.log",
            "python /tmp/parser.py /tmp/stdout.log /tmp/stderr.log /tmp/output.json",
        ]
    )
    return "\n".join(entry_lines) + "\n"


def remove_swebench_instance_image(image_name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "rmi", image_name],
            capture_output=True,
            text=True,
            timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("Error removing SWE-bench instance image %s", image_name, exc_info=True)
        return False

    if result.returncode == 0:
        logger.info("Removed SWE-bench instance image %s", image_name)
        return True

    logger.warning(
        "Failed to remove SWE-bench instance image %s: %s",
        image_name,
        result.stderr.strip(),
    )
    return False


def _image_cache_tag(image_key: str, *, kind: str) -> str:
    digest = hashlib.sha256(image_key.encode("utf-8")).hexdigest()
    return f"{kind}-{digest}"


def _docker_run_command(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _env_build_lock_key(test_spec: Any) -> str:
    env_image_key = getattr(test_spec, "env_image_key", None)
    if env_image_key:
        return str(env_image_key)
    return str(test_spec.instance_image_key)


def _env_build_lock_for(test_spec: Any) -> threading.Lock:
    key = _env_build_lock_key(test_spec)
    with _ENV_BUILD_LOCKS_GUARD:
        lock = _ENV_BUILD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ENV_BUILD_LOCKS[key] = lock
        return lock


def _read_log_tail(log_path: str | Path | None, limit: int = _SETUP_LOG_TAIL_CHARS) -> str | None:
    if not log_path:
        return None
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text:
        return None
    return text[-limit:]


def _cache_remote_ref(
    image_key: str,
    cache_config: SWEBenchEnvImageCacheConfig,
    *,
    kind: str,
) -> str:
    return cache_config.remote_ref_for(image_key, kind=kind)


def _pull_cached_image(
    image_key: str,
    cache_config: SWEBenchEnvImageCacheConfig,
    *,
    kind: str,
) -> bool:
    remote_ref = _cache_remote_ref(image_key, cache_config, kind=kind)
    logger.info("Trying SWE-bench %s cache pull for %s from %s", kind, image_key, remote_ref)
    try:
        pull = _docker_run_command(["pull", "--platform", "linux/amd64", remote_ref], timeout=600)
    except Exception:
        logger.warning(
            "Error pulling SWE-bench %s cache image %s for %s",
            kind,
            remote_ref,
            image_key,
            exc_info=True,
        )
        return False

    if pull.returncode != 0:
        stderr = pull.stderr.strip()
        if stderr:
            logger.info(
                "SWE-bench %s cache miss for %s from %s: %s",
                kind,
                image_key,
                remote_ref,
                stderr,
            )
        else:
            logger.info("SWE-bench %s cache miss for %s from %s", kind, image_key, remote_ref)
        return False

    try:
        tag = _docker_run_command(["tag", remote_ref, image_key], timeout=60)
    except Exception:
        logger.warning(
            "Error retagging SWE-bench %s cache image %s to %s",
            kind,
            remote_ref,
            image_key,
            exc_info=True,
        )
        return False

    if tag.returncode != 0:
        logger.warning(
            "Failed to retag SWE-bench %s cache image %s to %s: %s",
            kind,
            remote_ref,
            image_key,
            tag.stderr.strip(),
        )
        return False

    logger.info("Pulled SWE-bench %s cache image %s into %s", kind, remote_ref, image_key)
    return True


def _push_cached_image(
    image_key: str,
    cache_config: SWEBenchEnvImageCacheConfig,
    *,
    kind: str,
) -> bool:
    if not cache_config.write_enabled:
        return False

    remote_ref = _cache_remote_ref(image_key, cache_config, kind=kind)
    logger.info("Publishing SWE-bench %s image %s to %s", kind, image_key, remote_ref)
    try:
        tag = _docker_run_command(["tag", image_key, remote_ref], timeout=60)
    except Exception:
        logger.warning(
            "Error tagging SWE-bench %s image %s for cache push to %s",
            kind,
            image_key,
            remote_ref,
            exc_info=True,
        )
        return False

    if tag.returncode != 0:
        logger.warning(
            "Failed to tag SWE-bench %s image %s for cache push to %s: %s",
            kind,
            image_key,
            remote_ref,
            tag.stderr.strip(),
        )
        return False

    try:
        push = _docker_run_command(["push", remote_ref], timeout=1200)
    except Exception:
        logger.warning(
            "Error pushing SWE-bench %s image %s to %s",
            kind,
            image_key,
            remote_ref,
            exc_info=True,
        )
        return False

    if push.returncode != 0:
        logger.warning(
            "Failed to push SWE-bench %s image %s to %s: %s",
            kind,
            image_key,
            remote_ref,
            push.stderr.strip(),
        )
        return False

    logger.info("Published SWE-bench %s image %s to %s", kind, image_key, remote_ref)
    return True


def _normalize_setup_error(
    *,
    stage: str,
    detail: str,
    log_path: str | Path | None = None,
) -> SWEBenchContainerSetupError:
    log_path_str = str(log_path) if log_path else None
    return SWEBenchContainerSetupError(
        stage=stage,
        detail=detail.strip(),
        log_path=log_path_str,
        log_tail=_read_log_tail(log_path_str),
    )


def _normalize_build_image_error(exc: BuildImageError, *, stage: str) -> SWEBenchContainerSetupError:
    image_name = getattr(exc, "image_name", None)
    base_detail = getattr(exc, "super_str", None) or str(exc)
    detail = f"Error building image {image_name}: {base_detail}" if image_name else base_detail
    return _normalize_setup_error(stage=stage, detail=detail, log_path=getattr(exc, "log_path", None))


def _normalize_failed_env_build(failed_payloads: list[tuple[Any, ...]]) -> SWEBenchContainerSetupError:
    first_failure = failed_payloads[0] if failed_payloads else ()
    image_name = first_failure[0] if first_failure else None
    build_dir = first_failure[5] if len(first_failure) > 5 else None
    extra_failures = len(failed_payloads) - 1
    detail = "Environment image build failed"
    if image_name:
        detail = f"Error building image {image_name}: environment image build failed"
    if extra_failures > 0:
        detail = f"{detail} (+{extra_failures} more failures)"
    log_path = Path(build_dir) / "build_image.log" if build_dir else None
    return _normalize_setup_error(stage="build_env_images", detail=detail, log_path=log_path)


def _make_standard_test_spec(
    item: SWEBenchItem,
    *,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
) -> TestSpec:
    return make_test_spec(
        item.raw_instance,
        namespace=None,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        arch="x86_64",
    )


def _unique_env_test_specs(
    items: list[SWEBenchItem],
    *,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
) -> list[TestSpec]:
    specs_by_env: dict[str, TestSpec] = {}
    for item in items:
        if (item.raw_instance or {}).get("dockerhub_tag"):
            continue
        spec = _make_standard_test_spec(
            item,
            instance_image_tag=instance_image_tag,
            env_image_tag=env_image_tag,
        )
        specs_by_env.setdefault(_env_build_lock_key(spec), spec)
    return list(specs_by_env.values())


def prewarm_swebench_env_images(
    items: list[SWEBenchItem],
    *,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    max_workers: int = 1,
    env_image_cache: SWEBenchEnvImageCacheConfig | None = None,
) -> SWEBenchEnvPrewarmResult:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    _maybe_patch_swebench_builder()

    specs = _unique_env_test_specs(
        items,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
    )
    if not specs:
        return SWEBenchEnvPrewarmResult(unique_env_count=0, built_env_count=0)

    cached_env_keys: set[str] = set()
    if env_image_cache is not None:
        for spec in specs:
            env_image_key = _env_build_lock_key(spec)
            if _pull_cached_image(env_image_key, env_image_cache, kind="env"):
                cached_env_keys.add(env_image_key)

    docker_client = docker.from_env()
    try:
        successful, failed = build_env_images(
            docker_client,
            specs,
            force_rebuild=False,
            max_workers=max_workers,
        )
    except BuildImageError as exc:
        raise _normalize_build_image_error(exc, stage="build_env_images") from exc
    except SWEBenchContainerSetupError:
        raise
    except Exception as exc:
        raise _normalize_setup_error(
            stage="build_env_images",
            detail=str(exc) or "Unknown env image build failure",
        ) from exc
    finally:
        docker_client.close()

    if failed:
        raise _normalize_failed_env_build(failed)

    if env_image_cache is not None:
        for payload in successful:
            env_image_key = payload[0]
            if env_image_key in cached_env_keys:
                continue
            _push_cached_image(str(env_image_key), env_image_cache, kind="env")

    return SWEBenchEnvPrewarmResult(
        unique_env_count=len(specs),
        built_env_count=len(successful),
    )


class SWEBenchContainer:
    """Container backed by official SWE-bench instance images."""

    def __init__(
        self,
        item: SWEBenchItem,
        *,
        pro_eval_repo: str | None = None,
        dockerhub_username: str = "jefzda",
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
        env_image_cache: SWEBenchEnvImageCacheConfig | None = None,
    ):
        self.item = item
        self.pro_eval_repo = pro_eval_repo
        self.dockerhub_username = dockerhub_username
        self.instance_image_tag = instance_image_tag
        self.env_image_tag = env_image_tag
        self.env_image_cache = env_image_cache

        self.container_workdir = PRO_WORKDIR if self.is_pro else DOCKER_WORKDIR
        self.container_id: str | None = None
        self.initial_commit: str | None = None

        self._docker_client: docker.DockerClient | None = None
        self._container_obj: Any | None = None
        self._logger: Any | None = None
        self._test_spec: TestSpec | None = None
        self._image_name: str | None = None

    @property
    def is_pro(self) -> bool:
        return bool((self.item.raw_instance or {}).get("dockerhub_tag"))

    def start(self) -> str:
        _maybe_patch_swebench_builder()

        if self.is_pro:
            self._start_pro_container()
        else:
            self._start_standard_container()

        head_result = self.exec_run("git rev-parse HEAD", timeout=30)
        if head_result.exit_code != 0:
            raise RuntimeError(
                f"git rev-parse HEAD failed in container: {head_result.stderr.strip()}"
            )
        self.initial_commit = head_result.stdout.strip() or None
        if self.container_id is None:
            raise RuntimeError("Container failed to start")
        return self.container_id

    def _start_standard_container(self) -> None:
        self._docker_client = docker.from_env()
        self._test_spec = _make_standard_test_spec(
            self.item,
            env_image_tag=self.env_image_tag,
            instance_image_tag=self.instance_image_tag,
        )

        log_dir = Path(tempfile.mkdtemp(prefix=f"swebench-build-{uuid.uuid4().hex[:8]}-"))
        self._logger = setup_logger(self.item.instance_id, log_dir / "prepare_image.log")
        try:
            container = self._build_with_auto_env(log_dir)
            container.start()
        except Exception:
            close_logger(self._logger)
            self._logger = None
            raise

        self._image_name = self._test_spec.instance_image_key
        self._container_obj = container
        self.container_id = container.id

    def _build_with_auto_env(self, log_dir: Path):
        """Build the instance container, auto-building env images if missing."""
        run_id = uuid.uuid4().hex[:8]
        instance_image_key = str(self._test_spec.instance_image_key)

        def _build_container_once(*, logger_override, push_instance_cache: bool):
            container = build_container(
                self._test_spec, self._docker_client, run_id, logger_override, False, False
            )
            if push_instance_cache and self.env_image_cache is not None:
                _push_cached_image(instance_image_key, self.env_image_cache, kind="instance")
            return container

        if self.env_image_cache is not None and _pull_cached_image(
            instance_image_key, self.env_image_cache, kind="instance"
        ):
            try:
                return _build_container_once(
                    logger_override=self._logger,
                    push_instance_cache=False,
                )
            except BuildImageError:
                logger.warning(
                    "Cached SWE-bench instance image %s for %s did not produce a usable container; rebuilding locally",
                    instance_image_key,
                    self.item.instance_id,
                    exc_info=True,
                )

        try:
            return _build_container_once(
                logger_override=self._logger,
                push_instance_cache=True,
            )
        except BuildImageError as exc:
            if "not found" not in str(exc).lower():
                raise _normalize_build_image_error(exc, stage="build_container") from exc
        env_lock_key = _env_build_lock_key(self._test_spec)
        logger.info(
            "Env image %s missing for %s; waiting for rebuild lock",
            env_lock_key,
            self.item.instance_id,
        )
        with _env_build_lock_for(self._test_spec):
            try:
                return _build_container_once(
                    logger_override=self._logger,
                    push_instance_cache=True,
                )
            except BuildImageError as exc:
                if "not found" not in str(exc).lower():
                    raise _normalize_build_image_error(exc, stage="build_container") from exc
            if self.env_image_cache is not None and _pull_cached_image(
                env_lock_key, self.env_image_cache, kind="env"
            ):
                try:
                    return _build_container_once(
                        logger_override=self._logger,
                        push_instance_cache=True,
                    )
                except BuildImageError as exc:
                    logger.warning(
                        "Cached SWE-bench env image %s for %s did not produce a usable container; rebuilding locally",
                        env_lock_key,
                        self.item.instance_id,
                        exc_info=True,
                    )
            logger.info(
                "Env image %s missing for %s; building from scratch",
                env_lock_key,
                self.item.instance_id,
            )
            try:
                _, failed = build_env_images(
                    self._docker_client,
                    [self._test_spec],
                    force_rebuild=False,
                    max_workers=1,
                )
                if failed:
                    raise _normalize_failed_env_build(failed)
            except BuildImageError as exc:
                raise _normalize_build_image_error(exc, stage="build_env_images") from exc
            except SWEBenchContainerSetupError:
                raise
            except Exception as exc:
                raise _normalize_setup_error(
                    stage="build_env_images",
                    detail=str(exc) or "Unknown env image build failure",
                ) from exc
        if self.env_image_cache is not None:
            _push_cached_image(env_lock_key, self.env_image_cache, kind="env")
        env_logger = setup_logger(
            self.item.instance_id, log_dir / "rebuild_after_env.log"
        )
        try:
            return _build_container_once(
                logger_override=env_logger,
                push_instance_cache=True,
            )
        except BuildImageError as exc:
            raise _normalize_build_image_error(exc, stage="build_container") from exc
        finally:
            close_logger(env_logger)

    def _start_pro_container(self) -> None:
        image_uri = self._resolve_image_name()
        if image_uri is None:
            raise RuntimeError(f"Could not resolve Docker image for {self.item.instance_id}")

        pull = subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", image_uri],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"docker pull failed for {image_uri}: {pull.stderr.strip()}")

        run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--platform",
                "linux/amd64",
                "-w",
                self.container_workdir,
                "--entrypoint",
                "/bin/bash",
                image_uri,
                "-lc",
                "tail -f /dev/null",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if run.returncode != 0:
            raise RuntimeError(f"Docker run failed: {run.stderr.strip()}")

        self.container_id = run.stdout.strip()

    def _resolve_image_name(self) -> str | None:
        if self.is_pro:
            return dockerhub_image_for_instance(
                {self.item.instance_id: self.item.raw_instance},
                self.item.instance_id,
                dockerhub_username=self.dockerhub_username,
            )
        if self._test_spec is None:
            return None
        return self._test_spec.instance_image_key

    @property
    def image_name(self) -> str | None:
        if self._image_name is not None:
            return self._image_name
        try:
            return self._resolve_image_name()
        except Exception:
            logger.warning(
                "Failed to resolve SWE-bench instance image: %s",
                self.item.instance_id,
                exc_info=True,
            )
            return None

    def exec_run(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ):
        if not self.container_id:
            raise RuntimeError("Container not started")

        docker_cmd = ["docker", "exec"]
        if workdir:
            docker_cmd.extend(["-w", workdir])
        for key, value in (env or {}).items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.extend([self.container_id, "bash", "-lc", cmd])

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            from .docker_agents import ExecOutput

            return ExecOutput(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                timed_out=True,
            )

        from .docker_agents import ExecOutput

        return ExecOutput(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )

    def put_tar_archive(self, dest_dir: str, archive_bytes: bytes) -> bool:
        if not self.container_id:
            raise RuntimeError("Container not started")
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp.write(archive_bytes)
            tmp_path = Path(tmp.name)
        try:
            archive_name = f"/tmp/{tmp_path.name}"
            cp = subprocess.run(
                ["docker", "cp", str(tmp_path), f"{self.container_id}:{archive_name}"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if cp.returncode != 0:
                logger.error("docker cp for tar archive failed: %s", cp.stderr.strip())
                return False
            unpack = self.exec_run(
                f"mkdir -p {shlex.quote(dest_dir)} && "
                f"tar -xf {shlex.quote(archive_name)} -C {shlex.quote(dest_dir)} && "
                f"rm -f {shlex.quote(archive_name)}",
                timeout=180,
                workdir="/",
            )
            return unpack.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    def copy_to_host(self, source: str, dest: Path) -> bool:
        if not self.container_id:
            raise RuntimeError("Container not started")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["docker", "cp", f"{self.container_id}:{source}", str(dest)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            logger.warning("docker cp from container failed: %s", proc.stderr.strip())
            return False
        return True

    def read_text(self, path: str) -> str:
        result = self.exec_run(f"cat {shlex.quote(path)}", timeout=60, workdir="/")
        return result.stdout if result.exit_code == 0 else ""

    def diff(self, base: str = "HEAD") -> str:
        result = self.exec_run(
            f"git add -A && git diff --cached {shlex.quote(base)}"
            " -- . ':!.spell-trace'; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status",
            workdir=self.container_workdir,
        )
        if result.exit_code != 0:
            return ""
        return result.stdout

    def changed_files(self, base: str = "HEAD") -> list[str]:
        result = self.exec_run(
            f"git add -A && git diff --name-only --cached {shlex.quote(base)}"
            " -- . ':!.spell-trace'; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status",
            workdir=self.container_workdir,
        )
        if result.exit_code != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def run_eval(self, *, timeout: int) -> SameContainerEvalResult:
        if self.is_pro:
            return self._run_pro_eval(timeout=timeout)
        return self._run_standard_eval(timeout=timeout)

    def _run_standard_eval(self, *, timeout: int) -> SameContainerEvalResult:
        if self._test_spec is None:
            raise RuntimeError("Test spec not initialized")

        files = {"eval.sh": self._test_spec.eval_script}
        if not self.put_tar_archive("/tmp", _archive_bytes_for_files(files)):
            raise RuntimeError("Failed to copy eval script into container")

        log_path = "/tmp/same-container-eval.log"
        run = self.exec_run(
            f"chmod +x /tmp/eval.sh && /bin/bash /tmp/eval.sh > {log_path} 2>&1",
            timeout=timeout,
            workdir="/",
        )
        test_output = self.read_text(log_path)
        if run.timed_out and "Timeout" not in test_output:
            test_output = f"{test_output}\n\nTimeout after {timeout}s".strip()

        patch = self.diff(self.initial_commit or "HEAD")
        prediction = {
            KEY_INSTANCE_ID: self.item.instance_id,
            KEY_PREDICTION: patch,
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(test_output)
            tmp_path = Path(tmp.name)
        try:
            raw_report = get_eval_report(
                self._test_spec,
                prediction,
                str(tmp_path),
                include_tests_status=True,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        entry = (raw_report or {}).get(self.item.instance_id, {})
        return SameContainerEvalResult(
            resolved=entry.get("resolved"),
            test_output=test_output,
            raw_report=raw_report,
        )

    def _run_pro_eval(self, *, timeout: int) -> SameContainerEvalResult:
        if not self.pro_eval_repo:
            raise RuntimeError("SWE-bench Pro same-container eval requires --pro-eval-repo")

        eval_repo = ensure_scale_pro_eval_repo(self.pro_eval_repo)
        instance_id = self.item.instance_id
        scripts_dir = eval_repo / "run_scripts" / instance_id
        base_dockerfile = eval_repo / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"
        instance_dockerfile = (
            eval_repo / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"
        )

        env_exports = _env_exports_from_dockerfiles(base_dockerfile, instance_dockerfile)
        before_repo_set_cmd = str(
            (self.item.raw_instance or {}).get("before_repo_set_cmd", "")
        ).strip()
        selected_tests = _parse_string_list(
            (self.item.raw_instance or {}).get("selected_test_files_to_run", "[]")
        )
        files = {
            "run_script.sh": (scripts_dir / "run_script.sh").read_text(encoding="utf-8"),
            "parser.py": (scripts_dir / "parser.py").read_text(encoding="utf-8"),
            "entryscript.sh": _build_pro_eval_entryscript(
                container_workdir=self.container_workdir,
                env_exports=env_exports,
                before_repo_set_cmd=before_repo_set_cmd,
                selected_tests=selected_tests,
            ),
        }
        if not self.put_tar_archive("/tmp", _archive_bytes_for_files(files)):
            raise RuntimeError("Failed to copy SWE-bench Pro eval assets into container")

        run = self.exec_run(
            "chmod +x /tmp/run_script.sh /tmp/entryscript.sh && /bin/bash /tmp/entryscript.sh",
            timeout=timeout,
            workdir="/",
        )
        stdout = self.read_text("/tmp/stdout.log")
        stderr = self.read_text("/tmp/stderr.log")
        output_json = self.read_text("/tmp/output.json")
        if run.timed_out and "Timeout" not in stderr:
            stderr = f"{stderr}\nTimeout after {timeout}s".strip()

        parsed_output = json.loads(output_json) if output_json else {"tests": []}
        passed_tests = {
            str(test["name"])
            for test in parsed_output.get("tests", [])
            if test.get("status") == "PASSED"
        }
        fail_to_pass = set(
            _parse_string_list(
                (self.item.raw_instance or {}).get(
                    "fail_to_pass",
                    (self.item.raw_instance or {}).get("FAIL_TO_PASS", "[]"),
                )
            )
        )
        pass_to_pass = set(
            _parse_string_list(
                (self.item.raw_instance or {}).get(
                    "pass_to_pass",
                    (self.item.raw_instance or {}).get("PASS_TO_PASS", "[]"),
                )
            )
        )
        resolved = (fail_to_pass | pass_to_pass) <= passed_tests
        raw_report = {
            self.item.instance_id: {
                "resolved": resolved,
                "passed_tests": sorted(passed_tests),
                "fail_to_pass": sorted(fail_to_pass),
                "pass_to_pass": sorted(pass_to_pass),
                "output": parsed_output,
            }
        }
        combined_output = stdout
        if stderr:
            combined_output = f"{stdout}\n\n[stderr]\n{stderr}".strip()
        return SameContainerEvalResult(
            resolved=resolved,
            test_output=combined_output,
            raw_report=raw_report,
        )

    def stop(self) -> None:
        container_id = self.container_id
        cleanup_error: RuntimeError | None = None
        if container_id:
            stop_timed_out = False
            cleanup_failures: list[str] = []
            remove_failed = False
            try:
                stop_result = subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                    text=True,
                    timeout=_DOCKER_STOP_TIMEOUT_SECONDS,
                )
                if stop_result.returncode != 0:
                    logger.debug(
                        "docker stop failed for %s: %s",
                        container_id,
                        stop_result.stderr.strip(),
                    )
            except subprocess.TimeoutExpired:
                stop_timed_out = True
                logger.warning(
                    "docker stop timed out for %s after %ss; falling back to docker kill",
                    container_id,
                    _DOCKER_STOP_TIMEOUT_SECONDS,
                )
                try:
                    kill_result = subprocess.run(
                        ["docker", "kill", container_id],
                        capture_output=True,
                        text=True,
                        timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
                    )
                    if kill_result.returncode != 0:
                        kill_error = (
                            kill_result.stderr.strip()
                            or "docker kill returned non-zero exit status"
                        )
                        cleanup_failures.append(f"docker kill: {kill_error}")
                        logger.debug(
                            "docker kill failed for %s: %s",
                            container_id,
                            kill_error,
                        )
                except Exception as exc:
                    cleanup_failures.append(f"docker kill raised {exc!r}")
                    logger.debug("docker kill failed for %s", container_id, exc_info=True)
            except Exception:
                logger.debug("docker stop failed for %s", container_id, exc_info=True)
            try:
                remove = subprocess.run(
                    ["docker", "rm", container_id],
                    capture_output=True,
                    text=True,
                    timeout=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
                )
                if remove.returncode != 0:
                    remove_error = (
                        remove.stderr.strip() or "docker rm returned non-zero exit status"
                    )
                    if stop_timed_out:
                        remove_failed = True
                        cleanup_failures.append(f"docker rm: {remove_error}")
                    logger.debug(
                        "docker rm skipped for %s: %s",
                        container_id,
                        remove_error,
                    )
            except Exception as exc:
                if stop_timed_out:
                    remove_failed = True
                    cleanup_failures.append(f"docker rm raised {exc!r}")
                logger.debug("docker rm failed for %s", container_id, exc_info=True)
            if stop_timed_out and remove_failed:
                detail = "; ".join(cleanup_failures)
                logger.error(
                    "Failed to clean up timed-out SWE-bench container %s; container may still be running: %s",
                    container_id,
                    detail,
                )
                cleanup_error = RuntimeError(
                    f"Failed to clean up timed-out SWE-bench container {container_id}: {detail}"
                )
            else:
                self.container_id = None
                self.initial_commit = None
                self._container_obj = None
        else:
            self.container_id = None
            self.initial_commit = None
            self._container_obj = None
        if self._logger is not None:
            close_logger(self._logger)
            self._logger = None
        if cleanup_error is not None:
            raise cleanup_error

    def cleanup_image(self) -> None:
        image_name = self.image_name
        if image_name is None:
            return

        if remove_swebench_instance_image(image_name):
            self._image_name = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.stop()
