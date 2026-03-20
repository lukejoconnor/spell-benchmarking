"""Singularity-based container backend for SWE-bench.

Drop-in replacement for SWEBenchContainer for HPC environments (e.g. O2)
where Docker is unavailable but Singularity/Apptainer is installed.

Architecture:
- Base SIF images are cached in SINGULARITY_SIF_CACHE (env var or default)
- Per-task writable sandboxes are created fresh per instance (then deleted)
- exec_run uses `singularity exec --writable [--fakeroot]`
- File operations use direct filesystem access (sandbox is a plain directory)

Environment variables:
  SINGULARITY_SIF_CACHE   Directory for cached .sif files (default: ~/.singularity/swebench-cache)
  SINGULARITY_SANDBOX_ROOT Directory for per-task sandboxes (default: /tmp/swebench-sandboxes)
  SINGULARITY_BIN         Binary name: "singularity" or "apptainer" (auto-detected if unset)
  SINGULARITY_FAKEROOT    Set to "0" to disable --fakeroot (default: try fakeroot, fall back)
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from swebench.harness.constants import DOCKER_WORKDIR, KEY_INSTANCE_ID, KEY_PREDICTION
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from .docker_agents import ExecOutput
from .datasets import SWEBenchItem
from .swebench_container import SameContainerEvalResult, _archive_bytes_for_files

logger = logging.getLogger(__name__)

PRO_WORKDIR = "/app"


def _singularity_bin() -> str:
    explicit = os.environ.get("SINGULARITY_BIN")
    if explicit:
        return explicit
    for candidate in ("singularity", "apptainer"):
        if shutil.which(candidate):
            return candidate
    return "singularity"


def _sif_cache_dir() -> Path:
    env = os.environ.get("SINGULARITY_SIF_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".singularity" / "swebench-cache"


def _sandbox_root() -> Path:
    env = os.environ.get("SINGULARITY_SANDBOX_ROOT")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "swebench-sandboxes"


def _use_fakeroot() -> bool:
    return os.environ.get("SINGULARITY_FAKEROOT", "1") != "0"


class SingularitySWEBenchContainer:
    """Singularity-backed container for SWE-bench standard instances.

    Pulls the official swebench Docker image and builds a writable sandbox.
    Compatible interface with SWEBenchContainer from swebench_container.py.
    """

    def __init__(
        self,
        item: SWEBenchItem,
        *,
        instance_image_tag: str = "latest",
        env_image_tag: str = "latest",
    ):
        self.item = item
        self.instance_image_tag = instance_image_tag
        self.env_image_tag = env_image_tag
        self.container_workdir = DOCKER_WORKDIR  # "/testbed"
        self.sandbox_dir: Path | None = None
        self.initial_commit: str | None = None
        self._test_spec = None
        self._sing = _singularity_bin()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> str:
        self._test_spec = make_test_spec(
            self.item.raw_instance,
            namespace=None,
            env_image_tag=self.env_image_tag,
            instance_image_tag=self.instance_image_tag,
            arch="x86_64",
        )

        image_key = self._test_spec.instance_image_key  # e.g. swebench/sweb.eval.x86_64.django__django-11211:latest
        logger.info("SWE-bench image: %s", image_key)

        sandbox_name = f"{self.item.instance_id}_{uuid.uuid4().hex[:8]}"
        sandbox_root = _sandbox_root()
        sandbox_root.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir = sandbox_root / sandbox_name

        sif_path = self._ensure_sif(image_key)
        self._build_sandbox(sif_path)

        head_result = self.exec_run("git rev-parse HEAD", timeout=30)
        if head_result.exit_code != 0:
            raise RuntimeError(
                f"git rev-parse HEAD failed in sandbox: {head_result.stderr.strip()}"
            )
        self.initial_commit = head_result.stdout.strip() or None
        return str(self.sandbox_dir)

    def _ensure_sif(self, image_key: str) -> Path:
        """Pull Docker image to a cached .sif file if not already present."""
        cache_dir = _sif_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        safe_name = image_key.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = cache_dir / safe_name

        if sif_path.exists():
            logger.info("Using cached SIF: %s", sif_path)
            return sif_path

        logger.info("Pulling %s -> %s", image_key, sif_path)
        tmp_sif = sif_path.with_suffix(".tmp.sif")
        result = subprocess.run(
            [self._sing, "pull", "--force", str(tmp_sif), f"docker://{image_key}"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            tmp_sif.unlink(missing_ok=True)
            raise RuntimeError(
                f"singularity pull failed for {image_key}:\n{result.stderr.strip()}"
            )
        tmp_sif.rename(sif_path)
        return sif_path

    def _build_sandbox(self, sif_path: Path) -> None:
        logger.info("Building sandbox %s from %s", self.sandbox_dir, sif_path)
        result = subprocess.run(
            [self._sing, "build", "--sandbox", "--force", str(self.sandbox_dir), str(sif_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"singularity build --sandbox failed:\n{result.stderr.strip()}"
            )

    def stop(self) -> None:
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
        self.sandbox_dir = None
        self.initial_commit = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def exec_run(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecOutput:
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not started")

        sing_cmd = [self._sing, "exec", "--writable", "--no-home"]
        if _use_fakeroot():
            sing_cmd.append("--fakeroot")

        effective_workdir = workdir or self.container_workdir
        sing_cmd.extend(["--pwd", effective_workdir])

        for key, value in (env or {}).items():
            sing_cmd.extend(["--env", f"{key}={value}"])

        sing_cmd.extend([str(self.sandbox_dir), "bash", "-lc", cmd])

        try:
            result = subprocess.run(
                sing_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecOutput(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecOutput(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                timed_out=True,
            )

    # ------------------------------------------------------------------
    # File I/O — direct filesystem access (sandbox is a plain directory)
    # ------------------------------------------------------------------

    def put_tar_archive(self, dest_dir: str, archive_bytes: bytes) -> bool:
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not started")
        sandbox_dest = Path(str(self.sandbox_dir) + dest_dir)
        sandbox_dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp.write(archive_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                ["tar", "-xf", str(tmp_path), "-C", str(sandbox_dest)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                logger.error("tar extract failed: %s", result.stderr.strip())
                return False
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    def copy_to_host(self, source: str, dest: Path) -> bool:
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not started")
        sandbox_source = Path(str(self.sandbox_dir) + source.rstrip("/"))
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if sandbox_source.is_dir():
            shutil.copytree(sandbox_source, dest, dirs_exist_ok=True)
            return True
        if sandbox_source.is_file():
            shutil.copy2(sandbox_source, dest)
            return True
        logger.warning("Source not found in sandbox: %s", source)
        return False

    def read_text(self, path: str) -> str:
        if not self.sandbox_dir:
            return ""
        sandbox_path = Path(str(self.sandbox_dir) + path)
        if sandbox_path.exists():
            return sandbox_path.read_text(encoding="utf-8", errors="replace")
        return ""

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def diff(self, base: str = "HEAD") -> str:
        result = self.exec_run(
            f"git add -A && git diff --cached {shlex.quote(base)}; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status",
            workdir=self.container_workdir,
        )
        return result.stdout if result.exit_code == 0 else ""

    def changed_files(self, base: str = "HEAD") -> list[str]:
        result = self.exec_run(
            f"git add -A && git diff --name-only --cached {shlex.quote(base)}; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status",
            workdir=self.container_workdir,
        )
        if result.exit_code != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_eval(self, *, timeout: int) -> SameContainerEvalResult:
        if self._test_spec is None:
            raise RuntimeError("Test spec not initialized — call start() first")

        files = {"eval.sh": self._test_spec.eval_script}
        if not self.put_tar_archive("/tmp", _archive_bytes_for_files(files)):
            raise RuntimeError("Failed to copy eval script into sandbox")

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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.stop()
