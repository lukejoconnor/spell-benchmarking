"""Singularity-based container backend for SWE-bench.

Drop-in replacement for SWEBenchContainer for HPC environments (e.g. O2)
where Docker is unavailable but Singularity/Apptainer is installed.

Architecture:
- SIF images are stored in SINGULARITY_SIF_CACHE (often a restricted path
  managed by HPC admins, e.g. /n/app/containers/users/<user>/)
- Instead of writable sandboxes (which require write access to allowed
  container paths), we use **bind mounts**: copy /testbed, /etc, /tmp,
  and /usr/lib/locale out of the SIF to a temp directory, then bind-mount
  them back on every exec_run call.
- File operations use direct filesystem access on the bind-mount dirs.

Environment variables:
  SINGULARITY_SIF_CACHE   Directory for cached .sif files (default: ~/.singularity/swebench-cache)
  SINGULARITY_WORK_ROOT   Directory for per-task bind-mount workdirs (default: /tmp/swebench-work)
  SINGULARITY_BIN         Binary name: "singularity" or "apptainer" (auto-detected if unset)
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
    for candidate in ("apptainer", "singularity"):
        if shutil.which(candidate):
            return candidate
    # O2/HPC fallback: check common installation paths not in PATH
    for full_path in ("/usr/bin/apptainer", "/usr/local/bin/apptainer", "/opt/singularity/bin/singularity", "/usr/local/bin/singularity"):
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return "apptainer"


def _sif_cache_dir() -> Path:
    env = os.environ.get("SINGULARITY_SIF_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".singularity" / "swebench-cache"


def _work_root() -> Path:
    env = os.environ.get("SINGULARITY_WORK_ROOT")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "swebench-work"


class SingularitySWEBenchContainer:
    """Singularity-backed container for SWE-bench standard instances.

    Uses bind mounts instead of writable sandboxes to work around HPC
    environments where users cannot write to apptainer's allowed container
    paths. Copies /testbed, /etc, /usr/lib/locale, and /tmp out of the
    read-only SIF, then bind-mounts them back on every exec call.

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
        self.container_id: str | None = None  # compatibility — set to work_dir str
        self.work_dir: Path | None = None
        self.sif_path: Path | None = None
        self.initial_commit: str | None = None
        self._test_spec = None
        self._sing = _singularity_bin()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> str:
        self._test_spec = make_test_spec(
            self.item.raw_instance,
            namespace="swebench",
            env_image_tag=self.env_image_tag,
            instance_image_tag=self.instance_image_tag,
            arch="x86_64",
        )

        image_key = self._test_spec.instance_image_key
        logger.info("SWE-bench image: %s", image_key)

        self.sif_path = self._find_sif(image_key)

        work_name = f"{self.item.instance_id}_{uuid.uuid4().hex[:8]}"
        work_root = _work_root()
        work_root.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_root / work_name

        self._setup_bind_dirs()

        head_result = self.exec_run("git rev-parse HEAD", timeout=30)
        if head_result.exit_code != 0:
            raise RuntimeError(
                f"git rev-parse HEAD failed: {head_result.stderr.strip()}"
            )
        self.initial_commit = head_result.stdout.strip() or None
        self.container_id = str(self.work_dir)
        return self.container_id

    def _find_sif(self, image_key: str) -> Path:
        """Find a cached SIF file, or pull from Docker Hub if possible."""
        cache_dir = _sif_cache_dir()

        # Try matching by image_key-derived name
        safe_name = image_key.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = cache_dir / safe_name
        if sif_path.exists():
            logger.info("Using cached SIF: %s", sif_path)
            return sif_path

        # Also try matching by image_key substring in cache dir
        # (RC-built SIFs may omit the namespace prefix)
        if cache_dir.exists():
            # Extract the core image name (without namespace and tag)
            # e.g. "swebench/sweb.eval.x86_64.django_1776_django-11039:latest"
            #   -> "sweb.eval.x86_64.django_1776_django-11039"
            core = image_key.split("/")[-1].split(":")[0]
            for candidate in cache_dir.iterdir():
                if candidate.suffix == ".sif" and core in candidate.stem:
                    logger.info("Using SIF (pattern match): %s", candidate)
                    return candidate

        # Try to pull (may fail on HPC if cache_dir is not writable)
        if cache_dir.exists() and os.access(str(cache_dir), os.W_OK):
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

        raise RuntimeError(
            f"SIF not found for {image_key} in {cache_dir}. "
            f"On HPC, SIF files must be placed by admins in the allowed container path."
        )

    def _setup_bind_dirs(self) -> None:
        """Copy writable directories out of the SIF for bind mounting."""
        assert self.work_dir is not None and self.sif_path is not None

        dirs = {
            "testbed": "/testbed",
            "etc": "/etc",
            "usrlib_locale": "/usr/lib/locale",
            "tmp": None,  # just create empty
        }
        for local_name, container_path in dirs.items():
            local_dir = self.work_dir / local_name
            local_dir.mkdir(parents=True, exist_ok=True)
            if container_path:
                logger.info("Copying %s from SIF...", container_path)
                result = subprocess.run(
                    [self._sing, "exec", self.sif_path,
                     "cp", "-a", f"{container_path}/.", str(local_dir) + "/"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode != 0:
                    # Non-fatal for /etc (permission errors on shadow files are OK)
                    logger.warning(
                        "cp %s had errors (may be OK): %s",
                        container_path, result.stderr.strip()[:200],
                    )

        # Also ensure a writable /home dir for git config
        (self.work_dir / "home").mkdir(parents=True, exist_ok=True)

    def _bind_args(self) -> list[str]:
        """Return --bind arguments for exec_run."""
        assert self.work_dir is not None
        return [
            "--bind", f"{self.work_dir / 'testbed'}:/testbed",
            "--bind", f"{self.work_dir / 'tmp'}:/tmp",
            "--bind", f"{self.work_dir / 'etc'}:/etc",
            "--bind", f"{self.work_dir / 'usrlib_locale'}:/usr/lib/locale",
            "--bind", f"{self.work_dir / 'home'}:{Path.home()}",
        ]

    def stop(self) -> None:
        if self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir = None
        self.sif_path = None
        self.container_id = None
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
        if not self.work_dir or not self.sif_path:
            raise RuntimeError("Container not started")

        sing_cmd = [self._sing, "exec", "--no-home"]
        sing_cmd.extend(self._bind_args())

        effective_workdir = workdir or self.container_workdir
        sing_cmd.extend(["--pwd", effective_workdir])

        for key, value in (env or {}).items():
            sing_cmd.extend(["--env", f"{key}={value}"])

        sing_cmd.extend([str(self.sif_path), "bash", "-lc", cmd])

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
    # File I/O — direct filesystem access on bind-mount dirs
    # ------------------------------------------------------------------

    def _host_path(self, container_path: str) -> Path:
        """Map a container path to the corresponding host bind-mount path."""
        assert self.work_dir is not None
        if container_path.startswith("/testbed"):
            return self.work_dir / "testbed" / container_path[len("/testbed/"):]
        if container_path.startswith("/tmp"):
            return self.work_dir / "tmp" / container_path[len("/tmp/"):]
        if container_path.startswith("/etc"):
            return self.work_dir / "etc" / container_path[len("/etc/"):]
        # Fallback: use /tmp bind
        return self.work_dir / "tmp" / Path(container_path).name

    def put_tar_archive(self, dest_dir: str, archive_bytes: bytes) -> bool:
        if not self.work_dir:
            raise RuntimeError("Container not started")
        host_dest = self._host_path(dest_dir)
        host_dest.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp.write(archive_bytes)
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                ["tar", "-xf", str(tmp_path), "-C", str(host_dest)],
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
        if not self.work_dir:
            raise RuntimeError("Container not started")
        host_source = self._host_path(source)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if host_source.is_dir():
            shutil.copytree(host_source, dest, dirs_exist_ok=True)
            return True
        if host_source.is_file():
            shutil.copy2(host_source, dest)
            return True
        logger.warning("Source not found: %s (host: %s)", source, host_source)
        return False

    def read_text(self, path: str) -> str:
        if not self.work_dir:
            return ""
        host_path = self._host_path(path)
        if host_path.exists():
            return host_path.read_text(encoding="utf-8", errors="replace")
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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.stop()
