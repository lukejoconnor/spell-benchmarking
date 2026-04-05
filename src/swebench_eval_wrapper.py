"""Local wrapper for `swebench.harness.run_evaluation`.

This wrapper exists to patch image builds on macOS hosts where the Docker SDK
build path can ignore requested platform/arch and produce native arm64 images
for x86_64 specs.
"""

from __future__ import annotations

import os
import platform
import runpy
import subprocess
from pathlib import Path


def _build_image_via_cli(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client,  # kept for signature parity with swebench.harness.docker_build.build_image
    build_dir: Path,
    nocache: bool = False,
):
    del client

    from swebench.harness import docker_build as db

    logger = db.setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )
    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")

    try:
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w", encoding="utf-8") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile)

        cmd = [
            "docker",
            "build",
            "--platform",
            platform,
            "--tag",
            image_name,
            "--progress",
            "plain",
            "--rm",
            "--force-rm",
        ]
        if nocache:
            cmd.append("--no-cache")
        cmd.append(str(build_dir))

        env = os.environ.copy()
        env.setdefault("DOCKER_BUILDKIT", "1")

        logger.info(
            f"Building docker image {image_name} in {build_dir} with "
            f"platform {platform} via docker CLI"
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        buildlog = ""
        assert proc.stdout is not None
        for line in proc.stdout:
            clean_line = db.ansi_escape(line.rstrip("\n"))
            logger.info(clean_line)
            buildlog += clean_line + "\n"

        return_code = proc.wait()
        if return_code != 0:
            msg = f"docker build exited with code {return_code}"
            logger.error(msg)
            raise db.BuildImageError(image_name, msg, logger)

        logger.info("Image built successfully!")
    except db.BuildImageError:
        raise
    except Exception as exc:
        logger.error(f"Error building image {image_name}: {exc}")
        raise db.BuildImageError(image_name, str(exc), logger) from exc
    finally:
        db.close_logger(logger)


def _maybe_patch_swebench_builder() -> None:
    mode = os.environ.get("SPELL_SWEBENCH_BUILD_BACKEND", "auto").strip().lower()
    if mode not in {"auto", "cli", "sdk"}:
        mode = "auto"

    use_cli = mode == "cli" or (mode == "auto" and platform.system() == "Darwin")
    if use_cli:
        from swebench.harness import docker_build as db

        db.build_image = _build_image_via_cli

    _patch_pip_version_pin()


_PIP_PIN_APPLIED = False


def _patch_pip_version_pin() -> None:
    """Pin pip<24 in instance setup scripts to avoid PEP 660 and removed-flag errors.

    Pip 24+ removed the --no-use-pep517 flag and the legacy editable-install
    fallback.  The SWE-bench harness generates install commands that rely on
    both, so we inject ``pip install 'pip<24'`` after conda activation in every
    instance setup script.
    """
    global _PIP_PIN_APPLIED
    if _PIP_PIN_APPLIED:
        return
    _PIP_PIN_APPLIED = True

    from swebench.harness.test_spec import python as py_module

    _original_make_repo_script = py_module.make_repo_script_list_py

    def _patched_make_repo_script_list_py(specs, repo, repo_directory, base_commit, env_name):
        commands = _original_make_repo_script(specs, repo, repo_directory, base_commit, env_name)
        # Insert pip downgrade right after 'conda activate ...'
        for i, cmd in enumerate(commands):
            if cmd.startswith("conda activate "):
                commands.insert(i + 1, "python -m pip install 'pip<24' -qq")
                break
        return commands

    py_module.make_repo_script_list_py = _patched_make_repo_script_list_py


def main() -> None:
    _maybe_patch_swebench_builder()
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
