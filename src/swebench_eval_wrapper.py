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
    if not use_cli:
        return

    from swebench.harness import docker_build as db

    db.build_image = _build_image_via_cli


def main() -> None:
    _maybe_patch_swebench_builder()
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
