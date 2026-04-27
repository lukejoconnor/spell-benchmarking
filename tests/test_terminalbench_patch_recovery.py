from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.terminalbench_patch_recovery import (
    TEST_PATCH_COLLISION_RECOVERY_MARKER,
    build_patch_wrapper_script,
    parse_git_status_touched_test_files,
    scan_terminalbench_patch_collision_artifacts,
)


class TerminalBenchPatchRecoveryTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "app.py").write_text("baseline source\n", encoding="utf-8")
        (root / "tests" / "test_existing.py").write_text("baseline test\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True, capture_output=True)

    def _write_fake_patch(self, root: Path) -> Path:
        fake_patch = root / "fake-patch.sh"
        fake_patch.write_text(
            textwrap.dedent(
                """\
                #!/bin/sh
                while [ "${1:-}" = "--batch" ]; do
                  shift
                done
                payload="$(cat)"
                if printf '%s' "$payload" | grep -q 'tests/test_created.py'; then
                  if [ -e tests/test_created.py ]; then
                    echo 'File tests/test_created.py, which already exists!' >&2
                    exit 1
                  fi
                  printf 'official created test\\n' > tests/test_created.py
                  exit 0
                fi
                if printf '%s' "$payload" | grep -q 'tests/test_existing.py'; then
                  if grep -q 'agent changed test' tests/test_existing.py; then
                    echo 'Reversed (or previously applied) patch detected!' >&2
                    exit 1
                  fi
                  printf 'official existing test\\n' > tests/test_existing.py
                  exit 0
                fi
                echo 'Reversed (or previously applied) patch detected!' >&2
                exit 1
                """
            ),
            encoding="utf-8",
        )
        fake_patch.chmod(0o755)
        return fake_patch

    def _write_wrapper(self, root: Path, fake_patch: Path, touched_files: list[str]) -> Path:
        touched_path = root / "touched.txt"
        touched_path.write_text("\n".join(touched_files) + "\n", encoding="utf-8")
        wrapper = root / "patch-wrapper.sh"
        wrapper.write_text(
            build_patch_wrapper_script(str(fake_patch), touched_files_path=str(touched_path)),
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        return wrapper

    def test_parse_git_status_touched_test_files(self) -> None:
        status = "\n".join(
            [
                " M src/app.py",
                "?? tests/test_created.py",
                " M package/module_test.py",
                "R  tests/old.py -> tests/new.py",
            ]
        )

        self.assertEqual(
            parse_git_status_touched_test_files(status),
            ["tests/test_created.py", "package/module_test.py", "tests/new.py"],
        )

    def test_wrapper_removes_untracked_test_file_and_retries_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._init_repo(root)
            fake_patch = self._write_fake_patch(root)
            (root / "tests" / "test_created.py").write_text("agent test\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("model source change\n", encoding="utf-8")
            wrapper = self._write_wrapper(root, fake_patch, ["tests/test_created.py"])

            result = subprocess.run(
                [str(wrapper)],
                cwd=root,
                input="+++ b/tests/test_created.py\n",
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "tests" / "test_created.py").read_text(encoding="utf-8"), "official created test\n")
            self.assertEqual((root / "src" / "app.py").read_text(encoding="utf-8"), "model source change\n")
            self.assertIn(TEST_PATCH_COLLISION_RECOVERY_MARKER, result.stderr)
            self.assertIn("path=tests/test_created.py", result.stderr)

    def test_wrapper_restores_tracked_test_file_without_resetting_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._init_repo(root)
            fake_patch = self._write_fake_patch(root)
            (root / "tests" / "test_existing.py").write_text("agent changed test\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("model source change\n", encoding="utf-8")
            wrapper = self._write_wrapper(root, fake_patch, ["tests/test_existing.py"])

            result = subprocess.run(
                [str(wrapper)],
                cwd=root,
                input="diff --git a/tests/test_existing.py b/tests/test_existing.py\n+++ b/tests/test_existing.py\n",
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "tests" / "test_existing.py").read_text(encoding="utf-8"), "official existing test\n")
            self.assertEqual((root / "src" / "app.py").read_text(encoding="utf-8"), "model source change\n")
            self.assertIn("path=tests/test_existing.py", result.stderr)

    def test_wrapper_does_not_retry_non_test_patch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self._init_repo(root)
            fake_patch = self._write_fake_patch(root)
            (root / "src" / "app.py").write_text("model source change\n", encoding="utf-8")
            wrapper = self._write_wrapper(root, fake_patch, ["tests/test_existing.py"])

            result = subprocess.run(
                [str(wrapper)],
                cwd=root,
                input="diff --git a/src/app.py b/src/app.py\n+++ b/src/app.py\n",
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(TEST_PATCH_COLLISION_RECOVERY_MARKER, result.stderr)
            self.assertEqual((root / "src" / "app.py").read_text(encoding="utf-8"), "model source change\n")

    def test_scanner_detects_collision_timeout_but_not_near_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "terminalbench_20260427_000232_eb650234"
            affected_trial = run_dir / "swe-bench-langcodes" / "swe-bench-langcodes.1-of-1.codex"
            near_miss_trial = run_dir / "near-miss" / "near-miss.1-of-1.codex"
            (affected_trial / "sessions").mkdir(parents=True)
            (near_miss_trial / "sessions").mkdir(parents=True)
            (affected_trial / "sessions" / "tests.log").write_text(
                "File langcodes/tests/test_language.py, which already exists!\nTimed out after 600s\n",
                encoding="utf-8",
            )
            (near_miss_trial / "sessions" / "tests.log").write_text(
                "patching file tests/test_ok.py\nTimed out after 600s while running tests\n",
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "swe-bench-langcodes",
                                "trial_name": "swe-bench-langcodes.1-of-1.codex",
                                "failure_mode": "test_timeout",
                                "is_resolved": False,
                            },
                            {
                                "task_id": "near-miss",
                                "trial_name": "near-miss.1-of-1.codex",
                                "failure_mode": "test_timeout",
                                "is_resolved": False,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            records = scan_terminalbench_patch_collision_artifacts(root)

            self.assertEqual([record.task_id for record in records], ["swe-bench-langcodes"])
            self.assertEqual(records[0].status, "test_patch_collision_unrechecked")


if __name__ == "__main__":
    unittest.main()
