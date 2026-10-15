from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.docker_agents import ExecOutput
from src.featurebench_lib import _ensure_uv_in_container, install_deps_in_container


class FakeContainer:
    def __init__(self, outputs: list[ExecOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def exec_run(self, cmd: str, **kwargs) -> ExecOutput:
        self.calls.append((cmd, dict(kwargs)))
        if not self._outputs:
            raise AssertionError(f"Unexpected exec call: {cmd}")
        return self._outputs.pop(0)


class FeatureBenchLibTests(unittest.TestCase):
    @patch("src.featurebench_lib.ensure_python_packages", return_value=True)
    def test_ensure_uv_in_container_installs_uv_when_missing(self, ensure_packages) -> None:
        container = FakeContainer([ExecOutput(exit_code=1, stdout="", stderr="missing")])

        _ensure_uv_in_container(container, workdir="/workspace")

        self.assertEqual(
            container.calls,
            [("command -v uv >/dev/null 2>&1", {"timeout": 30, "workdir": "/workspace"})],
        )
        ensure_packages.assert_called_once_with(container, ["uv"])

    @patch("src.featurebench_lib.run_install_script", return_value=True)
    @patch("src.featurebench_lib.ensure_python_packages", return_value=True)
    def test_install_deps_in_container_uses_retrying_install_script(self, ensure_packages, run_install) -> None:
        container = FakeContainer([ExecOutput(exit_code=1, stdout="", stderr="missing")])
        item = {"repo_settings": '{"install": "python -m pip install -r requirements.txt"}'}

        install_deps_in_container(container, item, workdir="/repo")

        ensure_packages.assert_called_once_with(container, ["uv"])
        run_install.assert_called_once_with(
            container,
            "python -m pip install -r requirements.txt",
            timeout=600,
            workdir="/repo",
        )

    @patch("src.featurebench_lib.ensure_python_packages", return_value=False)
    def test_ensure_uv_in_container_raises_clear_error_when_install_fails(self, ensure_packages) -> None:
        container = FakeContainer([ExecOutput(exit_code=1, stdout="", stderr="missing")])

        with self.assertRaisesRegex(RuntimeError, "Failed to install uv in container"):
            _ensure_uv_in_container(container, workdir="/workspace")

        ensure_packages.assert_called_once_with(container, ["uv"])


if __name__ == "__main__":
    unittest.main()
