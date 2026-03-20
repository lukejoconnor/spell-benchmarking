from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.docker_agents import (
    ExecOutput,
    _load_swe_agent_trajectory,
    resolve_spell_model,
    spell_env_vars,
)


class DockerAgentsTests(unittest.TestCase):
    def test_resolve_spell_model_preserves_fireworks_prefix(self) -> None:
        self.assertEqual(resolve_spell_model("fireworks:glm-5"), "fireworks:glm-5")

    def test_spell_env_vars_forwards_fireworks_api_key(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test"}, clear=False):
            env = spell_env_vars("fireworks:glm-5")
        self.assertEqual(env["FIREWORKS_API_KEY"], "fw-test")

    def test_load_swe_agent_trajectory_prefers_newest_parseable_file(self) -> None:
        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del cmd, kwargs
                return ExecOutput(
                    exit_code=0,
                    stdout=(
                        "100.0\t/workspace/older.traj.json\n"
                        "200.0\t/workspace/newer.traj.json\n"
                    ),
                    stderr="",
                )

            def read_text(self, path: str) -> str:
                return {
                    "/workspace/older.traj.json": '{"answer":"old"}',
                    "/workspace/newer.traj.json": '{"answer":"new"}',
                }[path]

        path, traj = _load_swe_agent_trajectory(FakeContainer(), "")

        self.assertEqual(path, Path("/workspace/newer.traj.json"))
        self.assertEqual(traj, {"answer": "new"})


if __name__ == "__main__":
    unittest.main()
