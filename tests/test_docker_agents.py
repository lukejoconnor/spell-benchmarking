from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.docker_agents import (
    resolve_spell_model,
    spell_env_vars,
    swe_agent_env_vars,
    swe_agent_parse_usage,
    swe_agent_run_command,
)


class DockerAgentsTests(unittest.TestCase):
    def test_resolve_spell_model_preserves_fireworks_prefix(self) -> None:
        self.assertEqual(resolve_spell_model("fireworks:glm-5"), "fireworks:glm-5")

    def test_spell_env_vars_forwards_fireworks_api_key(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test"}, clear=False):
            env = spell_env_vars("fireworks:glm-5")
        self.assertEqual(env["FIREWORKS_API_KEY"], "fw-test")

    def test_swe_agent_run_command_sets_output_path_and_exit_flag(self) -> None:
        cmd = swe_agent_run_command(
            "Fix the bug",
            model="anthropic/claude-sonnet-4-6",
            budget=2.5,
            output_path=".mini.traj.json",
        )
        self.assertIn("mini -t 'Fix the bug'", cmd)
        self.assertIn("--exit-immediately", cmd)
        self.assertIn("-o .mini.traj.json", cmd)
        self.assertIn("-l 2.5", cmd)

    def test_swe_agent_env_vars_forward_openrouter_key(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            env = swe_agent_env_vars("openrouter/anthropic/claude-sonnet-4-6")
        self.assertEqual(env["OPENROUTER_API_KEY"], "or-test")

    def test_swe_agent_parse_usage_reads_instance_cost(self) -> None:
        usage = swe_agent_parse_usage({"info": {"model_stats": {"instance_cost": 0.42}}})
        assert usage is not None
        self.assertEqual(usage.cost_usd, 0.42)


if __name__ == "__main__":
    unittest.main()
