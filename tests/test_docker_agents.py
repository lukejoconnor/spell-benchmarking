from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.docker_agents import (
    ExecOutput,
    _load_swe_agent_trajectory,
    build_spell_init,
    codex_parse_usage,
    resolve_spell_model,
    spell_env_vars,
)


class DockerAgentsTests(unittest.TestCase):
    def test_resolve_spell_model_preserves_fireworks_prefix(self) -> None:
        self.assertEqual(resolve_spell_model("fireworks:glm-5"), "fireworks:glm-5")

    def test_build_spell_init_defaults_to_extend(self) -> None:
        init = build_spell_init("Solve it")

        self.assertIn('(quine prompt "Solve it")', init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_normalizes_shell_escaped_bang_in_trailing(self) -> None:
        init = build_spell_init("Solve it", trailing="'(\\!describe io)")

        self.assertIn("'(!describe io)", init)
        self.assertNotIn("\\!describe", init)

    def test_normalize_spell_trailing_preserves_backslash_bang_inside_strings(self) -> None:
        trailing = "'(do (think \"literal \\\\! stays\") (!describe io))"

        normalized = _normalize_spell_trailing(trailing)

        self.assertEqual(normalized, trailing)

    def test_codex_parse_usage_estimates_cost_and_counts_cached_tokens(self) -> None:
        usage = codex_parse_usage(
            '\n'.join(
                [
                    '{"type":"thread.started","thread_id":"t1"}',
                    '{"type":"turn.completed","usage":{"input_tokens":6804,"cached_input_tokens":5888,"output_tokens":6347}}',
                ]
            ),
            model="gpt-5-codex",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 6804)
        self.assertEqual(usage.cached_input_tokens, 5888)
        self.assertEqual(usage.total_input_tokens, 6804)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.065351, places=6)
        self.assertEqual(usage.cost_source, "estimated")
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
