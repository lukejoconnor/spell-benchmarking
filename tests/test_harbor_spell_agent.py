from __future__ import annotations

import asyncio
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from harbor_spell_agent import AgentContext, HarborSpellAgent
from src.docker_agents import (
    HARBOR_SPELL_BUNDLE_FORBIDDEN_PARTS,
    HARBOR_SPELL_BUNDLE_PATHS,
    spell_build_harbor_bundle,
)


class _ExecResult:
    def __init__(self, stdout: str = "", return_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.return_code = return_code


class _FakeEnvironment:
    async def exec(self, *args, **kwargs):
        return _ExecResult(
            stdout=json.dumps(
                {
                    "ok": True,
                    "result": "done",
                    "usage": {"total": {"input_tokens": 100, "output_tokens": 10, "cost": 0.456}},
                }
            ),
            return_code=0,
        )


class HarborSpellAgentTests(unittest.TestCase):
    def test_harbor_agent_imports_without_harbor_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = HarborSpellAgent(
                logs_dir=Path(tmp),
                model_name="openai-tc:gpt-5.4",
                reasoning_effort="medium",
            )

        self.assertEqual(agent.name(), "spell")
        self.assertEqual(agent.model_name, "openai-tc:gpt-5.4")
        self.assertEqual(agent.reasoning_effort, "medium")
        self.assertEqual(agent.agent_file, "/opt/spell/config/agents/io-tc.agent.edn")

    def test_harbor_bundle_contains_only_safe_spell_runtime_paths(self) -> None:
        bundle = spell_build_harbor_bundle()
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r") as tar:
            names = tar.getnames()

        self.assertIn("spell/deps.edn", names)
        self.assertIn("spell/src/spell/benchmark_api.clj", names)
        self.assertIn("spell/config/agents/io-tc.agent.edn", names)
        self.assertIn("spell/data/pricing.edn", names)
        for name in names:
            parts = set(Path(name).parts)
            self.assertFalse(
                parts & HARBOR_SPELL_BUNDLE_FORBIDDEN_PARTS,
                f"Forbidden member leaked into Harbor Spell bundle: {name}",
            )

    def test_harbor_bundle_path_allowlist_is_minimal(self) -> None:
        self.assertEqual(HARBOR_SPELL_BUNDLE_PATHS, ("deps.edn", "src", "config", "data/pricing.edn"))

    def test_run_leaves_context_empty_for_harbor_post_run_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = AgentContext()
            agent = HarborSpellAgent(logs_dir=Path(tmp), model_name="openai-tc:gpt-5.4")

            asyncio.run(agent.run("Solve it", _FakeEnvironment(), context))

        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.n_cache_tokens)
        self.assertIsNone(context.n_output_tokens)
        self.assertIsNone(context.cost_usd)
        self.assertIsNone(context.metadata)

    def test_post_run_accounting_comes_from_usage_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "spell-usage.json").write_text(
                json.dumps(
                    {
                        "schema": "spell.usage-snapshot.v1",
                        "usage": {
                            "total": {
                                "uncached_input_tokens": 100,
                                "cached_input_tokens": 25,
                                "cache_write_input_tokens": 5,
                                "visible_output_tokens": 10,
                                "reasoning_output_tokens": 2,
                                "cost": 0.123,
                            }
                        },
                        "records": [{"model": "gpt-5.4", "cost": 0.123}],
                    }
                ),
                encoding="utf-8",
            )
            context = AgentContext()
            agent = HarborSpellAgent(logs_dir=logs_dir, model_name="openai-tc:gpt-5.4")

            agent.populate_context_post_run(context)

        self.assertEqual(context.n_input_tokens, 130)
        self.assertEqual(context.n_cache_tokens, 25)
        self.assertEqual(context.n_output_tokens, 12)
        self.assertAlmostEqual(context.cost_usd or 0.0, 0.123)
        self.assertEqual(context.metadata["spell_accounting_source"], "usage_log")
        self.assertTrue(context.metadata["spell_usage_log_present"])
        self.assertEqual(context.metadata["spell_usage_log_records"], 1)

    def test_final_output_usage_is_not_accounting_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "spell-output.txt").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "done",
                        "usage": {
                            "total": {
                                "uncached_input_tokens": 100,
                                "visible_output_tokens": 10,
                                "cost": 0.456,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            context = AgentContext()
            agent = HarborSpellAgent(logs_dir=logs_dir, model_name="openai-tc:gpt-5.4")

            agent.populate_context_post_run(context)

        self.assertIsNone(context.n_input_tokens)
        self.assertIsNone(context.n_cache_tokens)
        self.assertIsNone(context.n_output_tokens)
        self.assertIsNone(context.cost_usd)
        self.assertFalse(context.metadata["spell_usage_log_present"])
        self.assertTrue(context.metadata["spell_ok"])
        self.assertEqual(context.metadata["spell_result_preview"], "done")
        self.assertIn("spell_usage", context.metadata)

    def test_final_metadata_does_not_override_usage_log_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "spell-usage.json").write_text(
                json.dumps(
                    {
                        "usage": {
                            "total": {
                                "uncached_input_tokens": 200,
                                "visible_output_tokens": 20,
                                "cost": 0.222,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (logs_dir / "spell-output.txt").write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "boom",
                        "usage": {
                            "total": {
                                "uncached_input_tokens": 999,
                                "visible_output_tokens": 999,
                                "cost": 9.99,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            context = AgentContext()
            agent = HarborSpellAgent(logs_dir=logs_dir, model_name="openai-tc:gpt-5.4")

            agent.populate_context_post_run(context)

        self.assertEqual(context.n_input_tokens, 200)
        self.assertEqual(context.n_output_tokens, 20)
        self.assertAlmostEqual(context.cost_usd or 0.0, 0.222)
        self.assertEqual(context.metadata["spell_accounting_source"], "usage_log")
        self.assertFalse(context.metadata["spell_ok"])
        self.assertEqual(context.metadata["spell_error"], "boom")


if __name__ == "__main__":
    unittest.main()
