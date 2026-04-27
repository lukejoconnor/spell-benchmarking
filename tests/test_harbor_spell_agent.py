from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from harbor_spell_agent import HarborSpellAgent
from src.docker_agents import (
    HARBOR_SPELL_BUNDLE_FORBIDDEN_PARTS,
    HARBOR_SPELL_BUNDLE_PATHS,
    spell_build_harbor_bundle,
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


if __name__ == "__main__":
    unittest.main()
