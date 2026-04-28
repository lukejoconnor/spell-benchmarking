from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from harbor_codex_agent import HarborCodexAgent


class HarborCodexAgentTests(unittest.TestCase):
    def test_agent_imports_without_harbor_dependency(self) -> None:
        self.assertEqual(HarborCodexAgent.name(), "spell-codex")

    def test_web_search_disabled_flag_is_codex_config(self) -> None:
        agent = HarborCodexAgent.__new__(HarborCodexAgent)

        self.assertEqual(agent._codex_web_disabled_flag(), "-c 'web_search=\"disabled\"'")


if __name__ == "__main__":
    unittest.main()
