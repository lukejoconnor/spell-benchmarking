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

    def test_run_preserves_harbor_prompt_template_decorator(self) -> None:
        source = Path(BENCHMARK_ROOT / "harbor_codex_agent.py").read_text(encoding="utf-8")

        self.assertIn("@with_prompt_template", source)

    def test_auth_json_is_not_uploaded_to_agent_logs_dir(self) -> None:
        source = Path(BENCHMARK_ROOT / "harbor_codex_agent.py").read_text(encoding="utf-8")

        self.assertIn('auth_target = "/tmp/codex-secrets/auth.json"', source)
        self.assertNotIn('EnvironmentPaths.agent_dir / "auth.json"', source)


if __name__ == "__main__":
    unittest.main()
