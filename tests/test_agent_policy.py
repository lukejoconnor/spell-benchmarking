from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_policy import (
    DEFAULT_AGENT_POLICY,
    classify_general_family,
    detect_spell_mode,
    resolve_spell_agent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AgentPolicyTests(unittest.TestCase):
    def test_detect_spell_mode(self) -> None:
        self.assertEqual(detect_spell_mode("codex-msg:gpt-5.3-codex"), "msg")
        self.assertEqual(detect_spell_mode("codex-tc:gpt-5.3-codex"), "tc")
        self.assertEqual(detect_spell_mode("anthropic-tc:claude-sonnet-4-5-20250929"), "tc")
        self.assertEqual(detect_spell_mode("anthropic-pf:claude-sonnet-4-5-20250929"), "msg")
        self.assertEqual(detect_spell_mode("gpt-5.3-codex"), "msg")

    def test_classify_general_family(self) -> None:
        self.assertEqual(classify_general_family({"aime", "hmmt"}), "general_math")
        self.assertEqual(classify_general_family({"project_euler"}), "general_math")
        self.assertEqual(classify_general_family({"longbench_short", "babilong_32k"}), "general_longcontext")
        self.assertEqual(classify_general_family({"longbench_short", "aime"}), "general_other")
        self.assertEqual(classify_general_family(set()), "general_other")

    def test_default_policy_paths_exist(self) -> None:
        for rel_path in DEFAULT_AGENT_POLICY.values():
            path = PROJECT_ROOT / rel_path
            self.assertTrue(path.exists(), f"Missing default policy file: {path}")

    def test_resolve_default_general_longcontext_tc(self) -> None:
        resolved = resolve_spell_agent(
            project_root=PROJECT_ROOT,
            bench_kind="general",
            model_spec="codex-tc:gpt-5.3-codex",
            item_datasets={"longbench_short"},
        )
        self.assertEqual(resolved.spell_mode, "tc")
        self.assertEqual(resolved.family, "general_longcontext")
        self.assertEqual(resolved.source, "default")
        self.assertEqual(resolved.policy_key, "tc:tc")
        self.assertTrue(resolved.agent_file.endswith("config/agents/io-tc.agent.edn"))

    def test_override_precedence(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".agent.edn", delete=False) as tf:
            tf.write(b"{:name temp}")
            temp_agent = Path(tf.name)

        try:
            resolved = resolve_spell_agent(
                project_root=PROJECT_ROOT,
                bench_kind="swebench",
                model_spec="codex-tc:gpt-5.3-codex",
                agent="config/agents/io-msg.agent.edn",
                agent_tc=str(temp_agent),
            )
            self.assertEqual(resolved.source, "agent_tc")
            self.assertEqual(resolved.agent_file, str(temp_agent.resolve()))
        finally:
            temp_agent.unlink(missing_ok=True)

    def test_missing_override_fails_fast(self) -> None:
        missing = PROJECT_ROOT / "config/agents/no-such.agent.edn"
        with self.assertRaises(FileNotFoundError):
            resolve_spell_agent(
                project_root=PROJECT_ROOT,
                bench_kind="featurebench",
                model_spec="codex-msg:gpt-5.3-codex",
                agent_msg=str(missing),
            )


if __name__ == "__main__":
    unittest.main()
