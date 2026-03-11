import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

import bench
from src.datasets import BenchmarkItem
from src.unified_adapters import build_spell_init


class SpellInitRegressionTest(unittest.TestCase):
    def test_build_spell_init_defaults_to_extend(self):
        init = build_spell_init("Solve it")

        self.assertIn("(quine prompt \"Solve it\")", init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_accepts_custom_trailing(self):
        init = build_spell_init(
            "Solve it",
            preamble='(quine sys-reminder "Use tools")',
            trailing="'(do (!describe io) (!extend))",
        )

        self.assertIn('(quine sys-reminder "Use tools")', init)
        self.assertIn("'(do (!describe io) (!extend))", init)
        self.assertNotIn("'(!extend))))", init)

    def test_fix_loop_variant_renders_prompt_text_into_trailing_expr(self):
        prompt = "Repository: demo/repo\n\n## Issue to Fix\n\nHandle newline-sensitive input."

        trailing = bench._render_swebench_trailing_expr(
            prompt,
            bench.SWEBENCH_VARIANTS["fix-loop"]["trailing"],
        )

        self.assertEqual(
            trailing,
            '\'(patterns/fix-loop "Repository: demo/repo\\n\\n## Issue to Fix\\n\\nHandle newline-sensitive input.")',
        )


class GeneralLoadingRegressionTest(unittest.TestCase):
    def test_project_euler_items_filter_uses_normalized_ids(self):
        loaded_item = BenchmarkItem(
            id="project_euler_0001",
            dataset="project_euler",
            context="",
            question="Q",
            answer="A",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "out.jsonl"
            with patch("bench.load_project_euler", return_value=[loaded_item]) as load_mock, \
                 patch("bench.click.echo") as echo_mock:
                bench.run_general.callback(
                    dataset="project_euler",
                    condition="one_shot",
                    model=None,
                    claude_model=None,
                    codex_model="codex",
                    n=None,
                    items="project_euler_1",
                    parallel=1,
                    budget=1.0,
                    agent=None,
                    agent_msg=None,
                    agent_tc=None,
                    depth=None,
                    max_tokens=None,
                    timeout=5,
                    thinking=False,
                    thinking_budget=10000,
                    reasoning_effort="high",
                    verbosity=None,
                    responses_api=False,
                    suffix_grammar=False,
                    web_search=False,
                    trace=False,
                    dry_run=True,
                    run_id="test-general",
                    output=str(output_path),
                )

        load_mock.assert_called_once_with(max_items=None, problem_numbers=[1])
        echoed_lines = [call.args[0] for call in echo_mock.call_args_list if call.args]
        self.assertIn("Loaded 1 items", echoed_lines)


if __name__ == "__main__":
    unittest.main()
