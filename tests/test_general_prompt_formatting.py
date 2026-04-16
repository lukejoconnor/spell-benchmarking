from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))


def _import_bench_with_stubbed_client():
    existing = sys.modules.get("bench")
    if existing is not None:
        return existing

    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise NotImplementedError

    stub.SpellBenchmarkClient = _DummyClient
    prior_client = sys.modules.get("spell_benchmark_client")
    sys.modules["spell_benchmark_client"] = stub
    try:
        return importlib.import_module("bench")
    finally:
        if prior_client is None:
            sys.modules.pop("spell_benchmark_client", None)
        else:
            sys.modules["spell_benchmark_client"] = prior_client
        sys.modules.pop("src.unified_adapters", None)


bench = _import_bench_with_stubbed_client()
from src.unified_adapters import build_spell_init
from src.datasets import (
    BenchmarkItem,
    LONGBENCH_SEARCH_GUIDANCE,
    format_benchmark_prompt,
)


def teardown_module(_module) -> None:
    sys.modules.pop("bench", None)
    sys.modules.pop("src.unified_adapters", None)
    stubbed_client = sys.modules.get("spell_benchmark_client")
    if stubbed_client is not None and getattr(stubbed_client, "__file__", None) is None:
        sys.modules.pop("spell_benchmark_client", None)


class GeneralPromptFormattingTest(unittest.TestCase):
    def test_math_trailing_is_describe_math(self) -> None:
        self.assertEqual(bench.MATH_TRAILING, "'(!describe math)")

    def test_math_prompt_is_shared_between_general_conditions(self) -> None:
        item = BenchmarkItem(
            id="math_1",
            dataset="math",
            context="",
            question="Compute 2+2.",
            answer="4",
        )

        prompt = format_benchmark_prompt(item)

        self.assertNotIn("uv run python3", prompt)
        self.assertIn("Problem:", prompt)

    def test_omni_math_prompt_matches_standard_math_format(self) -> None:
        item = BenchmarkItem(
            id="omni_math_1",
            dataset="omni_math",
            context="",
            question="Compute 2+2.",
            answer="4",
        )

        prompt = format_benchmark_prompt(item)

        self.assertIn("Return the answer with no explanation", prompt)

    def test_longbench_prompt_includes_search_guidance_for_context_files(self) -> None:
        item = BenchmarkItem(
            id="lb_1",
            dataset="longbench_short",
            context="Large document body",
            question="Find the answer.",
            answer="A",
        )

        prompt = format_benchmark_prompt(item, context_file="/tmp/context.txt")

        self.assertIn("Document path: /tmp/context.txt", prompt)
        self.assertIn(LONGBENCH_SEARCH_GUIDANCE, prompt)
        self.assertNotIn("!peek", prompt)
        self.assertNotIn("io/read-lines", prompt)
        self.assertNotIn("(think", prompt)
        self.assertNotIn("persist", prompt)

    def test_general_spell_trailing_applies_to_all_math_datasets(self) -> None:
        for ds in ("gsm8k", "math", "aime", "hmmt", "omni_math", "project_euler", "mathconstruct"):
            self.assertEqual(bench._general_spell_trailing(ds), bench.MATH_TRAILING, ds)
        self.assertIsNone(bench._general_spell_trailing("longbench_short"))

    def test_build_spell_init_accepts_setup_via_trailing(self) -> None:
        init = build_spell_init(
            "Repository: demo/repo",
            trailing="'(do (quine sys-reminder \"Recall: you are writing Spell code.\") (!extend))",
        )

        self.assertIn('(quine prompt "Repository: demo/repo")', init)
        self.assertIn('(quine sys-reminder "Recall: you are writing Spell code.")', init)
        self.assertIn("'(do (quine sys-reminder", init)

    def test_gaia_prompt_mentions_web_search(self) -> None:
        item = BenchmarkItem(
            id="gaia_1",
            dataset="gaia",
            context="",
            question="Who won?",
            answer="A",
        )

        prompt = format_benchmark_prompt(item)

        self.assertIn("Use web search, Python, and any tools needed", prompt)


if __name__ == "__main__":
    unittest.main()
