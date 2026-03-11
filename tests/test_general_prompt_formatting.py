from __future__ import annotations

import sys
import types

stub = types.ModuleType("spell_benchmark_client")
stub.SpellBenchmarkClient = object
sys.modules.setdefault("spell_benchmark_client", stub)

from src.unified_adapters import build_spell_init
from src.datasets import BenchmarkItem, format_benchmark_prompt


def test_omni_math_prompt_requires_final_answer_format() -> None:
    item = BenchmarkItem(
        id="omni_math_1",
        dataset="omni_math",
        context="",
        question="Compute 2+2.",
        answer="4",
    )

    prompt = format_benchmark_prompt(item)

    assert "Reply with ONLY the final answer on a single line" in prompt
    assert "\\boxed{2\\sqrt{3}}" in prompt


def test_long_context_prompt_adds_shared_search_advice() -> None:
    item = BenchmarkItem(
        id="longbench_1",
        dataset="longbench_short",
        context="alpha beta gamma",
        question="Which token appears second?",
        answer="beta",
        choices=["alpha", "beta", "gamma", "delta"],
    )

    prompt = format_benchmark_prompt(
        item,
        context_ref="Document path: /tmp/context.txt\nDocument size: 16 chars",
    )

    assert "The document may be very large." in prompt
    assert "Reply with ONLY the letter (A, B, C, or D)." in prompt


def test_gaia_prompt_toggles_web_search_wording() -> None:
    item = BenchmarkItem(
        id="gaia_1",
        dataset="gaia",
        context="",
        question="Who won?",
        answer="A",
        metadata={},
    )

    prompt_with_web = format_benchmark_prompt(item, web_search=True)
    prompt_without_web = format_benchmark_prompt(item, web_search=False)

    assert "Use web search, Python, and any tools needed" in prompt_with_web
    assert "non-web tools" not in prompt_with_web
    assert "Use Python and any available non-web tools needed" in prompt_without_web
    assert "Use web search" not in prompt_without_web


def test_build_spell_init_treats_prose_preamble_as_prompt_text() -> None:
    init = build_spell_init(
        "Repository: demo/repo",
        preamble="Use the team pattern immediately and decompose into 2-3 subtasks.",
    )

    assert '(quine prompt "Repository: demo/repo\\n\\nUse the team pattern immediately and decompose into 2-3 subtasks.")' in init
    assert init.count("Use the team pattern immediately") == 1
    assert "'(!extend)" in init


def test_build_spell_init_keeps_raw_spell_preamble_as_code() -> None:
    init = build_spell_init(
        "Repository: demo/repo",
        preamble='(quine sys-reminder "Recall: you are writing Spell code.")',
    )

    assert '(quine prompt "Repository: demo/repo")' in init
    assert '(quine sys-reminder "Recall: you are writing Spell code.")' in init
