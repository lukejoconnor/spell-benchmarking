from __future__ import annotations

import sys
import types

stub = types.ModuleType("spell_benchmark_client")


class _DummyClient:  # pragma: no cover - import wiring only
    pass


stub.SpellBenchmarkClient = _DummyClient
sys.modules.setdefault("spell_benchmark_client", stub)

from bench import OMNI_MATH_PREAMBLE, _format_general_prompt, _general_preamble
from src.datasets import BenchmarkItem, format_prompt_spell


def test_omni_math_preamble_excludes_subagent_guidance() -> None:
    assert "Sub-agent delegation" not in OMNI_MATH_PREAMBLE


def test_omni_math_spell_prompt_requires_final_answer_format() -> None:
    item = BenchmarkItem(
        id="omni_math_1",
        dataset="omni_math",
        context="",
        question="Compute 2+2.",
        answer="4",
    )

    prompt = format_prompt_spell(item)

    assert "Return ONLY the final answer on a single line" in prompt
    assert "\\boxed{2\\sqrt{3}}" in prompt


def test_general_preamble_applies_to_omni_math_and_project_euler() -> None:
    assert _general_preamble("omni_math") == OMNI_MATH_PREAMBLE
    assert _general_preamble("project_euler") == OMNI_MATH_PREAMBLE
    assert _general_preamble("math") is None


def test_gaia_prompt_mentions_web_search_when_enabled() -> None:
    item = BenchmarkItem(
        id="gaia_1",
        dataset="gaia",
        context="",
        question="Who won?",
        answer="42",
        metadata={"file_path": "/tmp/input.txt"},
    )

    prompt = _format_general_prompt(item, "claude_code", web_search=True)

    assert "Use web search, Python, and any tools needed" in prompt
    assert "Attached file: /tmp/input.txt" in prompt


def test_gaia_prompt_avoids_web_search_when_disabled() -> None:
    item = BenchmarkItem(
        id="gaia_2",
        dataset="gaia",
        context="",
        question="Who won?",
        answer="42",
    )

    prompt = _format_general_prompt(item, "codex", web_search=False)

    assert "Use web search" not in prompt
    assert "Use Python and any available non-web tools needed" in prompt
