from __future__ import annotations

from bench import OMNI_MATH_PREAMBLE, _general_preamble
from src.unified_adapters import build_spell_init
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
