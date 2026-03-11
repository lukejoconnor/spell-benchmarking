from __future__ import annotations

from src.datasets import BenchmarkItem, LONG_CONTEXT_PROMPT_ADVICE, format_benchmark_prompt


def test_shared_prompt_formats_omni_math_answer_instruction() -> None:
    item = BenchmarkItem(
        id="omni_math_1",
        dataset="omni_math",
        context="",
        question="Compute 2+2.",
        answer="4",
    )

    prompt = format_benchmark_prompt(item)

    assert "You may use Python for computation if helpful." in prompt
    assert "Reply with ONLY the final answer on a single line" in prompt
    assert "\\boxed{2\\sqrt{3}}" in prompt


def test_shared_prompt_adds_long_context_guidance_for_external_refs() -> None:
    item = BenchmarkItem(
        id="longbench_1",
        dataset="longbench_short",
        context="alpha beta gamma",
        question="Which choice is correct?",
        answer="A",
        choices=["Alpha", "Beta", "Gamma", "Delta"],
    )

    prompt = format_benchmark_prompt(
        item,
        context_ref="Read the document at /tmp/context.txt (16 chars)",
    )

    assert LONG_CONTEXT_PROMPT_ADVICE in prompt
    assert "Reply with ONLY the letter (A, B, C, or D)." in prompt


def test_shared_prompt_skips_long_context_guidance_for_inline_context() -> None:
    item = BenchmarkItem(
        id="longbench_2",
        dataset="longbench_short",
        context="alpha beta gamma",
        question="Which choice is correct?",
        answer="A",
        choices=["Alpha", "Beta", "Gamma", "Delta"],
    )

    prompt = format_benchmark_prompt(
        item,
        context_ref="Context:\nalpha beta gamma",
    )

    assert LONG_CONTEXT_PROMPT_ADVICE not in prompt
    assert "Choices:" in prompt


def test_shared_prompt_formats_gaia_with_tool_availability_toggle() -> None:
    item = BenchmarkItem(
        id="gaia_1",
        dataset="gaia",
        context="",
        question="Who won?",
        answer="someone",
        metadata={"file_path": "/tmp/input.txt"},
    )

    prompt_with_web = format_benchmark_prompt(item, web_search=True)
    prompt_without_web = format_benchmark_prompt(item, web_search=False)

    assert "Attached file: /tmp/input.txt" in prompt_with_web
    assert "Use web search, Python, and any tools needed to answer this question." in prompt_with_web
    assert "Use Python and any available non-web tools needed to answer this question." in prompt_without_web
