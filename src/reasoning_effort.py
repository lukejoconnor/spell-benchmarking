"""Reasoning-effort support matrix for benchmark adapters."""

from __future__ import annotations

SUPPORTED_SPELL_PROVIDER_PREFIXES = {
    "codex-tc",
    "openai-tc",
    "anthropic-pf",
    "anthropic-tc",
    "fireworks-tc",
}

# Anthropic uses budget_tokens (thinking) instead of reasoning_effort.
# Map effort levels to thinking budget values.
ANTHROPIC_THINKING_BUDGET: dict[str, int | None] = {
    "low": None,  # no extended thinking
    "medium": 10000,
    "high": 32000,
}

ANTHROPIC_PROVIDER_PREFIXES = {"anthropic-pf", "anthropic-tc"}


def is_anthropic_provider(model_spec: str | None) -> bool:
    return _provider_prefix(model_spec) in ANTHROPIC_PROVIDER_PREFIXES


def anthropic_thinking_budget(reasoning_effort: str | None) -> int | None:
    """Map reasoning effort to Anthropic thinking budget_tokens."""
    if not reasoning_effort:
        return None
    return ANTHROPIC_THINKING_BUDGET.get(reasoning_effort)


def _provider_prefix(model_spec: str | None) -> str | None:
    if not model_spec or ":" not in model_spec:
        return None
    provider, _ = model_spec.split(":", 1)
    return provider or None


def supports_reasoning_effort(condition: str, model_spec: str | None) -> bool:
    if condition in {"claude_code", "codex"}:
        return True
    if condition == "swe_agent":
        return False
    if condition in {"one_shot", "spell"}:
        return _provider_prefix(model_spec) in SUPPORTED_SPELL_PROVIDER_PREFIXES
    return False


def effective_reasoning_effort(
    condition: str,
    model_spec: str | None,
    reasoning_effort: str | None,
) -> str | None:
    if not reasoning_effort:
        return None
    if not supports_reasoning_effort(condition, model_spec):
        return None
    return reasoning_effort
