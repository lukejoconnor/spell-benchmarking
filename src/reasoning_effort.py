"""Reasoning-effort support matrix for benchmark adapters."""

from __future__ import annotations

SUPPORTED_SPELL_PROVIDER_PREFIXES = {
    "codex-tc",
    "openai-tc",
    "anthropic-pf",
    "anthropic-tc",
}

# Anthropic uses budget_tokens (thinking) instead of reasoning_effort.
# Map effort levels to thinking budget values.
ANTHROPIC_THINKING_BUDGET: dict[str, int | None] = {
    "low": None,  # no extended thinking
    "medium": 10000,
    "high": 32000,
}

ANTHROPIC_PROVIDER_PREFIXES = {"anthropic-pf", "anthropic-tc"}
FIREWORKS_QWEN_HIGH_REASONING_BUDGET = "32000"
FIREWORKS_REASONING_MODEL_MARKERS = {
    "deepseek-v4-pro",
    "glm-5p1",
    "qwen3p6-plus",
}
FIREWORKS_SWE_AGENT_REASONING_MODEL_MARKERS = {
    "deepseek-v4-pro",
}


def is_anthropic_provider(model_spec: str | None) -> bool:
    return _provider_prefix(model_spec) in ANTHROPIC_PROVIDER_PREFIXES


def anthropic_thinking_budget(reasoning_effort: str | None) -> int | None:
    """Map reasoning effort to Anthropic thinking budget_tokens."""
    if not reasoning_effort:
        return None
    return ANTHROPIC_THINKING_BUDGET.get(reasoning_effort)


def fireworks_reasoning_effort(model_spec: str | None, reasoning_effort: str | None) -> str | None:
    """Resolve benchmark effort levels to Fireworks provider reasoning values."""
    if not reasoning_effort:
        return None
    if _provider_prefix(model_spec) not in {"fireworks", "fireworks-tc"}:
        return reasoning_effort
    model = _model_name(model_spec)
    if not _fireworks_supports_reasoning(model):
        return None
    if model and "qwen3p6-plus" in model and reasoning_effort == "high":
        return FIREWORKS_QWEN_HIGH_REASONING_BUDGET
    return reasoning_effort


def _fireworks_supports_reasoning(model: str | None) -> bool:
    if not model:
        return False
    return any(marker in model for marker in FIREWORKS_REASONING_MODEL_MARKERS)


def _fireworks_swe_agent_supports_reasoning(model: str | None) -> bool:
    if not model:
        return False
    return any(marker in model for marker in FIREWORKS_SWE_AGENT_REASONING_MODEL_MARKERS)


def _provider_prefix(model_spec: str | None) -> str | None:
    if not model_spec or ":" not in model_spec:
        return None
    provider, _ = model_spec.split(":", 1)
    return provider or None


def _model_name(model_spec: str | None) -> str | None:
    if not model_spec or ":" not in model_spec:
        return model_spec
    _, model = model_spec.split(":", 1)
    return model or None


def supports_reasoning_effort(condition: str, model_spec: str | None) -> bool:
    if condition in {"claude_code", "codex"}:
        return True
    if condition == "swe_agent":
        return _fireworks_swe_agent_supports_reasoning(_model_name(model_spec))
    if condition in {"one_shot", "spell"}:
        provider = _provider_prefix(model_spec)
        if provider in SUPPORTED_SPELL_PROVIDER_PREFIXES:
            return True
        if provider in {"fireworks", "fireworks-tc"}:
            return _fireworks_supports_reasoning(_model_name(model_spec))
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
    return fireworks_reasoning_effort(model_spec, reasoning_effort)
