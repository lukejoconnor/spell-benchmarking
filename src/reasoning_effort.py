"""Reasoning-effort support matrix for benchmark adapters."""

from __future__ import annotations

SUPPORTED_SPELL_PROVIDER_PREFIXES = {
    "codex-msg",
    "codex-tc",
    "openai",
    "openai-tc",
}


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
