"""Shared Codex CLI usage parsing and cost estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Per-MTok pricing [input, output] for normalized benchmark cost estimation when
# the Codex CLI omits billed cost from its JSON stream.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5.1-codex": (1.25, 10.00),
    "gpt-5.2-codex": (1.75, 14.00),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.3": (1.75, 14.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
}


@dataclass(frozen=True, slots=True)
class CodexUsage:
    input_tokens: int | None = None
    cached_input_tokens: int = 0
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: str | None = None

    @property
    def total_input_tokens(self) -> int | None:
        if self.input_tokens is None:
            return None
        return self.input_tokens


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def estimate_cost(model: str | None, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if not model or tokens_in is None or tokens_out is None:
        return None
    for prefix, (in_rate, out_rate) in _MODEL_PRICING.items():
        if model.startswith(prefix):
            return tokens_in * in_rate / 1_000_000 + tokens_out * out_rate / 1_000_000
    return None


def normalize_codex_model(model_spec: str | None, fallback: str | None = None) -> str:
    raw = (model_spec or fallback or "codex").strip()
    if ":" in raw:
        _, raw = raw.split(":", 1)
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    if raw in {"", "codex"}:
        return "gpt-5-codex"
    return raw


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _has_usage_keys(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "input_tokens",
            "output_tokens",
            "total_input_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "total_cost_usd",
            "cost_usd",
            "cost",
        )
    )


def usage_payload_from_events(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    usage_candidate: dict[str, Any] | None = None
    for event in events:
        for dict_value in _iter_dicts(event):
            if _has_usage_keys(dict_value):
                usage_candidate = dict_value
    return usage_candidate


def usage_from_payload(payload: dict[str, Any] | None, *, model: str | None = None) -> CodexUsage | None:
    if not payload:
        return None

    base_input_raw = payload.get("input_tokens")
    input_tokens = int(base_input_raw) if isinstance(base_input_raw, (int, float)) else None
    cached_input_tokens = _as_int(payload.get("cached_input_tokens"))
    if cached_input_tokens == 0:
        cached_input_tokens = _as_int(payload.get("cache_creation_input_tokens")) + _as_int(
            payload.get("cache_read_input_tokens")
        )

    total_input_raw = payload.get("total_input_tokens")
    if input_tokens is None and isinstance(total_input_raw, (int, float)):
        input_tokens = int(total_input_raw)

    output_raw = payload.get("output_tokens")
    output_tokens = int(output_raw) if isinstance(output_raw, (int, float)) else None

    cost_raw = payload.get("total_cost_usd")
    if cost_raw is None:
        cost_raw = payload.get("cost_usd")
    if cost_raw is None:
        cost_raw = payload.get("cost")
    cost_usd = float(cost_raw) if isinstance(cost_raw, (int, float)) else None
    cost_source = "reported" if cost_usd is not None else None

    if cost_usd is None:
        cost_usd = estimate_cost(
            model,
            input_tokens,
            output_tokens,
        )
        if cost_usd is not None:
            cost_source = "estimated"

    if input_tokens is None and output_tokens is None and cost_usd is None:
        return None

    return CodexUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )


def usage_from_events(events: list[dict[str, Any]], *, model: str | None = None) -> CodexUsage | None:
    return usage_from_payload(usage_payload_from_events(events), model=model)
