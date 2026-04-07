"""Shared benchmark-side usage accounting helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
_TOKEN_RE = re.compile(r'\s+|[{}]|"(?:\\.|[^"\\])*"|[-+]?\d+(?:\.\d+)?')


def resolve_spell_root() -> Path:
    env_value = os.environ.get("SPELL_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    worktree_name = BENCHMARK_ROOT.parent.name
    if BENCHMARK_ROOT.parent.parent.name == "worktrees":
        spell_repo_root = BENCHMARK_ROOT.parents[4]
        paired_worktree = spell_repo_root / ".claude" / "worktrees" / worktree_name
        if (paired_worktree / "config" / "agents").exists():
            return paired_worktree

    for candidate in BENCHMARK_ROOT.parents:
        if (candidate / "config" / "agents").exists():
            return candidate
        sibling_candidate = candidate / "spell"
        if (sibling_candidate / "config" / "agents").exists():
            return sibling_candidate

    raise FileNotFoundError("Could not locate Spell repo root for shared pricing")


def _tokenize_edn(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            raise ValueError(f"Unsupported EDN near offset {pos}: {text[pos:pos + 40]!r}")
        token = match.group(0)
        pos = match.end()
        if token.isspace():
            continue
        tokens.append(token)
    return tokens


def _parse_edn_value(tokens: list[str], idx: int = 0):
    token = tokens[idx]
    if token == "{":
        out: dict[object, object] = {}
        idx += 1
        while tokens[idx] != "}":
            key, idx = _parse_edn_value(tokens, idx)
            value, idx = _parse_edn_value(tokens, idx)
            out[key] = value
        return out, idx + 1
    if token.startswith('"'):
        return json.loads(token), idx + 1
    if "." in token:
        return float(token), idx + 1
    return int(token), idx + 1


@lru_cache(maxsize=1)
def load_pricing_table() -> dict[str, dict[str, float]]:
    pricing_path = resolve_spell_root() / "data" / "pricing.edn"
    parsed, next_idx = _parse_edn_value(_tokenize_edn(pricing_path.read_text(encoding="utf-8")))
    if next_idx == 0 or not isinstance(parsed, dict):
        raise ValueError(f"Invalid pricing data in {pricing_path}")
    return parsed  # type: ignore[return-value]


def lookup_pricing(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    best_prefix = None
    best_cost = None
    for prefix, cost in load_pricing_table().items():
        if model.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
            best_prefix = prefix
            best_cost = cost
    return best_cost


@dataclass(frozen=True, slots=True)
class UsageBreakdown:
    uncached_input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    visible_output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: float | None = None
    cost_source: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.cached_input_tokens
            + self.cache_write_input_tokens
        )

    @property
    def input_tokens(self) -> int:
        return self.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.visible_output_tokens + self.reasoning_output_tokens

    @property
    def output_tokens(self) -> int:
        return self.visible_output_tokens


def usage_from_totals(
    *,
    total_input_tokens: int | None,
    total_output_tokens: int | None,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
    cost_usd: float | None = None,
    cost_source: str | None = None,
) -> UsageBreakdown | None:
    if total_input_tokens is None and total_output_tokens is None and cost_usd is None:
        return None
    total_in = max(0, total_input_tokens or 0)
    cached = max(0, cached_input_tokens)
    cache_write = max(0, cache_write_input_tokens)
    total_out = max(0, total_output_tokens or 0)
    reasoning = max(0, reasoning_output_tokens)
    return UsageBreakdown(
        uncached_input_tokens=max(total_in - cached - cache_write, 0),
        cached_input_tokens=cached,
        cache_write_input_tokens=cache_write,
        visible_output_tokens=total_out,
        reasoning_output_tokens=reasoning,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )


def sum_usages(*usages: UsageBreakdown | None) -> UsageBreakdown | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    costs = [usage.cost_usd for usage in present if usage.cost_usd is not None]
    cost_sources = {usage.cost_source for usage in present if usage.cost_source}
    return UsageBreakdown(
        uncached_input_tokens=sum(usage.uncached_input_tokens for usage in present),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in present),
        cache_write_input_tokens=sum(usage.cache_write_input_tokens for usage in present),
        visible_output_tokens=sum(usage.visible_output_tokens for usage in present),
        reasoning_output_tokens=sum(usage.reasoning_output_tokens for usage in present),
        cost_usd=sum(costs) if costs else None,
        cost_source=next(iter(cost_sources)) if len(cost_sources) == 1 else None,
    )


def estimate_cost(model: str | None, usage: UsageBreakdown | None) -> float | None:
    pricing = lookup_pricing(model)
    if pricing is None or usage is None:
        return None
    return (
        usage.uncached_input_tokens * float(pricing["input"]) / 1_000_000
        + usage.cached_input_tokens * float(pricing["cache-read-input"]) / 1_000_000
        + usage.cache_write_input_tokens * float(pricing["cache-write-input"]) / 1_000_000
        + usage.total_output_tokens * float(pricing["output"]) / 1_000_000
    )


def usage_record_fields(usage: UsageBreakdown | None) -> dict[str, int | None]:
    def optional(value: int) -> int | None:
        return value or None

    if usage is None:
        return {
            "uncached_input_tokens": None,
            "cached_input_tokens": None,
            "cache_write_input_tokens": None,
            "visible_output_tokens": None,
            "reasoning_output_tokens": None,
        }
    return {
        "uncached_input_tokens": optional(usage.uncached_input_tokens),
        "cached_input_tokens": optional(usage.cached_input_tokens),
        "cache_write_input_tokens": optional(usage.cache_write_input_tokens),
        "visible_output_tokens": optional(usage.visible_output_tokens),
        "reasoning_output_tokens": optional(usage.reasoning_output_tokens),
    }
