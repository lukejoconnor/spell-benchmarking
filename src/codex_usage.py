"""Shared Codex CLI usage parsing helpers."""

from __future__ import annotations

import json
from typing import Any

from .usage_accounting import (
    UsageBreakdown,
    estimate_cost as estimate_usage_cost,
    lookup_pricing,
    sum_usages,
    usage_from_totals,
)

CodexUsage = UsageBreakdown


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def estimate_cost(
    model: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    *,
    cached_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    reasoning_output_tokens: int | None = None,
) -> float | None:
    usage = usage_from_totals(
        total_input_tokens=tokens_in,
        total_output_tokens=tokens_out,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=max(0, reasoning_output_tokens or 0),
    )
    return estimate_usage_cost(model, usage)


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
            "reasoning_output_tokens",
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


def usage_from_payload(payload: dict[str, Any] | None, *, model: str | None = None) -> UsageBreakdown | None:
    if not payload:
        return None

    total_input_raw = payload.get("input_tokens")
    total_input_tokens = int(total_input_raw) if isinstance(total_input_raw, (int, float)) else None
    cached_input_tokens = _as_int(payload.get("cached_input_tokens"))
    cache_write_input_tokens = _as_int(payload.get("cache_creation_input_tokens"))
    if cached_input_tokens == 0:
        cached_input_tokens = _as_int(payload.get("cache_read_input_tokens"))

    alt_total_input_raw = payload.get("total_input_tokens")
    if total_input_tokens is None and isinstance(alt_total_input_raw, (int, float)):
        total_input_tokens = int(alt_total_input_raw)

    total_output_raw = payload.get("output_tokens")
    total_output_tokens = int(total_output_raw) if isinstance(total_output_raw, (int, float)) else None
    reasoning_raw = payload.get("reasoning_output_tokens")
    reasoning_output_tokens = int(reasoning_raw) if isinstance(reasoning_raw, (int, float)) else 0

    cost_raw = payload.get("total_cost_usd")
    if cost_raw is None:
        cost_raw = payload.get("cost_usd")
    if cost_raw is None:
        cost_raw = payload.get("cost")
    cost_usd = float(cost_raw) if isinstance(cost_raw, (int, float)) else None
    cost_source = "reported" if cost_usd is not None else None

    usage = usage_from_totals(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )
    if usage is None:
        return None

    if usage.cost_usd is None:
        estimated_cost = estimate_usage_cost(model, usage)
        usage = UsageBreakdown(
            uncached_input_tokens=usage.uncached_input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            visible_output_tokens=usage.visible_output_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            cost_usd=estimated_cost,
            cost_source="estimated" if estimated_cost is not None else None,
        )
    return usage


def usage_from_events(events: list[dict[str, Any]], *, model: str | None = None) -> UsageBreakdown | None:
    return usage_from_payload(usage_payload_from_events(events), model=model)


def extract_codex_thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
        thread = event.get("thread")
        if isinstance(thread, dict):
            nested_thread_id = thread.get("id")
            if isinstance(nested_thread_id, str) and nested_thread_id:
                return nested_thread_id
    return None


def parse_codex_rollout(text: str) -> dict[str, Any] | None:
    session_id: str | None = None
    latest_info: dict[str, Any] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        if event.get("type") == "session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                payload_id = payload.get("id")
                if isinstance(payload_id, str) and payload_id:
                    session_id = payload_id
            continue

        payload = event
        if event.get("type") == "event_msg":
            nested_payload = event.get("payload")
            if not isinstance(nested_payload, dict):
                continue
            payload = nested_payload

        if payload.get("type") != "token_count":
            continue

        info = payload.get("info")
        if not isinstance(info, dict):
            continue
        total_token_usage = info.get("total_token_usage")
        if not isinstance(total_token_usage, dict):
            continue

        latest_info = {
            "total_token_usage": total_token_usage,
            "last_token_usage": info.get("last_token_usage"),
            "model_context_window": info.get("model_context_window"),
            "session_id": session_id,
        }

    return latest_info


def iter_codex_token_count_infos(text: str) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    session_id: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        if event.get("type") == "session_meta":
            payload = event.get("payload")
            if isinstance(payload, dict):
                payload_id = payload.get("id")
                if isinstance(payload_id, str) and payload_id:
                    session_id = payload_id
            continue

        payload = event.get("payload") if event.get("type") == "event_msg" else event
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue

        info = payload.get("info")
        if not isinstance(info, dict) or not isinstance(info.get("total_token_usage"), dict):
            continue

        enriched = dict(info)
        if session_id:
            enriched["session_id"] = session_id
        infos.append(enriched)

    return infos


def _usage_payload_delta(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if previous is None:
        return dict(current)

    out: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    ):
        current_value = current.get(key)
        previous_value = previous.get(key)
        if isinstance(current_value, (int, float)):
            if isinstance(previous_value, (int, float)):
                out[key] = max(0, int(current_value) - int(previous_value))
            else:
                out[key] = int(current_value)
    return out


def codex_cost_accounting_from_rollout(
    text: str,
    *,
    model: str | None,
) -> dict[str, Any] | None:
    infos = iter_codex_token_count_infos(text)
    if not infos:
        return None

    normalized_model = normalize_codex_model(model)
    pricing = lookup_pricing(normalized_model)
    calls: list[dict[str, Any]] = []
    usages: list[UsageBreakdown] = []
    previous_total: dict[str, Any] | None = None
    warnings: list[str] = []

    for idx, info in enumerate(infos, start=1):
        total_usage = info.get("total_token_usage")
        if not isinstance(total_usage, dict):
            continue

        last_usage = info.get("last_token_usage")
        if isinstance(last_usage, dict):
            payload = last_usage
            source = "last_token_usage"
        else:
            payload = _usage_payload_delta(total_usage, previous_total)
            source = "delta_total_token_usage"
            warnings.append(f"call {idx}: missing last_token_usage; used cumulative delta")

        previous_total = total_usage
        usage = usage_from_payload(payload, model=normalized_model)
        if usage is None:
            warnings.append(f"call {idx}: token usage payload could not be normalized")
            continue

        calls.append(
            {
                "index": idx,
                "source": source,
                "input_tokens": usage.total_input_tokens,
                "uncached_input_tokens": usage.uncached_input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_write_input_tokens": usage.cache_write_input_tokens,
                "output_tokens": usage.total_output_tokens,
                "visible_output_tokens": usage.visible_output_tokens,
                "reasoning_output_tokens": usage.reasoning_output_tokens,
                "cost_usd": usage.cost_usd,
            }
        )
        usages.append(usage)

    aggregate = sum_usages(*usages)
    total_usage = infos[-1].get("total_token_usage")
    if isinstance(total_usage, dict) and aggregate is not None:
        final_usage = usage_from_payload(total_usage, model=normalized_model)
        if final_usage is not None:
            for field_name in ("total_input_tokens", "total_output_tokens", "cached_input_tokens"):
                if getattr(final_usage, field_name) != getattr(aggregate, field_name):
                    warnings.append(
                        f"per-call aggregate {field_name}={getattr(aggregate, field_name)} "
                        f"does not match final total {getattr(final_usage, field_name)}"
                    )

    total_cost = sum(call["cost_usd"] for call in calls if isinstance(call.get("cost_usd"), (int, float)))
    return {
        "model": normalized_model,
        "cost_source": "estimated_per_call" if calls and pricing is not None else None,
        "cost_usd": total_cost if calls and pricing is not None else None,
        "calls": calls,
        "aggregate": {
            "input_tokens": aggregate.total_input_tokens if aggregate else 0,
            "uncached_input_tokens": aggregate.uncached_input_tokens if aggregate else 0,
            "cached_input_tokens": aggregate.cached_input_tokens if aggregate else 0,
            "cache_write_input_tokens": aggregate.cache_write_input_tokens if aggregate else 0,
            "output_tokens": aggregate.total_output_tokens if aggregate else 0,
            "visible_output_tokens": aggregate.visible_output_tokens if aggregate else 0,
            "reasoning_output_tokens": aggregate.reasoning_output_tokens if aggregate else 0,
            "cost_usd": total_cost if calls and pricing is not None else None,
        },
        "pricing": {
            "rates_per_million": pricing,
            "source": "shared benchmark base pricing table",
        },
        "warnings": warnings,
    }


def usage_from_rollout(text: str, *, model: str | None = None) -> UsageBreakdown | None:
    parsed = parse_codex_rollout(text)
    if not parsed:
        return None
    payload = parsed.get("total_token_usage")
    if not isinstance(payload, dict):
        return None
    return usage_from_payload(payload, model=model)


def merge_codex_usage(
    stream: UsageBreakdown | None,
    rollout: UsageBreakdown | None,
    *,
    model: str | None = None,
) -> UsageBreakdown | None:
    del model
    if stream is None:
        return rollout
    if rollout is None:
        return stream

    if stream.cost_source == "reported":
        return UsageBreakdown(
            uncached_input_tokens=stream.uncached_input_tokens,
            cached_input_tokens=stream.cached_input_tokens or rollout.cached_input_tokens,
            cache_write_input_tokens=stream.cache_write_input_tokens or rollout.cache_write_input_tokens,
            visible_output_tokens=stream.visible_output_tokens,
            reasoning_output_tokens=(
                stream.reasoning_output_tokens
                if stream.reasoning_output_tokens
                else rollout.reasoning_output_tokens
            ),
            cost_usd=stream.cost_usd,
            cost_source=stream.cost_source,
        )

    if rollout.reasoning_output_tokens == 0:
        return rollout if rollout.cost_source == "reported" else stream

    if rollout.cost_source != "estimated":
        return rollout

    return UsageBreakdown(
        uncached_input_tokens=(
            stream.uncached_input_tokens
            if stream.total_input_tokens
            else rollout.uncached_input_tokens
        ),
        cached_input_tokens=rollout.cached_input_tokens or stream.cached_input_tokens,
        cache_write_input_tokens=rollout.cache_write_input_tokens or stream.cache_write_input_tokens,
        visible_output_tokens=(
            stream.visible_output_tokens
            if stream.total_output_tokens
            else rollout.visible_output_tokens
        ),
        reasoning_output_tokens=rollout.reasoning_output_tokens,
        cost_usd=rollout.cost_usd,
        cost_source="estimated_with_reasoning",
    )
