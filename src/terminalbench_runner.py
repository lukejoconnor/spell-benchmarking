"""Helpers for running Terminal-Bench through the unified benchmark CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .codex_usage import normalize_codex_model
from .docker_agents import discover_spell_root
from .provenance import benchmark_git_versions
from .reasoning_effort import effective_reasoning_effort
from .usage_accounting import (
    CLAUDE_REPORTED_COST_TOLERANCE_USD,
    UsageBreakdown,
    estimate_cost,
    usage_from_totals,
    usage_record_fields,
)
from .unified_results import UnifiedRunRecord, now_iso

TERMINALBENCH_OLD_CORE_DATASET = "terminal-bench-core==0.1.1"
TERMINALBENCH_HEAD_DATASET = "terminal-bench-core==head"
TERMINALBENCH_DATASET = TERMINALBENCH_OLD_CORE_DATASET
TERMINALBENCH_HEAD_REGISTRY = Path("terminal_bench") / "terminal-bench-core-head-registry.json"
TERMINALBENCH_HEAD_ALIASES = frozenset({"2.0", "head", "current", "latest"})
TERMINALBENCH_DATASET_ALIASES = {
    "old-core": TERMINALBENCH_OLD_CORE_DATASET,
    "core-0.1.1": TERMINALBENCH_OLD_CORE_DATASET,
    "0.1.1": TERMINALBENCH_OLD_CORE_DATASET,
    "1.1": TERMINALBENCH_OLD_CORE_DATASET,
    **{alias: TERMINALBENCH_HEAD_DATASET for alias in TERMINALBENCH_HEAD_ALIASES},
}
TERMINALBENCH_CONDITIONS = ("spell", "claude_code", "codex")
TERMINALBENCH_RESULTS_DIR = Path("results") / "terminal-bench"


@dataclass(frozen=True)
class TerminalBenchDatasetSpec:
    dataset: str
    local_registry_path: Path | None = None


@dataclass(frozen=True)
class TerminalBenchConditionSpec:
    agent_import_path: str
    default_model: str


CONDITION_SPECS: dict[str, TerminalBenchConditionSpec] = {
    "spell": TerminalBenchConditionSpec(
        agent_import_path="spell_agent:SpellAgent",
        default_model="anthropic-tc:opus46",
    ),
    "claude_code": TerminalBenchConditionSpec(
        agent_import_path="cc_agent:ClaudeCodeAgent",
        default_model="anthropic/claude-opus-4-6",
    ),
    "codex": TerminalBenchConditionSpec(
        agent_import_path="codex_agent:CodexAgent",
        default_model="openai/gpt-5.3-codex",
    ),
}


def resolve_terminalbench_model(condition: str, override: str | None) -> str:
    return override or CONDITION_SPECS[condition].default_model


def resolve_terminalbench_dataset(dataset: str | None) -> str:
    return resolve_terminalbench_dataset_spec(dataset).dataset


def resolve_terminalbench_dataset_spec(dataset: str | None) -> TerminalBenchDatasetSpec:
    if dataset is None:
        return TerminalBenchDatasetSpec(TERMINALBENCH_DATASET)
    requested = dataset.strip()
    resolved = TERMINALBENCH_DATASET_ALIASES.get(requested, requested)
    registry_path = TERMINALBENCH_HEAD_REGISTRY if requested in TERMINALBENCH_HEAD_ALIASES else None
    return TerminalBenchDatasetSpec(resolved, registry_path)


def parse_terminalbench_conditions(values: tuple[str, ...] | list[str]) -> list[str]:
    selected = list(values) if values else ["all"]
    if "all" in selected:
        return list(TERMINALBENCH_CONDITIONS)

    seen: set[str] = set()
    ordered: list[str] = []
    for value in selected:
        if value not in CONDITION_SPECS or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def terminalbench_output_file(run_id: str) -> Path:
    return TERMINALBENCH_RESULTS_DIR / f"{run_id}.jsonl"


def terminalbench_condition_run_id(run_id: str, condition: str, *, name: str | None = None) -> str:
    parts = [condition]
    if name:
        parts.append(_sanitize_terminalbench_segment(name, "name"))
    parts.append(run_id)
    return "_".join(parts)


def terminalbench_run_dir(run_id: str, condition: str, *, name: str | None = None) -> Path:
    return TERMINALBENCH_RESULTS_DIR / terminalbench_condition_run_id(run_id, condition, name=name)


def prepare_terminalbench_env(benchmark_root: Path) -> dict[str, str]:
    env = os.environ.copy()

    colima_sock = Path.home() / ".colima" / "default" / "docker.sock"
    if not env.get("DOCKER_HOST") and not Path("/var/run/docker.sock").exists() and colima_sock.exists():
        env["DOCKER_HOST"] = f"unix://{colima_sock}"

    terminal_bench_path = str((benchmark_root / "terminal_bench").resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        paths = existing_pythonpath.split(os.pathsep)
        if terminal_bench_path not in paths:
            env["PYTHONPATH"] = os.pathsep.join([terminal_bench_path, *paths])
    else:
        env["PYTHONPATH"] = terminal_bench_path

    if not env.get("SPELL_ROOT"):
        spell_root = discover_spell_root(search_from=benchmark_root)
        if spell_root is not None:
            env["SPELL_ROOT"] = str(spell_root)

    return env


def build_terminalbench_command(
    *,
    condition: str,
    run_id: str,
    model: str | None,
    dataset: str,
    task_ids: tuple[str, ...] | list[str],
    n_tasks: int | None,
    n_concurrent: int,
    budget: float,
    cleanup: bool,
    reasoning_effort: str,
    agent_file: str | None,
    trailing: str | None,
    trace: bool,
    output_root: Path = TERMINALBENCH_RESULTS_DIR,
) -> list[str]:
    spec = CONDITION_SPECS[condition]
    effective_effort = effective_reasoning_effort(condition, model or spec.default_model, reasoning_effort)
    dataset_spec = resolve_terminalbench_dataset_spec(dataset)
    cmd = [
        "tb",
        "run",
        "--agent-import-path",
        spec.agent_import_path,
        "--model",
        model or spec.default_model,
        "--dataset",
        dataset_spec.dataset,
        "--output-path",
        str(output_root),
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
    ]
    if dataset_spec.local_registry_path is not None:
        cmd.extend(["--local-registry-path", str(dataset_spec.local_registry_path)])

    if cleanup:
        cmd.append("--cleanup")
    else:
        cmd.append("--no-cleanup")

    # Set the harness-level agent timeout to 3600s (1 hour) to align with
    # other coding benchmarks (SWE-bench-style) and accommodate high-effort
    # reasoning runs. The tb CLI defaults to 360s which kills slow runs.
    cmd.extend(["--global-agent-timeout-sec", "3600"])

    for task_id in task_ids:
        cmd.extend(["-t", task_id])
    if n_tasks is not None:
        cmd.extend(["--n-tasks", str(n_tasks)])

    agent_kwargs: dict[str, Any] = {}
    if condition == "spell":
        agent_kwargs = {
            "budget": budget,
            # 600s per shell command — long enough for ML training jobs
            # (e.g., fasttext on 650k examples takes ~8 min). The previous
            # 120s value forced Spell agents to kill training mid-run on
            # the train-fasttext task, cycling through truncated attempts.
            "sh_timeout": 600,
            "agent_timeout": 3600,
            "trace": trace,
        }
        if effective_effort is not None:
            agent_kwargs["reasoning_effort"] = effective_effort
        if agent_file:
            agent_kwargs["agent_file"] = agent_file
        if trailing is not None:
            agent_kwargs["trailing"] = trailing
    elif condition == "claude_code":
        agent_kwargs = {"max_budget_usd": budget}
        if effective_effort is not None:
            agent_kwargs["reasoning_effort"] = effective_effort
    elif condition == "codex":
        agent_kwargs = {"agent_timeout": 3600}
        if effective_effort is not None:
            agent_kwargs["reasoning_effort"] = effective_effort

    for key, value in agent_kwargs.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        cmd.extend(["-k", f"{key}={rendered}"])

    return cmd


def prune_terminalbench_docker(env: dict[str, str]) -> None:
    subprocess.run(
        ["docker", "system", "prune", "-af", "--volumes"],
        cwd=None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def load_terminalbench_results(run_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Terminal-Bench results file not found: {results_path}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _latency_ms(started_at: str | None, ended_at: str | None) -> float | None:
    start = _parse_iso8601(started_at)
    end = _parse_iso8601(ended_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def _agent_log_dir(run_dir: Path, task_id: str, trial_name: str) -> Path:
    return run_dir / task_id / trial_name / "agent-logs"


def _load_agent_result(run_dir: Path, task_id: str, trial_name: str) -> dict[str, Any]:
    result_path = _agent_log_dir(run_dir, task_id, trial_name) / "result.json"
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _usage_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _terminalbench_usage(
    *,
    trial: dict[str, Any],
    agent_result: dict[str, Any],
    condition: str,
    model: str | None,
) -> UsageBreakdown | None:
    total_input_tokens = _usage_int(agent_result.get("total_input_tokens"))
    if total_input_tokens is None:
        total_input_tokens = _usage_int(agent_result.get("input_tokens"))
    if total_input_tokens is None:
        total_input_tokens = _usage_int(trial.get("total_input_tokens"))

    cached_input_tokens = _usage_int(agent_result.get("cached_input_tokens")) or 0
    cache_write_input_tokens = _usage_int(agent_result.get("cache_write_input_tokens")) or 0
    reasoning_output_tokens = _usage_int(agent_result.get("reasoning_output_tokens")) or 0

    total_output_tokens = _usage_int(agent_result.get("total_output_tokens"))
    if total_output_tokens is None:
        visible_output_tokens = _usage_int(agent_result.get("visible_output_tokens"))
        if visible_output_tokens is None:
            # Older agent logs recorded visible output under `output_tokens`.
            visible_output_tokens = _usage_int(agent_result.get("output_tokens"))
        if visible_output_tokens is not None:
            total_output_tokens = visible_output_tokens + reasoning_output_tokens

    if total_output_tokens is None:
        total_output_tokens = _usage_int(trial.get("total_output_tokens"))

    cost_usd = agent_result.get("cost_usd")
    if not isinstance(cost_usd, (int, float)):
        cost_usd = None

    usage = usage_from_totals(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cost_usd=cost_usd,
        cost_source=agent_result.get("cost_source"),
    )
    if usage is None:
        return usage

    if usage.cost_usd is not None and condition != "claude_code":
        return usage

    reported_cost = usage.cost_usd
    if reported_cost is not None:
        estimated_cost = estimate_cost(model, usage)
        if estimated_cost is not None and reported_cost < estimated_cost - CLAUDE_REPORTED_COST_TOLERANCE_USD:
            return UsageBreakdown(
                uncached_input_tokens=usage.uncached_input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_input_tokens=usage.cache_write_input_tokens,
                visible_output_tokens=usage.visible_output_tokens,
                reasoning_output_tokens=usage.reasoning_output_tokens,
                cost_usd=estimated_cost,
                cost_source="estimated",
            )
        if usage.cost_source is not None:
            return usage
        return UsageBreakdown(
            uncached_input_tokens=usage.uncached_input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            visible_output_tokens=usage.visible_output_tokens,
            reasoning_output_tokens=usage.reasoning_output_tokens,
            cost_usd=reported_cost,
            cost_source="reported",
        )

    estimated_cost = estimate_cost(model, usage)
    return UsageBreakdown(
        uncached_input_tokens=usage.uncached_input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_write_input_tokens=usage.cache_write_input_tokens,
        visible_output_tokens=usage.visible_output_tokens,
        reasoning_output_tokens=usage.reasoning_output_tokens,
        cost_usd=estimated_cost,
        cost_source="estimated" if estimated_cost is not None else None,
    )


def _copy_spell_trace_dir(
    run_dir: Path,
    *,
    task_id: str,
    trial_name: str,
    host_trace_root: Path | None,
) -> str | None:
    if host_trace_root is None:
        return None

    source = _agent_log_dir(run_dir, task_id, trial_name) / "spell-trace"
    if not source.exists():
        return None

    dest = host_trace_root / task_id
    if dest.exists():
        dest = dest / trial_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, dirs_exist_ok=True)
    return str(dest)


def _normalize_record_model(condition: str, model: str | None) -> str:
    raw = (model or "").strip()
    if condition == "codex":
        return normalize_codex_model(raw, raw)
    if "/" in raw:
        return raw.split("/", 1)[1]
    if ":" in raw:
        _, raw = raw.split(":", 1)
    return raw


def _condition_provider(condition: str, model: str | None) -> str | None:
    if condition == "claude_code":
        return "claude_code"
    if condition == "codex":
        return "codex"
    if not model or ":" not in model:
        return None
    provider, _ = model.split(":", 1)
    return provider or None


def _sanitize_terminalbench_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _terminalbench_failure_mode(trial: dict[str, Any]) -> str | None:
    failure_mode = trial.get("failure_mode")
    if not failure_mode or failure_mode in {"none", "unset"}:
        return None
    return str(failure_mode)


def _status_for_trial(trial: dict[str, Any]) -> str:
    if trial.get("is_resolved") is True:
        return "resolved"
    failure_mode = _terminalbench_failure_mode(trial)
    if failure_mode == "agent_timeout":
        return "timeout"
    if failure_mode:
        return "error"
    if trial.get("is_resolved") is False:
        return "unresolved"
    return "error"


def _error_type_for_trial(trial: dict[str, Any]) -> str | None:
    failure_mode = _terminalbench_failure_mode(trial)
    if failure_mode == "agent_timeout":
        return "timeout"
    return failure_mode


def _error_message_for_trial(trial: dict[str, Any]) -> str | None:
    failure_mode = _terminalbench_failure_mode(trial)
    if failure_mode is None:
        return None
    return f"Terminal-Bench failure: {failure_mode}"


def terminalbench_records_from_run(
    *,
    run_dir: Path,
    run_id: str,
    condition: str,
    dataset: str,
    model_spec: str | None,
    name: str | None,
    agent_file: str | None,
    requested_dataset: str | None = None,
    reasoning_effort: str | None = None,
    host_trace_root: Path | None = None,
) -> list[UnifiedRunRecord]:
    payload = load_terminalbench_results(run_dir)
    trials = payload.get("results") or []
    records: list[UnifiedRunRecord] = []
    resolved_dataset = resolve_terminalbench_dataset(dataset)
    resolved_model = resolve_terminalbench_model(condition, model_spec)
    normalized_model = _normalize_record_model(condition, resolved_model)
    recorded_reasoning_effort = effective_reasoning_effort(condition, resolved_model, reasoning_effort)

    for trial in trials:
        task_id = trial["task_id"]
        trial_name = trial["trial_name"]
        status = _status_for_trial(trial)
        error_type = _error_type_for_trial(trial)
        error_message = _error_message_for_trial(trial)
        agent_result = _load_agent_result(run_dir, task_id, trial_name)
        usage = _terminalbench_usage(
            trial=trial,
            agent_result=agent_result,
            condition=condition,
            model=normalized_model,
        )
        cached_input_tokens = usage.cached_input_tokens if usage else agent_result.get("cached_input_tokens")
        if not isinstance(cached_input_tokens, (int, float)):
            cached_input_tokens = None
        elif cached_input_tokens == 0:
            cached_input_tokens = None
        cost_usd = usage.cost_usd if usage else None
        reasoning_output_tokens = agent_result.get("reasoning_output_tokens")
        if not isinstance(reasoning_output_tokens, (int, float)):
            reasoning_output_tokens = usage.reasoning_output_tokens if usage else None
        usage_fields = usage_record_fields(usage)

        copied_trace_dir = None
        if condition == "spell":
            copied_trace_dir = _copy_spell_trace_dir(
                run_dir,
                task_id=task_id,
                trial_name=trial_name,
                host_trace_root=host_trace_root,
            )

        metadata = {
            "trial_name": trial_name,
            "failure_mode": trial.get("failure_mode"),
            "parser_results": trial.get("parser_results"),
            "recording_path": trial.get("recording_path"),
            "tb_run_dir": str(run_dir),
            "tb_agent_result": agent_result or None,
            "trace_dir": copied_trace_dir,
            "resolved_model": normalized_model if condition == "codex" else None,
            "cost_source": usage.cost_source if usage else None,
            "reasoning_output_tokens": reasoning_output_tokens if condition == "codex" else None,
            "requested_dataset": requested_dataset if requested_dataset and requested_dataset != resolved_dataset else None,
            "resolved_dataset": resolved_dataset if requested_dataset and requested_dataset != resolved_dataset else None,
            "versions": benchmark_git_versions(),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        records.append(
            UnifiedRunRecord(
                run_id=run_id,
                timestamp=now_iso(),
                benchmark="terminalbench",
                dataset=resolved_dataset,
                split=None,
                item_id=task_id,
                condition=condition,
                provider=_condition_provider(condition, resolved_model),
                model=normalized_model or None,
                agent_file=agent_file,
                status=status,
                response_text=None,
                patch=None,
                answer_extracted=None,
                correct=None if error_message else trial.get("is_resolved"),
                uncached_input_tokens=usage_fields["uncached_input_tokens"],
                cached_input_tokens=usage_fields["cached_input_tokens"],
                cache_write_input_tokens=usage_fields["cache_write_input_tokens"],
                visible_output_tokens=usage_fields["visible_output_tokens"],
                reasoning_output_tokens=usage_fields["reasoning_output_tokens"],
                cost_usd=cost_usd,
                latency_ms=_latency_ms(trial.get("agent_started_at"), trial.get("agent_ended_at")),
                error_type=error_type,
                error_message=error_message,
                metadata=metadata,
                name=name,
                reasoning_effort=recorded_reasoning_effort,
            )
        )

    return records
