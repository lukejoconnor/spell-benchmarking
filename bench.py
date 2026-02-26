#!/usr/bin/env python3
"""Unified benchmark CLI."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

try:
    import click
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: click. Run this CLI via `uv run python benchmarking/bench.py ...` "
        "or install benchmarking dependencies."
    ) from exc

from src.datasets import (
    DATASETS,
    BenchmarkItem,
    load_swebench_verified,
    load_swebench_lite,
    load_swebench_verified_mini,
    format_prompt_baseline,
    format_prompt_spell,
)
from src.evaluation import (
    extract_freeform_answer,
    extract_math_answer,
    extract_mcq_answer,
    extract_numeric_answer,
    math_equivalent,
    normalize_numeric,
)
from src.agent_policy import resolve_spell_agent
from src.featurebench_lib import (
    format_prompt as featurebench_format_prompt,
    install_deps,
    load_tasks,
    mask_functions,
    run_tests,
    setup_workspace,
)
from src.swebench_output import write_predictions_jsonl
from src.swebench_prep import setup_swebench_workspace, get_default_workspace_root
from src.workspace import RepoCache
from src.unified_adapters import (
    BaselineAdapter,
    ClaudeCodeAdapter,
    CodexAdapter,
    RunRequest,
    RunResponse,
    SpellAdapter,
    build_spell_init,
    run_requests_parallel,
)
from src.unified_results import (
    UnifiedRunRecord,
    append_jsonl,
    now_iso,
    parse_model_spec,
    summarize_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_DIR = PROJECT_ROOT / "benchmarking" / "data" / "contexts"
SWEBENCH_REMINDER_PREAMBLE = '(quine sys-reminder "Recall: you are writing Spell code, not natural language.")'


@dataclass(slots=True)
class _SwePrediction:
    instance_id: str
    patch: str


def _run_id(prefix: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def _default_output(run_id: str) -> Path:
    return Path("results") / "unified" / f"{run_id}.jsonl"


def _parse_conditions(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value == "all":
        return [c for c in allowed if c != "all"]
    return [value]


def _ensure_context_file(item: BenchmarkItem) -> str | None:
    if not item.context:
        return None
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTEXT_DIR / f"{item.id}.txt"
    path.write_text(item.context, encoding="utf-8")
    return str(path)


def _format_general_prompt(item: BenchmarkItem, condition: str) -> str:
    context_file = _ensure_context_file(item)
    if condition == "baseline":
        return format_prompt_baseline(item)
    if condition == "spell":
        return format_prompt_spell(item, context_file=context_file)

    parts: list[str] = []
    if item.context and context_file:
        parts.extend(
            [
                f"Read the document at {context_file} ({len(item.context):,} chars) and answer this question:",
                "",
                item.question.strip(),
            ]
        )
    else:
        parts.extend(["Solve this problem:", "", item.question.strip()])

    if item.choices:
        parts.append("")
        parts.append("Choices:")
        for i, choice in enumerate(item.choices):
            letter = chr(ord("A") + i)
            parts.append(f"  {letter}. {choice}")
        parts.append("")
        parts.append("Reply with ONLY the letter (A, B, C, or D). No explanation.")
    elif item.dataset in ("gsm8k", "math", "aime", "hmmt", "omni_math"):
        parts.append("")
        parts.append("You may use Python via `uv run python3` if helpful.")
        parts.append("Reply with ONLY the final answer. No explanation.")
    else:
        parts.append("")
        parts.append("Reply with ONLY the answer. No explanation.")

    return "\n".join(parts)


def _make_spell_init(prompt: str, preamble: str | None = None) -> str:
    # Some benchmark datasets (e.g. Omni-MATH) provide a raw open Spell prefix.
    # Keep those as-is; otherwise wrap NL prompt in the completion init template.
    if prompt.lstrip().startswith("("):
        return prompt
    return build_spell_init(prompt, preamble=preamble)


def _evaluate_general(item: BenchmarkItem, response_text: str | None) -> tuple[str | None, bool | None]:
    response = (response_text or "").strip()
    if not response:
        return None, False

    if item.choices:
        answer = extract_mcq_answer(response)
        return answer, (answer == item.answer) if answer else False

    if item.dataset == "gsm8k":
        answer = extract_numeric_answer(response)
        if answer is None:
            return None, False
        return answer, normalize_numeric(answer) == normalize_numeric(item.answer)

    if item.dataset in ("math", "aime", "hmmt", "omni_math"):
        answer = extract_math_answer(response)
        if answer is None:
            return None, False
        return answer, math_equivalent(answer, item.answer)

    answer = extract_freeform_answer(response)
    return answer, answer.strip().lower() == item.answer.strip().lower()


def _provider_model_for_condition(condition: str, model_spec: str | None) -> tuple[str | None, str | None]:
    if condition == "claude_code":
        if model_spec and ":" in model_spec:
            _, model = model_spec.split(":", 1)
            return "claude_code", model
        return "claude_code", model_spec
    if condition == "codex":
        if model_spec and ":" in model_spec:
            _, model = model_spec.split(":", 1)
            return "codex", model
        return "codex", (model_spec or "codex")
    return parse_model_spec(model_spec)


def _mk_record(
    *,
    run_id: str,
    benchmark: str,
    dataset: str,
    split: str | None,
    item_id: str,
    condition: str,
    model_spec: str | None,
    agent_file: str | None,
    response: RunResponse,
    response_text: str | None,
    patch: str | None,
    answer_extracted: str | None,
    correct: bool | None,
    metadata: dict[str, Any] | None = None,
) -> UnifiedRunRecord:
    provider, model = _provider_model_for_condition(condition, model_spec)
    return UnifiedRunRecord(
        run_id=run_id,
        timestamp=now_iso(),
        benchmark=benchmark,
        dataset=dataset,
        split=split,
        item_id=item_id,
        condition=condition,
        provider=provider,
        model=model,
        agent_file=agent_file,
        status=response.status,
        response_text=response_text,
        patch=patch,
        answer_extracted=answer_extracted,
        correct=correct,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        error_type=response.error_type,
        error_message=response.error_message,
        metadata={**(metadata or {}), **response.metadata},
    )


def _build_adapter(condition: str, *, model: str, claude_model: str, codex_model: str, budget: float):
    if condition == "spell":
        return SpellAdapter(project_root=PROJECT_ROOT, model_spec=model)
    if condition == "baseline":
        return BaselineAdapter(project_root=PROJECT_ROOT, model_spec=model)
    if condition == "claude_code":
        return ClaudeCodeAdapter(model=claude_model, budget=budget)
    if condition == "codex":
        return CodexAdapter(project_root=PROJECT_ROOT, model_spec=codex_model)
    raise ValueError(f"Unknown condition: {condition}")


def _print_summary(output_file: Path, run_id: str, condition: str) -> None:
    rows = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("run_id") == run_id and row.get("condition") == condition:
                rows.append(row)
    summary = summarize_records(rows)
    click.echo(f"Summary ({condition}): total={summary['total']} errors={summary['errors']} ")
    if summary["accuracy"] is not None:
        click.echo(f"  accuracy={summary['accuracy']:.1%}")
    if summary["total_cost_usd"] is not None:
        click.echo(f"  cost=${summary['total_cost_usd']:.4f}")


def _resolve_spell_agent_or_die(
    *,
    bench_kind: str,
    model_spec: str | None,
    item_datasets: set[str] | None = None,
    agent: str | None = None,
    agent_message: str | None = None,
    agent_toolcall: str | None = None,
):
    try:
        return resolve_spell_agent(
            project_root=PROJECT_ROOT,
            bench_kind=bench_kind,  # type: ignore[arg-type]
            model_spec=model_spec,
            item_datasets=item_datasets,
            agent=agent,
            agent_message=agent_message,
            agent_toolcall=agent_toolcall,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.group()
def cli() -> None:
    """Unified benchmark runner."""


@cli.command("general")
@click.option("--dataset", default="longbench_short", type=click.Choice(list(DATASETS.keys()) + ["all"]))
@click.option("--condition", default="baseline", type=click.Choice(["baseline", "spell", "claude_code", "codex", "all"]))
@click.option("--model", default="anthropic:claude-sonnet-4-5-20250929", help="provider:model for baseline/spell")
@click.option("--claude-model", default="sonnet", help="Model for Claude Code comparator")
@click.option("--codex-model", default="codex", help="Model for Codex harness comparator")
@click.option("--n", type=int, default=None)
@click.option("--items", default=None, help="Comma-separated item ids")
@click.option("--parallel", type=int, default=5, show_default=True)
@click.option("--budget", type=float, default=1.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-message", default=None, help="Spell message-mode agent override")
@click.option("--agent-toolcall", default=None, help="Spell toolcall-mode agent override")
@click.option("--depth", type=int, default=None)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=300, show_default=True)
@click.option("--thinking/--no-thinking", default=False, show_default=True)
@click.option("--thinking-budget", type=int, default=10000, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--verbosity", type=click.Choice(["low", "auto"]), default=None)
@click.option("--responses-api/--no-responses-api", default=False, show_default=True)
@click.option("--suffix-grammar/--no-suffix-grammar", default=False, show_default=True)
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None, help="JSONL output path")
def run_general(
    dataset: str,
    condition: str,
    model: str,
    claude_model: str,
    codex_model: str,
    n: int | None,
    items: str | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_message: str | None,
    agent_toolcall: str | None,
    depth: int | None,
    max_tokens: int | None,
    timeout: int,
    thinking: bool,
    thinking_budget: int,
    reasoning_effort: str | None,
    verbosity: str | None,
    responses_api: bool,
    suffix_grammar: bool,
    trace: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("general")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_conditions(condition, ("baseline", "spell", "claude_code", "codex", "all"))

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")

    if dataset == "all":
        loaded: list[BenchmarkItem] = []
        for loader in DATASETS.values():
            loaded.extend(loader())
    else:
        loaded = DATASETS[dataset]()

    if items:
        ids = {x.strip() for x in items.split(",") if x.strip()}
        loaded = [item for item in loaded if item.id in ids]
    elif n:
        loaded = loaded[:n]
    item_datasets = {item.dataset for item in loaded}

    click.echo(f"Loaded {len(loaded)} items")
    if dry_run:
        spell_resolution = None
        if "spell" in conditions:
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="general",
                model_spec=model,
                item_datasets=item_datasets,
                agent=agent,
                agent_message=agent_message,
                agent_toolcall=agent_toolcall,
            )
        click.echo("Dry run config:")
        click.echo(
            json.dumps(
                {
                    "dataset": dataset,
                    "conditions": conditions,
                    "model": model,
                    "claude_model": claude_model,
                    "codex_model": codex_model,
                    "parallel": parallel,
                    "budget": budget,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                    "spell_mode": spell_resolution.spell_mode if spell_resolution else None,
                    "agent_source": spell_resolution.source if spell_resolution else None,
                    "agent_policy_key": spell_resolution.policy_key if spell_resolution else None,
                },
                indent=2,
            )
        )
        if loaded:
            click.echo(f"First item: {loaded[0].id}")
        return

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond == "spell":
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="general",
                model_spec=model,
                item_datasets=item_datasets,
                agent=agent,
                agent_message=agent_message,
                agent_toolcall=agent_toolcall,
            )
            click.echo(
                f"  spell_mode={spell_resolution.spell_mode} "
                f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
            )

        requests: list[RunRequest] = []
        for item in loaded:
            prompt = _format_general_prompt(item, cond)
            requests.append(
                RunRequest(
                    benchmark="general",
                    dataset=dataset,
                    item_id=item.id,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "baseline": model, "claude_code": claude_model, "codex": codex_model}[cond],
                    timeout=timeout,
                    init=_make_spell_init(prompt) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget if thinking else None,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    responses_api=responses_api,
                    suffix_grammar=suffix_grammar,
                    trace=trace,
                    metadata={"add_dir": CONTEXT_DIR},
                )
            )

        completed = {"count": 0}

        def progress(_idx: int, req: RunRequest, resp: RunResponse) -> None:
            completed["count"] += 1
            status = "E" if resp.error_message else "."
            click.echo(f"  [{completed['count']}/{len(requests)}] {req.item_id[:48]} {status}")

        responses = run_requests_parallel(requests, adapter, max_parallel=parallel, progress_callback=progress)

        for item, req, resp in zip(loaded, requests, responses):
            extracted, correct = _evaluate_general(item, resp.response_text)
            record = _mk_record(
                run_id=run_id,
                benchmark="general",
                dataset=dataset,
                split=None,
                item_id=item.id,
                condition=cond,
                model_spec=req.model_spec,
                agent_file=req.agent_file,
                response=resp,
                response_text=resp.response_text,
                patch=None,
                answer_extracted=extracted,
                correct=correct,
                metadata={
                    "question": item.question,
                    "expected_answer": item.answer,
                    **(
                        {
                            "spell_mode": spell_resolution.spell_mode,
                            "resolved_agent_file": spell_resolution.agent_file,
                            "agent_source": spell_resolution.source,
                            "agent_policy_key": spell_resolution.policy_key,
                        }
                        if (cond == "spell" and spell_resolution)
                        else {}
                    ),
                },
            )
            append_jsonl(output_file, record)

        _print_summary(output_file, run_id, cond)


@cli.command("swebench")
@click.option("--dataset", default="mini", type=click.Choice(["mini", "lite", "verified"]))
@click.option("--condition", default="both", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--model", default="anthropic:claude-opus-4-5-20251101", help="provider:model for Spell")
@click.option("--claude-model", default="opus", help="Model for Claude Code comparator")
@click.option("--codex-model", default="codex")
@click.option("--n", type=int, default=None)
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=10.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-message", default=None, help="Spell message-mode agent override")
@click.option("--agent-toolcall", default=None, help="Spell toolcall-mode agent override")
@click.option("--depth", type=int, default=8, show_default=True)
@click.option("--max-tokens", type=int, default=16384, show_default=True)
@click.option("--timeout", type=int, default=600, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--workspace-root", default=None)
@click.option("--keep-workspaces", is_flag=True, default=False)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_swebench(
    dataset: str,
    condition: str,
    model: str,
    claude_model: str,
    codex_model: str,
    n: int | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_message: str | None,
    agent_toolcall: str | None,
    depth: int,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None,
    workspace_root: str | None,
    keep_workspaces: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("swebench")
    output_file = Path(output) if output else _default_output(run_id)

    if condition == "both":
        conditions = ["spell", "claude_code"]
    elif condition == "all":
        conditions = ["spell", "claude_code", "codex"]
    else:
        conditions = [condition]

    if dataset == "mini":
        items = load_swebench_verified_mini(50)
    elif dataset == "lite":
        items = load_swebench_lite()
    else:
        items = load_swebench_verified()

    if n:
        items = items[:n]

    root = Path(workspace_root) if workspace_root else get_default_workspace_root()
    root.mkdir(parents=True, exist_ok=True)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Workspace root: {root}")
    click.echo(f"Loaded {len(items)} items")
    if dry_run:
        spell_resolution = None
        if "spell" in conditions:
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="swebench",
                model_spec=model,
                agent=agent,
                agent_message=agent_message,
                agent_toolcall=agent_toolcall,
            )
        click.echo(
            json.dumps(
                {
                    "dataset": dataset,
                    "conditions": conditions,
                    "model": model,
                    "claude_model": claude_model,
                    "codex_model": codex_model,
                    "budget": budget,
                    "depth": depth,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                    "spell_mode": spell_resolution.spell_mode if spell_resolution else None,
                    "agent_source": spell_resolution.source if spell_resolution else None,
                    "agent_policy_key": spell_resolution.policy_key if spell_resolution else None,
                    "parallel": parallel,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First item: {items[0].instance_id}")
        return

    cache = RepoCache(root)

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond == "spell":
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="swebench",
                model_spec=model,
                agent=agent,
                agent_message=agent_message,
                agent_toolcall=agent_toolcall,
            )
            click.echo(
                f"  spell_mode={spell_resolution.spell_mode} "
                f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
            )

        def process(item) -> UnifiedRunRecord:
            wt = setup_swebench_workspace(cache, item, tag=cond)
            ws_path = wt.path
            try:
                prompt_parts = [
                    f"Repository: {item.repo}",
                    "",
                    "## Issue to Fix",
                    "",
                    item.problem_statement.strip(),
                    "",
                ]
                if item.hints_text:
                    prompt_parts.extend(["## Hints", "", item.hints_text.strip(), ""])

                prompt_parts.extend(
                    [
                        "## Instructions",
                        "",
                        "Fix this bug with minimal code changes.",
                        "Do not commit any changes.",
                        "Run focused checks/tests when possible.",
                    ]
                )
                prompt = "\n".join(prompt_parts)

                req = RunRequest(
                    benchmark="swebench",
                    dataset=dataset,
                    item_id=item.instance_id,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}[cond],
                    timeout=timeout,
                    init=(_make_spell_init(prompt, SWEBENCH_REMINDER_PREAMBLE)
                          if cond == "spell" else None),
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    workspace=ws_path,
                )
                resp = adapter.run(req)
                patch = wt.diff()
                changed_files = wt.changed_files()

                status = resp.status
                if status == "ok":
                    status = "patch_generated" if patch.strip() else "no_patch"
                resp = RunResponse(
                    status=status,
                    response_text=resp.response_text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cost_usd=resp.cost_usd,
                    latency_ms=resp.latency_ms,
                    error_message=resp.error_message,
                    error_type=resp.error_type,
                    metadata={**resp.metadata, "changed_files": changed_files},
                )

                return _mk_record(
                    run_id=run_id,
                    benchmark="swebench",
                    dataset=dataset,
                    split=dataset,
                    item_id=item.instance_id,
                    condition=cond,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=resp,
                    response_text=resp.response_text,
                    patch=patch,
                    answer_extracted=None,
                    correct=None,
                    metadata={
                        "repo": item.repo,
                        "base_commit": item.base_commit,
                        **(
                            {
                                "spell_mode": spell_resolution.spell_mode,
                                "resolved_agent_file": spell_resolution.agent_file,
                                "agent_source": spell_resolution.source,
                                "agent_policy_key": spell_resolution.policy_key,
                            }
                            if (cond == "spell" and spell_resolution)
                            else {}
                        ),
                    },
                )
            finally:
                if not keep_workspaces:
                    wt.remove()

        if parallel <= 1:
            records = []
            for i, item in enumerate(items):
                click.echo(f"  [{i+1}/{len(items)}] {item.instance_id}")
                records.append(process(item))
        else:
            records = [None] * len(items)
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(process, item): idx for idx, item in enumerate(items)}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    done += 1
                    records[idx] = fut.result()
                    click.echo(f"  [{done}/{len(items)}] {items[idx].instance_id}")
            records = [r for r in records if r is not None]

        for record in records:
            append_jsonl(output_file, record)

        _print_summary(output_file, run_id, cond)

        pred_path = Path("results") / "swebench" / f"{cond}_{dataset}_{run_id}.jsonl"
        swe_results = [_SwePrediction(instance_id=record.item_id, patch=record.patch or "") for record in records]
        pred_model_name = {
            "spell": model,
            "claude_code": f"claude_code:{claude_model}",
            "codex": codex_model,
        }[cond]
        write_predictions_jsonl(swe_results, pred_path, model_name=pred_model_name)
        click.echo(f"Predictions: {pred_path}")


@cli.command("featurebench")
@click.option("--condition", default="claude_code", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--split", default="lite")
@click.option("--task-ids", type=int, multiple=True)
@click.option("--model", default="anthropic:claude-sonnet-4-5-20250929")
@click.option("--claude-model", default="claude-sonnet-4-5-20250929")
@click.option("--codex-model", default="codex")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=10.0, show_default=True)
@click.option("--agent", default=None)
@click.option("--agent-message", default=None, help="Spell message-mode agent override")
@click.option("--agent-toolcall", default=None, help="Spell toolcall-mode agent override")
@click.option("--depth", type=int, default=12, show_default=True)
@click.option("--max-tokens", type=int, default=16384, show_default=True)
@click.option("--timeout", type=int, default=600, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--workspace-root", default="/tmp/fb_ws")
@click.option("--skip-install", is_flag=True, default=False)
@click.option("--skip-tests", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_featurebench(
    condition: str,
    split: str,
    task_ids: tuple[int, ...],
    model: str,
    claude_model: str,
    codex_model: str,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_message: str | None,
    agent_toolcall: str | None,
    depth: int,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None,
    workspace_root: str,
    skip_install: bool,
    skip_tests: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("featurebench")
    output_file = Path(output) if output else _default_output(run_id)

    if condition == "both":
        conditions = ["spell", "claude_code"]
    elif condition == "all":
        conditions = ["spell", "claude_code", "codex"]
    else:
        conditions = [condition]

    tasks = load_tasks(split=split, task_indices=list(task_ids) if task_ids else None)
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = RepoCache(root)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Workspace root: {root}")
    click.echo(f"Loaded {len(tasks)} tasks")

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond == "spell":
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="featurebench",
                model_spec=model,
                agent=agent,
                agent_message=agent_message,
                agent_toolcall=agent_toolcall,
            )
            click.echo(
                f"  spell_mode={spell_resolution.spell_mode} "
                f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
            )

        def process(item: dict) -> UnifiedRunRecord:
            wt = setup_workspace(cache, item, tag=cond)
            try:
                if not skip_install:
                    install_deps(wt.path, item)
                mask_functions(wt.path, item["problem_statement"])
                prompt = featurebench_format_prompt(item, wt.path)

                req = RunRequest(
                    benchmark="featurebench",
                    dataset=split,
                    item_id=item["instance_id"],
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}[cond],
                    timeout=timeout,
                    init=_make_spell_init(prompt) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    workspace=wt.path,
                )
                resp = adapter.run(req)
                patch = wt.diff()
                changed_files = wt.changed_files()

                test_results = []
                tests_passed: bool | None = None
                if not skip_tests and patch.strip():
                    fail_to_pass = item.get("FAIL_TO_PASS", [])
                    if fail_to_pass:
                        test_results = run_tests(wt.path, fail_to_pass, repo=item.get("repo"))
                        tests_passed = all(t["passed"] for t in test_results)

                status = resp.status
                if status == "ok":
                    if patch.strip():
                        status = "tests_passed" if tests_passed else "patch_generated"
                    else:
                        status = "no_patch"

                resp = RunResponse(
                    status=status,
                    response_text=resp.response_text,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    cost_usd=resp.cost_usd,
                    latency_ms=resp.latency_ms,
                    error_message=resp.error_message,
                    error_type=resp.error_type,
                    metadata={**resp.metadata, "changed_files": changed_files, "test_results": test_results},
                )

                return _mk_record(
                    run_id=run_id,
                    benchmark="featurebench",
                    dataset=split,
                    split=split,
                    item_id=item["instance_id"],
                    condition=cond,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=resp,
                    response_text=resp.response_text,
                    patch=patch,
                    answer_extracted=None,
                    correct=tests_passed,
                    metadata={
                        "repo": item.get("repo"),
                        **(
                            {
                                "spell_mode": spell_resolution.spell_mode,
                                "resolved_agent_file": spell_resolution.agent_file,
                                "agent_source": spell_resolution.source,
                                "agent_policy_key": spell_resolution.policy_key,
                            }
                            if (cond == "spell" and spell_resolution)
                            else {}
                        ),
                    },
                )
            finally:
                wt.remove()

        if parallel <= 1:
            records = []
            for i, item in enumerate(tasks):
                click.echo(f"  [{i+1}/{len(tasks)}] {item['instance_id']}")
                records.append(process(item))
        else:
            records = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(process, item): idx for idx, item in enumerate(tasks)}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    done += 1
                    records[idx] = fut.result()
                    click.echo(f"  [{done}/{len(tasks)}] {tasks[idx]['instance_id']}")
            records = [r for r in records if r is not None]

        for record in records:
            append_jsonl(output_file, record)

        _print_summary(output_file, run_id, cond)


@cli.command("summarize")
@click.option("--input", "input_path", required=True, help="Input JSONL path")
@click.option("--run-id", default=None, help="Optional run-id filter")
@click.option("--condition", default=None, help="Optional condition filter")
def summarize(input_path: str, run_id: str | None, condition: str | None) -> None:
    path = Path(input_path)
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id and row.get("run_id") != run_id:
                continue
            if condition and row.get("condition") != condition:
                continue
            rows.append(row)

    summary = summarize_records(rows)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
