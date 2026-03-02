#!/usr/bin/env python3
"""Unified benchmark CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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
    ExercismItem,
    ScienceAgentBenchItem,
    SWEBenchItem,
    load_exercism,
    load_scienceagentbench,
    load_swebench_verified,
    load_swebench_lite,
    load_swebench_pro,
    load_swebench_verified_mini,
    format_prompt_baseline,
    format_prompt_spell,
)
from src.evaluation import (
    extract_freeform_answer,
    extract_math_answer,
    extract_mcq_answer,
    extract_numeric_answer,
    gaia_question_scorer,
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
from src.docker_adapters import (
    DockerClaudeCodeAdapter,
    DockerCodexAdapter,
    DockerSpellAdapter,
)
from src.unified_results import (
    UnifiedRunRecord,
    append_jsonl,
    now_iso,
    parse_model_spec,
    summarize_records,
)

BENCHMARK_ROOT = Path(__file__).resolve().parent


def _resolve_spell_root() -> Path:
    env_spell_root = os.environ.get("SPELL_ROOT")
    if env_spell_root:
        return Path(env_spell_root).expanduser().resolve()

    # Nested checkout mode: spell/benchmarking/bench.py
    nested_candidate = BENCHMARK_ROOT.parent
    if (nested_candidate / "config" / "agents").exists():
        return nested_candidate

    # Standalone fallback: leave as benchmark root and let agent resolution
    # surface a precise error if config files are unavailable.
    return BENCHMARK_ROOT


SPELL_ROOT = _resolve_spell_root()
CONTEXT_DIR = BENCHMARK_ROOT / "data" / "contexts"
SWEBENCH_REMINDER_PREAMBLE = '(quine sys-reminder "Recall: you are writing Spell code, not natural language.")'
SWEBENCH_DOCKER_PREAMBLE = (
    '(quine sys-reminder "Recall: you are writing Spell code, not natural language. '
    'You are working inside a Docker container. All file and system operations MUST use '
    '(io/sh ...) shell commands. Do NOT use io/spit, io/slurp, io/read-file, '
    'io/write-file, io/ls, io/exists?, io/mkdir — they operate on the host, not the '
    'container. Use shell equivalents via io/sh instead.")'
)
OMNI_MATH_PREAMBLE = (
    '(quine strategy "Solve the problem using a combination of the following strategies:\\n'
    '- Pure reasoning via think blocks\\n'
    '- Numerical computation using Spell (defn, loop/recur, map/reduce, math/)\\n'
    '- Context pruning to manage your chain of thoughts (rethink + !extend)\\n'
    '- Sub-agent delegation for exploring alternative approaches")'
)


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


def _parse_swebench_conditions(value: str) -> list[str]:
    if value == "both":
        return ["spell", "claude_code"]
    if value == "all":
        return ["spell", "claude_code", "codex"]
    return [value]


def _swebench_dataset_name(dataset: str) -> str:
    mapping = {
        "mini": "princeton-nlp/SWE-bench_Verified",
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "pro": "ScaleAI/SWE-bench_Pro",
    }
    return mapping[dataset]


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
    if item.dataset == "gaia":
        parts.append(item.question.strip())
        file_path = (item.metadata or {}).get('file_path', '')
        if file_path:
            parts.extend(["", f"Attached file: {file_path}"])
        parts.append("")
        parts.append("Use web search, Python, and any tools needed to answer this question.")
        parts.append("Reply with ONLY the final answer — a number, a short string, or a comma-separated list. No explanation.")
    elif item.context and context_file:
        parts.extend(
            [
                f"Read the document at {context_file} ({len(item.context):,} chars) and answer this question:",
                "",
                item.question.strip(),
            ]
        )
    else:
        parts.extend(["Solve this problem:", "", item.question.strip()])

    if item.dataset == "gaia":
        pass  # Already formatted above
    elif item.choices:
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


def _general_preamble(dataset: str) -> str | None:
    return {"omni_math": OMNI_MATH_PREAMBLE}.get(dataset)


def _format_optional_swebench_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, indent=2, ensure_ascii=False).strip()
    except TypeError:
        return str(value).strip()


def _format_swebench_prompt(item: SWEBenchItem) -> str:
    parts = [
        f"Repository: {item.repo}",
        "",
        "## Issue to Fix",
        "",
        item.problem_statement.strip(),
        "",
    ]

    hints_text = (item.hints_text or "").strip()
    if hints_text:
        parts.extend(["## Hints", "", hints_text, ""])

    requirements = _format_optional_swebench_text((item.metadata or {}).get("requirements"))
    if requirements:
        parts.extend(["## Requirements", "", requirements, ""])

    interface = _format_optional_swebench_text((item.metadata or {}).get("interface"))
    if interface:
        parts.extend(["## Interface Contract", "", interface, ""])

    parts.extend(
        [
            "## Instructions",
            "",
            "Fix this bug with minimal code changes.",
            "Do not commit any changes.",
            "Run focused checks/tests when possible.",
        ]
    )
    return "\n".join(parts)


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

    if item.dataset == "gaia":
        # GAIA: use official scorer with normalization
        # Extract answer: last line of response (models put answer last)
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        answer = lines[-1] if lines else response
        # Strip markdown citation links: "34689([url](https://...))" -> "34689"
        answer = re.sub(r'\s*\(\[.*?\]\(.*?\)\)', '', answer).strip()
        # Strip trailing markdown links: "[source](url)" at end
        answer = re.sub(r'\s*\[.*?\]\(.*?\)\s*$', '', answer).strip()
        return answer, gaia_question_scorer(answer, item.answer)

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


def _require_model(condition: str, model: str | None, flag_name: str = "--model") -> str:
    if model is None:
        raise click.ClickException(f"{flag_name} is required for condition '{condition}'")
    return model


def _build_adapter(condition: str, *, model: str | None, claude_model: str | None, codex_model: str, budget: float, docker: bool = False):
    if docker:
        return _build_docker_adapter(condition, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
    if condition == "spell":
        return SpellAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
    if condition == "baseline":
        return BaselineAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
    if condition == "claude_code":
        return ClaudeCodeAdapter(model=_require_model(condition, claude_model, "--claude-model"), budget=budget)
    if condition == "codex":
        return CodexAdapter(project_root=SPELL_ROOT, model_spec=codex_model)
    raise ValueError(f"Unknown condition: {condition}")


def _build_docker_adapter(condition: str, *, model: str | None, claude_model: str | None, codex_model: str, budget: float):
    if condition == "spell":
        return DockerSpellAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
    if condition == "claude_code":
        return DockerClaudeCodeAdapter(model=_require_model(condition, claude_model, "--claude-model"), budget=budget)
    if condition == "codex":
        return DockerCodexAdapter(model_spec=codex_model)
    if condition == "baseline":
        # Baseline (single LLM call) doesn't need Docker isolation
        return BaselineAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
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
    agent_msg: str | None = None,
    agent_tc: str | None = None,
):
    try:
        return resolve_spell_agent(
            project_root=SPELL_ROOT,
            bench_kind=bench_kind,  # type: ignore[arg-type]
            model_spec=model_spec,
            item_datasets=item_datasets,
            agent=agent,
            agent_msg=agent_msg,
            agent_tc=agent_tc,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def _build_swebench_eval_plan(
    *,
    dataset: str,
    condition: str,
    run_id: str,
    predictions_dir: str,
    dataset_name: str | None,
    split: str,
    max_workers: int,
    timeout: int,
    namespace: str,
    instance_image_tag: str,
    env_image_tag: str,
    final_cache_level: str,
    final_clean: bool,
    verify_predictions: bool,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    conditions = _parse_swebench_conditions(condition)
    resolved_dataset_name = dataset_name or _swebench_dataset_name(dataset)
    pred_root = Path(predictions_dir)

    plan: list[dict[str, Any]] = []
    for idx, cond in enumerate(conditions):
        pred_path = pred_root / f"{cond}_{dataset}_{run_id}.jsonl"
        if verify_predictions and not pred_path.exists():
            raise click.ClickException(f"Predictions file not found: {pred_path}")

        is_last = idx == len(conditions) - 1
        cache_level = final_cache_level if is_last else "env"
        clean = final_clean if is_last else False
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            resolved_dataset_name,
            "--split",
            split,
            "--predictions_path",
            str(pred_path),
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout),
            "--run_id",
            run_id,
            "--namespace",
            namespace,
            "--instance_image_tag",
            instance_image_tag,
            "--env_image_tag",
            env_image_tag,
            "--force_rebuild",
            "false",
            "--cache_level",
            cache_level,
            "--clean",
            "true" if clean else "false",
        ]
        plan.append(
            {
                "condition": cond,
                "predictions_path": str(pred_path),
                "cache_level": cache_level,
                "clean": clean,
                "command": cmd,
            }
        )

    return resolved_dataset_name, conditions, plan


def _run_swebench_eval_plan(plan: list[dict[str, Any]]) -> None:
    env = os.environ.copy()
    colima_sock = Path.home() / ".colima" / "default" / "docker.sock"
    if not env.get("DOCKER_HOST") and not Path("/var/run/docker.sock").exists() and colima_sock.exists():
        env["DOCKER_HOST"] = f"unix://{colima_sock}"
        click.echo(f"Using DOCKER_HOST fallback for evaluation: {env['DOCKER_HOST']}")

    for step in plan:
        cond = step["condition"]
        click.echo(
            f"\n--- Evaluating {cond} "
            f"(cache_level={step['cache_level']} clean={step['clean']}) ---"
        )
        try:
            subprocess.run(step["command"], check=True, cwd=BENCHMARK_ROOT, env=env)
        except subprocess.CalledProcessError as exc:
            raise click.ClickException(
                f"swebench evaluation failed for condition '{cond}' with exit code {exc.returncode}"
            ) from exc


@click.group()
def cli() -> None:
    """Unified benchmark runner."""


@cli.command("general")
@click.option("--dataset", default="longbench_short", type=click.Choice(list(DATASETS.keys()) + ["all"]))
@click.option("--condition", default="baseline", type=click.Choice(["baseline", "spell", "claude_code", "codex", "all"]))
@click.option("--model", default=None, help="provider:model for baseline/spell (required for those conditions)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex", help="Model for Codex harness comparator")
@click.option("--n", type=int, default=None)
@click.option("--items", default=None, help="Comma-separated item ids")
@click.option("--parallel", type=int, default=5, show_default=True)
@click.option("--budget", type=float, default=1.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
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
    agent_msg: str | None,
    agent_tc: str | None,
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
                agent_msg=agent_msg,
                agent_tc=agent_tc,
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
                agent_msg=agent_msg,
                agent_tc=agent_tc,
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
                    model_spec={"spell": model, "baseline": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    timeout=timeout,
                    init=_make_spell_init(prompt, preamble=_general_preamble(item.dataset)) if cond == "spell" else None,
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
@click.option("--dataset", default="mini", type=click.Choice(["mini", "lite", "verified", "pro"]))
@click.option("--condition", default="all", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex")
@click.option("--n", type=int, default=None)
@click.option("--items", "item_ids", default=None, help="Comma-separated SWE-bench instance IDs to run")
@click.option("--seed", type=int, default=None, help="Random seed for sampling --n items")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=10.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=1800, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--workspace-root", default=None)
@click.option("--keep-workspaces", is_flag=True, default=False)
@click.option("--docker", is_flag=True, default=False, help="Run agents inside Docker containers for isolation")
@click.option("--trace/--no-trace", default=False, show_default=True, help="Record Spell execution traces")
@click.option("--evaluate/--no-evaluate", default=True, show_default=True, help="Run swebench.harness evaluation after generating predictions")
@click.option("--eval-predictions-dir", default="results/swebench", show_default=True)
@click.option("--eval-dataset-name", default=None, help="Override swebench.harness dataset_name")
@click.option("--eval-split", default="test", show_default=True)
@click.option("--eval-max-workers", type=int, default=4, show_default=True)
@click.option("--eval-timeout", type=int, default=1800, show_default=True)
@click.option(
    "--eval-namespace",
    default="none",
    show_default=True,
    help='Image namespace passed to swebench.harness.run_evaluation ("none" builds locally)',
)
@click.option("--eval-instance-image-tag", default="latest", show_default=True)
@click.option("--eval-env-image-tag", default="latest", show_default=True)
@click.option(
    "--eval-final-cache-level",
    type=click.Choice(["none", "base", "env", "instance"]),
    default="base",
    show_default=True,
)
@click.option("--eval-final-clean/--no-eval-final-clean", default=True, show_default=True)
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
    item_ids: str | None,
    seed: int | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None,
    workspace_root: str | None,
    keep_workspaces: bool,
    docker: bool,
    trace: bool,
    evaluate: bool,
    eval_predictions_dir: str,
    eval_dataset_name: str | None,
    eval_split: str,
    eval_max_workers: int,
    eval_timeout: int,
    eval_namespace: str,
    eval_instance_image_tag: str,
    eval_env_image_tag: str,
    eval_final_cache_level: str,
    eval_final_clean: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("swebench")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_swebench_conditions(condition)

    swebench_loaders = {
        "mini": lambda: load_swebench_verified_mini(50),
        "lite": load_swebench_lite,
        "verified": load_swebench_verified,
        "pro": load_swebench_pro,
    }
    all_items = swebench_loaders[dataset]()

    requested_ids: set[str] | None = None
    if item_ids:
        requested_ids = {x.strip() for x in item_ids.split(",") if x.strip()}
        if not requested_ids:
            raise click.ClickException("--items was provided but no valid instance IDs were parsed")
        before = len(all_items)
        all_items = [item for item in all_items if item.instance_id in requested_ids]
        found_ids = {item.instance_id for item in all_items}
        missing = sorted(requested_ids - found_ids)
        click.echo(f"Filtered items by --items: {len(all_items)}/{before} selected")
        if missing:
            click.echo(f"Warning: {len(missing)} requested IDs were not found in dataset '{dataset}'")
            for miss in missing[:10]:
                click.echo(f"  missing: {miss}")
            if len(missing) > 10:
                click.echo(f"  ... and {len(missing) - 10} more")
    elif n:
        if seed is not None:
            import random as _rng
            rng = _rng.Random(seed)
            all_items = rng.sample(all_items, min(n, len(all_items)))
        else:
            all_items = all_items[:n]
    items = all_items

    root = Path(workspace_root) if workspace_root else get_default_workspace_root()
    root.mkdir(parents=True, exist_ok=True)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Workspace root: {root}")
    if docker:
        click.echo("Docker mode: agents run inside containers")
    click.echo(f"Loaded {len(items)} items")
    if dry_run:
        spell_resolution = None
        if "spell" in conditions:
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="swebench",
                model_spec=model,
                agent=agent,
                agent_msg=agent_msg,
                agent_tc=agent_tc,
            )
        click.echo(
            json.dumps(
                {
                    "dataset": dataset,
                    "conditions": conditions,
                    "model": model,
                    "claude_model": claude_model,
                    "codex_model": codex_model,
                    "items": sorted(requested_ids) if requested_ids else None,
                    "budget": budget,
                    "depth": depth,
                    "trace": trace,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                    "spell_mode": spell_resolution.spell_mode if spell_resolution else None,
                    "agent_source": spell_resolution.source if spell_resolution else None,
                    "agent_policy_key": spell_resolution.policy_key if spell_resolution else None,
                    "parallel": parallel,
                    "evaluate": evaluate,
                    "eval_predictions_dir": eval_predictions_dir,
                    "eval_dataset_name": eval_dataset_name,
                    "eval_split": eval_split,
                    "eval_max_workers": eval_max_workers,
                    "eval_timeout": eval_timeout,
                    "eval_namespace": eval_namespace,
                    "eval_instance_image_tag": eval_instance_image_tag,
                    "eval_env_image_tag": eval_env_image_tag,
                    "eval_final_cache_level": eval_final_cache_level,
                    "eval_final_clean": eval_final_clean,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First item: {items[0].instance_id}")
        return

    cache = RepoCache(root)

    for cond in conditions:
        click.echo(f"\n--- Running {cond}{' (docker)' if docker else ''} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget, docker=docker)
        spell_resolution = None
        if cond == "spell":
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="swebench",
                model_spec=model,
                agent=agent,
                agent_msg=agent_msg,
                agent_tc=agent_tc,
            )
            click.echo(
                f"  spell_mode={spell_resolution.spell_mode} "
                f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
            )

        def process(item) -> UnifiedRunRecord:
            wt = None
            try:
                wt = setup_swebench_workspace(cache, item, tag=cond)
                ws_path = wt.path
                prompt = _format_swebench_prompt(item)

                req = RunRequest(
                    benchmark="swebench",
                    dataset=dataset,
                    item_id=item.instance_id,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    timeout=timeout,
                    init=(_make_spell_init(prompt, SWEBENCH_DOCKER_PREAMBLE if docker else SWEBENCH_REMINDER_PREAMBLE)
                          if cond == "spell" else None),
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    trace=trace if cond == "spell" else False,
                    workspace=ws_path,
                )
                resp = adapter.run(req)

                # In Docker mode, the patch lives inside the container and is
                # returned in resp.metadata by the Docker adapter. On the host
                # (non-Docker), we read it from the worktree.
                if docker and resp.metadata.get("docker"):
                    patch = resp.metadata.get("patch", "")
                    changed_files = resp.metadata.get("changed_files", [])
                else:
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
            except Exception as exc:
                # Keep long swebench runs alive even when one instance has an
                # unreachable base commit or workspace setup failure.
                err_text = str(exc)
                return _mk_record(
                    run_id=run_id,
                    benchmark="swebench",
                    dataset=dataset,
                    split=dataset,
                    item_id=item.instance_id,
                    condition=cond,
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    response=RunResponse(
                        status="error",
                        response_text="",
                        tokens_in=None,
                        tokens_out=None,
                        cost_usd=None,
                        latency_ms=None,
                        error_message=err_text,
                        error_type="workspace_error",
                        metadata={},
                    ),
                    response_text="",
                    patch=None,
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
                if wt is not None and not keep_workspaces:
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
            "spell": model or "unknown",
            "claude_code": f"claude_code:{claude_model or 'unknown'}",
            "codex": codex_model,
        }[cond]
        write_predictions_jsonl(swe_results, pred_path, model_name=pred_model_name)
        click.echo(f"Predictions: {pred_path}")

    if evaluate:
        click.echo("\n=== Running SWE-bench evaluation ===")
        resolved_dataset_name, eval_conditions, plan = _build_swebench_eval_plan(
            dataset=dataset,
            condition=condition,
            run_id=run_id,
            predictions_dir=eval_predictions_dir,
            dataset_name=eval_dataset_name,
            split=eval_split,
            max_workers=eval_max_workers,
            timeout=eval_timeout,
            namespace=eval_namespace,
            instance_image_tag=eval_instance_image_tag,
            env_image_tag=eval_env_image_tag,
            final_cache_level=eval_final_cache_level,
            final_clean=eval_final_clean,
            verify_predictions=True,
        )
        click.echo(
            json.dumps(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "dataset_name": resolved_dataset_name,
                    "conditions": eval_conditions,
                    "namespace": eval_namespace,
                    "plan": plan,
                },
                indent=2,
            )
        )
        _run_swebench_eval_plan(plan)


@cli.command("swebench-eval")
@click.option("--dataset", default="mini", type=click.Choice(["mini", "lite", "verified", "pro"]))
@click.option("--condition", default="all", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--run-id", required=True, help="Run ID used in results/swebench/<condition>_<dataset>_<run-id>.jsonl")
@click.option("--predictions-dir", default="results/swebench", show_default=True)
@click.option("--dataset-name", default=None, help="Override swebench.harness dataset_name")
@click.option("--split", default="test", show_default=True)
@click.option("--max-workers", type=int, default=4, show_default=True)
@click.option("--timeout", type=int, default=1800, show_default=True)
@click.option(
    "--namespace",
    default="none",
    show_default=True,
    help='Image namespace passed to swebench.harness.run_evaluation ("none" builds locally)',
)
@click.option("--instance-image-tag", default="latest", show_default=True)
@click.option("--env-image-tag", default="latest", show_default=True)
@click.option("--final-cache-level", type=click.Choice(["none", "base", "env", "instance"]), default="base", show_default=True)
@click.option("--final-clean/--no-final-clean", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
def run_swebench_eval(
    dataset: str,
    condition: str,
    run_id: str,
    predictions_dir: str,
    dataset_name: str | None,
    split: str,
    max_workers: int,
    timeout: int,
    namespace: str,
    instance_image_tag: str,
    env_image_tag: str,
    final_cache_level: str,
    final_clean: bool,
    dry_run: bool,
) -> None:
    resolved_dataset_name, conditions, plan = _build_swebench_eval_plan(
        dataset=dataset,
        condition=condition,
        run_id=run_id,
        predictions_dir=predictions_dir,
        dataset_name=dataset_name,
        split=split,
        max_workers=max_workers,
        timeout=timeout,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
        final_cache_level=final_cache_level,
        final_clean=final_clean,
        verify_predictions=True,
    )

    click.echo(
        json.dumps(
            {
                "run_id": run_id,
                "dataset": dataset,
                "dataset_name": resolved_dataset_name,
                "conditions": conditions,
                "namespace": namespace,
                "plan": plan,
            },
            indent=2,
        )
    )
    if dry_run:
        return

    _run_swebench_eval_plan(plan)


@cli.command("featurebench")
@click.option("--condition", default="claude_code", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--split", default="lite")
@click.option("--task-ids", type=int, multiple=True)
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=10.0, show_default=True)
@click.option("--agent", default=None)
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=1800, show_default=True)
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
    agent_msg: str | None,
    agent_tc: str | None,
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
                agent_msg=agent_msg,
                agent_tc=agent_tc,
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
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
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


SAB_BENCHMARK_DIR = BENCHMARK_ROOT / "data" / "scienceagentbench" / "benchmark" / "benchmark"


def _format_sab_prompt(item: ScienceAgentBenchItem, workspace: Path, use_knowledge: bool = False) -> str:
    """Format a ScienceAgentBench task as a prompt for code generation.

    Mirrors the prompt structure from the original ScienceAgent but adapted
    for our adapters (Claude Code, Spell, Codex).
    """
    dataset_root = item.dataset_folder_tree.split("\n")[0].strip().lstrip("|-- ").rstrip("/")
    dataset_path = str(workspace / dataset_root)

    parts = [
        "You are an expert Python programming assistant that helps scientist users "
        "to write high-quality code to solve their tasks.",
        "",
        "Write a complete, self-contained Python program and save it as `solution.py` "
        f"in the current directory ({workspace}).",
        "Do not use interactive commands like `!pip install`.",
        "Do not suggest partial code or line changes — write the full program.",
        "",
        "## Task",
        "",
        item.task_inst,
    ]

    if use_knowledge and item.domain_knowledge:
        parts.extend(["", "## Domain Knowledge", "", item.domain_knowledge])

    parts.extend([
        "",
        "## Dataset",
        "",
        f"You can access the dataset at `{dataset_path}`. Here is the directory structure:",
        "```",
        item.dataset_folder_tree,
        "```",
    ])

    if item.dataset_preview:
        preview = item.dataset_preview
        if len(preview) > 8000:
            preview = preview[:8000] + "\n... (preview truncated)"
        parts.extend([
            "",
            "Here are some helpful previews for the dataset file(s):",
            preview,
        ])

    parts.extend([
        "",
        "## Output",
        "",
        f"Save your output to `{item.output_fname}`.",
        "Make sure the output directory exists before writing (use `os.makedirs` if needed).",
    ])

    return "\n".join(parts)


def _extract_python_program(response_text: str | None, workspace: Path | None = None) -> str | None:
    """Extract Python code from agent response or workspace files.

    For agents like Claude Code that write files directly, checks the workspace
    for .py files. For baseline/Spell that return markdown, extracts from code blocks.
    """
    import re

    # First check workspace for written Python files (Claude Code, Codex pattern)
    if workspace and workspace.exists():
        py_files = sorted(workspace.glob("*.py"), key=lambda p: p.stat().st_mtime)
        if py_files:
            # Take the most recently modified .py file
            return py_files[-1].read_text(encoding="utf-8")

    if not response_text:
        return None
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if response looks like pure Python (no markdown), return as-is
    stripped = response_text.strip()
    if stripped.startswith(("import ", "from ", "#", "def ", "class ")):
        return stripped
    return None


def _validate_python_syntax(code: str) -> tuple[bool, str | None]:
    """Check if code is valid Python. Returns (valid, error_message)."""
    import ast
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


@cli.command("scienceagentbench")
@click.option("--condition", default="claude_code", type=click.Choice(["spell", "claude_code", "codex", "both", "all"]))
@click.option("--n", type=int, default=None, help="Limit to N items")
@click.option("--instance-ids", default=None, help="Comma-separated instance IDs")
@click.option("--domain", default=None, help="Filter by domain (partial match)")
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator")
@click.option("--codex-model", default="codex")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=5.0, show_default=True)
@click.option("--agent", default=None)
@click.option("--agent-msg", default=None)
@click.option("--agent-tc", default=None)
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=900, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--use-knowledge/--no-knowledge", default=False, show_default=True, help="Include domain knowledge in prompt")
@click.option("--preamble", default=None, help="Spell init preamble expression (inserted before !extend)")
@click.option("--workspace-root", default="/tmp/sab_ws")
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_scienceagentbench(
    condition: str,
    n: int | None,
    instance_ids: str | None,
    domain: str | None,
    model: str,
    claude_model: str,
    codex_model: str,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None,
    use_knowledge: bool,
    preamble: str | None,
    workspace_root: str,
    run_id: str | None,
    output: str | None,
) -> None:
    """Run ScienceAgentBench: data-driven scientific code generation."""
    import re
    import shutil

    run_id = run_id or _run_id("sab")
    output_file = Path(output) if output else _default_output(run_id)

    if condition == "both":
        conditions = ["spell", "claude_code"]
    elif condition == "all":
        conditions = ["spell", "claude_code", "codex"]
    else:
        conditions = [condition]

    items = load_scienceagentbench(domain=domain)

    if instance_ids:
        ids = {x.strip() for x in instance_ids.split(",") if x.strip()}
        items = [item for item in items if item.instance_id in ids]
    elif n:
        items = items[:n]

    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    # Check for benchmark data
    has_benchmark_data = SAB_BENCHMARK_DIR.exists() and (SAB_BENCHMARK_DIR / "datasets").exists()

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Workspace root: {root}")
    click.echo(f"Items: {len(items)}")
    click.echo(f"Domains: {', '.join(sorted({i.domain for i in items}))}")
    click.echo(f"Benchmark data: {'found' if has_benchmark_data else 'NOT FOUND (syntax-only eval)'}")

    def setup_item_workspace(item: ScienceAgentBenchItem, tag: str) -> Path:
        ws = root / f"sab_{item.instance_id}_{tag}"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        # Copy real dataset files if available, otherwise create placeholders
        dataset_root = item.dataset_folder_tree.split("\n")[0].strip().lstrip("|-- ").rstrip("/")
        src_data = SAB_BENCHMARK_DIR / "datasets" / dataset_root
        if has_benchmark_data and src_data.exists():
            # Symlink to save disk space; fall back to copy if symlink fails
            try:
                os.symlink(src_data, ws / dataset_root)
            except OSError:
                shutil.copytree(src_data, ws / dataset_root)
        else:
            for line in item.dataset_folder_tree.split("\n"):
                line = line.strip().lstrip("|").strip().lstrip("-").strip()
                if not line:
                    continue
                path = ws / line.rstrip("/")
                if line.endswith("/"):
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
        # Create output directory
        out_dir = ws / Path(item.output_fname).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return ws

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond == "spell":
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="general",
                model_spec=model,
                agent=agent,
                agent_msg=agent_msg,
                agent_tc=agent_tc,
            )
            click.echo(
                f"  spell_mode={spell_resolution.spell_mode} "
                f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
            )

        def process(item: ScienceAgentBenchItem) -> UnifiedRunRecord:
            import subprocess as _sp
            ws = setup_item_workspace(item, cond)
            try:
                prompt = _format_sab_prompt(item, ws, use_knowledge=use_knowledge)

                req = RunRequest(
                    benchmark="scienceagentbench",
                    dataset="scienceagentbench",
                    item_id=item.instance_id,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    timeout=timeout,
                    init=_make_spell_init(prompt, preamble=preamble) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    workspace=ws,
                )
                resp = adapter.run(req)

                # Extract Python program from response
                program = _extract_python_program(resp.response_text, workspace=ws)
                syntax_valid = False
                syntax_error = None
                if program:
                    syntax_valid, syntax_error = _validate_python_syntax(program)
                    # Save program to workspace
                    pred_path = ws / f"pred_{item.gold_program_name}"
                    pred_path.write_text(program, encoding="utf-8")

                # Execution-based evaluation
                exec_success = False
                output_produced = False
                exec_error = None
                if program and syntax_valid and has_benchmark_data:
                    # Run the program in the workspace
                    pred_path = ws / f"pred_{item.gold_program_name}"
                    try:
                        exec_result = _sp.run(
                            ["uv", "run", "python3", str(pred_path)],
                            cwd=ws, capture_output=True, timeout=900,
                        )
                        if exec_result.returncode == 0:
                            exec_success = True
                        else:
                            exec_error = exec_result.stderr.decode("utf-8", errors="replace")[-500:]
                    except _sp.TimeoutExpired:
                        exec_error = "Program execution timed out (900s)"

                    # Check if output file was produced
                    output_path = ws / item.output_fname
                    output_produced = output_path.exists() and output_path.stat().st_size > 0

                status = resp.status
                if status == "ok":
                    if not program:
                        status = "no_program"
                    elif not syntax_valid:
                        status = "syntax_error"
                    elif has_benchmark_data:
                        if output_produced:
                            status = "output_produced"
                        elif exec_success:
                            status = "exec_ok_no_output"
                        else:
                            status = "exec_failed"
                    else:
                        status = "program_generated"

                # correct = output was produced (best we can check without eval scripts)
                correct = output_produced if has_benchmark_data else (syntax_valid if program else False)

                return _mk_record(
                    run_id=run_id,
                    benchmark="scienceagentbench",
                    dataset="scienceagentbench",
                    split=None,
                    item_id=item.instance_id,
                    condition=cond,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=RunResponse(
                        status=status,
                        response_text=resp.response_text,
                        tokens_in=resp.tokens_in,
                        tokens_out=resp.tokens_out,
                        cost_usd=resp.cost_usd,
                        latency_ms=resp.latency_ms,
                        error_message=resp.error_message or syntax_error or exec_error,
                        error_type=resp.error_type,
                        metadata=resp.metadata,
                    ),
                    response_text=resp.response_text,
                    patch=program,
                    answer_extracted=None,
                    correct=correct,
                    metadata={
                        "domain": item.domain,
                        "subtask_categories": item.subtask_categories,
                        "output_fname": item.output_fname,
                        "gold_program_name": item.gold_program_name,
                        "syntax_valid": syntax_valid,
                        "program_extracted": program is not None,
                        "exec_success": exec_success,
                        "output_produced": output_produced,
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
                # Keep workspaces for inspection during pilot
                pass

        if parallel <= 1:
            records = []
            for i, item in enumerate(items):
                click.echo(f"  [{i+1}/{len(items)}] #{item.instance_id} {item.domain[:30]}: {item.task_inst[:50]}...")
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
                    click.echo(f"  [{done}/{len(items)}] #{items[idx].instance_id}")
            records = [r for r in records if r is not None]

        for record in records:
            append_jsonl(output_file, record)

        # Print summary
        _print_summary(output_file, run_id, cond)

        # Save extracted programs for potential evaluation
        pred_dir = Path("results") / "scienceagentbench" / f"{cond}_{run_id}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for item, record in zip(items, records):
            if record.patch:
                pred_path = pred_dir / f"pred_{item.gold_program_name}"
                pred_path.write_text(record.patch, encoding="utf-8")
        click.echo(f"  Programs saved to: {pred_dir}")


EXERCISM_PYTEST_TIMEOUT = 60


def _setup_exercism_workspace(item: ExercismItem, tag: str) -> Path:
    """Create a temp directory with exercise files. Returns workspace path."""
    import tempfile
    ws = Path(tempfile.mkdtemp(prefix=f"exercism-{item.slug}-{tag}-"))
    (ws / item.stub_file).write_text(item.stub_content, encoding="utf-8")
    (ws / item.test_file).write_text(item.test_content, encoding="utf-8")
    for ef in item.extra_files:
        (ws / ef["name"]).write_text(ef["content"], encoding="utf-8")
    return ws


def _cleanup_exercism_workspace(ws: Path) -> None:
    import shutil
    if ws.exists():
        shutil.rmtree(ws)


def _format_exercism_prompt(item: ExercismItem, ws: Path) -> str:
    stub_path = ws / item.stub_file
    test_path = ws / item.test_file
    return (
        "You are solving a Python programming exercise.\n\n"
        "## Instructions\n\n"
        f"{item.instructions}\n\n"
        "## Task\n\n"
        f"The solution file is at {stub_path}. Edit it to implement the solution.\n\n"
        "Current contents:\n"
        f"```python\n{item.stub_content}```\n\n"
        f"Read the test file at {test_path} before starting.\n"
        f"Verify with: uv tool run pytest {test_path} -v\n\n"
        "Only use standard libraries. Do not modify the test file."
    )


def _format_exercism_retry_prompt(test_output: str, ws: Path, item: ExercismItem) -> str:
    stub_path = ws / item.stub_file
    test_path = ws / item.test_file
    return (
        "The tests failed with this output:\n\n"
        f"```\n{test_output}```\n\n"
        "The tests are correct — do not modify them. "
        f"Fix the code in {stub_path} to resolve the errors.\n"
        f"Verify with: uv tool run pytest {test_path} -v"
    )


def _format_exercism_baseline_prompt(item: ExercismItem) -> str:
    """Format prompt for baseline (single LLM call, no tool use)."""
    return (
        "You are solving a Python programming exercise.\n\n"
        "## Instructions\n\n"
        f"{item.instructions}\n\n"
        "## Stub\n\n"
        f"```python\n{item.stub_content}```\n\n"
        "## Tests\n\n"
        f"```python\n{item.test_content}```\n\n"
        "Write the complete solution. Return ONLY the Python code, no explanation."
    )


def _run_exercism_pytest(test_path: Path, timeout: int = EXERCISM_PYTEST_TIMEOUT) -> tuple[bool, str]:
    """Run pytest on the test file. Returns (passed, output)."""
    try:
        proc = subprocess.run(
            ["uv", "tool", "run", "pytest", str(test_path), "-v"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: pytest exceeded {timeout} seconds"


@cli.command("exercism")
@click.option("--condition", default="spell", type=click.Choice(["baseline", "spell", "claude_code", "codex", "all"]))
@click.option("--difficulty", default=None, help="Difficulty range, e.g. '1-3' or '5'")
@click.option("--slugs", default=None, help="Comma-separated exercise slugs")
@click.option("--n", type=int, default=None, help="Limit to N exercises")
@click.option("--model", default=None, help="provider:model for baseline/spell (required for those conditions)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex", help="Model for Codex harness comparator")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=2.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=8, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=300, show_default=True)
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--no-retry", is_flag=True, default=False, help="Skip retry on test failure")
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None, help="JSONL output path")
def run_exercism(
    condition: str,
    difficulty: str | None,
    slugs: str | None,
    n: int | None,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int | None,
    timeout: int,
    reasoning_effort: str | None,
    no_retry: bool,
    trace: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    """Run Exercism Python benchmark with two-attempt protocol."""
    run_id = run_id or _run_id("exercism")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_conditions(condition, ("baseline", "spell", "claude_code", "codex", "all"))

    # Parse difficulty
    diff_min, diff_max = None, None
    if difficulty:
        parts = difficulty.split("-")
        if len(parts) == 2:
            diff_min, diff_max = int(parts[0]), int(parts[1])
        else:
            diff_min = diff_max = int(parts[0])

    slug_set = {s.strip() for s in slugs.split(",") if s.strip()} if slugs else None

    items = load_exercism(
        difficulty_min=diff_min,
        difficulty_max=diff_max,
        slugs=slug_set,
        max_items=n,
    )

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Exercises: {len(items)}")
    click.echo(f"Retry: {'disabled' if no_retry else 'enabled'}")

    if dry_run:
        spell_resolution = None
        if "spell" in conditions:
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="exercism",
                model_spec=model,
                agent=agent,
                agent_msg=agent_msg,
                agent_tc=agent_tc,
            )
        click.echo(
            json.dumps(
                {
                    "conditions": conditions,
                    "model": model,
                    "claude_model": claude_model,
                    "codex_model": codex_model,
                    "budget": budget,
                    "depth": depth,
                    "no_retry": no_retry,
                    "trace": trace,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First: {items[0].slug} (d={items[0].difficulty})")
        return

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond in ("spell", "baseline"):
            spell_resolution = _resolve_spell_agent_or_die(
                bench_kind="exercism",
                model_spec=model,
                agent=agent,
                agent_msg=agent_msg,
                agent_tc=agent_tc,
            )
            if cond == "spell":
                click.echo(
                    f"  spell_mode={spell_resolution.spell_mode} "
                    f"agent={spell_resolution.agent_file} source={spell_resolution.source}"
                )

        def process(item: ExercismItem) -> UnifiedRunRecord:
            ws = _setup_exercism_workspace(item, cond)
            try:
                test_path = ws / item.test_file
                total_cost = 0.0
                total_tokens_in = 0
                total_tokens_out = 0
                attempt = 0
                passed = False

                # === Attempt 1 ===
                attempt = 1
                if cond == "baseline":
                    prompt = _format_exercism_baseline_prompt(item)
                else:
                    prompt = _format_exercism_prompt(item, ws)

                req = RunRequest(
                    benchmark="exercism",
                    dataset="exercism",
                    item_id=item.slug,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "baseline": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    timeout=timeout,
                    init=_make_spell_init(prompt) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if (spell_resolution and cond == "spell") else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    trace=trace if cond == "spell" else False,
                    workspace=ws if cond != "baseline" else None,
                )
                resp1 = adapter.run(req)

                # For baseline, extract code from response and write to file
                if cond == "baseline" and resp1.response_text:
                    code = _extract_python_program(resp1.response_text)
                    if code:
                        (ws / item.stub_file).write_text(code, encoding="utf-8")

                total_cost += resp1.cost_usd or 0.0
                total_tokens_in += resp1.tokens_in or 0
                total_tokens_out += resp1.tokens_out or 0

                passed, test_output = _run_exercism_pytest(test_path)

                # === Attempt 2 (retry) ===
                resp2 = None
                if not passed and not no_retry:
                    attempt = 2
                    retry_prompt = _format_exercism_retry_prompt(test_output, ws, item)

                    req2 = RunRequest(
                        benchmark="exercism",
                        dataset="exercism",
                        item_id=item.slug,
                        condition=cond,
                        prompt=retry_prompt,
                        model_spec=req.model_spec,
                        timeout=timeout,
                        init=_make_spell_init(retry_prompt) if cond == "spell" else None,
                        agent_file=req.agent_file,
                        budget=budget,
                        depth=depth,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        trace=trace if cond == "spell" else False,
                        workspace=ws if cond != "baseline" else None,
                    )
                    resp2 = adapter.run(req2)

                    if cond == "baseline" and resp2.response_text:
                        code = _extract_python_program(resp2.response_text)
                        if code:
                            (ws / item.stub_file).write_text(code, encoding="utf-8")

                    total_cost += resp2.cost_usd or 0.0
                    total_tokens_in += resp2.tokens_in or 0
                    total_tokens_out += resp2.tokens_out or 0

                    passed, test_output = _run_exercism_pytest(test_path)

                last_resp = resp2 if resp2 else resp1
                latency_ms = (resp1.latency_ms or 0.0) + ((resp2.latency_ms or 0.0) if resp2 else 0.0)

                status = "passed" if passed else last_resp.status
                if not passed and not last_resp.error_message:
                    status = "tests_failed"

                return _mk_record(
                    run_id=run_id,
                    benchmark="exercism",
                    dataset="exercism",
                    split=None,
                    item_id=item.slug,
                    condition=cond,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=RunResponse(
                        status=status,
                        response_text=last_resp.response_text,
                        tokens_in=total_tokens_in if total_tokens_in else None,
                        tokens_out=total_tokens_out if total_tokens_out else None,
                        cost_usd=total_cost if total_cost else None,
                        latency_ms=latency_ms,
                        error_message=last_resp.error_message,
                        error_type=last_resp.error_type,
                        metadata={**last_resp.metadata, "test_output": test_output[-2000:]},
                    ),
                    response_text=last_resp.response_text,
                    patch=None,
                    answer_extracted=None,
                    correct=passed,
                    metadata={
                        "difficulty": item.difficulty,
                        "attempt": attempt,
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
                _cleanup_exercism_workspace(ws)

        if parallel <= 1:
            records = []
            for i, item in enumerate(items):
                click.echo(f"  [{i+1}/{len(items)}] {item.slug} (d={item.difficulty})")
                record = process(item)
                records.append(record)
                mark = "PASS" if record.correct else "FAIL"
                cost = record.cost_usd or 0.0
                click.echo(f"    {mark} (${cost:.4f})")
        else:
            records = [None] * len(items)
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(process, item): idx for idx, item in enumerate(items)}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    done += 1
                    records[idx] = fut.result()
                    item = items[idx]
                    mark = "PASS" if records[idx].correct else "FAIL"
                    click.echo(f"  [{done}/{len(items)}] {item.slug} (d={item.difficulty}) {mark}")
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
