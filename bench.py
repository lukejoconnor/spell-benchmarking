#!/usr/bin/env python3
"""Unified benchmark CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, UTC
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    evaluate_mathconstruct_response,
    load_exercism,
    load_mathconstruct,
    load_project_euler,
    load_scienceagentbench,
    load_swebench_verified,
    load_swebench_lite,
    load_swebench_pro,
    load_swebench_verified_mini,
    format_benchmark_prompt,
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
    OneShotAdapter,
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
from src.swebench_pro_eval import (
    check_docker_manifest_access,
    convert_predictions_jsonl_to_scale_json,
    dockerhub_image_for_instance,
    ensure_scale_pro_eval_repo,
    load_raw_samples,
    write_raw_sample_jsonl,
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
SCALE_PRO_EVAL_REPO_DEFAULT = "data/swebench_pro_eval/SWE-bench_Pro-os"
SCALE_PRO_OUTPUT_DIR_DEFAULT = "results/swebench_pro_eval"
SWEBENCH_MIN_FREE_DISK_GB = 32


@dataclass
class ProEvalConfig:
    """Groups ScaleAI SWE-bench Pro evaluation options."""

    eval_repo: str = SCALE_PRO_EVAL_REPO_DEFAULT
    raw_sample_path: str | None = None
    output_dir: str = SCALE_PRO_OUTPUT_DIR_DEFAULT
    dockerhub_username: str = "jefzda"
    strip_trace_artifacts: bool = True
    use_local_docker: bool = True
    block_network: bool = False
    redo: bool = False


SWEBENCH_REMINDER_PREAMBLE = '(quine sys-reminder "Recall: you are writing Spell code, not natural language.")'
SWEBENCH_DOCKER_PREAMBLE = (
    '(quine sys-reminder "Recall: you are writing Spell code, not natural language. '
    'You are working inside a Docker container. All file and system operations MUST use '
    '(io/sh ...) shell commands. Do NOT use io/spit, io/slurp, io/read-file, '
    'io/write-file, io/ls, io/exists?, io/mkdir — they operate on the host, not the '
    'container. Use shell equivalents via io/sh instead.")'
)

# Named SWE-bench intervention variants for experimentation.
# Each variant specifies raw Spell preamble expressions, trailing expression,
# and optional agent override.
SWEBENCH_VARIANTS: dict[str, dict[str, str | None]] = {
    "describe-all": {
        "preamble_expr": SWEBENCH_REMINDER_PREAMBLE,
        "trailing": "'(do (!describe io agents globals) (!extend))",
        "agent": "config/agents/cli.agent.edn",
    },
    "orchestrator": {
        "preamble_expr": (
            SWEBENCH_REMINDER_PREAMBLE
            + ' (quine instruction "You are the orchestrator agent. Your role is to decompose this '
            'problem and delegate ALL work to sub-agents. Do NOT make code edits yourself. '
            'Step 1: Spawn explorer sub-agents using (agents/spawn !llm-self \\"task\\") to '
            'understand the codebase — read relevant files, find the right locations. '
            'Step 2: Synthesize explorer findings into a plan. '
            'Step 3: Spawn worker sub-agents to implement each part of the plan. '
            'Workers should use io/sh, io/read-file, io/write-file for edits. '
            'Use agents/!ask to collect results. Use globals/* for shared state.")'
        ),
        "trailing": "'(do (!describe io agents globals) (!extend))",
        "agent": "config/agents/cli.agent.edn",
    },
    "describe-io": {
        "preamble_expr": SWEBENCH_REMINDER_PREAMBLE,
        "trailing": "'(do (!describe io) (!extend))",
        "agent": None,
    },
    "context-mgmt": {
        "preamble_expr": (
            SWEBENCH_REMINDER_PREAMBLE
            + ' (quine instruction "Actively manage your context window to avoid context bloat '
            'and context rot, which is the primary cause of failure on complex tasks. '
            'After reading large files or getting long shell outputs, use (rethink N \\"reason\\") '
            'to prune stale reasoning and tool results before calling (!extend). '
            'Use (!peek name expr) instead of (!call-now name expr) for large file reads — '
            'peek bindings auto-prune on the next extension unless you persist what you need '
            'with (def keep (subvec peeked-data start end)). '
            'Keep your active context focused on only the information needed for the current step.")'
        ),
        "trailing": "'(do (!describe io) (!extend))",
        "agent": None,
    },
    "fix-loop": {
        "preamble_expr": None,
        "trailing": "'(patterns/fix-loop prompt)",
        "agent": "config/agents/cli.agent.edn",
    },
}


@dataclass(slots=True)
class _SwePrediction:
    instance_id: str
    patch: str


def _run_id(prefix: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def _default_output(run_id: str) -> Path:
    return Path("results") / "unified" / f"{run_id}.jsonl"


def _resolve_trace_dir(run_id: str, trace: bool) -> Path | None:
    if not trace:
        return None
    trace_dir = (Path("traces") / run_id).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def _sanitize_trace_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _trace_dir_for_request(
    base_trace_dir: Path | None,
    condition: str,
    item_id: str,
    *,
    attempt: int | None = None,
) -> Path | None:
    if base_trace_dir is None:
        return None

    path = (
        base_trace_dir
        / _sanitize_trace_segment(condition, "condition")
        / _sanitize_trace_segment(item_id, "item")
    )
    if attempt is not None:
        path = path / f"attempt-{attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _swebench_worktree_tag(condition: str, run_id: str, *, variant: str | None = None) -> str:
    parts = [_sanitize_trace_segment(condition, "condition")]
    if variant:
        parts.append(_sanitize_trace_segment(variant, "variant"))
    parts.append(_sanitize_trace_segment(run_id, "run"))
    return ".".join(parts)


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


GENERAL_WEB_SEARCH_CONDITIONS = frozenset({"claude_code", "codex"})
GAIA_DATASET_PREFIX = "gaia"


def _resolve_web_search_default(item_datasets: set[str], requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return any(dataset == GAIA_DATASET_PREFIX or dataset.startswith(f"{GAIA_DATASET_PREFIX}_") for dataset in item_datasets)


def _general_condition_web_search(condition: str, enabled: bool) -> bool:
    return enabled and condition in GENERAL_WEB_SEARCH_CONDITIONS


def _context_ref(item: BenchmarkItem, condition: str, context_file: str | None) -> str | None:
    if not item.context:
        return None

    n_chars = len(item.context)
    if condition == "one_shot":
        return f"Context:\n{item.context}"
    if not context_file:
        return None
    if condition == "spell":
        return f"Document path: {context_file}\nDocument size: {n_chars:,} chars"
    return f"Read the document at {context_file} ({n_chars:,} chars)"


def _format_general_prompt(item: BenchmarkItem, condition: str, *, web_search: bool = False) -> str:
    context_file = None if condition == "one_shot" else _ensure_context_file(item)
    return format_benchmark_prompt(
        item,
        context_ref=_context_ref(item, condition, context_file),
        web_search=web_search,
    )


def _make_spell_init(
    prompt: str,
    preamble: str | None = None,
    trailing: str | None = None,
    *,
    preamble_expr: str | None = None,
    prompt_preamble: str | None = None,
) -> str:
    # Some benchmark datasets (e.g. Omni-MATH) provide a raw open Spell prefix.
    # Keep those as-is; otherwise wrap NL prompt in the completion init template.
    if prompt.lstrip().startswith("("):
        return prompt
    return build_spell_init(
        prompt,
        preamble=preamble,
        trailing=trailing,
        preamble_expr=preamble_expr,
        prompt_preamble=prompt_preamble,
    )

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
            "Do not install packages or modify project dependencies.",
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

    if item.dataset == "project_euler":
        expected_raw = (item.answer or "").strip()
        if not expected_raw:
            # Some upstream Project Euler rows currently have missing gold answers.
            # Keep these unscored until the source dataset is fixed.
            return response, None

        def _to_decimal(raw: str) -> Decimal | None:
            cleaned = raw.strip().replace(",", "").replace("$", "")
            if cleaned.startswith("+"):
                cleaned = cleaned[1:]
            try:
                dec = Decimal(cleaned)
            except InvalidOperation:
                return None
            return dec if dec.is_finite() else None

        expected_decimal = _to_decimal(expected_raw)
        if expected_decimal is not None:
            answer = extract_numeric_answer(response)
            if answer is None:
                direct_decimal = _to_decimal(response)
                if direct_decimal is None:
                    return None, False
                return response, direct_decimal == expected_decimal

            got_decimal = _to_decimal(answer)
            if got_decimal is None:
                return answer, False

            if got_decimal == got_decimal.to_integral_value():
                try:
                    extracted = str(int(got_decimal))
                except (OverflowError, ValueError):
                    extracted = answer
            else:
                extracted = answer
            return extracted, got_decimal == expected_decimal

        def _normalize_text(raw: str) -> str:
            return "".join(ch.lower() for ch in raw if ch.isalnum())

        answer = response.strip()
        if not answer:
            return None, False
        return answer, _normalize_text(answer) == _normalize_text(expected_raw)

    if item.dataset == "mathconstruct":
        answer, is_correct, _details = evaluate_mathconstruct_response(item, response)
        return answer, bool(is_correct)

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
    if condition == "one_shot":
        return OneShotAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
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
    if condition == "one_shot":
        # One-shot (single LLM call) doesn't need Docker isolation
        return OneShotAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
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


def _default_swebench_pro_raw_sample_path(dataset_name: str, split: str) -> Path:
    dataset_slug = dataset_name.replace("/", "_")
    return BENCHMARK_ROOT / "data" / "swebench_pro_eval" / f"{dataset_slug}_{split}.jsonl"


def _build_swebench_pro_command(
    *,
    raw_sample_path: Path,
    patch_path: Path,
    output_dir: Path,
    max_workers: int,
    dockerhub_username: str,
    use_local_docker: bool,
    block_network: bool,
    redo: bool,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--with",
        "pandas>=1.5.0",
        "--with",
        "tqdm>=4.64.0",
        "--with",
        "docker>=6.0.0",
        "--with",
        "datasets>=2.14.0",
        "python",
        "swe_bench_pro_eval.py",
        "--raw_sample_path",
        str(raw_sample_path),
        "--patch_path",
        str(patch_path),
        "--output_dir",
        str(output_dir),
        "--scripts_dir",
        "run_scripts",
        "--num_workers",
        str(max_workers),
        "--dockerhub_username",
        dockerhub_username,
    ]
    if use_local_docker:
        cmd.append("--use_local_docker")
    if block_network:
        cmd.append("--block_network")
    if redo:
        cmd.append("--redo")
    return cmd


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
    pro: ProEvalConfig,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    conditions = _parse_swebench_conditions(condition)
    resolved_dataset_name = dataset_name or _swebench_dataset_name(dataset)
    pred_root = Path(predictions_dir)

    plan: list[dict[str, Any]] = []
    if dataset == "pro":
        eval_repo_path = Path(pro.eval_repo)
        raw_sample = Path(pro.raw_sample_path) if pro.raw_sample_path else _default_swebench_pro_raw_sample_path(
            resolved_dataset_name, split
        )
        output_root = Path(pro.output_dir) / run_id

        for cond in conditions:
            pred_path = pred_root / f"{cond}_{dataset}_{run_id}.jsonl"
            if verify_predictions and not pred_path.exists():
                raise click.ClickException(f"Predictions file not found: {pred_path}")

            converted_path = output_root / f"{cond}_{dataset}_{run_id}.patches.json"
            condition_output_dir = output_root / cond
            cmd = _build_swebench_pro_command(
                raw_sample_path=raw_sample.resolve(),
                patch_path=converted_path.resolve(),
                output_dir=condition_output_dir.resolve(),
                max_workers=max_workers,
                dockerhub_username=pro.dockerhub_username,
                use_local_docker=pro.use_local_docker,
                block_network=pro.block_network,
                redo=pro.redo,
            )
            plan.append(
                {
                    "mode": "scaleai_pro",
                    "condition": cond,
                    "predictions_path": str(pred_path),
                    "converted_patch_path": str(converted_path),
                    "output_dir": str(condition_output_dir),
                    "eval_repo_path": str(eval_repo_path),
                    "raw_sample_path": str(raw_sample),
                    "dataset_name": resolved_dataset_name,
                    "split": split,
                    "dockerhub_username": pro.dockerhub_username,
                    "strip_trace_artifacts": pro.strip_trace_artifacts,
                    "command": cmd,
                }
            )
        return resolved_dataset_name, conditions, plan

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
            "src.swebench_eval_wrapper",
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
                "mode": "standard",
                "condition": cond,
                "predictions_path": str(pred_path),
                "cache_level": cache_level,
                "clean": clean,
                "command": cmd,
            }
        )

    return resolved_dataset_name, conditions, plan


def _run_swebench_pro_eval_step(step: dict[str, Any], env: dict[str, str]) -> None:
    cond = step["condition"]
    eval_repo_path = Path(step["eval_repo_path"])
    raw_sample_path = Path(step["raw_sample_path"])
    pred_path = Path(step["predictions_path"])
    converted_path = Path(step["converted_patch_path"])
    output_dir = Path(step["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n--- Evaluating {cond} (ScaleAI SWE-bench Pro) ---")

    try:
        eval_repo = ensure_scale_pro_eval_repo(eval_repo_path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if not raw_sample_path.exists():
        click.echo(f"Materializing raw sample metadata: {raw_sample_path}")
        write_raw_sample_jsonl(
            raw_sample_path,
            dataset_name=step["dataset_name"],
            split=step["split"],
        )

    conversion_stats = convert_predictions_jsonl_to_scale_json(
        pred_path,
        converted_path,
        default_prefix=cond,
        strip_trace_artifacts=bool(step.get("strip_trace_artifacts", True)),
    )
    click.echo(
        f"Prepared {conversion_stats['count']} Pro patches -> {converted_path} "
        f"(trace sections removed: {conversion_stats['removed_trace_sections']})"
    )

    raw_samples = load_raw_samples(raw_sample_path)
    converted_payload = json.loads(converted_path.read_text(encoding="utf-8"))
    if not converted_payload:
        raise click.ClickException(f"No converted Pro patches produced from {pred_path}")

    sample_instance_id = converted_payload[0]["instance_id"]
    try:
        image_uri = dockerhub_image_for_instance(
            raw_samples,
            sample_instance_id,
            dockerhub_username=step["dockerhub_username"],
        )
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    ok, err = check_docker_manifest_access(image_uri, env=env)
    if not ok:
        hint = ""
        err_lower = err.lower()
        if "unauthorized" in err_lower or "authentication required" in err_lower:
            hint = " (try `docker login` if these images are private in your environment)"
        raise click.ClickException(
            f"Cannot access Docker image {image_uri}: {err}{hint}"
        )
    click.echo(f"Docker image access check passed: {image_uri}")

    try:
        subprocess.run(step["command"], check=True, cwd=eval_repo, env=env)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"SWE-bench Pro evaluation failed for condition '{cond}' with exit code {exc.returncode}"
        ) from exc

    eval_results_path = output_dir / "eval_results.json"
    if not eval_results_path.exists():
        raise click.ClickException(
            f"SWE-bench Pro evaluation finished but missing {eval_results_path}"
        )

    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    total = len(eval_results)
    resolved = sum(1 for v in eval_results.values() if v)
    accuracy = (resolved / total) if total else 0.0
    click.echo(
        f"Pro eval summary ({cond}): {accuracy:.1%} ({resolved}/{total}) "
        f"[{eval_results_path}]"
    )


def _ensure_swebench_eval_free_disk(path: Path = BENCHMARK_ROOT) -> None:
    _, _, free_bytes = shutil.disk_usage(path)
    min_free_bytes = SWEBENCH_MIN_FREE_DISK_GB * (1024**3)
    if free_bytes < min_free_bytes:
        free_gb = free_bytes / (1024**3)
        raise click.ClickException(
            "Insufficient free disk space for SWE-bench install/eval. "
            f"Detected free space: {free_gb:.1f} GB. "
            f"Required minimum: {SWEBENCH_MIN_FREE_DISK_GB} GB. "
            "Try `docker system df`, then `docker system prune`, "
            "or increase Colima disk size."
        )


def _run_swebench_eval_plan(plan: list[dict[str, Any]]) -> None:
    _ensure_swebench_eval_free_disk()

    env = os.environ.copy()
    colima_sock = Path.home() / ".colima" / "default" / "docker.sock"
    if not env.get("DOCKER_HOST") and not Path("/var/run/docker.sock").exists() and colima_sock.exists():
        env["DOCKER_HOST"] = f"unix://{colima_sock}"
        click.echo(f"Using DOCKER_HOST fallback for evaluation: {env['DOCKER_HOST']}")

    for step in plan:
        if step.get("mode") == "scaleai_pro":
            _run_swebench_pro_eval_step(step, env)
            continue

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
@click.option("--condition", default="one_shot", type=click.Choice(["one_shot", "spell", "claude_code", "codex", "all"]))
@click.option("--model", default=None, help="provider:model for one_shot/spell (required for those conditions)")
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
@click.option(
    "--web-search/--no-web-search",
    default=None,
    help="Enable live web search (defaults to on for GAIA, off otherwise)",
)
@click.option("--trace/--no-trace", default=True, show_default=True)
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
    web_search: bool | None,
    trace: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("general")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_conditions(condition, ("one_shot", "spell", "claude_code", "codex", "all"))

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")

    requested_ids = {x.strip() for x in (items or "").split(",") if x.strip()}

    if dataset == "project_euler":
        if requested_ids:
            requested_ids = {
                f"project_euler_{int(match.group(1)):04d}" if (match := re.match(r"project_euler_(\d+)$", item_id)) else item_id
                for item_id in requested_ids
            }
        requested_numbers: list[int] | None = None
        if requested_ids:
            requested_numbers = []
            for item_id in sorted(requested_ids):
                match = re.match(r"project_euler_(\d+)$", item_id)
                if match:
                    requested_numbers.append(int(match.group(1)))
        loaded = load_project_euler(
            max_items=n if (n and not requested_numbers) else None,
            problem_numbers=requested_numbers,
        )
    elif dataset == "mathconstruct":
        requested_problem_names: list[str] | None = None
        if requested_ids:
            requested_problem_names = []
            for item_id in sorted(requested_ids):
                prefix = "mathconstruct_"
                if item_id.startswith(prefix) and len(item_id) > len(prefix):
                    requested_problem_names.append(item_id[len(prefix):])
        loaded = load_mathconstruct(
            max_items=n if (n and not requested_problem_names) else None,
            problem_names=requested_problem_names,
        )
    elif dataset == "all":
        loaded: list[BenchmarkItem] = []
        for loader in DATASETS.values():
            loaded.extend(loader())
    else:
        loaded = DATASETS[dataset]()

    if requested_ids:
        loaded = [item for item in loaded if item.id in requested_ids]
    if n:
        loaded = loaded[:n]
    item_datasets = {item.dataset for item in loaded}
    effective_web_search = _resolve_web_search_default(item_datasets, web_search)

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
                    "web_search_requested": effective_web_search,
                    "web_search_by_condition": {
                        cond: _general_condition_web_search(cond, effective_web_search) for cond in conditions
                    },
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

    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

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

        condition_web_search = _general_condition_web_search(cond, effective_web_search)
        requests: list[RunRequest] = []
        for item in loaded:
            prompt = _format_general_prompt(item, cond, web_search=condition_web_search)
            requests.append(
                RunRequest(
                    benchmark="general",
                    dataset=dataset,
                    item_id=item.id,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "one_shot": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
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
                    web_search=condition_web_search,
                    trace=trace,
                    trace_dir=(
                        _trace_dir_for_request(trace_dir, cond, item.id)
                        if cond == "spell"
                        else trace_dir
                    ),
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
                    "web_search": condition_web_search,
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
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--trace/--no-trace", default=True, show_default=True, help="Record execution traces")
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
@click.option(
    "--pro-eval-repo",
    default=SCALE_PRO_EVAL_REPO_DEFAULT,
    show_default=True,
    help="Path for local clone of ScaleAI SWE-bench Pro eval assets",
)
@click.option(
    "--pro-raw-sample-path",
    default=None,
    help="Optional CSV/JSONL raw sample metadata path for SWE-bench Pro",
)
@click.option(
    "--pro-output-dir",
    default=SCALE_PRO_OUTPUT_DIR_DEFAULT,
    show_default=True,
    help="Output directory for SWE-bench Pro eval artifacts",
)
@click.option(
    "--pro-dockerhub-username",
    default="jefzda",
    show_default=True,
    help="DockerHub namespace containing sweap-images",
)
@click.option(
    "--pro-strip-trace-artifacts/--no-pro-strip-trace-artifacts",
    default=True,
    show_default=True,
    help="Strip trace artifact files from patches before SWE-bench Pro eval",
)
@click.option(
    "--pro-use-local-docker/--no-pro-use-local-docker",
    default=True,
    show_default=True,
    help="Use local Docker runtime for SWE-bench Pro eval",
)
@click.option(
    "--pro-block-network/--no-pro-block-network",
    default=False,
    show_default=True,
    help="Block network access inside SWE-bench Pro eval containers",
)
@click.option(
    "--pro-redo/--no-pro-redo",
    default=False,
    show_default=True,
    help="Re-run SWE-bench Pro eval even if output files already exist",
)
@click.option(
    "--preamble",
    default=None,
    help="Prompt steering text appended inside the quoted Spell task prompt; raw Spell forms are still accepted for backward compatibility",
)
@click.option(
    "--preamble-expr",
    default=None,
    help="Raw Spell init expression inserted before the trailing expression",
)
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides default '(!extend))")
@click.option("--variant", default=None, type=click.Choice(list(SWEBENCH_VARIANTS.keys())),
              help="Named intervention variant (overrides preamble/trailing/agent)")
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
    web_search: bool,
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
    pro_eval_repo: str,
    pro_raw_sample_path: str | None,
    pro_output_dir: str,
    pro_dockerhub_username: str,
    pro_strip_trace_artifacts: bool,
    pro_use_local_docker: bool,
    pro_block_network: bool,
    pro_redo: bool,
    preamble: str | None,
    preamble_expr: str | None,
    trailing: str | None,
    variant: str | None,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    # Apply named variant overrides (variant takes precedence over manual preamble/trailing).
    if variant:
        vconf = SWEBENCH_VARIANTS[variant]
        preamble_expr = preamble_expr or vconf.get("preamble_expr") or vconf.get("preamble")
        trailing = trailing or vconf.get("trailing")
        agent = agent or vconf.get("agent")
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
                    "web_search": web_search,
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
                    "pro_eval_repo": pro_eval_repo,
                    "pro_raw_sample_path": pro_raw_sample_path,
                    "pro_output_dir": pro_output_dir,
                    "pro_dockerhub_username": pro_dockerhub_username,
                    "pro_strip_trace_artifacts": pro_strip_trace_artifacts,
                    "pro_use_local_docker": pro_use_local_docker,
                    "pro_block_network": pro_block_network,
                    "pro_redo": pro_redo,
                    "preamble": preamble,
                    "trailing": trailing,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First item: {items[0].instance_id}")
        return

    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

    cache = RepoCache(root)

    for cond in conditions:
        click.echo(f"\n--- Running {cond}{' (docker)' if docker else ''} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget, docker=docker)
        worktree_tag = _swebench_worktree_tag(cond, run_id, variant=variant)
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
                wt = setup_swebench_workspace(cache, item, tag=worktree_tag)
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
                    init=(_make_spell_init(
                              prompt,
                              preamble=preamble,
                              preamble_expr=preamble_expr or (
                                  SWEBENCH_DOCKER_PREAMBLE if docker else SWEBENCH_REMINDER_PREAMBLE
                              ),
                              trailing=trailing,
                          ) if cond == "spell" else None),
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=web_search if cond == "codex" else False,
                    trace=trace,
                    trace_dir=(
                        _trace_dir_for_request(trace_dir, cond, item.instance_id)
                        if cond == "spell"
                        else trace_dir
                    ),
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
                    # Diff against base_commit to capture both uncommitted edits
                    # and changes committed inside the worktree (e.g. by fix-loop).
                    patch = wt.diff(item.base_commit)
                    changed_files = wt.changed_files(item.base_commit)

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
                        "web_search": web_search,
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
        pro_config = ProEvalConfig(
            eval_repo=pro_eval_repo,
            raw_sample_path=pro_raw_sample_path,
            output_dir=pro_output_dir,
            dockerhub_username=pro_dockerhub_username,
            strip_trace_artifacts=pro_strip_trace_artifacts,
            use_local_docker=pro_use_local_docker,
            block_network=pro_block_network,
            redo=pro_redo,
        )
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
            pro=pro_config,
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
@click.option(
    "--pro-eval-repo",
    default=SCALE_PRO_EVAL_REPO_DEFAULT,
    show_default=True,
    help="Path for local clone of ScaleAI SWE-bench Pro eval assets",
)
@click.option(
    "--pro-raw-sample-path",
    default=None,
    help="Optional CSV/JSONL raw sample metadata path for SWE-bench Pro",
)
@click.option(
    "--pro-output-dir",
    default=SCALE_PRO_OUTPUT_DIR_DEFAULT,
    show_default=True,
    help="Output directory for SWE-bench Pro eval artifacts",
)
@click.option(
    "--pro-dockerhub-username",
    default="jefzda",
    show_default=True,
    help="DockerHub namespace containing sweap-images",
)
@click.option(
    "--pro-strip-trace-artifacts/--no-pro-strip-trace-artifacts",
    default=True,
    show_default=True,
    help="Strip trace artifact files from patches before SWE-bench Pro eval",
)
@click.option(
    "--pro-use-local-docker/--no-pro-use-local-docker",
    default=True,
    show_default=True,
    help="Use local Docker runtime for SWE-bench Pro eval",
)
@click.option(
    "--pro-block-network/--no-pro-block-network",
    default=False,
    show_default=True,
    help="Block network access inside SWE-bench Pro eval containers",
)
@click.option(
    "--pro-redo/--no-pro-redo",
    default=False,
    show_default=True,
    help="Re-run SWE-bench Pro eval even if output files already exist",
)
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
    pro_eval_repo: str,
    pro_raw_sample_path: str | None,
    pro_output_dir: str,
    pro_dockerhub_username: str,
    pro_strip_trace_artifacts: bool,
    pro_use_local_docker: bool,
    pro_block_network: bool,
    pro_redo: bool,
    dry_run: bool,
) -> None:
    pro_config = ProEvalConfig(
        eval_repo=pro_eval_repo,
        raw_sample_path=pro_raw_sample_path,
        output_dir=pro_output_dir,
        dockerhub_username=pro_dockerhub_username,
        strip_trace_artifacts=pro_strip_trace_artifacts,
        use_local_docker=pro_use_local_docker,
        block_network=pro_block_network,
        redo=pro_redo,
    )
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
        pro=pro_config,
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
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--trace/--no-trace", default=True, show_default=True)
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
    web_search: bool,
    trace: bool,
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
    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

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
                    web_search=web_search if cond == "codex" else False,
                    trace=trace,
                    trace_dir=(
                        _trace_dir_for_request(trace_dir, cond, item["instance_id"])
                        if cond == "spell"
                        else trace_dir
                    ),
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
    for .py files. For one_shot/Spell that return markdown, extracts from code blocks.
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
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--use-knowledge/--no-knowledge", default=False, show_default=True, help="Include domain knowledge in prompt")
@click.option(
    "--preamble",
    default=None,
    help="Prompt steering text appended inside the quoted Spell task prompt; raw Spell forms are still accepted for backward compatibility",
)
@click.option(
    "--preamble-expr",
    default=None,
    help="Raw Spell init expression inserted before the trailing expression",
)
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
    web_search: bool,
    use_knowledge: bool,
    preamble: str | None,
    preamble_expr: str | None,
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
                    init=(
                        _make_spell_init(prompt, preamble=preamble, preamble_expr=preamble_expr)
                        if cond == "spell"
                        else None
                    ),
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=web_search if cond == "codex" else False,
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


def _format_exercism_one_shot_prompt(item: ExercismItem) -> str:
    """Format prompt for one-shot (single LLM call, no tool use)."""
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
@click.option("--condition", default="spell", type=click.Choice(["one_shot", "spell", "claude_code", "codex", "all"]))
@click.option("--difficulty", default=None, help="Difficulty range, e.g. '1-3' or '5'")
@click.option("--slugs", default=None, help="Comma-separated exercise slugs")
@click.option("--n", type=int, default=None, help="Limit to N exercises")
@click.option("--model", default=None, help="provider:model for one_shot/spell (required for those conditions)")
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
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
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
    web_search: bool,
    trace: bool,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    """Run Exercism Python benchmark with two-attempt protocol."""
    run_id = run_id or _run_id("exercism")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_conditions(condition, ("one_shot", "spell", "claude_code", "codex", "all"))

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
                    "web_search": web_search,
                    "trace": trace,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First: {items[0].slug} (d={items[0].difficulty})")
        return

    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
        adapter = _build_adapter(cond, model=model, claude_model=claude_model, codex_model=codex_model, budget=budget)
        spell_resolution = None
        if cond == "spell":
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
                if cond == "one_shot":
                    prompt = _format_exercism_one_shot_prompt(item)
                else:
                    prompt = _format_exercism_prompt(item, ws)

                req = RunRequest(
                    benchmark="exercism",
                    dataset="exercism",
                    item_id=item.slug,
                    condition=cond,
                    prompt=prompt,
                    model_spec={"spell": model, "one_shot": model, "claude_code": claude_model, "codex": codex_model}.get(cond),
                    timeout=timeout,
                    init=_make_spell_init(prompt) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if (spell_resolution and cond == "spell") else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=web_search if cond == "codex" else False,
                    trace=trace if cond == "spell" else False,
                    trace_dir=(
                        _trace_dir_for_request(trace_dir, cond, item.slug, attempt=1)
                        if (cond == "spell" and trace_dir is not None)
                        else None
                    ),
                    workspace=ws if cond != "one_shot" else None,
                )
                resp1 = adapter.run(req)

                # For one_shot, extract code from response and write to file
                if cond == "one_shot" and resp1.response_text:
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
                        web_search=web_search if cond == "codex" else False,
                        trace=trace if cond == "spell" else False,
                        trace_dir=(
                            _trace_dir_for_request(trace_dir, cond, item.slug, attempt=2)
                            if (cond == "spell" and trace_dir is not None)
                            else None
                        ),
                        workspace=ws if cond != "one_shot" else None,
                    )
                    resp2 = adapter.run(req2)

                    if cond == "one_shot" and resp2.response_text:
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
