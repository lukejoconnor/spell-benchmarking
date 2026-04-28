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
    load_tasks,
    mask_functions,
    setup_workspace,
)
from src.swebench_output import write_predictions_jsonl
from src.swebench_container import (
    SWEBenchContainerSetupError,
    SWEBenchEnvImageCacheConfig,
    prewarm_swebench_env_images,
    remove_swebench_instance_image,
)
from src.workspace import RepoCache
from src.unified_exercism import format_unified_exercism_prompt, run_unified_exercism
from src.unified_featurebench import run_unified_featurebench
from src.unified_swebench import run_unified_swebench
from src.unified_adapters import (
    OneShotAdapter,
    ClaudeCodeAdapter,
    CodexAdapter,
    SweAgentAdapter,
    RunRequest,
    RunResponse,
    SpellAdapter,
    build_spell_init,
    run_requests_parallel,
)
from src.provenance import BENCHMARK_ROOT, benchmark_git_versions, resolve_spell_root
from src.reasoning_effort import effective_reasoning_effort
from src.usage_accounting import sum_usages, usage_record_fields
from src.unified_results import (
    UnifiedRunRecord,
    append_jsonl,
    load_jsonl,
    now_iso,
    parse_model_spec,
    summarize_records,
)
from src.terminalbench_runner import (
    TERMINALBENCH_DATASET,
    build_terminalbench_command,
    parse_terminalbench_conditions,
    prepare_terminalbench_env,
    prune_terminalbench_docker,
    resolve_terminalbench_dataset_spec,
    resolve_terminalbench_model,
    terminalbench_condition_run_id,
    terminalbench_output_file,
    terminalbench_records_from_run,
    terminalbench_run_dir,
)
from src.harbor_terminalbench_runner import (
    HARBOR_TERMINALBENCH_DATASET,
    HarborTerminalBenchComplianceError,
    build_harbor_terminalbench_commands,
)
from src.swebench_pro_eval import (
    check_docker_manifest_access,
    convert_predictions_jsonl_to_scale_json,
    dockerhub_image_for_instance,
    ensure_scale_pro_eval_repo,
    load_raw_samples,
    write_raw_sample_jsonl,
)

SPELL_ROOT = resolve_spell_root()
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


MATH_TRAILING = "'(!describe math)"

@dataclass(slots=True)
class _SwePrediction:
    instance_id: str
    patch: str


def _make_swebench_env_cache_config(
    repository: str | None,
    *,
    read_only: bool,
) -> SWEBenchEnvImageCacheConfig | None:
    if not repository:
        return None
    return SWEBenchEnvImageCacheConfig(repository=repository, write_enabled=not read_only)

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


def _trace_dir_for_condition(
    base_trace_dir: Path | None,
    condition: str,
    *,
    name: str | None = None,
) -> Path | None:
    if base_trace_dir is None:
        return None

    path = base_trace_dir / _sanitize_trace_segment(condition, "condition")
    if name:
        path = path / _sanitize_trace_segment(name, "name")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trace_dir_for_request(
    base_trace_dir: Path | None,
    condition: str,
    item_id: str,
    *,
    name: str | None = None,
    attempt: int | None = None,
) -> Path | None:
    path = _trace_dir_for_condition(base_trace_dir, condition, name=name)
    if path is None:
        return None

    path = path / _sanitize_trace_segment(item_id, "item")
    if attempt is not None:
        path = path / f"attempt-{attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_conditions(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value == "all":
        return [c for c in allowed if c != "all"]
    return [value]


def _parse_swebench_conditions(value: str) -> list[str]:
    if value == "both":
        return ["spell", "claude_code"]
    if value == "all":
        return ["spell", "claude_code", "codex", "swe_agent"]
    return [value]


def _parse_terminalbench_conditions_option(values: tuple[str, ...]) -> list[str]:
    if not values:
        return parse_terminalbench_conditions(["all"])
    return parse_terminalbench_conditions(list(values))


def _cleanup_swebench_instance_images(image_names: set[str]) -> None:
    if not image_names:
        return

    click.echo(f"Cleaning up {len(image_names)} SWE-bench instance image(s)")
    for image_name in sorted(image_names):
        if not remove_swebench_instance_image(image_name):
            click.echo(f"Warning: failed to remove SWE-bench instance image {image_name}")


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


def _general_context_file(condition: str, item: BenchmarkItem) -> str | None:
    if condition == "one_shot":
        return None
    return _ensure_context_file(item)


def _format_general_prompt(item: BenchmarkItem, condition: str) -> str:
    return format_benchmark_prompt(item, context_file=_general_context_file(condition, item))


def _make_spell_init(prompt: str, trailing: str | None = None) -> str:
    # Some benchmark datasets provide a raw open Spell prefix. Keep those as-is.
    if prompt.lstrip().startswith("("):
        return prompt
    return build_spell_init(prompt, trailing)


_MATH_DATASETS = frozenset({
    "gsm8k", "math", "aime", "hmmt", "omni_math", "mathconstruct", "project_euler",
})

# Per-dataset default timeouts (seconds).  Datasets not listed use _DEFAULT_TIMEOUT.
_DEFAULT_TIMEOUT = 300
_DATASET_TIMEOUTS: dict[str, int] = {
    # General benchmarks
    "omni_math":        3600,
    "omni_math_hard":   3600,
    "browsecomp_plus":  1800,
    "browsecomp_plus_pilot": 1800,
    "aime_2025":         600,
    "aime_2026":         600,
    "hmmt_feb_2025":     600,
    "hmmt_nov_2025":     600,
    "project_euler":     600,
    "mathconstruct":     600,
    # Coding benchmarks
    "swebench":         1800,
    "swebench_pro":    14400,
    "swebench_eval":    1800,
    "featurebench":     1800,
    "exercism":          300,
    "scienceagentbench": 900,
}


def _general_spell_trailing(dataset: str) -> str | None:
    if dataset in _MATH_DATASETS:
        return MATH_TRAILING
    return None


def _condition_label(condition: str, name: str | None = None) -> str:
    return f"{condition}/{name}" if name else condition


def _swebench_predictions_filename(condition: str, dataset: str, run_id: str, *, name: str | None = None) -> str:
    parts = [condition]
    if name:
        parts.append(_sanitize_trace_segment(name, "name"))
    parts.extend([dataset, run_id])
    return "_".join(parts) + ".jsonl"


def _swebench_eval_run_id(run_id: str, *, name: str | None = None) -> str:
    if not name:
        return run_id
    return f"{run_id}__{_sanitize_trace_segment(name, 'name')}"


def _swebench_converted_patch_filename(condition: str, dataset: str, run_id: str, *, name: str | None = None) -> str:
    return _swebench_predictions_filename(condition, dataset, run_id, name=name).removesuffix(".jsonl") + ".patches.json"


def _validate_general_configuration(
    dataset: str,
    conditions: list[str],
    item_datasets: set[str],
    *,
    web_search_enabled: bool,
) -> None:
    if "browsecomp_plus" in item_datasets and "one_shot" in conditions:
        raise click.ClickException(
            "BrowseComp-Plus does not support `one_shot`: that condition inlines the full "
            "document pack into the prompt instead of giving the model file access. "
            "Use `spell`, `claude_code`, or `codex` so the 1000-document context stays in an external file."
        )

    uses_gaia = dataset == GAIA_DATASET_PREFIX or any(
        item_dataset == GAIA_DATASET_PREFIX or item_dataset.startswith(f"{GAIA_DATASET_PREFIX}_")
        for item_dataset in item_datasets
    )
    if not uses_gaia:
        return
    if not web_search_enabled:
        raise click.ClickException("GAIA requires web search. Re-run without `--no-web-search`.")
    unsupported = [cond for cond in conditions if cond not in GENERAL_WEB_SEARCH_CONDITIONS]
    if unsupported:
        unsupported_list = ", ".join(unsupported)
        raise click.ClickException(
            "GAIA currently supports only web-enabled conditions "
            f"({', '.join(sorted(GENERAL_WEB_SEARCH_CONDITIONS))}); got unsupported condition(s): {unsupported_list}."
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


def _format_swebench_prompt(item: SWEBenchItem, workspace: str | Path | None = None) -> str:
    location = str(workspace) if workspace else "/testbed"

    # Build PR description from item fields.
    pr_parts = [item.problem_statement.strip()]
    hints_text = (item.hints_text or "").strip()
    if hints_text:
        pr_parts.extend(["", "## Hints", "", hints_text])
    requirements = _format_optional_swebench_text((item.metadata or {}).get("requirements"))
    if requirements:
        pr_parts.extend(["", "## Requirements", "", requirements])
    interface = _format_optional_swebench_text((item.metadata or {}).get("interface"))
    if interface:
        pr_parts.extend(["", "## Interface Contract", "", interface])
    pr_description = "\n".join(pr_parts)

    return f"""\
<uploaded_files>
{location}
</uploaded_files>
I've uploaded a python code repository in the directory {location} (not in /tmp/inputs). Consider the following PR description:

<pr_description>
{pr_description}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!

Your task is to make the minimal changes to non-tests files in the {location} directory to ensure the <pr_description> is satisfied.

Follow these steps to resolve the issue:
1. As a first step, it would be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
6. Run select tests from the repo to make sure that your fix doesn't break anything else.

Platform: Linux (Debian-based).

TIPS:
- You must make changes in the {location} directory in order to ensure the requirements specified in the <pr_description> are met. Leaving the directory unchanged is not a valid solution.
- When you need to find information about the codebase, use "grep" and "find" to search for relevant files and code with the bash tool
- Use your bash tool to set up any necessary environment variables, such as those needed to run tests.
"""


def _swebench_prompt_workspace(item: SWEBenchItem) -> Path:
    if (item.raw_instance or {}).get("dockerhub_tag"):
        return Path("/app")
    return Path("/testbed")


def _evaluate_general(item: BenchmarkItem, response_text: str | None) -> tuple[str | None, bool | None, str | None]:
    response = (response_text or "").strip()
    if not response:
        return None, False, None

    if item.choices:
        answer = extract_mcq_answer(response)
        return answer, (answer == item.answer) if answer else False, None

    if item.dataset == "gsm8k":
        answer = extract_numeric_answer(response)
        if answer is None:
            return None, False, None
        return answer, normalize_numeric(answer) == normalize_numeric(item.answer), None

    if item.dataset == "project_euler":
        expected_raw = (item.answer or "").strip()
        if not expected_raw:
            # Some upstream Project Euler rows currently have missing gold answers.
            # Keep these unscored until the source dataset is fixed.
            return response, None, None

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
                    return None, False, None
                return response, direct_decimal == expected_decimal, None

            got_decimal = _to_decimal(answer)
            if got_decimal is None:
                return answer, False, None

            if got_decimal == got_decimal.to_integral_value():
                try:
                    extracted = str(int(got_decimal))
                except (OverflowError, ValueError):
                    extracted = answer
            else:
                extracted = answer
            return extracted, got_decimal == expected_decimal, None

        def _normalize_text(raw: str) -> str:
            return "".join(ch.lower() for ch in raw if ch.isalnum())

        answer = response.strip()
        if not answer:
            return None, False, None
        return answer, _normalize_text(answer) == _normalize_text(expected_raw), None

    if item.dataset == "mathconstruct":
        answer, is_correct, _details = evaluate_mathconstruct_response(item, response)
        return answer, bool(is_correct), None

    if item.dataset in ("math", "aime", "hmmt", "omni_math"):
        answer = extract_math_answer(response)
        if answer is None:
            return None, False, None
        return answer, math_equivalent(answer, item.answer), None

    if item.dataset == "gaia":
        # GAIA: use official scorer with normalization
        # Extract answer: last line of response (models put answer last)
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        answer = lines[-1] if lines else response
        # Strip markdown citation links: "34689([url](https://...))" -> "34689"
        answer = re.sub(r'\s*\(\[.*?\]\(.*?\)\)', '', answer).strip()
        # Strip trailing markdown links: "[source](url)" at end
        answer = re.sub(r'\s*\[.*?\]\(.*?\)\s*$', '', answer).strip()
        return answer, gaia_question_scorer(answer, item.answer), None

    if item.dataset == "browsecomp_plus":
        from src.browsecomp_plus import extract_exact_answer, judge_response
        extracted = extract_exact_answer(response)
        outcome = judge_response(
            question=item.question,
            response=response,
            correct_answer=item.answer,
            judge_model=os.environ.get("BROWSECOMP_PLUS_JUDGE_MODEL", "gpt-4.1-mini"),
        )
        judge_extracted = outcome.extracted or extracted
        if outcome.correct is None:
            return judge_extracted, None, outcome.error or "BrowseComp judge returned an unparseable response"
        return judge_extracted, outcome.correct, None

    answer = extract_freeform_answer(response)
    return answer, answer.strip().lower() == item.answer.strip().lower(), None


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
    name: str | None = None,
    model_spec: str | None,
    agent_file: str | None,
    response: RunResponse,
    response_text: str | None,
    patch: str | None,
    answer_extracted: str | None,
    correct: bool | None,
    metadata: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> UnifiedRunRecord:
    provider, model = _provider_model_for_condition(condition, model_spec)
    recorded_reasoning_effort = effective_reasoning_effort(condition, model_spec, reasoning_effort)
    usage_fields = usage_record_fields(response.usage)
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
        uncached_input_tokens=usage_fields["uncached_input_tokens"],
        cached_input_tokens=usage_fields["cached_input_tokens"],
        cache_write_input_tokens=usage_fields["cache_write_input_tokens"],
        visible_output_tokens=usage_fields["visible_output_tokens"],
        reasoning_output_tokens=usage_fields["reasoning_output_tokens"],
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        error_type=response.error_type,
        error_message=response.error_message,
        metadata={**(metadata or {}), **response.metadata, "versions": benchmark_git_versions()},
        name=name,
        reasoning_effort=recorded_reasoning_effort,
    )


def _rewrap_response(
    response: RunResponse,
    *,
    status: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunResponse:
    return RunResponse(
        status=status or response.status,
        response_text=response.response_text,
        usage=response.usage,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        error_message=response.error_message,
        error_type=response.error_type,
        metadata=metadata if metadata is not None else response.metadata,
    )


def _mark_evaluation_error(
    response: RunResponse,
    *,
    error_type: str,
    error_message: str,
    metadata: dict[str, Any] | None = None,
) -> RunResponse:
    rewritten = _rewrap_response(response, status="error", metadata=metadata)
    rewritten.error_type = error_type
    rewritten.error_message = error_message
    return rewritten


def _require_model(condition: str, model: str | None, flag_name: str = "--model") -> str:
    if model is None:
        raise click.ClickException(f"{flag_name} is required for condition '{condition}'")
    return model


def _condition_model_spec(
    condition: str,
    *,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
) -> str | None:
    return {
        "spell": model,
        "one_shot": model,
        "claude_code": claude_model,
        "codex": codex_model,
        "swe_agent": swe_agent_model,
    }.get(condition)


def _build_adapter(
    condition: str,
    *,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
    budget: float,
):
    if condition == "spell":
        return SpellAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
    if condition == "one_shot":
        return OneShotAdapter(project_root=SPELL_ROOT, model_spec=_require_model(condition, model))
    if condition == "claude_code":
        return ClaudeCodeAdapter(model=_require_model(condition, claude_model, "--claude-model"), budget=budget)
    if condition == "codex":
        return CodexAdapter(project_root=SPELL_ROOT, model_spec=codex_model)
    if condition == "swe_agent":
        return SweAgentAdapter(model=_require_model(condition, swe_agent_model, "--swe-agent-model"), budget=budget)
    raise ValueError(f"Unknown condition: {condition}")


def _print_summary(
    output_file: Path,
    run_id: str,
    condition: str,
    *,
    name: str | None = None,
    dataset: str | None = None,
) -> None:
    rows = _matching_unified_rows(
        output_file,
        run_id=run_id,
        condition=condition,
        name=name,
        dataset=dataset,
        tolerate_invalid=True,
    )
    summary = summarize_records(rows)
    click.echo(f"Summary ({_condition_label(condition, name)}): total={summary['total']} errors={summary['errors']} ")
    if summary["accuracy"] is not None:
        click.echo(f"  accuracy={summary['accuracy']:.1%}")
    if summary["total_cost_usd"] is not None:
        click.echo(f"  cost=${summary['total_cost_usd']:.4f}")


def _matching_unified_rows(
    output_file: Path,
    *,
    run_id: str,
    condition: str | None = None,
    name: str | None = None,
    dataset: str | None = None,
    tolerate_invalid: bool = False,
) -> list[dict[str, Any]]:
    rows = load_jsonl(output_file, tolerate_invalid=tolerate_invalid)
    return [
        row for row in rows
        if row.get("run_id") == run_id
        and (condition is None or row.get("condition") == condition)
        and row.get("name") == name
        and (dataset is None or row.get("dataset") == dataset)
    ]


def _latest_rows_by_item(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if isinstance(item_id, str) and item_id:
            latest[item_id] = row
    return list(latest.values())


def _swebench_predictions_from_output(
    output_file: Path,
    *,
    run_id: str,
    condition: str,
    name: str | None = None,
    dataset: str | None = None,
) -> list[_SwePrediction]:
    rows = _latest_rows_by_item(
        _matching_unified_rows(
            output_file,
            run_id=run_id,
            condition=condition,
            name=name,
            dataset=dataset,
            tolerate_invalid=True,
        )
    )
    return [_SwePrediction(instance_id=row["item_id"], patch=row.get("patch") or "") for row in rows]


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
    name: str | None = None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    conditions = _parse_swebench_conditions(condition)
    resolved_dataset_name = dataset_name or _swebench_dataset_name(dataset)
    pred_root = Path(predictions_dir)
    eval_run_id = _swebench_eval_run_id(run_id, name=name)

    plan: list[dict[str, Any]] = []
    if dataset == "pro":
        eval_repo_path = Path(pro.eval_repo)
        raw_sample = Path(pro.raw_sample_path) if pro.raw_sample_path else _default_swebench_pro_raw_sample_path(
            resolved_dataset_name, split
        )
        output_root = Path(pro.output_dir) / eval_run_id

        for cond in conditions:
            pred_path = pred_root / _swebench_predictions_filename(cond, dataset, run_id, name=name)
            if verify_predictions and not pred_path.exists():
                raise click.ClickException(f"Predictions file not found: {pred_path}")

            converted_path = output_root / _swebench_converted_patch_filename(cond, dataset, run_id, name=name)
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
                    "eval_run_id": eval_run_id,
                    "name": name,
                    "command": cmd,
                }
            )
        return resolved_dataset_name, conditions, plan

    for idx, cond in enumerate(conditions):
        pred_path = pred_root / _swebench_predictions_filename(cond, dataset, run_id, name=name)
        if verify_predictions and not pred_path.exists():
            raise click.ClickException(f"Predictions file not found: {pred_path}")

        is_last = idx == len(conditions) - 1
        cache_level = final_cache_level if is_last else "env"
        clean = final_clean if is_last else False
        report_dir = BENCHMARK_ROOT / "results" / "swebench_eval_reports"
        report_dir = report_dir / _sanitize_trace_segment(cond, "condition")
        if name:
            report_dir = report_dir / _sanitize_trace_segment(name, "name")
        report_dir.mkdir(parents=True, exist_ok=True)
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
            eval_run_id,
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
            "--report_dir",
            str(report_dir),
        ]
        plan.append(
            {
                "mode": "standard",
                "condition": cond,
                "predictions_path": str(pred_path),
                "cache_level": cache_level,
                "clean": clean,
                "eval_run_id": eval_run_id,
                "name": name,
                "report_dir": str(report_dir),
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
        if step.get("report_dir"):
            click.echo(f"SWE-bench eval reports written to: {step['report_dir']}")


@click.group()
def cli() -> None:
    """Unified benchmark runner."""


@cli.command("general")
@click.option("--dataset", default="longbench_short", type=click.Choice(list(DATASETS.keys()) + ["all"]))
@click.option("--condition", default="one_shot", type=click.Choice(["one_shot", "spell", "claude_code", "codex", "swe_agent", "all"]))
@click.option("--model", default=None, help="provider:model for one_shot/spell (required for those conditions)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex", help="Model for Codex harness comparator")
@click.option("--swe-agent-model", default=None, help="LiteLLM model for mini-swe-agent comparator")
@click.option("--n", type=int, default=None)
@click.option("--items", default=None, help="Comma-separated item ids")
@click.option("--parallel", type=int, default=5, show_default=True)
@click.option("--budget", type=float, default=1.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=None)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Per-item timeout in seconds (default: dataset-dependent, see _DATASET_TIMEOUTS)")
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
@click.option("--name", default=None, help="Label for this run variant (flows into traces, predictions, and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides dataset default or falls back to '(!extend))")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None, help="JSONL output path")
def run_general(
    dataset: str,
    condition: str,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
    n: int | None,
    items: str | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int | None,
    max_tokens: int | None,
    timeout: int | None,
    thinking: bool,
    thinking_budget: int,
    reasoning_effort: str | None,
    verbosity: str | None,
    responses_api: bool,
    suffix_grammar: bool,
    web_search: bool | None,
    trace: bool,
    name: str | None = None,
    trailing: str | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    output: str | None = None,
) -> None:
    run_id = run_id or _run_id("general")
    output_file = Path(output) if output else _default_output(run_id)
    if timeout is None:
        timeout = _DATASET_TIMEOUTS.get(dataset, _DEFAULT_TIMEOUT)
    conditions = _parse_conditions(condition, ("one_shot", "spell", "claude_code", "codex", "swe_agent", "all"))

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    if name:
        click.echo(f"Name: {name}")

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
    _validate_general_configuration(
        dataset,
        conditions,
        item_datasets,
        web_search_enabled=effective_web_search,
    )

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
                    "swe_agent_model": swe_agent_model,
                    "parallel": parallel,
                    "budget": budget,
                    "name": name,
                    "trailing": trailing,
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
        adapter = _build_adapter(
            cond,
            model=model,
            claude_model=claude_model,
            codex_model=codex_model,
            swe_agent_model=swe_agent_model,
            budget=budget,
        )
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
            prompt = _format_general_prompt(item, cond)
            requests.append(
                RunRequest(
                    benchmark="general",
                    dataset=dataset,
                    item_id=item.id,
                    condition=cond,
                    prompt=prompt,
                    model_spec=_condition_model_spec(
                        cond,
                        model=model,
                        claude_model=claude_model,
                        codex_model=codex_model,
                        swe_agent_model=swe_agent_model,
                    ),
                    timeout=timeout,
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
                    init=_make_spell_init(prompt, trailing if trailing is not None else _general_spell_trailing(item.dataset)) if cond == "spell" else None,
                    trace_dir=_trace_dir_for_request(trace_dir, cond, item.id, name=name),
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
            extracted, correct, eval_error = _evaluate_general(item, resp.response_text)
            if eval_error:
                resp = _mark_evaluation_error(
                    resp,
                    error_type="judge_error",
                    error_message=eval_error,
                    metadata={**resp.metadata, "judge_error": eval_error},
                )
            record = _mk_record(
                run_id=run_id,
                benchmark="general",
                dataset=dataset,
                split=None,
                item_id=item.id,
                condition=cond,
                name=name,
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
                reasoning_effort=reasoning_effort,
            )
            append_jsonl(output_file, record)

        _print_summary(output_file, run_id, cond, name=name)


@cli.command("swebench")
@click.option("--dataset", default="mini", type=click.Choice(["mini", "lite", "verified", "pro"]))
@click.option("--condition", default="all", type=click.Choice(["spell", "claude_code", "codex", "swe_agent", "both", "all"]))
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex")
@click.option("--swe-agent-model", default=None, help="LiteLLM model for mini-swe-agent comparator")
@click.option("--n", type=int, default=None)
@click.option("--items", "item_ids", default=None, help="Comma-separated SWE-bench instance IDs to run")
@click.option("--seed", type=int, default=None, help="Random seed for sampling --n items")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=None, help="Spell budget (default: 10 for mini/lite/verified, 80 for pro)")
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Per-item timeout seconds (default: from _DATASET_TIMEOUTS)")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default=None, help="Reasoning effort (default: provider default, i.e. no --effort flag for CC)")
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--trace/--no-trace", default=True, show_default=True, help="Record execution traces")
@click.option("--clean-eval/--no-clean-eval", default=False, show_default=True, help="Re-run patches in a fresh official eval container after generation")
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
    "--env-image-cache-repository",
    default=None,
    envvar="SPELL_SWEBENCH_ENV_IMAGE_CACHE_REPOSITORY",
    help="Artifact Registry repository for SWE-bench env/instance image caching",
)
@click.option(
    "--env-image-cache-read-only/--no-env-image-cache-read-only",
    default=False,
    show_default=True,
    envvar="SPELL_SWEBENCH_ENV_IMAGE_CACHE_READ_ONLY",
    help="Pull from the SWE-bench image cache but do not publish new env or instance images",
)
@click.option(
    "--prewarm-envs/--no-prewarm-envs",
    default=False,
    show_default=True,
    help="Prebuild unique SWE-bench env images once before item generation",
)
@click.option(
    "--env-build-workers",
    type=int,
    default=1,
    show_default=True,
    help="Parallel workers for SWE-bench env-image prewarm",
)
@click.option(
    "--prewarm-only",
    is_flag=True,
    default=False,
    help="Build env images for the selected SWE-bench items, then exit without generation",
)
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
    "--keep-images/--no-keep-images",
    default=False,
    show_default=True,
    help="Retain per-item SWE-bench instance images after generation for debugging",
)
@click.option("--name", default=None, help="Label for this run variant (flows into traces, predictions, and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides default '(!extend))")
@click.option(
    "--backend",
    type=click.Choice(["docker", "singularity"]),
    default="docker",
    show_default=True,
    help="Container backend: 'docker' (default) or 'singularity' (for HPC clusters without Docker)",
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_swebench(
    dataset: str,
    condition: str,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
    n: int | None,
    item_ids: str | None,
    seed: int | None,
    parallel: int,
    budget: float | None,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int,
    timeout: int | None,
    reasoning_effort: str | None,
    web_search: bool,
    trace: bool,
    clean_eval: bool,
    eval_predictions_dir: str,
    eval_dataset_name: str | None,
    eval_split: str,
    eval_max_workers: int,
    eval_timeout: int,
    eval_namespace: str,
    eval_instance_image_tag: str,
    eval_env_image_tag: str,
    env_image_cache_repository: str | None,
    env_image_cache_read_only: bool,
    prewarm_envs: bool,
    env_build_workers: int,
    prewarm_only: bool,
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
    keep_images: bool,
    name: str | None = None,
    trailing: str | None = None,
    backend: str = "docker",
    dry_run: bool = False,
    run_id: str | None = None,
    output: str | None = None,
) -> None:
    run_id = run_id or _run_id("swebench")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_swebench_conditions(condition)

    # Resolve timeout from central map
    if timeout is None:
        key = f"swebench_{dataset}"
        timeout = _DATASET_TIMEOUTS.get(key, _DATASET_TIMEOUTS.get("swebench", _DEFAULT_TIMEOUT))
    # Pro-specific budget default
    if dataset == "pro":
        if budget is None:
            budget = 80.0
    else:
        if budget is None:
            budget = 10.0

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

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    if name:
        click.echo(f"Name: {name}")
    click.echo("Container mode: official SWE-bench instance images")
    click.echo(f"Loaded {len(items)} items")
    if env_build_workers < 1:
        raise click.ClickException("--env-build-workers must be at least 1")
    env_image_cache = _make_swebench_env_cache_config(
        env_image_cache_repository,
        read_only=env_image_cache_read_only,
    )
    if backend != "docker" and env_image_cache is not None:
        raise click.ClickException("SWE-bench env-image cache is only supported with the docker backend")
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
                    "swe_agent_model": swe_agent_model,
                    "items": sorted(requested_ids) if requested_ids else None,
                    "budget": budget,
                    "depth": depth,
                    "web_search": web_search,
                    "trace": trace,
                    "keep_images": keep_images,
                    "agent": spell_resolution.agent_file if spell_resolution else None,
                    "spell_mode": spell_resolution.spell_mode if spell_resolution else None,
                    "agent_source": spell_resolution.source if spell_resolution else None,
                    "agent_policy_key": spell_resolution.policy_key if spell_resolution else None,
                    "parallel": parallel,
                    "clean_eval": clean_eval,
                    "eval_predictions_dir": eval_predictions_dir,
                    "eval_dataset_name": eval_dataset_name,
                    "eval_split": eval_split,
                    "eval_max_workers": eval_max_workers,
                    "eval_timeout": eval_timeout,
                    "eval_namespace": eval_namespace,
                    "eval_instance_image_tag": eval_instance_image_tag,
                    "eval_env_image_tag": eval_env_image_tag,
                    "env_image_cache_repository": env_image_cache_repository,
                    "env_image_cache_read_only": env_image_cache_read_only,
                    "prewarm_envs": prewarm_envs,
                    "env_build_workers": env_build_workers,
                    "prewarm_only": prewarm_only,
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
                    "name": name,
                    "trailing": trailing,
                },
                indent=2,
            )
        )
        if items:
            click.echo(f"First item: {items[0].instance_id}")
        return

    if dataset == "pro":
        ensure_scale_pro_eval_repo(pro_eval_repo)
        click.echo(f"Pro eval repo ready: {pro_eval_repo}")

    if backend != "docker" and (prewarm_envs or prewarm_only):
        raise click.ClickException("SWE-bench env prewarm is only supported with the docker backend")

    if prewarm_envs or prewarm_only:
        click.echo(
            f"Prewarming SWE-bench env images with {env_build_workers} worker"
            f"{'' if env_build_workers == 1 else 's'}"
        )
        prewarm_kwargs: dict[str, Any] = {
            "instance_image_tag": eval_instance_image_tag,
            "env_image_tag": eval_env_image_tag,
            "max_workers": env_build_workers,
        }
        if env_image_cache is not None:
            prewarm_kwargs["env_image_cache"] = env_image_cache
        prewarm_result = prewarm_swebench_env_images(items, **prewarm_kwargs)
        click.echo(
            "Prewarm summary: "
            f"{prewarm_result.unique_env_count} unique env image"
            f"{'' if prewarm_result.unique_env_count == 1 else 's'}, "
            f"{prewarm_result.built_env_count} built this run"
        )
        if prewarm_only:
            return

    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

    images_to_cleanup: set[str] = set()
    try:
        for cond in conditions:
            click.echo(f"\n--- Running {cond} ---")
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
                try:
                    prompt = _format_swebench_prompt(item, workspace=_swebench_prompt_workspace(item))
                    request_trace_dir = _trace_dir_for_request(trace_dir, cond, item.instance_id, name=name)

                    req = RunRequest(
                        benchmark="swebench",
                        dataset=dataset,
                        item_id=item.instance_id,
                        condition=cond,
                        prompt=prompt,
                        model_spec=_condition_model_spec(
                            cond,
                            model=model,
                            claude_model=claude_model,
                            codex_model=codex_model,
                            swe_agent_model=swe_agent_model,
                        ),
                        timeout=timeout,
                        init=(_make_spell_init(prompt, trailing) if cond == "spell" else None),
                        agent_file=spell_resolution.agent_file if spell_resolution else None,
                        budget=budget,
                        depth=depth,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        web_search=web_search if cond == "codex" else False,
                        trace=trace,
                        trace_dir=request_trace_dir,
                    )
                    resp = run_unified_swebench(
                        condition=cond,
                        prompt=prompt,
                        model_spec=req.model_spec,
                        timeout=timeout,
                        init=req.init,
                        agent_file=req.agent_file,
                        budget=budget,
                        depth=depth,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        web_search=req.web_search,
                        trace=trace,
                        trace_dir=req.trace_dir,
                        eval_timeout=eval_timeout,
                        item=item,
                        pro_eval_repo=pro_eval_repo if dataset == "pro" else None,
                        pro_dockerhub_username=pro_dockerhub_username,
                        instance_image_tag=eval_instance_image_tag,
                        env_image_tag=eval_env_image_tag,
                        backend=backend,
                        **(
                            {"env_image_cache": env_image_cache}
                            if env_image_cache is not None
                            else {}
                        ),
                    )
                    patch = resp.metadata.get("patch", "")
                    changed_files = resp.metadata.get("changed_files", [])

                    status = resp.status
                    if status == "ok":
                        status = "patch_generated" if patch.strip() else "no_patch"
                    resp = _rewrap_response(
                        resp,
                        status=status,
                        metadata={**resp.metadata, "changed_files": changed_files},
                    )

                    return _mk_record(
                        run_id=run_id,
                        benchmark="swebench",
                        dataset=dataset,
                        split=dataset,
                        item_id=item.instance_id,
                        condition=cond,
                        name=name,
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
                        reasoning_effort=reasoning_effort,
                    )
                except Exception as exc:
                    # Keep long SWE-bench runs alive even when one instance fails
                    # during container setup or same-container evaluation.
                    err_text = str(exc)
                    err_metadata: dict[str, Any] = {}
                    if isinstance(exc, SWEBenchContainerSetupError):
                        err_metadata = exc.metadata()
                    return _mk_record(
                        run_id=run_id,
                        benchmark="swebench",
                        dataset=dataset,
                        split=dataset,
                        item_id=item.instance_id,
                        condition=cond,
                        name=name,
                        model_spec=_condition_model_spec(
                            cond,
                            model=model,
                            claude_model=claude_model,
                            codex_model=codex_model,
                            swe_agent_model=swe_agent_model,
                        ),
                        agent_file=spell_resolution.agent_file if spell_resolution else None,
                        response=RunResponse(
                            status="error",
                            response_text="",
                            tokens_in=None,
                            tokens_out=None,
                            cost_usd=None,
                            latency_ms=None,
                            error_message=err_text,
                            error_type="container_error",
                            metadata=err_metadata,
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
                        reasoning_effort=reasoning_effort,
                    )

            existing_rows = _matching_unified_rows(
                output_file,
                run_id=run_id,
                condition=cond,
                name=name,
                dataset=dataset,
                tolerate_invalid=True,
            )
            completed_item_ids = {
                item_id
                for row in existing_rows
                if isinstance((item_id := row.get("item_id")), str) and item_id
            }
            pending_items = [item for item in items if item.instance_id not in completed_item_ids]
            if completed_item_ids:
                click.echo(
                    f"  Resume: {len(completed_item_ids)}/{len(items)} items already recorded; "
                    f"running {len(pending_items)} pending"
                )

            def persist_record(record: UnifiedRunRecord) -> None:
                image_name = record.metadata.get("instance_image_name")
                if isinstance(image_name, str) and image_name:
                    images_to_cleanup.add(image_name)
                append_jsonl(output_file, record, flush=True, fsync=True)

            completed_so_far = len(completed_item_ids)
            if parallel <= 1:
                for item in pending_items:
                    record = process(item)
                    persist_record(record)
                    completed_so_far += 1
                    click.echo(f"  [{completed_so_far}/{len(items)}] {item.instance_id}")
            else:
                with ThreadPoolExecutor(max_workers=parallel) as pool:
                    futures = {pool.submit(process, item): item for item in pending_items}
                    for fut in as_completed(futures):
                        item = futures[fut]
                        record = fut.result()
                        persist_record(record)
                        completed_so_far += 1
                        click.echo(f"  [{completed_so_far}/{len(items)}] {item.instance_id}")

            if not pending_items:
                click.echo("  No pending items; rebuilding predictions from existing unified output")

            _print_summary(output_file, run_id, cond, name=name, dataset=dataset)

            pred_path = Path("results") / "swebench" / _swebench_predictions_filename(cond, dataset, run_id, name=name)
            swe_results = _swebench_predictions_from_output(
                output_file,
                run_id=run_id,
                condition=cond,
                name=name,
                dataset=dataset,
            )
            pred_model_name = {
                "spell": model or "unknown",
                "claude_code": f"claude_code:{claude_model or 'unknown'}",
                "codex": codex_model,
                "swe_agent": f"swe_agent:{swe_agent_model or 'unknown'}",
            }[cond]
            write_predictions_jsonl(swe_results, pred_path, model_name=pred_model_name)
            click.echo(f"Predictions: {pred_path}")

        if clean_eval:
            click.echo("\n=== Running clean SWE-bench evaluation ===")
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
                name=name,
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
    finally:
        if not keep_images and backend == "docker":
            _cleanup_swebench_instance_images(images_to_cleanup)


@cli.command("swebench-eval")
@click.option("--dataset", default="mini", type=click.Choice(["mini", "lite", "verified", "pro"]))
@click.option("--condition", default="all", type=click.Choice(["spell", "claude_code", "codex", "swe_agent", "both", "all"]))
@click.option("--run-id", required=True, help="Run ID used in results/swebench/<condition>[_<name>]_<dataset>_<run-id>.jsonl")
@click.option("--name", default=None, help="Optional run label used in predictions filenames")
@click.option("--predictions-dir", default="results/swebench", show_default=True)
@click.option("--dataset-name", default=None, help="Override swebench.harness dataset_name")
@click.option("--split", default="test", show_default=True)
@click.option("--max-workers", type=int, default=4, show_default=True)
@click.option("--timeout", type=int, default=None, help="Eval timeout seconds (default: from _DATASET_TIMEOUTS)")
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
    name: str | None = None,
    predictions_dir: str = "results/swebench",
    dataset_name: str | None = None,
    split: str = "test",
    max_workers: int = 4,
    timeout: int | None = None,
    namespace: str = "none",
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    final_cache_level: str = "base",
    final_clean: bool = True,
    pro_eval_repo: str = SCALE_PRO_EVAL_REPO_DEFAULT,
    pro_raw_sample_path: str | None = None,
    pro_output_dir: str = SCALE_PRO_OUTPUT_DIR_DEFAULT,
    pro_dockerhub_username: str = "jefzda",
    pro_strip_trace_artifacts: bool = True,
    pro_use_local_docker: bool = True,
    pro_block_network: bool = False,
    pro_redo: bool = False,
    dry_run: bool = False,
) -> None:
    if timeout is None:
        timeout = _DATASET_TIMEOUTS.get("swebench_eval", _DEFAULT_TIMEOUT)
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
        name=name,
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
@click.option("--condition", default="claude_code", type=click.Choice(["spell", "claude_code", "codex", "swe_agent", "both", "all"]))
@click.option("--split", default="lite")
@click.option("--task-ids", type=int, multiple=True)
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex")
@click.option("--swe-agent-model", default=None, help="LiteLLM model for mini-swe-agent comparator")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=10.0, show_default=True)
@click.option("--agent", default=None)
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Per-item timeout seconds (default: from _DATASET_TIMEOUTS)")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--workspace-root", default="/tmp/fb_ws")
@click.option("--skip-install", is_flag=True, default=False)
@click.option("--skip-tests", is_flag=True, default=False)
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--trace/--no-trace", default=True, show_default=True)
@click.option("--name", default=None, help="Label for this run variant (flows into traces and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides default '(!extend))")
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_featurebench(
    condition: str,
    split: str,
    task_ids: tuple[int, ...],
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int,
    timeout: int | None,
    reasoning_effort: str | None,
    workspace_root: str,
    skip_install: bool,
    skip_tests: bool,
    web_search: bool,
    trace: bool,
    name: str | None = None,
    trailing: str | None = None,
    run_id: str | None = None,
    output: str | None = None,
) -> None:
    if timeout is None:
        timeout = _DATASET_TIMEOUTS.get("featurebench", _DEFAULT_TIMEOUT)
    run_id = run_id or _run_id("featurebench")
    output_file = Path(output) if output else _default_output(run_id)

    if condition == "both":
        conditions = ["spell", "claude_code"]
    elif condition == "all":
        conditions = ["spell", "claude_code", "codex", "swe_agent"]
    else:
        conditions = [condition]

    tasks = load_tasks(split=split, task_indices=list(task_ids) if task_ids else None)
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    cache = RepoCache(root)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    if name:
        click.echo(f"Name: {name}")
    click.echo(f"Workspace root: {root}")
    click.echo("Container mode: shared software benchmark container infra")
    click.echo(f"Loaded {len(tasks)} tasks")
    trace_dir = _resolve_trace_dir(run_id, trace)
    if trace_dir:
        click.echo(f"Trace dir: {trace_dir}")

    for cond in conditions:
        click.echo(f"\n--- Running {cond} ---")
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
                mask_functions(wt.path, item["problem_statement"])
                prompt_workspace = Path("/workspace")
                prompt = featurebench_format_prompt(item, prompt_workspace)

                req = RunRequest(
                    benchmark="featurebench",
                    dataset=split,
                    item_id=item["instance_id"],
                    condition=cond,
                    prompt=prompt,
                    model_spec=_condition_model_spec(
                        cond,
                        model=model,
                        claude_model=claude_model,
                        codex_model=codex_model,
                        swe_agent_model=swe_agent_model,
                    ),
                    timeout=timeout,
                    init=_make_spell_init(prompt, trailing) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if spell_resolution else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=web_search if cond == "codex" else False,
                    trace=trace,
                    trace_dir=_trace_dir_for_request(trace_dir, cond, item["instance_id"], name=name),
                    workspace=wt.path,
                )
                test_results = []
                tests_passed: bool | None = None
                resp, tests_passed, test_results = run_unified_featurebench(
                    condition=cond,
                    prompt=prompt,
                    model_spec=req.model_spec,
                    timeout=timeout,
                    init=req.init,
                    agent_file=req.agent_file,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=req.web_search,
                    trace=trace,
                    trace_dir=req.trace_dir,
                    workspace=wt.path,
                    item=item,
                    skip_install=skip_install,
                    skip_tests=skip_tests,
                )
                patch = resp.metadata.get("patch", "")
                changed_files = resp.metadata.get("changed_files", [])

                status = resp.status
                if status == "ok":
                    if patch.strip():
                        status = "tests_passed" if tests_passed else "patch_generated"
                    else:
                        status = "no_patch"

                resp = _rewrap_response(
                    resp,
                    status=status,
                    metadata={**resp.metadata, "changed_files": changed_files, "test_results": test_results},
                )

                return _mk_record(
                    run_id=run_id,
                    benchmark="featurebench",
                    dataset=split,
                    split=split,
                    item_id=item["instance_id"],
                    condition=cond,
                    name=name,
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
                    reasoning_effort=reasoning_effort,
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

        _print_summary(output_file, run_id, cond, name=name)


@cli.command("terminalbench")
@click.option(
    "--condition",
    "condition_values",
    multiple=True,
    type=click.Choice(["spell", "claude_code", "codex", "all"]),
    default=("all",),
    help="Condition(s) to run. Repeat the flag to run multiple explicit conditions.",
)
@click.option(
    "--dataset",
    default=TERMINALBENCH_DATASET,
    show_default=True,
    help="Terminal-Bench dataset or alias. Aliases: old-core/core-0.1.1/1.1 -> terminal-bench-core==0.1.1; 2.0/head/current/latest -> terminal-bench-core==head via the bundled 2.x registry.",
)
@click.option("--model", default=None, help="provider:model override for Spell")
@click.option("--claude-model", default=None, help="Model override for Claude Code")
@click.option("--codex-model", default=None, help="Model override for Codex CLI")
@click.option("--items", "task_ids", multiple=True, help="Specific Terminal-Bench task IDs")
@click.option("-t", "task_ids_short", multiple=True, help="Specific Terminal-Bench task IDs")
@click.option("--n", "n_tasks", type=int, default=None, help="Limit to N tasks")
@click.option("--budget", type=float, default=5.0, show_default=True, help="Spell budget in USD")
@click.option("--n-concurrent", type=int, default=1, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--cleanup/--no-cleanup", default=True, show_default=True)
@click.option("--trace/--no-trace", default=True, show_default=True, help="Extract Spell traces to traces/<run-id>/spell/")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high", "xhigh"]), default="high", show_default=True)
@click.option(
    "--agent-timeout-sec",
    type=int,
    default=3600,
    show_default=True,
    help="Per-task wall-clock cap passed to the tb harness and condition agents.",
)
@click.option("--name", default=None, help="Label for this run variant (flows into output directories and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (default: '(!extend))")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
@click.option("--output", default=None, help="JSONL output path")
def run_terminalbench(
    condition_values: tuple[str, ...],
    dataset: str,
    model: str | None,
    claude_model: str | None,
    codex_model: str | None,
    task_ids: tuple[str, ...],
    task_ids_short: tuple[str, ...],
    n_tasks: int | None,
    budget: float,
    n_concurrent: int,
    agent: str | None,
    cleanup: bool,
    trace: bool,
    reasoning_effort: str,
    agent_timeout_sec: int,
    name: str | None,
    trailing: str | None,
    dry_run: bool,
    run_id: str | None,
    output: str | None,
) -> None:
    run_id = run_id or _run_id("terminalbench")
    output_file = Path(output) if output else terminalbench_output_file(run_id)
    conditions = _parse_terminalbench_conditions_option(condition_values)
    requested_dataset = dataset
    resolved_dataset = resolve_terminalbench_dataset_spec(dataset).dataset
    requested_task_ids = tuple(task_ids) + tuple(task_ids_short)

    click.echo(f"Run ID: {run_id}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Dataset: {resolved_dataset}")
    if requested_dataset != resolved_dataset:
        click.echo(f"Requested dataset: {requested_dataset}")
    if name:
        click.echo(f"Name: {name}")

    env = prepare_terminalbench_env(BENCHMARK_ROOT)
    if env.get("DOCKER_HOST"):
        click.echo(f"DOCKER_HOST: {env['DOCKER_HOST']}")

    models_by_condition = {
        "spell": model,
        "claude_code": claude_model,
        "codex": codex_model,
    }
    commands: list[dict[str, Any]] = []
    for cond in conditions:
        tb_run_id = terminalbench_condition_run_id(run_id, cond, name=name)
        cmd = build_terminalbench_command(
            condition=cond,
            run_id=tb_run_id,
            model=models_by_condition.get(cond),
            dataset=requested_dataset,
            task_ids=requested_task_ids,
            n_tasks=n_tasks,
            n_concurrent=n_concurrent,
            budget=budget,
            cleanup=cleanup,
            reasoning_effort=reasoning_effort,
            agent_timeout_sec=agent_timeout_sec,
            agent_file=agent,
            trailing=trailing,
            trace=trace,
        )
        commands.append(
            {
                "condition": cond,
                "tb_run_id": tb_run_id,
                "model": resolve_terminalbench_model(cond, models_by_condition.get(cond)),
                "run_dir": str(terminalbench_run_dir(run_id, cond, name=name)),
                "command": cmd,
            }
        )

    if dry_run:
        click.echo("Dry run config:")
        click.echo(
            json.dumps(
                {
                    "dataset": resolved_dataset,
                    "requested_dataset": requested_dataset,
                    "resolved_dataset": resolved_dataset,
                    "conditions": conditions,
                    "task_ids": list(requested_task_ids),
                    "n_tasks": n_tasks,
                    "n_concurrent": n_concurrent,
                    "budget": budget,
                    "cleanup": cleanup,
                    "trace": trace,
                    "reasoning_effort": reasoning_effort,
                    "name": name,
                    "trailing": trailing,
                    "commands": commands,
                },
                indent=2,
            )
        )
        return

    host_trace_root = _resolve_trace_dir(run_id, trace)
    if host_trace_root:
        click.echo(f"Trace dir: {host_trace_root}")

    for idx, command_info in enumerate(commands):
        cond = command_info["condition"]
        click.echo(f"\n--- Running {cond} ---")
        subprocess.run(command_info["command"], check=True, cwd=BENCHMARK_ROOT, env=env)

        records = terminalbench_records_from_run(
            run_dir=terminalbench_run_dir(run_id, cond, name=name),
            run_id=run_id,
            condition=cond,
            dataset=resolved_dataset,
            model_spec=command_info["model"],
            name=name,
            agent_file=agent if cond == "spell" else None,
            requested_dataset=requested_dataset,
            reasoning_effort=reasoning_effort,
            host_trace_root=_trace_dir_for_condition(host_trace_root, cond, name=name) if cond == "spell" else None,
        )
        for record in records:
            append_jsonl(output_file, record)

        click.echo(
            f"  Imported {len(records)} Terminal-Bench result(s) from "
            f"{terminalbench_run_dir(run_id, cond, name=name)}"
        )
        _print_summary(output_file, run_id, cond, name=name)

        if cleanup and idx < len(commands) - 1:
            click.echo("  Pruning Docker before next condition...")
            prune_terminalbench_docker(env)


@cli.command("harbor-terminalbench")
@click.option("--condition", default="spell", show_default=True, type=click.Choice(["spell", "codex", "all"]))
@click.option("--model", default="openai-tc:gpt-5.4", show_default=True, help="Spell provider:model")
@click.option("--codex-model", default="gpt-5.4", show_default=True, help="Model for custom Harbor Codex agent")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="medium", show_default=True)
@click.option("-k", "--trials", type=int, default=5, show_default=True, help="Trials per task")
@click.option("--dataset", default=HARBOR_TERMINALBENCH_DATASET, show_default=True, help="Harbor dataset name@version")
@click.option("--jobs-dir", type=click.Path(path_type=Path), default=None, help="Root directory for Harbor job artifacts")
@click.option("--n-concurrent", type=int, default=4, show_default=True, help="Harbor concurrent trial count")
@click.option("--task", default=None, help="Harbor registry task selector; dev mode only")
@click.option("--include-task-name", multiple=True, help="Harbor task-name include glob; dev mode only")
@click.option("--exclude-task-name", multiple=True, help="Harbor task-name exclude glob; dev mode only")
@click.option("--n", "n_tasks", type=int, default=None, help="Limit to N tasks; dev mode only")
@click.option("--budget", type=float, default=5.0, show_default=True, help="Spell budget in USD")
@click.option("--dev-smoke", is_flag=True, default=False, help="Permit non-submission smoke settings such as k=1 or task filters")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--run-id", default=None)
def run_harbor_terminalbench(
    condition: str,
    model: str,
    codex_model: str,
    reasoning_effort: str,
    trials: int,
    dataset: str,
    jobs_dir: Path | None,
    n_concurrent: int,
    task: str | None,
    include_task_name: tuple[str, ...],
    exclude_task_name: tuple[str, ...],
    n_tasks: int | None,
    budget: float,
    dev_smoke: bool,
    dry_run: bool,
    run_id: str | None,
) -> None:
    run_id = run_id or _run_id("harbor_terminalbench")
    try:
        commands = build_harbor_terminalbench_commands(
            condition=condition,
            run_id=run_id,
            model=model,
            codex_model=codex_model,
            reasoning_effort=reasoning_effort,
            trials=trials,
            jobs_dir=jobs_dir,
            dataset=dataset,
            dev_smoke=dev_smoke,
            n_concurrent=n_concurrent,
            task=task,
            include_task_names=include_task_name,
            exclude_task_names=exclude_task_name,
            n_tasks=n_tasks,
            budget=budget,
        )
    except HarborTerminalBenchComplianceError as exc:
        raise click.ClickException(str(exc)) from exc

    mode = "dev" if dev_smoke else "submission"
    click.echo(f"Run ID: {run_id}")
    click.echo(f"Mode: {mode}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Trials per task: {trials}")
    click.echo("Compliance: " + ("non-submission dev run" if dev_smoke else "submission command guarded"))

    rendered = [
        {
            "condition": item.condition,
            "job_name": item.job_name,
            "model": item.model,
            "jobs_dir": str(item.jobs_dir),
            "submission_compliant": item.submission_compliant,
            "command": item.command,
        }
        for item in commands
    ]
    if dry_run:
        click.echo("Dry run config:")
        click.echo(json.dumps({"commands": rendered}, indent=2))
        return

    for item in commands:
        click.echo(f"\n--- Running Harbor Terminal-Bench {item.condition} ---")
        click.echo(f"Job dir: {item.jobs_dir}")
        subprocess.run(item.command, check=True, cwd=BENCHMARK_ROOT)


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
        "## Environment",
        "",
        "Platform: Linux (Debian-based).",
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
@click.option("--condition", default="claude_code", type=click.Choice(["spell", "claude_code", "codex", "swe_agent", "both", "all"]))
@click.option("--n", type=int, default=None, help="Limit to N items")
@click.option("--instance-ids", default=None, help="Comma-separated instance IDs")
@click.option("--domain", default=None, help="Filter by domain (partial match)")
@click.option("--model", default=None, help="provider:model for Spell (required for spell condition)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator")
@click.option("--codex-model", default="codex")
@click.option("--swe-agent-model", default=None, help="LiteLLM model for mini-swe-agent comparator")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=5.0, show_default=True)
@click.option("--agent", default=None)
@click.option("--agent-msg", default=None)
@click.option("--agent-tc", default=None)
@click.option("--depth", type=int, default=50, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Per-item timeout seconds (default: from _DATASET_TIMEOUTS)")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--use-knowledge/--no-knowledge", default=False, show_default=True, help="Include domain knowledge in prompt")
@click.option("--workspace-root", default="/tmp/sab_ws")
@click.option("--name", default=None, help="Label for this run variant (flows into output labels and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides default '(!extend))")
@click.option("--run-id", default=None)
@click.option("--output", default=None)
def run_scienceagentbench(
    condition: str,
    n: int | None,
    instance_ids: str | None,
    domain: str | None,
    model: str | None,
    claude_model: str | None,
    codex_model: str,
    swe_agent_model: str | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int,
    timeout: int | None,
    reasoning_effort: str | None,
    web_search: bool,
    use_knowledge: bool,
    workspace_root: str,
    name: str | None = None,
    trailing: str | None = None,
    run_id: str | None = None,
    output: str | None = None,
) -> None:
    """Run ScienceAgentBench: data-driven scientific code generation."""
    import re
    import shutil

    if timeout is None:
        timeout = _DATASET_TIMEOUTS.get("scienceagentbench", _DEFAULT_TIMEOUT)
    run_id = run_id or _run_id("sab")
    output_file = Path(output) if output else _default_output(run_id)

    if condition == "both":
        conditions = ["spell", "claude_code"]
    elif condition == "all":
        conditions = ["spell", "claude_code", "codex", "swe_agent"]
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
    if name:
        click.echo(f"Name: {name}")
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
        adapter = _build_adapter(
            cond,
            model=model,
            claude_model=claude_model,
            codex_model=codex_model,
            swe_agent_model=swe_agent_model,
            budget=budget,
        )
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
                    model_spec=_condition_model_spec(
                        cond,
                        model=model,
                        claude_model=claude_model,
                        codex_model=codex_model,
                        swe_agent_model=swe_agent_model,
                    ),
                    timeout=timeout,
                    init=(_make_spell_init(prompt, trailing) if cond == "spell" else None),
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
                    name=name,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=RunResponse(
                        status=status,
                        response_text=resp.response_text,
                        usage=resp.usage,
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
                    reasoning_effort=reasoning_effort,
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
        _print_summary(output_file, run_id, cond, name=name)

        # Save extracted programs for potential evaluation
        pred_dir_name = f"{cond}_{_sanitize_trace_segment(name, 'name')}_{run_id}" if name else f"{cond}_{run_id}"
        pred_dir = Path("results") / "scienceagentbench" / pred_dir_name
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
        "## Environment\n\n"
        "Platform: Linux (Debian-based).\n\n"
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
@click.option("--condition", default="spell", type=click.Choice(["one_shot", "spell", "claude_code", "codex", "swe_agent", "all"]))
@click.option("--difficulty", default=None, help="Difficulty range, e.g. '1-3' or '5'")
@click.option("--slugs", default=None, help="Comma-separated exercise slugs")
@click.option("--n", type=int, default=None, help="Limit to N exercises")
@click.option("--model", default=None, help="provider:model for one_shot/spell (required for those conditions)")
@click.option("--claude-model", default=None, help="Model for Claude Code comparator (required for claude_code condition)")
@click.option("--codex-model", default="codex", help="Model for Codex harness comparator")
@click.option("--swe-agent-model", default=None, help="LiteLLM model for mini-swe-agent comparator")
@click.option("--parallel", type=int, default=1, show_default=True)
@click.option("--budget", type=float, default=2.0, show_default=True)
@click.option("--agent", default=None, help="Spell agent override")
@click.option("--agent-msg", default=None, help="Spell msg-mode agent override")
@click.option("--agent-tc", default=None, help="Spell tc-mode agent override")
@click.option("--depth", type=int, default=8, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--timeout", type=int, default=None, help="Per-item timeout seconds (default: from _DATASET_TIMEOUTS)")
@click.option("--reasoning-effort", type=click.Choice(["low", "medium", "high"]), default="high", show_default=True)
@click.option("--no-retry", is_flag=True, default=False, help="Skip retry on test failure")
@click.option("--web-search/--no-web-search", default=False, show_default=True, help="Enable Codex live web search")
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--name", default=None, help="Label for this run variant (flows into traces and JSONL records)")
@click.option("--trailing", default=None, help="Custom Spell trailing expression (overrides default '(!extend))")
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
    swe_agent_model: str | None,
    parallel: int,
    budget: float,
    agent: str | None,
    agent_msg: str | None,
    agent_tc: str | None,
    depth: int,
    max_tokens: int | None,
    timeout: int | None,
    reasoning_effort: str | None,
    no_retry: bool,
    web_search: bool,
    trace: bool,
    name: str | None = None,
    trailing: str | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    output: str | None = None,
) -> None:
    """Run Exercism Python benchmark with two-attempt protocol."""
    if timeout is None:
        timeout = _DATASET_TIMEOUTS.get("exercism", _DEFAULT_TIMEOUT)
    run_id = run_id or _run_id("exercism")
    output_file = Path(output) if output else _default_output(run_id)
    conditions = _parse_conditions(condition, ("one_shot", "spell", "claude_code", "codex", "swe_agent", "all"))

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
    if name:
        click.echo(f"Name: {name}")
    click.echo(f"Exercises: {len(items)}")
    click.echo("Container mode: shared software benchmark container infra (except one_shot)")
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
                    "swe_agent_model": swe_agent_model,
                    "budget": budget,
                    "depth": depth,
                    "no_retry": no_retry,
                    "name": name,
                    "trailing": trailing,
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
        adapter = _build_adapter(
            cond,
            model=model,
            claude_model=claude_model,
            codex_model=codex_model,
            swe_agent_model=swe_agent_model,
            budget=budget,
        ) if cond == "one_shot" else None
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
                if cond != "one_shot":
                    outcome = run_unified_exercism(
                        condition=cond,
                        item=item,
                        workspace=ws,
                        model_spec=_condition_model_spec(
                            cond,
                            model=model,
                            claude_model=claude_model,
                            codex_model=codex_model,
                            swe_agent_model=swe_agent_model,
                        ),
                        timeout=timeout,
                        init=_make_spell_init(format_unified_exercism_prompt(item), trailing) if cond == "spell" else None,
                        agent_file=spell_resolution.agent_file if (spell_resolution and cond == "spell") else None,
                        budget=budget,
                        depth=depth,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        web_search=web_search if cond == "codex" else False,
                        trace=trace,
                        trace_dir=_trace_dir_for_request(trace_dir, cond, item.slug, name=name, attempt=1),
                        no_retry=no_retry,
                    )
                    last_resp = outcome.response
                    status = "passed" if outcome.passed else last_resp.status
                    if not outcome.passed and not last_resp.error_message:
                        status = "tests_failed"
                    return _mk_record(
                        run_id=run_id,
                        benchmark="exercism",
                        dataset="exercism",
                        split=None,
                        item_id=item.slug,
                        condition=cond,
                        name=name,
                        model_spec=_condition_model_spec(
                            cond,
                            model=model,
                            claude_model=claude_model,
                            codex_model=codex_model,
                            swe_agent_model=swe_agent_model,
                        ),
                        agent_file=spell_resolution.agent_file if (spell_resolution and cond == "spell") else None,
                        response=_rewrap_response(
                            last_resp,
                            status=status,
                            metadata={**last_resp.metadata, "test_output": outcome.test_output[-2000:]},
                        ),
                        response_text=last_resp.response_text,
                        patch=None,
                        answer_extracted=None,
                        correct=outcome.passed,
                        metadata={
                            "difficulty": item.difficulty,
                            "attempt": outcome.attempts,
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
                        reasoning_effort=reasoning_effort,
                    )

                test_path = ws / item.test_file
                total_cost = 0.0
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
                    model_spec=_condition_model_spec(
                        cond,
                        model=model,
                        claude_model=claude_model,
                        codex_model=codex_model,
                        swe_agent_model=swe_agent_model,
                    ),
                    timeout=timeout,
                    init=_make_spell_init(prompt, trailing) if cond == "spell" else None,
                    agent_file=spell_resolution.agent_file if (spell_resolution and cond == "spell") else None,
                    budget=budget,
                    depth=depth,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    web_search=web_search if cond == "codex" else False,
                    trace=trace,
                    trace_dir=_trace_dir_for_request(trace_dir, cond, item.slug, name=name, attempt=1),
                    workspace=ws if cond != "one_shot" else None,
                )
                assert adapter is not None
                resp1 = adapter.run(req)

                # For one_shot, extract code from response and write to file
                if cond == "one_shot" and resp1.response_text:
                    code = _extract_python_program(resp1.response_text)
                    if code:
                        (ws / item.stub_file).write_text(code, encoding="utf-8")

                total_cost += resp1.cost_usd or 0.0

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
                        init=_make_spell_init(retry_prompt, trailing) if cond == "spell" else None,
                        agent_file=req.agent_file,
                        budget=budget,
                        depth=depth,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        web_search=web_search if cond == "codex" else False,
                        trace=trace,
                        trace_dir=_trace_dir_for_request(trace_dir, cond, item.slug, name=name, attempt=2),
                        workspace=ws if cond != "one_shot" else None,
                    )
                    resp2 = adapter.run(req2)

                    if cond == "one_shot" and resp2.response_text:
                        code = _extract_python_program(resp2.response_text)
                        if code:
                            (ws / item.stub_file).write_text(code, encoding="utf-8")

                    total_cost += resp2.cost_usd or 0.0

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
                    name=name,
                    model_spec=req.model_spec,
                    agent_file=req.agent_file,
                    response=RunResponse(
                        status=status,
                        response_text=last_resp.response_text,
                        usage=sum_usages(resp1.usage, resp2.usage),
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
                    reasoning_effort=reasoning_effort,
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

        _print_summary(output_file, run_id, cond, name=name)


@cli.command("summarize")
@click.option("--input", "input_path", required=True, help="Input JSONL path")
@click.option("--run-id", default=None, help="Optional run-id filter")
@click.option("--condition", default=None, help="Optional condition filter")
@click.option("--name", default=None, help="Optional name filter")
def summarize(input_path: str, run_id: str | None, condition: str | None, name: str | None) -> None:
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
            if name is not None and row.get("name") != name:
                continue
            rows.append(row)

    summary = summarize_records(rows)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
