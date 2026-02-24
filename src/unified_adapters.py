"""Unified adapter interface for benchmark execution conditions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spell_benchmark_client import SpellBenchmarkClient


@dataclass(slots=True)
class RunRequest:
    benchmark: str
    dataset: str
    item_id: str
    condition: str
    prompt: str
    model_spec: str | None
    timeout: int
    agent_file: str | None = None
    budget: float | None = None
    depth: int | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    responses_api: bool = False
    suffix_grammar: bool = False
    grammar_max_chars: int | None = None
    trace: bool = False
    workspace: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResponse:
    status: str
    response_text: str | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    latency_ms: float | None
    error_message: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RunnerAdapter(Protocol):
    condition: str

    def run(self, req: RunRequest) -> RunResponse:
        ...


class SpellAdapter:
    condition = "spell"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec
        self.client = SpellBenchmarkClient(project_root=project_root)

    def run(self, req: RunRequest) -> RunResponse:
        payload = {
            "mode": "spell",
            "prompt": req.prompt,
            "model": req.model_spec or self.model_spec,
            "agent": req.agent_file,
            "budget": req.budget,
            "depth": req.depth,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort,
            "verbosity": req.verbosity,
            "responses_api": req.responses_api,
            "suffix_grammar": req.suffix_grammar,
            "grammar_max_chars": req.grammar_max_chars,
            "trace": req.trace,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        response = self.client.run(payload, timeout=req.timeout, cwd=req.workspace or self.project_root)
        usage_total = (response.usage or {}).get("total", {})

        if response.ok:
            return RunResponse(
                status="ok",
                response_text=response.result,
                tokens_in=usage_total.get("input_tokens"),
                tokens_out=usage_total.get("output_tokens"),
                cost_usd=usage_total.get("cost"),
                latency_ms=response.latency_ms,
                metadata={"trace_dir": response.trace_dir, "raw": response.raw},
            )

        return RunResponse(
            status="timeout" if response.error_type == "timeout" else "error",
            response_text=response.result,
            tokens_in=usage_total.get("input_tokens"),
            tokens_out=usage_total.get("output_tokens"),
            cost_usd=usage_total.get("cost"),
            latency_ms=response.latency_ms,
            error_message=response.error,
            error_type=response.error_type,
            metadata={"raw": response.raw},
        )


class BaselineAdapter:
    condition = "baseline"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec
        self.client = SpellBenchmarkClient(project_root=project_root)

    def run(self, req: RunRequest) -> RunResponse:
        payload = {
            "mode": "baseline",
            "prompt": req.prompt,
            "model": req.model_spec or self.model_spec,
            "budget": req.budget,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort,
            "verbosity": req.verbosity,
            "responses_api": req.responses_api,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        response = self.client.run(payload, timeout=req.timeout, cwd=req.workspace or self.project_root)
        usage_total = (response.usage or {}).get("total", {})

        if response.ok:
            return RunResponse(
                status="ok",
                response_text=response.result,
                tokens_in=usage_total.get("input_tokens"),
                tokens_out=usage_total.get("output_tokens"),
                cost_usd=usage_total.get("cost"),
                latency_ms=response.latency_ms,
                metadata={"raw": response.raw},
            )

        return RunResponse(
            status="timeout" if response.error_type == "timeout" else "error",
            response_text=response.result,
            tokens_in=usage_total.get("input_tokens"),
            tokens_out=usage_total.get("output_tokens"),
            cost_usd=usage_total.get("cost"),
            latency_ms=response.latency_ms,
            error_message=response.error,
            error_type=response.error_type,
            metadata={"raw": response.raw},
        )


class ClaudeCodeAdapter:
    condition = "claude_code"

    def __init__(self, model: str, budget: float):
        self.model = model
        self.budget = budget

    def run(self, req: RunRequest) -> RunResponse:
        model = req.model_spec or self.model
        if model and ":" in model:
            _, model = model.split(":", 1)

        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--model",
            model,
            "--max-budget-usd",
            str(req.budget if req.budget is not None else self.budget),
            req.prompt,
        ]

        if add_dir := req.metadata.get("add_dir"):
            cmd[1:1] = ["--add-dir", str(add_dir)]

        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=req.timeout,
                cwd=str(req.workspace) if req.workspace else None,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return RunResponse(
                status="timeout",
                response_text="",
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=(time.time() - start) * 1000,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        latency_ms = (time.time() - start) * 1000
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        parsed: dict[str, Any] | None = None
        if stdout:
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None

        if proc.returncode == 0 and parsed:
            usage = parsed.get("usage", {})
            tokens_in = (
                usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
            )
            tokens_out = usage.get("output_tokens", 0)
            return RunResponse(
                status="ok",
                response_text=parsed.get("result", ""),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=parsed.get("total_cost_usd"),
                latency_ms=latency_ms,
                metadata={
                    "num_turns": parsed.get("num_turns"),
                    "session_id": parsed.get("session_id"),
                    "stderr": stderr,
                },
            )

        error_message = stderr or f"Exit code {proc.returncode}"
        if parsed is None and stdout:
            error_message = f"Invalid JSON output from claude CLI: {stdout[:400]}"

        return RunResponse(
            status="error",
            response_text=(parsed or {}).get("result", "") if isinstance(parsed, dict) else "",
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            latency_ms=latency_ms,
            error_message=error_message,
            error_type="cli_error",
            metadata={"stdout": stdout[:4000], "stderr": stderr[:4000]},
        )


class CodexAdapter:
    condition = "codex"

    def __init__(self, model: str):
        self.model = model

    def run(self, req: RunRequest) -> RunResponse:
        return RunResponse(
            status="not_implemented",
            response_text="",
            tokens_in=None,
            tokens_out=None,
            cost_usd=None,
            latency_ms=0.0,
            error_message=(
                "Codex adapter scaffold exists but backend is not enabled yet. "
                "Configure codex CLI/API invocation in benchmarking/src/unified_adapters.py"
            ),
            error_type="not_implemented",
            metadata={"model": req.model_spec or self.model},
        )


def run_requests_parallel(
    requests: list[RunRequest],
    adapter: RunnerAdapter,
    max_parallel: int,
    progress_callback: Callable[[int, RunRequest, RunResponse], None] | None = None,
) -> list[RunResponse]:
    if max_parallel <= 1:
        out: list[RunResponse] = []
        for idx, req in enumerate(requests):
            resp = adapter.run(req)
            out.append(resp)
            if progress_callback:
                progress_callback(idx, req, resp)
        return out

    out: list[RunResponse | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(adapter.run, req): idx for idx, req in enumerate(requests)}
        for fut in as_completed(futures):
            idx = futures[fut]
            req = requests[idx]
            try:
                resp = fut.result()
            except Exception as exc:  # pragma: no cover
                resp = RunResponse(
                    status="error",
                    response_text="",
                    tokens_in=None,
                    tokens_out=None,
                    cost_usd=None,
                    latency_ms=0.0,
                    error_message=str(exc),
                    error_type="adapter_exception",
                )
            out[idx] = resp
            if progress_callback:
                progress_callback(idx, req, resp)

    return [r for r in out if r is not None]
