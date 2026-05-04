"""Unified adapter interface for benchmark execution conditions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from .docker_agents import (
    build_spell_init,
    filter_codex_raw_events,
    swe_agent_env_vars,
    swe_agent_run_command,
    _extract_swe_agent_cost,
    _extract_swe_agent_response,
)
from .codex_usage import (
    extract_codex_thread_id,
    merge_codex_usage,
    normalize_codex_model,
    parse_codex_rollout,
    usage_from_events,
    usage_from_rollout,
)
from .usage_accounting import (
    UsageBreakdown,
    claude_code_usage_from_events,
    claude_code_usage_from_text,
    usage_from_totals,
)
from .reasoning_effort import effective_reasoning_effort

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def _resolve_spell_root() -> Path:
    env_spell_root = os.environ.get("SPELL_ROOT")
    if env_spell_root:
        return Path(env_spell_root).expanduser().resolve()

    nested_candidate = BENCHMARK_ROOT.parent
    if (nested_candidate / "config" / "agents").exists():
        return nested_candidate

    sibling_candidate = BENCHMARK_ROOT.parent / "spell"
    if (sibling_candidate / "config" / "agents").exists():
        return sibling_candidate

    return BENCHMARK_ROOT


PROJECT_ROOT = _resolve_spell_root()
CODEX_NO_CUSTOM_SCRIPT = BENCHMARK_ROOT / "scripts" / "codex-no-custom.sh"
for p in (PROJECT_ROOT, BENCHMARK_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from spell_benchmark_client import SpellBenchmarkClient

DEFAULT_REASONING_EFFORT = "high"
SOFTWARE_BENCHMARKS = {"swebench", "featurebench", "scienceagentbench", "exercism"}
NONSOFTWARE_DEFAULT_FORMAT: dict[str, list[str]] = {"required": ["result"]}

def _benchmark_api_clj_cmd(project_root: Path) -> list[str]:
    # Allow spell.benchmark-api invocations from external workspace CWDs
    # by injecting this repo as a local dep.
    sdeps = '{:deps {local/spell {:local/root "%s"}}}' % project_root
    return ["clj", "-Sdeps", sdeps, "-M", "-m", "spell.benchmark-api"]


@contextmanager
def _benchmark_run_cwd(workspace: Path | None):
    if workspace is not None:
        yield workspace
        return

    with tempfile.TemporaryDirectory(prefix="spell-bench-") as tmp_dir:
        yield Path(tmp_dir)

def _default_format_for_spell(req: "RunRequest") -> dict[str, list[str]] | None:
    if req.benchmark in SOFTWARE_BENCHMARKS:
        return None
    return NONSOFTWARE_DEFAULT_FORMAT


@dataclass(slots=True)
class RunRequest:
    benchmark: str
    dataset: str
    item_id: str
    condition: str
    prompt: str
    model_spec: str | None
    timeout: int
    init: str | None = None
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
    format_spec: dict[str, Any] | None = None
    trace: bool = False
    trace_dir: Path | None = None
    workspace: Path | None = None
    web_search: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResponse:
    status: str
    response_text: str | None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached_input_tokens: int | None = None
    usage: UsageBreakdown | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    error_message: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.usage is None and (
            self.tokens_in is not None or self.tokens_out is not None or self.cached_input_tokens is not None
        ):
            self.usage = usage_from_totals(
                total_input_tokens=self.tokens_in,
                total_output_tokens=self.tokens_out,
                cached_input_tokens=self.cached_input_tokens or 0,
                cost_usd=self.cost_usd,
            )
        elif self.usage is not None:
            self.tokens_in = self.usage.total_input_tokens or None
            self.tokens_out = self.usage.output_tokens or None
            total_cached_tokens = self.usage.cached_input_tokens + self.usage.cache_write_input_tokens
            self.cached_input_tokens = total_cached_tokens or None


class RunnerAdapter(Protocol):
    condition: str

    def run(self, req: RunRequest) -> RunResponse:
        ...


def _usage_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _spell_usage_breakdown(usage_total: dict[str, Any]) -> UsageBreakdown | None:
    uncached_input_tokens = _usage_int(usage_total.get("uncached_input_tokens"))
    cached_input_tokens = _usage_int(usage_total.get("cached_input_tokens"))
    cache_write_input_tokens = _usage_int(usage_total.get("cache_write_input_tokens"))
    visible_output_tokens = _usage_int(usage_total.get("visible_output_tokens"))
    reasoning_output_tokens = _usage_int(usage_total.get("reasoning_output_tokens"))

    if not any(
        (
            uncached_input_tokens,
            cached_input_tokens,
            cache_write_input_tokens,
            visible_output_tokens,
            reasoning_output_tokens,
        )
    ):
        uncached_input_tokens = _usage_int(usage_total.get("input_tokens"))
        cached_input_tokens = _usage_int(usage_total.get("cache_read_input_tokens"))
        cache_write_input_tokens = _usage_int(usage_total.get("cache_creation_input_tokens"))
        visible_output_tokens = _usage_int(usage_total.get("output_tokens"))

    return usage_from_totals(
        total_input_tokens=uncached_input_tokens + cached_input_tokens + cache_write_input_tokens,
        total_output_tokens=visible_output_tokens + reasoning_output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cost_usd=(float(usage_total["cost"]) if isinstance(usage_total.get("cost"), (int, float)) else None),
    )


class SpellAdapter:
    condition = "spell"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec
        self.client = SpellBenchmarkClient(project_root=project_root)
        self.client.clj_cmd = _benchmark_api_clj_cmd(project_root)

    def run(self, req: RunRequest) -> RunResponse:
        format_spec = req.format_spec or _default_format_for_spell(req)
        model_spec = req.model_spec or self.model_spec
        reasoning_effort = effective_reasoning_effort(
            self.condition,
            model_spec,
            req.reasoning_effort or DEFAULT_REASONING_EFFORT,
        )
        payload = {
            "mode": "spell",
            "model": model_spec,
            "agent": req.agent_file,
            "budget": req.budget,
            "depth": req.depth,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": reasoning_effort,
            "verbosity": req.verbosity,
            "responses_api": req.responses_api,
            "suffix_grammar": req.suffix_grammar,
            "grammar_max_chars": req.grammar_max_chars,
            "format": format_spec,
            "trace": req.trace,
        }
        if req.trace_dir is not None:
            payload["trace_dir"] = str(req.trace_dir)
        if req.init is not None:
            payload["init"] = req.init
        else:
            payload["prompt"] = req.prompt
        payload = {k: v for k, v in payload.items() if v is not None}

        # Use a temp directory when no workspace is provided so generated
        # files (solve.py, compiled binaries, etc.) don't pollute the repo root.
        with _benchmark_run_cwd(req.workspace) as run_cwd:
            response = self.client.run(payload, timeout=req.timeout, cwd=run_cwd)
        usage_total = (response.usage or {}).get("total", {})
        usage = _spell_usage_breakdown(usage_total)

        if response.ok:
            return RunResponse(
                status="ok",
                response_text=response.result,
                usage=usage,
                cost_usd=usage_total.get("cost"),
                latency_ms=response.latency_ms,
                metadata={"trace_dir": response.trace_dir, "raw": response.raw},
            )

        return RunResponse(
            status="timeout" if response.error_type == "timeout" else "error",
            response_text=response.result,
            usage=usage,
            cost_usd=usage_total.get("cost"),
            latency_ms=response.latency_ms,
            error_message=response.error,
            error_type=response.error_type,
            metadata={"raw": response.raw},
        )


class OneShotAdapter:
    condition = "one_shot"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec
        self.client = SpellBenchmarkClient(project_root=project_root)
        self.client.clj_cmd = _benchmark_api_clj_cmd(project_root)

    def run(self, req: RunRequest) -> RunResponse:
        model_spec = req.model_spec or self.model_spec
        reasoning_effort = effective_reasoning_effort(
            self.condition,
            model_spec,
            req.reasoning_effort or DEFAULT_REASONING_EFFORT,
        )
        payload = {
            "mode": "one_shot",
            "prompt": req.prompt,
            "model": model_spec,
            "budget": req.budget,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": reasoning_effort,
            "verbosity": req.verbosity,
            "responses_api": req.responses_api,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        with _benchmark_run_cwd(req.workspace) as run_cwd:
            response = self.client.run(payload, timeout=req.timeout, cwd=run_cwd)
        usage_total = (response.usage or {}).get("total", {})
        usage = _spell_usage_breakdown(usage_total)

        if response.ok:
            return RunResponse(
                status="ok",
                response_text=response.result,
                usage=usage,
                cost_usd=usage_total.get("cost"),
                latency_ms=response.latency_ms,
                metadata={"raw": response.raw},
            )

        return RunResponse(
            status="timeout" if response.error_type == "timeout" else "error",
            response_text=response.result,
            usage=usage,
            cost_usd=usage_total.get("cost"),
            latency_ms=response.latency_ms,
            error_message=response.error,
            error_type=response.error_type,
            metadata={"raw": response.raw},
        )


class SweAgentAdapter:
    condition = "swe_agent"

    def __init__(self, model: str, budget: float):
        self.model = model
        self.budget = budget

    def run(self, req: RunRequest) -> RunResponse:
        model = req.model_spec or self.model
        budget = req.budget if req.budget is not None else self.budget
        env = os.environ.copy()
        env.update(swe_agent_env_vars())
        runtime_base = Path(os.environ.get("MINI_SWE_AGENT_HOST_RUNTIME_BASE", "/tmp/bench-agent-runtime/mini-host"))
        runtime_base.mkdir(parents=True, exist_ok=True)
        host_bin = os.environ.get("MINI_SWE_AGENT_HOST_BIN", str(Path.home() / ".local" / "bin"))

        start = time.time()
        try:
            with _benchmark_run_cwd(req.workspace) as run_cwd, tempfile.TemporaryDirectory(prefix="run-", dir=runtime_base) as runtime_tmp:
                runtime_root = Path(runtime_tmp)
                cmd = swe_agent_run_command(
                    req.prompt,
                    model=model,
                    budget=budget,
                    workdir=str(run_cwd),
                    runtime_bin=host_bin,
                    runtime_root=str(runtime_root),
                )
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=req.timeout,
                    cwd=None,
                    env=env,
                    shell=True,
                )
                latency_ms = (time.time() - start) * 1000
                stdout = (proc.stdout or "").strip()
                stderr = (proc.stderr or "").strip()
                traj_path, traj = _load_swe_agent_trajectory(runtime_root / "trajectories")
                response_text = _extract_swe_agent_response(traj) if traj else stdout
                cost_usd = _extract_swe_agent_cost(traj) if traj else None

                trace_file: Path | None = None
                if req.trace_dir is not None:
                    trace_payload = {
                        "item_id": req.item_id,
                        "condition": self.condition,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "model": model,
                        "prompt": req.prompt,
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": proc.returncode,
                        "latency_ms": latency_ms,
                        "trajectory_path": str(traj_path) if traj_path else None,
                        "trajectory": traj,
                    }
                    trace_file = _write_trace(req.trace_dir, req.item_id, self.condition, trace_payload)

                metadata = {
                    "trajectory_path": str(traj_path) if traj_path else None,
                    "trace_file": str(trace_file) if trace_file else None,
                    "stdout_tail": stdout[-4000:],
                    "stderr_tail": stderr[-4000:],
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}

                if proc.returncode == 0 and traj is not None:
                    return RunResponse(
                        status="ok",
                        response_text=response_text or None,
                        usage=None,
                        cost_usd=cost_usd,
                        latency_ms=latency_ms,
                        metadata=metadata,
                    )

                error_message = stderr or f"Exit code {proc.returncode}"
                error_type = "parse_error" if proc.returncode == 0 and traj is None else "mini_cli_error"
                if proc.returncode == 0 and traj is None:
                    error_message = f"No .traj.json file found under {run_cwd}"

                return RunResponse(
                    status="error",
                    response_text=response_text or None,
                    usage=None,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    error_message=error_message,
                    error_type=error_type,
                    metadata=metadata,
                )
        except subprocess.TimeoutExpired:
            return RunResponse(
                status="timeout",
                response_text="",
                usage=None,
                cost_usd=None,
                latency_ms=(time.time() - start) * 1000,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
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

        oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if not oauth_token:
            return RunResponse(
                status="error",
                response_text=None,
                usage=None,
                cost_usd=None,
                latency_ms=0.0,
                error_message=(
                    "CLAUDE_CODE_OAUTH_TOKEN is required for Claude Code benchmark runs. "
                    "Run `claude setup-token` and export/store that token; Claude Code harnesses "
                    "do not accept ANTHROPIC_API_KEY or CLAUDE_JSON_B64."
                ),
                error_type="auth_error",
            )

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"ANTHROPIC_API_KEY", "CLAUDE_JSON_B64", "CLAUDECODE"}
        }
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        start = time.time()
        debug_text = None
        session_text = None
        try:
            with tempfile.TemporaryDirectory(prefix="spell-claude-code-home-") as isolated_home:
                env["HOME"] = isolated_home
                debug_file = Path(isolated_home) / "claude-debug.log"
                cmd = [
                    "claude",
                    "--setting-sources",
                    "",
                    "--debug-file",
                    str(debug_file),
                    "-p",
                    "--output-format",
                    "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",
                    "--model",
                    model,
                    "--max-budget-usd",
                    str(req.budget if req.budget is not None else self.budget),
                ]
                if req.reasoning_effort:
                    cmd.extend(["--effort", req.reasoning_effort])
                cmd.append(req.prompt)

                if add_dir := req.metadata.get("add_dir"):
                    cmd[1:1] = ["--add-dir", str(add_dir)]

                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=req.timeout,
                        cwd=str(req.workspace) if req.workspace else None,
                        env=env,
                    )
                except subprocess.TimeoutExpired as exc:
                    stdout = (exc.stdout or "").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                    stderr = (exc.stderr or "").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                    if debug_file.exists():
                        debug_text = debug_file.read_text(encoding="utf-8", errors="replace")
                    tmp_events, _ = _parse_jsonl_events(stdout.strip())
                    tmp_session_id = None
                    for event in tmp_events:
                        candidate_session_id = event.get("session_id") or event.get("sessionId")
                        if isinstance(candidate_session_id, str) and candidate_session_id:
                            tmp_session_id = candidate_session_id
                            break
                    if tmp_session_id:
                        session_matches = list((Path(isolated_home) / ".claude" / "projects").rglob(f"{tmp_session_id}.jsonl"))
                        if session_matches:
                            session_text = session_matches[0].read_text(encoding="utf-8", errors="replace")
                    response_usage = _best_claude_usage_from_ledgers(
                        debug_text=debug_text,
                        session_text=session_text,
                        model=model,
                    )
                    if response_usage is None:
                        response_usage = claude_code_usage_from_events(
                            tmp_events,
                            model=model,
                            assistant_cost_source="estimated_partial",
                        )
                    return RunResponse(
                        status="timeout",
                        response_text="",
                        usage=response_usage,
                        cost_usd=response_usage.cost_usd if response_usage else None,
                        latency_ms=(time.time() - start) * 1000,
                        error_message=f"Timeout ({req.timeout}s)",
                        error_type="timeout",
                        metadata={
                            "session_id": tmp_session_id,
                            "cost_source": response_usage.cost_source if response_usage else None,
                            "claude_usage_ledger_sources": [
                                source
                                for source, text in (("debug", debug_text), ("ledger", session_text))
                                if text
                            ],
                            "stderr": stderr[:4000],
                        },
                    )
                if debug_file.exists():
                    debug_text = debug_file.read_text(encoding="utf-8", errors="replace")
                tmp_events, _ = _parse_jsonl_events((proc.stdout or "").strip())
                tmp_session_id = None
                for event in tmp_events:
                    candidate_session_id = event.get("session_id") or event.get("sessionId")
                    if isinstance(candidate_session_id, str) and candidate_session_id:
                        tmp_session_id = candidate_session_id
                        break
                if tmp_session_id:
                    session_matches = list((Path(isolated_home) / ".claude" / "projects").rglob(f"{tmp_session_id}.jsonl"))
                    if session_matches:
                        session_text = session_matches[0].read_text(encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return RunResponse(
                status="timeout",
                response_text="",
                usage=None,
                cost_usd=None,
                latency_ms=(time.time() - start) * 1000,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        latency_ms = (time.time() - start) * 1000
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        events, _ = _parse_jsonl_events(stdout)

        result_event: dict[str, Any] | None = None
        for event in reversed(events):
            if event.get("type") == "result":
                result_event = event
                break

        session_id = result_event.get("session_id") if isinstance(result_event, dict) else None
        if session_id is None:
            for event in events:
                candidate_session_id = event.get("session_id") or event.get("sessionId")
                if isinstance(candidate_session_id, str) and candidate_session_id:
                    session_id = candidate_session_id
                    break
        if session_id and session_text is None:
            session_matches = list((Path(env["HOME"]) / ".claude" / "projects").rglob(f"{session_id}.jsonl"))
            if session_matches:
                session_text = session_matches[0].read_text(encoding="utf-8", errors="replace")

        if result_event:
            response_usage = claude_code_usage_from_events(events, model=model)
        else:
            response_usage = _best_claude_usage_from_ledgers(
                debug_text=debug_text,
                session_text=session_text,
                model=model,
            )
            if response_usage is None:
                response_usage = claude_code_usage_from_events(
                    events,
                    model=model,
                    assistant_cost_source="estimated_partial",
                )
        cost_usd = response_usage.cost_usd if response_usage else None
        response_text = str(result_event.get("result", "")) if isinstance(result_event, dict) else ""
        num_turns = result_event.get("num_turns") if isinstance(result_event, dict) else None

        trace_file: Path | None = None
        if req.trace_dir is not None:
            trace_payload = {
                "item_id": req.item_id,
                "condition": self.condition,
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model,
                "prompt": req.prompt,
                "events": events,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "latency_ms": latency_ms,
            }
            trace_file = _write_trace(req.trace_dir, req.item_id, self.condition, trace_payload)

        metadata = {
            "num_turns": num_turns,
            "session_id": session_id,
            "cost_source": response_usage.cost_source if response_usage else None,
            "claude_usage_ledger_sources": [
                source
                for source, text in (("debug", debug_text), ("ledger", session_text))
                if text
            ],
            "stderr": stderr[:4000],
            "trace_file": str(trace_file) if trace_file else None,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        if proc.returncode == 0 and result_event:
            return RunResponse(
                status="ok",
                response_text=response_text,
                usage=response_usage,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata=metadata,
            )

        error_message = stderr or f"Exit code {proc.returncode}"
        if not result_event and stdout:
            error_message = f"No result event found in Claude stream-json output: {stdout[:400]}"

        return RunResponse(
            status="error",
            response_text=response_text,
            usage=response_usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error_message=error_message,
            error_type="cli_error",
            metadata=metadata,
        )


def _parse_jsonl_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    non_json_lines: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                events.append(parsed)
            else:
                non_json_lines.append(stripped)
        except json.JSONDecodeError:
            non_json_lines.append(stripped)
    return events, non_json_lines


def _best_claude_usage_from_ledgers(
    *,
    debug_text: str | None,
    session_text: str | None,
    model: str | None,
) -> UsageBreakdown | None:
    usages: list[UsageBreakdown] = []
    if debug_text:
        usage = claude_code_usage_from_text(
            debug_text,
            model=model,
            assistant_cost_source="estimated_debug",
        )
        if usage is not None:
            usages.append(usage)
    if session_text:
        usage = claude_code_usage_from_text(
            session_text,
            model=model,
            assistant_cost_source="estimated_ledger",
        )
        if usage is not None:
            usages.append(usage)
    if not usages:
        return None
    return max(usages, key=lambda usage: (usage.total_output_tokens, usage.total_input_tokens))


def _extract_codex_error(events: list[dict[str, Any]], stderr: str, return_code: int) -> str:
    for event in reversed(events):
        if event.get("type") == "turn.failed":
            err = event.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        if event.get("type") == "error":
            msg = event.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "error":
            msg = item.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    return stderr or f"Exit code {return_code}"


def _extract_codex_response_text(events: list[dict[str, Any]]) -> str:
    def content_text(v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            for key in ("text", "value", "content"):
                text = content_text(v.get(key))
                if text:
                    return text
            return ""
        if isinstance(v, list):
            parts = [content_text(x) for x in v]
            parts = [p for p in parts if p]
            return "\n".join(parts).strip()
        return ""

    for event in reversed(events):
        item = event.get("item")
        if isinstance(item, dict):
            role = item.get("role")
            item_type = item.get("type")
            if role == "assistant" or item_type in {"assistant_message", "message"}:
                text = content_text(item)
                if text:
                    return text
    return ""


def _write_trace(trace_dir: Path, item_id: str, condition: str, data: dict[str, Any]) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{item_id}_{condition}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def _find_swe_agent_trajectory(search_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in search_dir.rglob("*"):
        if path.is_file() and path.name.endswith(".traj.json"):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _load_swe_agent_trajectory(search_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = _find_swe_agent_trajectory(search_dir)
    if path is None:
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, None
    return path, data if isinstance(data, dict) else None


def _find_codex_rollout_for_thread(sessions_root: Path, thread_id: str | None) -> Path | None:
    if not thread_id or not sessions_root.exists():
        return None
    matches = list(sessions_root.rglob(f"rollout-*-{thread_id}.jsonl"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


class CodexAdapter:
    condition = "codex"

    def __init__(self, project_root: Path, model_spec: str = "codex", sessions_root: Path | None = None):
        self.project_root = project_root
        self.model_spec = model_spec
        self.sessions_root = sessions_root or (Path.home() / ".codex" / "sessions")

    def run(self, req: RunRequest) -> RunResponse:
        model = normalize_codex_model(req.model_spec, self.model_spec)
        reasoning_effort = req.reasoning_effort or DEFAULT_REASONING_EFFORT
        add_dir = req.metadata.get("add_dir")

        # Use config override for sandbox mode. In some Codex-hosted environments,
        # `--sandbox workspace-write` can be ignored due inherited wrapper settings.
        # `-c sandbox_mode="workspace-write"` is applied reliably.
        codex_launcher = str(CODEX_NO_CUSTOM_SCRIPT) if CODEX_NO_CUSTOM_SCRIPT.exists() else "codex"
        base_cmd = [
            codex_launcher,
            "--ask-for-approval",
            "never",
            "-c",
            'sandbox_mode="workspace-write"',
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if req.web_search:
            base_cmd.append("--search")
        else:
            base_cmd.extend(["-c", 'web_search="disabled"'])
        if add_dir:
            base_cmd.extend(["--add-dir", str(add_dir)])

        with tempfile.NamedTemporaryFile(prefix="codex_last_message_", suffix=".txt", delete=False) as last_msg_file:
            last_msg_path = Path(last_msg_file.name)

        cmd = [
            *base_cmd,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-m",
            model,
            "--output-last-message",
            str(last_msg_path),
            req.prompt,
        ]

        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=req.timeout,
                cwd=str(req.workspace or self.project_root),
            )
        except subprocess.TimeoutExpired:
            last_msg_path.unlink(missing_ok=True)
            return RunResponse(
                status="timeout",
                response_text="",
                usage=None,
                cost_usd=None,
                latency_ms=(time.time() - start) * 1000,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        latency_ms = (time.time() - start) * 1000
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        events, non_json_lines = _parse_jsonl_events(stdout)
        filtered_events = filter_codex_raw_events(events)
        stream_usage = usage_from_events(events, model=model)
        thread_id = extract_codex_thread_id(events)
        rollout_path = _find_codex_rollout_for_thread(self.sessions_root, thread_id)
        rollout_text = None
        if rollout_path is not None and rollout_path.exists():
            try:
                rollout_text = rollout_path.read_text(encoding="utf-8")
            except OSError:
                rollout_text = None
        rollout_info = parse_codex_rollout(rollout_text) if rollout_text else None
        rollout_usage = usage_from_rollout(rollout_text, model=model) if rollout_text else None
        usage = merge_codex_usage(stream_usage, rollout_usage, model=model)
        cached_input_tokens = usage.cached_input_tokens if usage else None
        cost_usd = usage.cost_usd if usage else None
        reasoning_output_tokens = usage.reasoning_output_tokens if usage else None

        if thread_id is None:
            rollout_status = "missing_thread_id"
        elif rollout_path is None:
            rollout_status = "missing"
        elif rollout_info is None:
            rollout_status = "no_token_count_event"
        else:
            rollout_status = "ok"

        response_text = ""
        if last_msg_path.exists():
            try:
                response_text = last_msg_path.read_text(encoding="utf-8").strip()
            finally:
                last_msg_path.unlink(missing_ok=True)
        if not response_text:
            response_text = _extract_codex_response_text(events)

        trace_file: Path | None = None
        if req.trace_dir is not None:
            trace_payload = {
                "item_id": req.item_id,
                "condition": self.condition,
                "timestamp": datetime.now(UTC).isoformat(),
                "model": model,
                "prompt": req.prompt,
                "events": events,
                "non_json_lines": non_json_lines,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "latency_ms": latency_ms,
            }
            trace_file = _write_trace(req.trace_dir, req.item_id, self.condition, trace_payload)

        metadata = {
            "requested_model": (req.model_spec or self.model_spec or "codex"),
            "resolved_model": model,
            "cost_source": usage.cost_source if usage else None,
            "cached_input_tokens": cached_input_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "reasoning_effort": reasoning_effort,
            "no_custom_config": codex_launcher != "codex",
            "thread_id": thread_id,
            "rollout_path": str(rollout_path) if rollout_path else None,
            "rollout_token_count": rollout_info,
            "rollout_status": rollout_status,
            "raw": {
                "events": filtered_events,
                "non_json_lines": non_json_lines,
                "rollout_token_count": rollout_info,
                "rollout_path": str(rollout_path) if rollout_path else None,
            },
            "stderr": stderr[:4000],
            "trace_file": str(trace_file) if trace_file else None,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None and v != [] and v != ""}

        if proc.returncode == 0:
            return RunResponse(
                status="ok",
                response_text=response_text,
                usage=usage,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata=metadata,
            )

        error_message = _extract_codex_error(events, stderr, proc.returncode)
        return RunResponse(
            status="error",
            response_text=response_text,
            usage=usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error_message=error_message,
            error_type="codex_cli_error",
            metadata=metadata,
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
                    usage=None,
                    cost_usd=None,
                    latency_ms=0.0,
                    error_message=str(exc),
                    error_type="adapter_exception",
                )
            out[idx] = resp
            if progress_callback:
                progress_callback(idx, req, resp)

    return [r for r in out if r is not None]
