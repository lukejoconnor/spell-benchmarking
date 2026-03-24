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
    swe_agent_env_vars,
    swe_agent_run_command,
    _extract_swe_agent_cost,
    _extract_swe_agent_response,
)

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def _ensure_java_home() -> None:
    """Ensure JAVA_HOME is set so clj can find the JVM on macOS."""
    if os.environ.get("JAVA_HOME"):
        return
    # Try /usr/libexec/java_home (macOS standard)
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            os.environ["JAVA_HOME"] = java_home
            os.environ["PATH"] = os.path.join(java_home, "bin") + os.pathsep + os.environ.get("PATH", "")
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Homebrew fallback
    for candidate in [
        "/usr/local/opt/openjdk/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home",
    ]:
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = os.path.join(candidate, "bin") + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_java_home()


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

# Per-MTok pricing [input, output] for cost estimation when the CLI
# doesn't report cost directly (e.g. Codex subscription mode).
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5.1-codex": (1.25, 10.00),
    "gpt-5.2-codex": (1.75, 14.00),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.3": (1.75, 14.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
}


def _estimate_cost(model: str, tokens_in: int | None, tokens_out: int | None) -> float | None:
    """Estimate cost from token counts using prefix-matched pricing."""
    if tokens_in is None or tokens_out is None:
        return None
    for prefix, (in_rate, out_rate) in _MODEL_PRICING.items():
        if model.startswith(prefix):
            return tokens_in * in_rate / 1_000_000 + tokens_out * out_rate / 1_000_000
    return None


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


def _escape_for_spell(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def build_spell_init(prompt: str, trailing: str | None = None) -> str:
    escaped_prompt = _escape_for_spell(prompt)
    trailing_expr = trailing or "'(!extend)"
    return (
        f'(quine completion (eval (do (quine prompt "{escaped_prompt}") '
        f"{trailing_expr})))"
    )


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
        self.client.clj_cmd = _benchmark_api_clj_cmd(project_root)

    def run(self, req: RunRequest) -> RunResponse:
        format_spec = req.format_spec or _default_format_for_spell(req)
        payload = {
            "mode": "spell",
            "model": req.model_spec or self.model_spec,
            "agent": req.agent_file,
            "budget": req.budget,
            "depth": req.depth,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort or DEFAULT_REASONING_EFFORT,
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


def build_raw_init(
    prompt: str,
    namespace: str,
    fn: str = "run",
    *,
    map_arg: bool = False,
    extra_map_opts: dict[str, str] | None = None,
) -> str:
    """Build a raw Spell init expression that calls (namespace/fn task).

    map_arg=True:  (eval (quote (ns/fn {:task "..."})))
    map_arg=False: (eval (quote (ns/fn "...")))
    extra_map_opts: additional keyword/value pairs to include in the map arg
    """
    escaped = _escape_for_spell(prompt)
    if map_arg:
        opts_str = f':task "{escaped}"'
        for k, v in (extra_map_opts or {}).items():
            opts_str += f' {k} "{_escape_for_spell(v)}"'
        arg = "{" + opts_str + "}"
    else:
        arg = f'"{escaped}"'
    return f"(eval (quote ({namespace}/{fn} {arg})))"


class RawInitSpellAdapter:
    """Spell adapter that builds a raw init expression instead of a quine-wrapped prompt.

    Use this for namespaced entry points like react/run, context-folding/run, rlm/run, etc.
    that should be invoked directly without the quine completion wrapper.
    """

    def __init__(
        self,
        condition: str,
        project_root: Path,
        model_spec: str,
        namespace: str,
        fn: str = "run",
        map_arg: bool = False,
        default_agent: str | None = None,
    ):
        self.condition = condition
        self.project_root = project_root
        self.model_spec = model_spec
        self.namespace = namespace
        self.fn = fn
        self.map_arg = map_arg
        self.default_agent = default_agent
        self.client = SpellBenchmarkClient(project_root=project_root)
        self.client.clj_cmd = _benchmark_api_clj_cmd(project_root)

    def run(self, req: RunRequest) -> RunResponse:
        search_url = getattr(self, "_search_url", None)
        extra_map_opts = {":search-url": search_url} if search_url else None
        init = build_raw_init(req.prompt, self.namespace, self.fn, map_arg=self.map_arg, extra_map_opts=extra_map_opts)
        agent_file = req.agent_file or (
            str(self.project_root / self.default_agent) if self.default_agent else None
        )
        payload = {
            "mode": "spell",
            "model": req.model_spec or self.model_spec,
            "agent": agent_file,
            "init": init,
            "budget": req.budget,
            "depth": req.depth,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort or DEFAULT_REASONING_EFFORT,
            "verbosity": req.verbosity,
            "trace": req.trace,
            "retries": [0, 10, 30, 60],  # aggressive retries for rate limits
        }
        if req.trace_dir is not None:
            payload["trace_dir"] = str(req.trace_dir)
        payload = {k: v for k, v in payload.items() if v is not None}

        # Use spell project root as CWD (not a temp dir) so config/web.edn is found.
        with _benchmark_run_cwd(req.workspace or self.project_root) as run_cwd:
            response = self.client.run(payload, timeout=req.timeout, cwd=run_cwd)
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


def make_context_folding_adapter(
    project_root: Path,
    model_spec: str,
    search_url: str | None = None,
    agent_file: str | None = None,
) -> RawInitSpellAdapter:
    adapter = RawInitSpellAdapter(
        condition="context_folding",
        project_root=project_root,
        model_spec=model_spec,
        namespace="context-folding",
        fn="run",
        map_arg=True,
        default_agent=agent_file or "config/agents/context-folding.agent.edn",
    )
    adapter._search_url = search_url
    return adapter


def make_react_adapter(project_root: Path, model_spec: str) -> RawInitSpellAdapter:
    return RawInitSpellAdapter(
        condition="react",
        project_root=project_root,
        model_spec=model_spec,
        namespace="react",
        fn="run",
        map_arg=False,
        default_agent="config/agents/react.agent.edn",
    )


class OneShotAdapter:
    condition = "one_shot"

    def __init__(self, project_root: Path, model_spec: str):
        self.project_root = project_root
        self.model_spec = model_spec
        self.client = SpellBenchmarkClient(project_root=project_root)
        self.client.clj_cmd = _benchmark_api_clj_cmd(project_root)

    def run(self, req: RunRequest) -> RunResponse:
        payload = {
            "mode": "one_shot",
            "prompt": req.prompt,
            "model": req.model_spec or self.model_spec,
            "budget": req.budget,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking_budget,
            "reasoning_effort": req.reasoning_effort or DEFAULT_REASONING_EFFORT,
            "verbosity": req.verbosity,
            "responses_api": req.responses_api,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        with _benchmark_run_cwd(req.workspace) as run_cwd:
            response = self.client.run(payload, timeout=req.timeout, cwd=run_cwd)
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

        start = time.time()
        try:
            with _benchmark_run_cwd(req.workspace) as run_cwd:
                cmd = swe_agent_run_command(
                    req.prompt,
                    model=model,
                    budget=budget,
                    workdir=str(run_cwd),
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
                traj_path, traj = _load_swe_agent_trajectory(run_cwd)
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
                        tokens_in=None,
                        tokens_out=None,
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
                    tokens_in=None,
                    tokens_out=None,
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
                tokens_in=None,
                tokens_out=None,
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

        cmd = [
            "claude",
            "--setting-sources",
            "",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
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
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        events, _ = _parse_jsonl_events(stdout)

        result_event: dict[str, Any] | None = None
        for event in reversed(events):
            if event.get("type") == "result":
                result_event = event
                break

        def as_int(v: Any) -> int:
            return int(v) if isinstance(v, (int, float)) else 0

        usage = result_event.get("usage", {}) if isinstance(result_event, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        tokens_in = as_int(usage.get("input_tokens")) + as_int(usage.get("cache_creation_input_tokens")) + as_int(
            usage.get("cache_read_input_tokens")
        )
        tokens_out = as_int(usage.get("output_tokens"))
        cost_raw = result_event.get("total_cost_usd") if isinstance(result_event, dict) else None
        cost_usd = float(cost_raw) if isinstance(cost_raw, (int, float)) else None
        response_text = str(result_event.get("result", "")) if isinstance(result_event, dict) else ""
        num_turns = result_event.get("num_turns") if isinstance(result_event, dict) else None
        session_id = result_event.get("session_id") if isinstance(result_event, dict) else None

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
            "stderr": stderr[:4000],
            "trace_file": str(trace_file) if trace_file else None,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        if proc.returncode == 0 and result_event:
            return RunResponse(
                status="ok",
                response_text=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
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
            tokens_in=tokens_in if result_event else None,
            tokens_out=tokens_out if result_event else None,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error_message=error_message,
            error_type="cli_error",
            metadata=metadata,
        )


def _normalize_codex_model(model_spec: str | None, fallback: str | None = None) -> str:
    raw = (model_spec or fallback or "codex").strip()
    if ":" in raw:
        _, raw = raw.split(":", 1)
    # Keep "codex" as the user-facing default, but resolve to the canonical
    # Codex CLI model id to avoid metadata fallback warnings.
    if raw in {"", "codex"}:
        return "gpt-5-codex"
    return raw


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


def _iter_dicts(v: Any):
    if isinstance(v, dict):
        yield v
        for child in v.values():
            yield from _iter_dicts(child)
    elif isinstance(v, list):
        for child in v:
            yield from _iter_dicts(child)


def _extract_codex_usage(events: list[dict[str, Any]], model: str = "") -> tuple[int | None, int | None, float | None]:
    def as_int(v: Any) -> int:
        return int(v) if isinstance(v, (int, float)) else 0

    usage_candidate: dict[str, Any] | None = None
    for event in events:
        for d in _iter_dicts(event):
            if any(k in d for k in ("input_tokens", "output_tokens", "total_tokens", "total_cost_usd", "cost_usd", "cost")):
                usage_candidate = d

    if not usage_candidate:
        return None, None, None

    base_input = usage_candidate.get("input_tokens")
    cache_creation = usage_candidate.get("cache_creation_input_tokens", 0)
    cache_read = usage_candidate.get("cache_read_input_tokens", 0)
    tokens_in = None
    if isinstance(base_input, (int, float)):
        tokens_in = as_int(base_input) + as_int(cache_creation) + as_int(cache_read)

    output = usage_candidate.get("output_tokens")
    tokens_out = int(output) if isinstance(output, (int, float)) else None

    cost = usage_candidate.get("total_cost_usd")
    if cost is None:
        cost = usage_candidate.get("cost_usd")
    if cost is None:
        cost = usage_candidate.get("cost")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else None

    # Fall back to token-based estimation when CLI doesn't report cost
    if cost_usd is None and model:
        cost_usd = _estimate_cost(model, tokens_in, tokens_out)

    return tokens_in, tokens_out, cost_usd


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


def _extract_codex_thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str):
                return thread_id
    return None


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


class CodexAdapter:
    condition = "codex"

    def __init__(self, project_root: Path, model_spec: str = "codex"):
        self.project_root = project_root
        self.model_spec = model_spec

    def run(self, req: RunRequest) -> RunResponse:
        model = _normalize_codex_model(req.model_spec, self.model_spec)
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
            "--ephemeral",
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
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                latency_ms=(time.time() - start) * 1000,
                error_message=f"Timeout ({req.timeout}s)",
                error_type="timeout",
            )

        latency_ms = (time.time() - start) * 1000
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        events, non_json_lines = _parse_jsonl_events(stdout)
        tokens_in, tokens_out, cost_usd = _extract_codex_usage(events, model=model)
        thread_id = _extract_codex_thread_id(events)

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
            "reasoning_effort": reasoning_effort,
            "no_custom_config": codex_launcher != "codex",
            "thread_id": thread_id,
            "events_tail": events[-20:],
            "non_json_stdout_tail": non_json_lines[-20:],
            "stderr": stderr[:4000],
            "trace_file": str(trace_file) if trace_file else None,
        }
        metadata = {k: v for k, v in metadata.items() if v}

        if proc.returncode == 0:
            return RunResponse(
                status="ok",
                response_text=response_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                metadata=metadata,
            )

        error_message = _extract_codex_error(events, stderr, proc.returncode)
        return RunResponse(
            status="error",
            response_text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
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
