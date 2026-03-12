"""Shared container-agent helpers for software benchmarks.

This module holds benchmark-agnostic pieces shared by Terminal-Bench and the
bench.py unified software benchmark paths:

- container lifecycle / exec helpers for copied workspaces
- agent install/env/command construction
- token / cost parsing
- Spell bundle preparation for in-container execution
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_WORKDIR = "/workspace"
SPELL_AGENT_IN_CONTAINER = "/opt/spell/config/agents/io-tc.agent.edn"
SPELL_SDEPS = '{:deps {local/spell {:local/root "/opt/spell"}}}'
SPELL_BUNDLE_PATHS = ("deps.edn", "src", "config")

CC_ALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep",
    "WebFetch", "NotebookEdit",
]

LITELLM_TO_SPELL_MODEL = {
    "anthropic/claude-opus-4-6-20250826": "anthropic-tc:opus46",
    "anthropic/claude-opus-4-6": "anthropic-tc:opus46",
    "anthropic/claude-opus-4-5-20251101": "anthropic-tc:opus",
    "anthropic/claude-opus-4-5": "anthropic-tc:opus",
    "anthropic/claude-sonnet-4-5-20250929": "anthropic-tc:sonnet",
    "anthropic/claude-sonnet-4-5": "anthropic-tc:sonnet",
    "anthropic/claude-haiku-4-5-20251001": "anthropic-tc:haiku",
    "anthropic/claude-haiku-4-5": "anthropic-tc:haiku",
    "anthropic/claude-sonnet-4-20250514": "anthropic-tc:sonnet",
    "anthropic/claude-opus-4-20250514": "anthropic-tc:opus",
}

NVM_NODE_INSTALL = r"""
set -e
if [ -d "$HOME/.nvm" ]; then
    source "$HOME/.nvm/nvm.sh" 2>/dev/null || true
    if command -v node >/dev/null 2>&1; then
        echo "node already installed: $(node --version)"
        exit 0
    fi
fi
apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install 22
echo "node installed: $(node --version)"
"""

SPELL_RUNTIME_INSTALL = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if ! command -v java >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl default-jre >/dev/null
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache ca-certificates curl bash openjdk17-jre >/dev/null
  elif command -v yum >/dev/null 2>&1; then
    yum install -y -q ca-certificates curl java-17-openjdk-headless >/dev/null
  else
    echo "Unable to install Java runtime: unsupported package manager" >&2
    exit 1
  fi
fi

if ! (command -v clojure >/dev/null 2>&1 && clojure -Sdescribe >/dev/null 2>&1) && ! command -v clj >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq curl >/dev/null
    elif command -v apk >/dev/null 2>&1; then
      apk add --no-cache curl >/dev/null
    elif command -v yum >/dev/null 2>&1; then
      yum install -y -q curl >/dev/null
    fi
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "Unable to install Clojure CLI: curl not available" >&2
    exit 1
  fi
  tmpdir="$(mktemp -d)"
  curl -fsSL https://download.clojure.org/install/linux-install-1.11.1.1413.sh -o "$tmpdir/install-clojure.sh"
  chmod +x "$tmpdir/install-clojure.sh"
  "$tmpdir/install-clojure.sh" >/dev/null
  rm -rf "$tmpdir"
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$HOME/.codex"
if [ -n "${CODEX_AUTH_JSON_B64:-}" ]; then
  printf '%s' "$CODEX_AUTH_JSON_B64" | base64 -d > "$HOME/.codex/auth.json"
fi

if [ ! -d /opt/spell ]; then
  echo "Spell bundle missing at /opt/spell" >&2
  exit 1
fi

if command -v clojure >/dev/null 2>&1 && clojure -Sdescribe >/dev/null 2>&1; then
  CLOJURE_BIN=clojure
elif command -v clj >/dev/null 2>&1; then
  CLOJURE_BIN=clj
else
  echo "Clojure CLI not available after installation" >&2
  exit 1
fi

"$CLOJURE_BIN" -Sdeps '{:deps {local/spell {:local/root "/opt/spell"}}}' -M -e "(require 'spell.benchmark-api)" >/dev/null

echo "spell runtime ready"
"""


@dataclass(slots=True)
class ExecOutput:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(slots=True)
class ParsedUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens


@dataclass(slots=True)
class AgentRunResult:
    output: ExecOutput
    usage: ParsedUsage | None = None
    response_text: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    raw: dict[str, Any] | None = None
    trace_dir: str | None = None


def parse_jsonl_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


class DockerWorkspaceContainer:
    """Container wrapper for copied workspaces.

    Worktrees are copied instead of bind-mounted because git worktrees carry
    host-path references in `.git` that are invalid inside the container.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        image: str = DEFAULT_IMAGE,
        container_workdir: str = DEFAULT_WORKDIR,
    ):
        self.workspace = Path(workspace)
        self.image = image
        self.container_workdir = container_workdir
        self.container_id: str | None = None
        self.initial_commit: str | None = None

    def start(self) -> str:
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--rm",
                "-w", self.container_workdir,
                self.image,
                "sleep", "infinity",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed: {result.stderr.strip()}")

        self.container_id = result.stdout.strip()
        self.exec_run(
            "command -v git >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1)",
            timeout=240,
            workdir="/",
        )
        self.exec_run(
            f"mkdir -p {shlex.quote(self.container_workdir)}",
            timeout=60,
            workdir="/",
        )

        cp_result = subprocess.run(
            ["docker", "cp", f"{self.workspace}/.", f"{self.container_id}:{self.container_workdir}"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if cp_result.returncode != 0:
            raise RuntimeError(f"docker cp failed: {cp_result.stderr.strip()}")

        git_result = self.exec_run(
            "rm -rf .git && git init -q && "
            "git config user.email 'bench@local' && "
            "git config user.name 'bench' && "
            "git add -A && git commit -q -m 'base'",
            timeout=120,
        )
        if git_result.exit_code != 0:
            raise RuntimeError(f"git init failed in container: {git_result.stderr.strip()}")

        head_result = self.exec_run("git rev-parse HEAD", timeout=30)
        if head_result.exit_code != 0:
            raise RuntimeError(f"git rev-parse HEAD failed in container: {head_result.stderr.strip()}")
        self.initial_commit = head_result.stdout.strip() or None

        return self.container_id

    def exec_run(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecOutput:
        if not self.container_id:
            raise RuntimeError("Container not started")

        docker_cmd = ["docker", "exec"]
        if workdir:
            docker_cmd.extend(["-w", workdir])
        for key, value in (env or {}).items():
            docker_cmd.extend(["-e", f"{key}={value}"])
        docker_cmd.extend([self.container_id, "bash", "-lc", cmd])

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecOutput(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            return ExecOutput(
                exit_code=124,
                stdout="",
                stderr=f"Timeout after {timeout}s",
                timed_out=True,
            )

    def exec(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecOutput:
        return self.exec_run(cmd, timeout=timeout, env=env, workdir=workdir)

    def put_tar_archive(self, dest_dir: str, archive_bytes: bytes) -> bool:
        if not self.container_id:
            raise RuntimeError("Container not started")
        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            tmp.write(archive_bytes)
            tmp_path = Path(tmp.name)
        try:
            archive_name = f"/tmp/{tmp_path.name}"
            cp = subprocess.run(
                ["docker", "cp", str(tmp_path), f"{self.container_id}:{archive_name}"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if cp.returncode != 0:
                logger.error("docker cp for tar archive failed: %s", cp.stderr.strip())
                return False
            unpack = self.exec_run(
                f"mkdir -p {shlex.quote(dest_dir)} && "
                f"tar -xf {shlex.quote(archive_name)} -C {shlex.quote(dest_dir)} && "
                f"rm -f {shlex.quote(archive_name)}",
                timeout=180,
                workdir="/",
            )
            return unpack.exit_code == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    def put_archive(self, dest_dir: str, archive_bytes: bytes) -> bool:
        return self.put_tar_archive(dest_dir, archive_bytes)

    def copy_to_host(self, source: str, dest: Path) -> bool:
        if not self.container_id:
            raise RuntimeError("Container not started")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["docker", "cp", f"{self.container_id}:{source}", str(dest)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            logger.warning("docker cp from container failed: %s", proc.stderr.strip())
            return False
        return True

    def read_text(self, path: str) -> str:
        result = self.exec_run(f"cat {shlex.quote(path)}", timeout=60, workdir="/")
        return result.stdout if result.exit_code == 0 else ""

    def host_to_container_path(self, host_path: Path | str) -> str:
        rel = Path(host_path).relative_to(self.workspace)
        return str(Path(self.container_workdir) / rel)

    def diff(self, base: str = "HEAD") -> str:
        result = self.exec_run(
            f"git add -A && git diff --cached {shlex.quote(base)}; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status"
        )
        if result.exit_code != 0:
            return ""
        return result.stdout

    def changed_files(self, base: str = "HEAD") -> list[str]:
        result = self.exec_run(
            f"git add -A && git diff --name-only --cached {shlex.quote(base)}; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status"
        )
        if result.exit_code != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def stop(self) -> None:
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "stop", self.container_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.container_id = None
        self.initial_commit = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.stop()


def install_nvm_node(container: DockerWorkspaceContainer) -> bool:
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        result = container.exec_run(NVM_NODE_INSTALL, timeout=300)
        if result.exit_code == 0:
            return True
        logger.error("NVM/Node install failed (attempt %s): %s", attempt, result.stderr[:500])
        if attempt < max_attempts:
            time.sleep(2)

    fallback = container.exec_run(
        "apt-get update -qq && apt-get install -y -qq nodejs npm >/dev/null 2>&1 && "
        'echo "node installed via apt: $(node --version 2>/dev/null || echo missing)"',
        timeout=300,
        workdir="/",
    )
    return fallback.exit_code == 0


def container_exec(
    container: Any,
    cmd: str,
    *,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
    workdir: str | None = None,
) -> ExecOutput:
    if isinstance(container, DockerWorkspaceContainer):
        return container.exec_run(cmd, timeout=timeout, env=env, workdir=workdir)

    if timeout:
        shell_cmd = f"timeout {timeout}s bash -c {shlex.quote(cmd)}"
    else:
        shell_cmd = cmd

    result = container.exec_run(
        ["bash", "-c", shell_cmd],
        environment=env or {},
        workdir=workdir,
        demux=True,
    )
    stdout_bytes, stderr_bytes = result.output or (b"", b"")
    stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
    stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
    timed_out = result.exit_code == 124 and timeout is not None
    return ExecOutput(
        exit_code=result.exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
    )


def install_nvm_node_exec(exec_fn: Callable[..., ExecOutput]) -> bool:
    result = exec_fn(NVM_NODE_INSTALL, timeout=300)
    if result.exit_code == 0:
        return True
    fallback = exec_fn(
        "apt-get update -qq && apt-get install -y -qq nodejs npm >/dev/null 2>&1",
        timeout=300,
        workdir="/",
    )
    return fallback.exit_code == 0


def ensure_uv(container: DockerWorkspaceContainer) -> bool:
    result = container.exec_run(
        "python -m pip install -q --disable-pip-version-check uv",
        timeout=180,
    )
    return result.exit_code == 0


def ensure_python_packages(container: DockerWorkspaceContainer, packages: list[str]) -> bool:
    if not packages:
        return True
    quoted = " ".join(shlex.quote(package) for package in packages)
    result = container.exec_run(
        f"python -m pip install -q --disable-pip-version-check {quoted}",
        timeout=300,
    )
    return result.exit_code == 0


def run_install_script(
    container: DockerWorkspaceContainer,
    script: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    workdir: str | None = None,
) -> bool:
    result = container.exec_run(script, timeout=timeout, env=env, workdir=workdir)
    if result.exit_code != 0:
        logger.error("Install script failed: %s", result.stderr[:500])
        return False
    return True


def _load_claude_json_b64() -> str | None:
    if os.environ.get("CLAUDE_JSON_B64"):
        return os.environ["CLAUDE_JSON_B64"]
    path = Path.home() / ".claude.json"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def cc_install_script(*, version: str | None = None) -> str:
    resolved_version = version or "latest"
    return f"""
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
npm install -g @anthropic-ai/claude-code@{resolved_version}
if [ -n "${{CLAUDE_JSON_B64:-}}" ]; then
  printf '%s' "$CLAUDE_JSON_B64" | base64 -d > "$HOME/.claude.json"
fi
echo "claude installed: $(claude --version 2>/dev/null || echo unknown)"
"""


def cc_env_vars(
    *,
    model: str | None = None,
    include_background: bool = True,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if include_background:
        env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
        env["ENABLE_BACKGROUND_TASKS"] = "1"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    claude_json_b64 = _load_claude_json_b64()
    if claude_json_b64:
        env["CLAUDE_JSON_B64"] = claude_json_b64

    if model:
        env["ANTHROPIC_MODEL"] = model

    return env


def cc_run_command(
    prompt: str,
    *,
    model: str | None = None,
    budget: float | None = None,
    allowed_tools: list[str] | None = None,
    setting_sources: str = "",
) -> str:
    parts = ['[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"', "&&", "claude"]
    parts.extend(["--setting-sources", shlex.quote(setting_sources)])
    parts.extend(["--verbose", "--output-format", "stream-json"])
    if budget is not None:
        parts.extend(["--max-budget-usd", str(budget)])
    if model:
        parts.extend(["--model", shlex.quote(model)])
    tools = allowed_tools or CC_ALLOWED_TOOLS
    parts.extend(["--allowedTools", " ".join(tools)])
    parts.extend(["-p", shlex.quote(prompt)])
    return " ".join(parts)


def cc_parse_usage(stdout: str) -> ParsedUsage | None:
    events = parse_jsonl_events(stdout)
    result_event = None
    for event in reversed(events):
        if event.get("type") == "result":
            result_event = event
            break
    if not result_event:
        return None

    usage = result_event.get("usage", {})

    def _int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    input_tokens = _int(usage.get("input_tokens", 0))
    cached = _int(usage.get("cache_creation_input_tokens", 0)) + _int(usage.get("cache_read_input_tokens", 0))
    output_tokens = _int(usage.get("output_tokens", 0))
    cost = result_event.get("total_cost_usd")
    if cost is None:
        cost = result_event.get("cost_usd")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else None
    return ParsedUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _load_codex_auth_b64() -> str | None:
    if os.environ.get("CODEX_AUTH_JSON_B64"):
        return os.environ["CODEX_AUTH_JSON_B64"]
    path = Path.home() / ".codex" / "auth.json"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def codex_install_script(*, version: str | None = None) -> str:
    resolved_version = version or "latest"
    return f"""
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
npm install -g @openai/codex@{resolved_version}
mkdir -p "$HOME/.codex"
if [ -n "${{CODEX_AUTH_JSON_B64:-}}" ]; then
  printf '%s' "$CODEX_AUTH_JSON_B64" | base64 -d > "$HOME/.codex/auth.json"
elif [ -n "${{OPENAI_API_KEY:-}}" ]; then
  echo '{{"OPENAI_API_KEY": "'$OPENAI_API_KEY'"}}' > "$HOME/.codex/auth.json"
fi
echo "codex installed: $(codex --version 2>/dev/null || echo unknown)"
"""


def codex_env_vars() -> dict[str, str]:
    env: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    auth_b64 = _load_codex_auth_b64()
    if auth_b64:
        env["CODEX_AUTH_JSON_B64"] = auth_b64
    return env


def resolve_codex_model(model: str | None) -> str:
    if not model or model == "codex":
        return "gpt-5-codex"
    if ":" in model:
        _, model = model.split(":", 1)
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def codex_run_command(
    prompt: str,
    *,
    model: str,
    workdir: str | None = None,
    web_search: bool = False,
    ephemeral: bool = True,
) -> str:
    parts = ['[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"', "&&", "codex"]
    if web_search:
        parts.append("--search")
    else:
        parts.extend(["-c", shlex.quote('web_search="disabled"')])
    parts.append("exec")
    parts.extend(["--json"])
    if ephemeral:
        parts.append("--ephemeral")
    parts.extend(["--skip-git-repo-check", "--sandbox", "danger-full-access"])
    if workdir:
        parts.extend(["--cwd", shlex.quote(workdir)])
    parts.extend(["-m", shlex.quote(model), shlex.quote(prompt)])
    return " ".join(parts)


def codex_parse_usage(stdout: str) -> ParsedUsage | None:
    events = parse_jsonl_events(stdout)
    usage_candidate = None
    for event in events:
        for dict_value in iter_dicts(event):
            if any(key in dict_value for key in ("input_tokens", "output_tokens")):
                usage_candidate = dict_value
    if not usage_candidate:
        return None

    def _int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    total_in = _int(usage_candidate.get("input_tokens", 0))
    cached = _int(usage_candidate.get("cached_input_tokens", 0))
    output_tokens = _int(usage_candidate.get("output_tokens", 0))
    cost = usage_candidate.get("total_cost_usd") or usage_candidate.get("cost_usd")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else None
    return ParsedUsage(
        input_tokens=total_in - cached,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def resolve_spell_model(model_name: str | None) -> str:
    if model_name is None:
        return "anthropic-tc:opus46"
    if any(model_name.startswith(prefix + ":") for prefix in ("anthropic-tc", "anthropic-pf", "openai", "codex-msg", "codex-tc")):
        return model_name
    if model_name in LITELLM_TO_SPELL_MODEL:
        return LITELLM_TO_SPELL_MODEL[model_name]
    if "/" in model_name:
        return f"anthropic-tc:{model_name.split('/', 1)[1]}"
    return f"anthropic-tc:{model_name}"


def _is_spell_root(path: Path) -> bool:
    return (
        (path / "deps.edn").exists()
        and (path / "src" / "spell" / "benchmark_api.clj").exists()
        and (path / "config" / "agents").exists()
    )


def discover_spell_root(explicit: str | None = None, *, search_from: Path | None = None) -> Path | None:
    def _lineage(anchor: Path) -> list[Path]:
        base = anchor if anchor.is_dir() else anchor.parent
        return [base, *base.parents]

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    env_spell_root = os.environ.get("SPELL_ROOT")
    if env_spell_root:
        candidates.append(Path(env_spell_root).expanduser().resolve())

    if search_from is not None:
        resolved = Path(search_from).resolve()
        lineage = _lineage(resolved)
        candidates.extend(lineage)
        candidates.extend(parent / "spell" for parent in lineage)

    here = Path(__file__).resolve()
    here_lineage = _lineage(here)
    candidates.extend(here_lineage)
    candidates.extend(parent / "spell" for parent in here_lineage)
    candidates.append(Path.cwd())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_spell_root(candidate):
            return candidate
    return None


def spell_build_bundle(spell_root: Path | None = None) -> bytes:
    resolved_root = spell_root or discover_spell_root()
    if resolved_root is None:
        raise RuntimeError(
            "Could not locate Spell source tree. Set SPELL_ROOT to the spell repo root "
            "(must contain deps.edn, src/spell/benchmark_api.clj, and config/agents)."
        )

    missing = [rel for rel in SPELL_BUNDLE_PATHS if not (resolved_root / rel).exists()]
    if missing:
        raise RuntimeError(
            f"Spell source tree at {resolved_root} is missing required paths: {', '.join(missing)}"
        )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in SPELL_BUNDLE_PATHS:
            tar.add(resolved_root / rel, arcname=f"spell/{rel}")
    return buf.getvalue()


def spell_pre_install(
    container: DockerWorkspaceContainer,
    bundle: bytes,
) -> bool:
    prep = container.exec_run("mkdir -p /opt && rm -rf /opt/spell", timeout=60, workdir="/")
    if prep.exit_code != 0:
        return False
    return container.put_tar_archive("/opt", bundle)


def spell_install_script() -> str:
    return SPELL_RUNTIME_INSTALL


def spell_env_vars(
    model_name: str,
    *,
    sh_timeout: int = 120,
) -> dict[str, str]:
    env: dict[str, str] = {
        "SPELL_SH_TIMEOUT": str(sh_timeout),
    }
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key)
        if value:
            env[key] = value

    if model_name.startswith(("codex-msg:", "codex-tc:")):
        auth_b64 = _load_codex_auth_b64()
        if auth_b64:
            env["CODEX_AUTH_JSON_B64"] = auth_b64
    return env


def _resolve_agent_for_container(agent_file: str | None, *, spell_root: Path | None = None) -> str:
    """Map bundled Spell agent paths to their /opt/spell container equivalent."""
    if agent_file is None:
        return SPELL_AGENT_IN_CONTAINER
    # If already a container path, pass through
    if agent_file.startswith("/opt/spell/"):
        return agent_file

    resolved_root = spell_root or discover_spell_root()
    if resolved_root is None:
        return agent_file

    resolved_agent = Path(agent_file).expanduser().resolve()
    bundled_agents_dir = (resolved_root / "config" / "agents").resolve()
    try:
        resolved_agent.relative_to(bundled_agents_dir)
    except ValueError:
        return agent_file

    relative_agent = resolved_agent.relative_to(resolved_root.resolve())
    return str(PurePosixPath("/opt/spell") / PurePosixPath(relative_agent.as_posix()))


def spell_run_command(
    *,
    prompt: str | None,
    model_name: str,
    agent_file: str | None = None,
    budget: float | None = None,
    depth: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    trace: bool = False,
    trace_dir: str | None = None,
    init: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "mode": "spell",
        "model": model_name,
        "budget": budget,
        "agent": _resolve_agent_for_container(agent_file),
        "depth": depth,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "trace": trace,
        "trace_dir": trace_dir,
    }
    if init is not None:
        payload["init"] = init
    elif prompt is not None:
        payload["prompt"] = prompt
    payload = {key: value for key, value in payload.items() if value is not None}
    payload_json = json.dumps(payload, separators=(",", ":"))
    return (
        'export PATH="$HOME/.local/bin:$PATH" && '
        'CLOJURE_BIN="$(if command -v clojure >/dev/null 2>&1 && clojure -Sdescribe >/dev/null 2>&1; '
        'then echo clojure; elif command -v clj >/dev/null 2>&1; then echo clj; else echo clojure; fi)" && '
        f"printf %s {shlex.quote(payload_json)} | "
        f"$CLOJURE_BIN -Sdeps {shlex.quote(SPELL_SDEPS)} -M -m spell.benchmark-api "
        "--request - --response -"
    )


def spell_parse_benchmark_response(stdout: str) -> dict[str, Any] | None:
    raw = (stdout or "").strip()
    if not raw:
        return None

    candidates = [raw]
    candidates.extend(reversed(raw.splitlines()))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("ok" in parsed or "usage" in parsed):
            return parsed
    return None


def copy_container_tree_to_host(
    container: DockerWorkspaceContainer,
    source: str,
    dest: Path,
) -> bool:
    normalized = source.rstrip("/")
    if not normalized:
        return False

    candidates: list[str] = []
    if normalized.startswith("/"):
        candidates.append(normalized)
    else:
        candidates.append(str(PurePosixPath(container.container_workdir) / normalized))
        candidates.append(normalized)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        copy_source = candidate if candidate.endswith("/.") else f"{candidate}/."
        if container.copy_to_host(copy_source, dest):
            return True

    logger.warning("Could not copy container path '%s' to host dir %s", source, dest)
    return False


def copy_spell_traces_to_host(
    container: DockerWorkspaceContainer,
    raw_response: dict[str, Any] | None,
    host_trace_dir: Path | None,
) -> str | None:
    if host_trace_dir is None or not raw_response:
        return None

    trace_dir = raw_response.get("trace_dir")
    if not trace_dir:
        return None

    dest = Path(host_trace_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if copy_container_tree_to_host(container, trace_dir, dest):
        return str(dest)
    return None


def spell_parse_usage(stdout: str) -> ParsedUsage | None:
    parsed = spell_parse_benchmark_response(stdout)
    if not parsed:
        return None

    usage = (parsed.get("usage") or {}).get("total", {})

    def _int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    input_tokens = _int(usage.get("input_tokens", 0))
    cached = _int(usage.get("cache_creation_input_tokens", 0)) + _int(usage.get("cache_read_input_tokens", 0))
    output_tokens = _int(usage.get("output_tokens", 0))
    cost = usage.get("cost")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else None
    return ParsedUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def _install_agent_runtime(
    container: DockerWorkspaceContainer,
    *,
    condition: Literal["spell", "claude_code", "codex"],
    version: str | None = None,
    model_spec: str | None = None,
) -> bool:
    if condition in {"claude_code", "codex"} and not install_nvm_node(container):
        return False

    if condition == "claude_code":
        return run_install_script(container, cc_install_script(version=version), env=cc_env_vars(model=model_spec))
    if condition == "codex":
        return run_install_script(container, codex_install_script(version=version), env=codex_env_vars())
    if condition == "spell":
        spell_root = discover_spell_root()
        bundle = spell_build_bundle(spell_root)
        if not spell_pre_install(container, bundle):
            return False
        return run_install_script(
            container,
            spell_install_script(),
            env=spell_env_vars(resolve_spell_model(model_spec)),
            timeout=600,
        )
    raise ValueError(f"Unsupported condition: {condition}")


def run_agent_in_container(
    container: DockerWorkspaceContainer,
    *,
    condition: Literal["spell", "claude_code", "codex"],
    prompt: str,
    timeout: int,
    model_spec: str | None,
    budget: float | None = None,
    agent_file: str | None = None,
    depth: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    web_search: bool = False,
    trace: bool = False,
    trace_dir: Path | None = None,
    trace_subdir: str | None = None,
    init: str | None = None,
    version: str | None = None,
) -> AgentRunResult:
    if not _install_agent_runtime(container, condition=condition, version=version, model_spec=model_spec):
        return AgentRunResult(
            output=ExecOutput(exit_code=1, stdout="", stderr="Agent install failed"),
            error_message="Agent install failed",
            error_type="install_error",
        )

    if condition == "claude_code":
        resolved_model = model_spec
        if resolved_model and "/" in resolved_model:
            resolved_model = resolved_model.removeprefix("anthropic/")
        env = cc_env_vars(model=resolved_model)
        output = container.exec_run(
            cc_run_command(
                prompt,
                model=resolved_model,
                budget=budget,
            ),
            timeout=timeout,
            env=env,
        )
        usage = cc_parse_usage(output.stdout)
        return AgentRunResult(
            output=output,
            usage=usage,
            error_message=output.stderr[:2000] if output.exit_code != 0 else None,
            error_type="cli_error" if output.exit_code != 0 else None,
        )

    if condition == "codex":
        resolved_model = resolve_codex_model(model_spec)
        env = codex_env_vars()
        output = container.exec_run(
            codex_run_command(
                prompt,
                model=resolved_model,
                workdir=container.container_workdir,
                web_search=web_search,
            ),
            timeout=timeout,
            env=env,
        )
        usage = codex_parse_usage(output.stdout)
        return AgentRunResult(
            output=output,
            usage=usage,
            error_message=output.stderr[:2000] if output.exit_code != 0 else None,
            error_type="codex_cli_error" if output.exit_code != 0 else None,
        )

    resolved_model = resolve_spell_model(model_spec)
    env = spell_env_vars(resolved_model)
    container_trace_dir = None
    if trace_dir is not None:
        container_trace_dir = ".spell-trace"
        if trace_subdir:
            container_trace_dir = f"{container_trace_dir}/{trace_subdir}"
    output = container.exec_run(
        spell_run_command(
            prompt=prompt,
            model_name=resolved_model,
            agent_file=agent_file,
            budget=budget,
            depth=depth,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            trace=trace,
            trace_dir=container_trace_dir,
            init=init,
        ),
        timeout=timeout,
        env=env,
    )
    parsed = spell_parse_benchmark_response(output.stdout)
    usage = spell_parse_usage(output.stdout)
    persisted_trace_dir = copy_spell_traces_to_host(container, parsed, trace_dir)
    return AgentRunResult(
        output=output,
        usage=usage,
        response_text=parsed.get("result") if parsed else None,
        error_message=((parsed or {}).get("error") if parsed else output.stderr[:2000]) or None,
        error_type=(parsed or {}).get("error_type") if parsed else "parse_error",
        raw=parsed,
        trace_dir=persisted_trace_dir,
    )


# Backward-compatible alias for older in-flight refactors.
execute_agent_in_container = run_agent_in_container
