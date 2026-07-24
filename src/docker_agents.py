"""Shared container-agent helpers for software benchmarks.

This module holds benchmark-agnostic pieces shared by Terminal-Bench and the
bench.py software benchmark container paths:

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
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Literal

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
    estimate_cost,
    usage_from_totals,
)

logger = logging.getLogger(__name__)

ParsedUsage = UsageBreakdown

DEFAULT_IMAGE = "python:3.11-slim"
DEFAULT_WORKDIR = "/workspace"
SPELL_AGENT_IN_CONTAINER = "/opt/spell/config/agents/io-tc.agent.edn"
SPELL_SDEPS = '{:deps {local/spell {:local/root "/opt/spell"}}}'
SPELL_BUNDLE_PATHS = ("deps.edn", "src", "config", "data")
HARBOR_SPELL_BUNDLE_PATHS = ("deps.edn", "src", "config", "data/pricing.edn")
HARBOR_SPELL_BUNDLE_FORBIDDEN_PARTS = frozenset(
    {
        "benchmarking",
        "terminal_bench",
        "terminal-bench",
        "results",
        "traces",
        "logs",
        "notebook",
        "workspace",
        "submissions",
    }
)
CODEX_RUNTIME_ROOT = "/tmp/bench-agent-runtime/codex"
CODEX_RUNTIME_CODEX_HOME = f"{CODEX_RUNTIME_ROOT}/home"
CODEX_RUNTIME_NVM_DIR = f"{CODEX_RUNTIME_ROOT}/nvm"
CODEX_RUNTIME_NPM_PREFIX = f"{CODEX_RUNTIME_ROOT}/npm-global"
CODEX_RUNTIME_NPM_CACHE = f"{CODEX_RUNTIME_ROOT}/npm-cache"
UV_RUNTIME_BIN = "/tmp/bench-agent-runtime/uv-bin"
MINI_RUNTIME_PREFIX = "/opt/mini"
MINI_RUNTIME_BIN = f"{MINI_RUNTIME_PREFIX}/bin"
MINI_RUNTIME_ROOT = "/tmp/bench-agent-runtime/mini"
MINI_RUNTIME_HOME = f"{MINI_RUNTIME_ROOT}/home"
MINI_RUNTIME_CONFIG_ROOT = f"{MINI_RUNTIME_ROOT}/config"
MINI_RUNTIME_CONFIG_DIR = f"{MINI_RUNTIME_CONFIG_ROOT}/mini-swe-agent"
MINI_RUNTIME_CACHE_ROOT = f"{MINI_RUNTIME_ROOT}/cache"
MINI_RUNTIME_TRAJECTORY_DIR = f"{MINI_RUNTIME_ROOT}/trajectories"
MINI_RUNTIME_BUNDLE_BASE_IMAGE = "python:3.11-slim-bullseye"
APT_FETCH_RETRY_MAX_ATTEMPTS = 3
APT_FETCH_RETRY_SLEEP_SECONDS = 5
INSTALL_TIMEOUT_CLEANUP_TIMEOUT_SECONDS = 30
RETRYABLE_APT_FETCH_MARKERS = (
    "connection timed out",
    "temporary failure resolving",
    "could not connect",
    "connection failed",
    "network is unreachable",
)
RETRYABLE_APT_MIRROR_MARKERS = (
    "file has unexpected size",
    "hash sum mismatch",
)
RETRYABLE_PIP_HASH_MISMATCH_MARKER = "these packages do not match the hashes"
INSTALL_TIMEOUT_RETRY_CLEANUP_SCRIPT = r"""
if command -v pkill >/dev/null 2>&1; then
  pkill -TERM -f '(^|/)(apt|apt-get|dpkg)( |$)' >/dev/null 2>&1 || true
  sleep 2
  pkill -KILL -f '(^|/)(apt|apt-get|dpkg)( |$)' >/dev/null 2>&1 || true
fi
for lock_path in \
  /var/lib/dpkg/lock-frontend \
  /var/lib/dpkg/lock \
  /var/cache/apt/archives/lock \
  /var/lib/apt/lists/lock
do
  for _attempt in 1 2 3 4 5; do
    [ ! -e "$lock_path" ] && break
    sleep 1
  done
done
if command -v dpkg >/dev/null 2>&1; then
  dpkg --configure -a >/dev/null 2>&1 || true
fi
"""

CLAUDE_RUNTIME_ROOT = "/tmp/bench-agent-runtime/claude"
CLAUDE_RUNTIME_NVM_DIR = f"{CLAUDE_RUNTIME_ROOT}/nvm"
CLAUDE_RUNTIME_NPM_PREFIX = f"{CLAUDE_RUNTIME_ROOT}/npm-global"
CLAUDE_RUNTIME_NPM_CACHE = f"{CLAUDE_RUNTIME_ROOT}/npm-cache"
CLAUDE_DEBUG_LOG_PATH = f"{CLAUDE_RUNTIME_ROOT}/debug.log"
NODE_BINARY_VERSION = "v22.22.2"
MIN_NODE_MAJOR_VERSION = 18
CLAUDE_CLI_MISSING_MARKER = "Claude Code CLI not available after installation"

CC_ALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep",
    "WebFetch", "NotebookEdit",
]
CC_PAPER_COMPLIANT_ALLOWED_TOOLS = [tool for tool in CC_ALLOWED_TOOLS if tool != "WebFetch"]

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


def _nvm_source_with_tmp_npm_prefix() -> str:
    return r"""
if [ "${NPM_CONFIG_PREFIX+x}" = x ]; then
    NPM_CONFIG_PREFIX_SAVED="$NPM_CONFIG_PREFIX"
    NPM_CONFIG_PREFIX_WAS_SET=1
else
    NPM_CONFIG_PREFIX_SAVED=
    NPM_CONFIG_PREFIX_WAS_SET=
fi
unset NPM_CONFIG_PREFIX
if [ -s "$NVM_DIR/nvm.sh" ]; then
    source "$NVM_DIR/nvm.sh"
fi
if [ -n "${NPM_CONFIG_PREFIX_WAS_SET:-}" ]; then
    export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"
fi
unset NPM_CONFIG_PREFIX_SAVED NPM_CONFIG_PREFIX_WAS_SET
""".strip()


def _install_curl_shell() -> str:
    return r"""
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::AllowInsecureRepositories=true update -qq && apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq curl >/dev/null
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache curl >/dev/null 2>&1
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y -q curl >/dev/null 2>&1
elif command -v yum >/dev/null 2>&1; then
    yum install -y -q curl >/dev/null 2>&1
else
    echo "Unable to install curl: unsupported package manager" >&2
    exit 1
fi
""".strip()


def _install_uv_shell() -> str:
    return f"""
set -e
uv_path="$(command -v uv || true)"
if [ -z "$uv_path" ]; then
  export UV_UNMANAGED_INSTALL="${{UV_UNMANAGED_INSTALL:-{UV_RUNTIME_BIN}}}"
  uv_installed_with_standalone=
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_UNMANAGED_INSTALL" sh
    uv_installed_with_standalone=1
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_UNMANAGED_INSTALL" sh
    uv_installed_with_standalone=1
  elif command -v python >/dev/null 2>&1 && python -m pip --version >/dev/null 2>&1; then
    python -m pip install -q --disable-pip-version-check uv
  elif command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install -q --disable-pip-version-check uv
  else
    {_install_curl_shell()}
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_UNMANAGED_INSTALL" sh
      uv_installed_with_standalone=1
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_UNMANAGED_INSTALL" sh
      uv_installed_with_standalone=1
    else
      echo "Unable to install uv: neither curl, wget, nor pip is available" >&2
      exit 1
    fi
  fi
  if [ -n "$uv_installed_with_standalone" ]; then
    uv_path="${{UV_UNMANAGED_INSTALL%/}}/uv"
  else
    uv_path="$(command -v uv || true)"
  fi
fi
if [ -z "$uv_path" ] && [ -x "$HOME/.local/bin/uv" ]; then
  uv_path="$HOME/.local/bin/uv"
fi
if [ -z "$uv_path" ] && [ -x /usr/local/bin/uv ]; then
  uv_path=/usr/local/bin/uv
fi
if [ -z "$uv_path" ] && [ -x /usr/bin/uv ]; then
  uv_path=/usr/bin/uv
fi
if [ -z "$uv_path" ]; then
  uv_path="$(command -v uv || true)"
fi
if [ -z "$uv_path" ] || [ ! -x "$uv_path" ]; then
  echo "uv not available after installation" >&2
  exit 1
fi
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/env" <<EOF
export PATH="$(dirname "$uv_path"):\\$PATH"
EOF
""".strip()


UV_BOOTSTRAP_COMMAND = _install_uv_shell()


def _install_nodejs_shell(*, log_version: bool = False) -> str:
    script = r"""
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::AllowInsecureRepositories=true update -qq && apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq nodejs npm >/dev/null
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache nodejs npm >/dev/null 2>&1
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y -q nodejs npm >/dev/null 2>&1
elif command -v yum >/dev/null 2>&1; then
    yum install -y -q nodejs npm >/dev/null 2>&1
else
    echo "Unable to install Node.js runtime: unsupported package manager" >&2
    exit 1
fi
""".strip()
    if log_version:
        script += '\necho "node installed via package manager: $(node --version 2>/dev/null || echo missing)"'
    return script

def _claude_runtime_shell_setup() -> str:
    return f"""
export CLAUDE_RUNTIME_ROOT="${{CLAUDE_RUNTIME_ROOT:-{CLAUDE_RUNTIME_ROOT}}}"
export NVM_DIR="${{NVM_DIR:-{CLAUDE_RUNTIME_NVM_DIR}}}"
export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CLAUDE_RUNTIME_NPM_PREFIX}}}"
export NPM_CONFIG_CACHE="${{NPM_CONFIG_CACHE:-{CLAUDE_RUNTIME_NPM_CACHE}}}"
export npm_config_cache="${{npm_config_cache:-$NPM_CONFIG_CACHE}}"
mkdir -p "$CLAUDE_RUNTIME_ROOT" "$NVM_DIR" "$NPM_CONFIG_PREFIX" "$npm_config_cache"
%s
export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
""".strip() % _nvm_source_with_tmp_npm_prefix()


def _verify_nodejs_shell() -> str:
    return f"""
set -e
{_claude_runtime_shell_setup()}
if ! command -v node >/dev/null 2>&1; then
  echo "node not available after installation" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not available after installation" >&2
  exit 1
fi
node_version="$(node --version 2>/dev/null || true)"
node_major="${{node_version#v}}"
node_major="${{node_major%%.*}}"
case "$node_major" in
  ""|*[!0-9]*)
    echo "Unable to parse node version after installation: ${{node_version:-unknown}}" >&2
    exit 1
    ;;
esac
if [ "$node_major" -lt {MIN_NODE_MAJOR_VERSION} ]; then
  echo "node too old after installation: $node_version (need >={MIN_NODE_MAJOR_VERSION})" >&2
  exit 1
fi
echo "node verified: $node_version"
echo "npm verified: $(npm --version 2>/dev/null || echo unknown)"
""".strip()


def _install_nodejs_binary_shell(*, version: str = NODE_BINARY_VERSION) -> str:
    quoted_version = shlex.quote(version)
    return f"""
set -e
{_claude_runtime_shell_setup()}
node_version={quoted_version}
case "$(uname -m)" in
  x86_64|amd64)
    node_arch=x64
    ;;
  aarch64|arm64)
    node_arch=arm64
    ;;
  *)
    echo "Unsupported architecture for Node.js binary install: $(uname -m)" >&2
    exit 1
    ;;
esac
if ! command -v curl >/dev/null 2>&1; then
  {_install_curl_shell()}
fi
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
tarball="node-${{node_version}}-linux-${{node_arch}}.tar.gz"
url="https://nodejs.org/dist/${{node_version}}/${{tarball}}"
curl -fsSL "$url" -o "$tmpdir/$tarball"
mkdir -p "$NPM_CONFIG_PREFIX"
tar -xzf "$tmpdir/$tarball" -C "$NPM_CONFIG_PREFIX" --strip-components=1
"$NPM_CONFIG_PREFIX/bin/node" --version >/dev/null
"$NPM_CONFIG_PREFIX/bin/npm" --version >/dev/null
echo "node installed via official binary: $($NPM_CONFIG_PREFIX/bin/node --version)"
""".strip()


def _cleanup_nodejs_binary_shell() -> str:
    return f"""
set -e
{_claude_runtime_shell_setup()}
if [ -n "${{NPM_CONFIG_PREFIX:-}}" ] && [ -d "$NPM_CONFIG_PREFIX" ]; then
  find "$NPM_CONFIG_PREFIX" -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +
fi
""".strip()


NVM_NODE_INSTALL = r"""
set -e
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
mkdir -p "$NVM_DIR"
if [ -d "$NVM_DIR" ]; then
    %s 2>/dev/null || true
    if command -v node >/dev/null 2>&1; then
        echo "node already installed: $(node --version)"
        exit 0
    fi
fi
%s
export PROFILE=/dev/null
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
%s
nvm install 22
echo "node installed: $(node --version)"
""" % (
    _nvm_source_with_tmp_npm_prefix(),
    _install_curl_shell(),
    _nvm_source_with_tmp_npm_prefix(),
)

SPELL_RUNTIME_INSTALL = r"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if ! command -v java >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::AllowInsecureRepositories=true update -qq
    apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq ca-certificates curl default-jre >/dev/null
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
      apt-get -o Acquire::AllowInsecureRepositories=true update -qq
      apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq curl >/dev/null
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
class AgentRunResult:
    output: ExecOutput
    usage: UsageBreakdown | None = None
    response_text: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    raw: dict[str, Any] | None = None
    trace_dir: str | None = None
    metadata: dict[str, Any] | None = None


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
            "(apt-get -o Acquire::AllowInsecureRepositories=true update -qq && apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq git >/dev/null 2>&1)",
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
            f"git add -A && git diff --cached {shlex.quote(base)}"
            " -- . ':!.spell-trace'; "
            "status=$?; git reset -q >/dev/null 2>&1; exit $status"
        )
        if result.exit_code != 0:
            return ""
        return result.stdout

    def changed_files(self, base: str = "HEAD") -> list[str]:
        result = self.exec_run(
            f"git add -A && git diff --name-only --cached {shlex.quote(base)}"
            " -- . ':!.spell-trace'; "
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


def install_nvm_node(
    container: DockerWorkspaceContainer,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    return install_nvm_node_exec(container.exec_run, env=env)


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


def _exec_output_excerpt(result: ExecOutput, *, limit: int = 500) -> str:
    message = result.stderr.strip() or result.stdout.strip()
    if not message:
        message = f"exit_code={result.exit_code}"
    return message[:limit]


def _format_install_attempt_log(label: str, attempt: int, result: ExecOutput) -> str:
    return (
        f"== {label} attempt {attempt} ==\n"
        f"exit_code: {result.exit_code}\n"
        f"timed_out: {str(result.timed_out).lower()}\n"
        "-- stdout --\n"
        f"{result.stdout}"
        f"{'' if result.stdout.endswith(chr(10)) or not result.stdout else chr(10)}"
        "-- stderr --\n"
        f"{result.stderr}"
        f"{'' if result.stderr.endswith(chr(10)) or not result.stderr else chr(10)}"
    )


def _append_install_attempt_log(path: Path, label: str, attempt: int, result: ExecOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_format_install_attempt_log(label, attempt, result))


def _is_retryable_install_timeout(result: ExecOutput) -> bool:
    return result.timed_out or result.exit_code == 124


def _is_retryable_apt_fetch_failure(stdout: str, stderr: str) -> bool:
    output = "\n".join(part for part in (stdout, stderr) if part).lower()
    if any(marker in output for marker in RETRYABLE_APT_MIRROR_MARKERS):
        return True
    return "failed to fetch" in output and any(marker in output for marker in RETRYABLE_APT_FETCH_MARKERS)


def _is_retryable_pip_hash_mismatch(stdout: str, stderr: str) -> bool:
    output = "\n".join(part for part in (stdout, stderr) if part).lower()
    return RETRYABLE_PIP_HASH_MISMATCH_MARKER in output


def _pip_install_command(packages: list[str], *, no_cache_dir: bool = False) -> str:
    quoted = " ".join(shlex.quote(package) for package in packages)
    cache_flag = " --no-cache-dir" if no_cache_dir else ""
    return f"python -m pip install -q --disable-pip-version-check{cache_flag} {quoted}"


def _best_effort_timeout_retry_cleanup(
    exec_fn: Callable[..., ExecOutput],
    *,
    env: dict[str, str] | None = None,
) -> None:
    try:
        result = exec_fn(
            INSTALL_TIMEOUT_RETRY_CLEANUP_SCRIPT,
            timeout=INSTALL_TIMEOUT_CLEANUP_TIMEOUT_SECONDS,
            env=dict(env) if env else None,
            workdir="/",
        )
    except Exception:
        logger.warning("Install timeout cleanup raised unexpectedly", exc_info=True)
        return
    if result.exit_code != 0:
        logger.warning(
            "Install timeout cleanup exited non-zero: %s",
            _exec_output_excerpt(result),
        )


def _run_install_with_apt_retry(
    exec_fn: Callable[..., ExecOutput],
    script: str,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    label: str,
    attempt_log_path: Path | None = None,
) -> ExecOutput:
    last_result = ExecOutput(exit_code=1, stdout="", stderr="install did not run")
    current_env = dict(env or {})
    for attempt in range(1, APT_FETCH_RETRY_MAX_ATTEMPTS + 1):
        exec_env = dict(current_env) if current_env else None
        last_result = exec_fn(script, timeout=timeout, env=exec_env, workdir=workdir)
        if attempt_log_path is not None:
            _append_install_attempt_log(attempt_log_path, label, attempt, last_result)
        if last_result.exit_code == 0:
            return last_result
        retryable_timeout = _is_retryable_install_timeout(last_result)
        retryable_apt = _is_retryable_apt_fetch_failure(last_result.stdout, last_result.stderr)
        retryable_pip_hash = _is_retryable_pip_hash_mismatch(last_result.stdout, last_result.stderr)
        if attempt >= APT_FETCH_RETRY_MAX_ATTEMPTS or not (retryable_timeout or retryable_apt or retryable_pip_hash):
            return last_result
        if retryable_timeout:
            logger.warning(
                "%s timed out (attempt %s/%s), retrying whole script: %s",
                label,
                attempt,
                APT_FETCH_RETRY_MAX_ATTEMPTS,
                _exec_output_excerpt(last_result),
            )
            _best_effort_timeout_retry_cleanup(exec_fn, env=current_env)
        elif retryable_pip_hash:
            logger.warning(
                "%s hit pip hash mismatch (attempt %s/%s), retrying with cache disabled: %s",
                label,
                attempt,
                APT_FETCH_RETRY_MAX_ATTEMPTS,
                _exec_output_excerpt(last_result),
            )
            current_env["PIP_NO_CACHE_DIR"] = "1"
            current_env["UV_NO_CACHE"] = "1"
        else:
            logger.warning(
                "%s hit transient apt fetch failure (attempt %s/%s): %s",
                label,
                attempt,
                APT_FETCH_RETRY_MAX_ATTEMPTS,
                _exec_output_excerpt(last_result),
            )
        time.sleep(APT_FETCH_RETRY_SLEEP_SECONDS)
    return last_result


def install_nvm_node_exec(
    exec_fn: Callable[..., ExecOutput],
    *,
    env: dict[str, str] | None = None,
) -> bool:
    def verify(label: str) -> bool:
        verify_result = exec_fn(
            _verify_nodejs_shell(),
            timeout=60,
            env=dict(env) if env else None,
            workdir=None,
        )
        if verify_result.exit_code == 0:
            return True
        logger.warning("%s succeeded but Node/npm verification failed: %s", label, _exec_output_excerpt(verify_result))
        return False

    result = _run_install_with_apt_retry(
        exec_fn,
        NVM_NODE_INSTALL,
        timeout=300,
        env=env,
        label="NVM/Node install",
    )
    if result.exit_code == 0 and verify("NVM/Node install"):
        return True
    if result.exit_code != 0:
        logger.warning("NVM/Node install failed, falling back to official Node binary: %s", _exec_output_excerpt(result))
    else:
        logger.warning("NVM/Node install did not provide Node >=%s and npm, falling back to official Node binary", MIN_NODE_MAJOR_VERSION)
    binary = _run_install_with_apt_retry(
        exec_fn,
        _install_nodejs_binary_shell(),
        timeout=300,
        env=env,
        workdir="/",
        label="Node binary install",
    )
    if binary.exit_code == 0 and verify("Node binary install"):
        return True
    if binary.exit_code != 0:
        logger.warning("Node binary install failed, falling back to package manager: %s", _exec_output_excerpt(binary))
    else:
        logger.warning("Node binary install did not provide Node >=%s and npm, falling back to package manager", MIN_NODE_MAJOR_VERSION)
    cleanup = exec_fn(
        _cleanup_nodejs_binary_shell(),
        timeout=60,
        env=dict(env) if env else None,
        workdir=None,
    )
    if cleanup.exit_code != 0:
        logger.warning("Node binary cleanup failed before package-manager fallback: %s", _exec_output_excerpt(cleanup))
    fallback = _run_install_with_apt_retry(
        exec_fn,
        _install_nodejs_shell(log_version=True),
        timeout=300,
        env=env,
        workdir="/",
        label="Node fallback install",
    )
    if fallback.exit_code != 0:
        logger.error("Node fallback install failed: %s", _exec_output_excerpt(fallback))
        return False
    return verify("Node fallback install")


def ensure_uv(container: DockerWorkspaceContainer) -> bool:
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        result = container.exec_run(UV_BOOTSTRAP_COMMAND, timeout=180)
        if result.exit_code == 0:
            return True
        logger.error("uv install failed (attempt %s): %s", attempt, result.stderr[:500])
        if attempt < max_attempts:
            time.sleep(2)
    return False


def ensure_uv_exec(
    exec_fn: Callable[..., ExecOutput],
    *,
    env: dict[str, str] | None = None,
    workdir: str | None = None,
) -> bool:
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        result = exec_fn(
            UV_BOOTSTRAP_COMMAND,
            timeout=180,
            env=env,
            workdir=workdir,
        )
        if result.exit_code == 0:
            return True
        logger.error("uv install failed (attempt %s): %s", attempt, result.stderr[:500])
        if attempt < max_attempts:
            time.sleep(2)
    return False


def ensure_python_packages(container: DockerWorkspaceContainer, packages: list[str]) -> bool:
    if not packages:
        return True
    result = container.exec_run(
        _pip_install_command(packages),
        timeout=300,
    )
    if result.exit_code == 0:
        return True
    if not _is_retryable_pip_hash_mismatch(result.stdout, result.stderr):
        return False
    logger.warning(
        "Python package install hit pip hash mismatch, retrying without cache: %s",
        _exec_output_excerpt(result),
    )
    retry_result = container.exec_run(
        _pip_install_command(packages, no_cache_dir=True),
        timeout=300,
    )
    if retry_result.exit_code != 0:
        logger.error("Python package install failed after cache-busting retry: %s", _exec_output_excerpt(retry_result))
    return retry_result.exit_code == 0


def run_install_script(
    container: DockerWorkspaceContainer,
    script: str,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    workdir: str | None = None,
    install_log_path: Path | None = None,
) -> bool:
    if install_log_path is not None:
        Path(install_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(install_log_path).write_text("", encoding="utf-8")
    result = _run_install_with_apt_retry(
        container.exec_run,
        script,
        timeout=timeout,
        env=env,
        workdir=workdir,
        label="Install script",
        attempt_log_path=Path(install_log_path) if install_log_path is not None else None,
    )
    if result.exit_code != 0:
        logger.error("Install script failed: %s", _exec_output_excerpt(result))
        return False
    return True


def cc_install_script(*, version: str | None = None) -> str:
    resolved_version = version or "latest"
    return f"""
set -euo pipefail
{_claude_runtime_shell_setup()}
npm install -g @anthropic-ai/claude-code@{resolved_version}
if ! command -v claude >/dev/null 2>&1; then
  echo "{CLAUDE_CLI_MISSING_MARKER}" >&2
  exit 1
fi
echo "claude installed: $(claude --version 2>/dev/null || echo unknown)"
"""


def cc_env_vars(
    *,
    model: str | None = None,
    include_background: bool = True,
) -> dict[str, str]:
    env: dict[str, str] = {
        "CLAUDE_RUNTIME_ROOT": CLAUDE_RUNTIME_ROOT,
        "NVM_DIR": CLAUDE_RUNTIME_NVM_DIR,
        "NPM_CONFIG_PREFIX": CLAUDE_RUNTIME_NPM_PREFIX,
        "NPM_CONFIG_CACHE": CLAUDE_RUNTIME_NPM_CACHE,
        "npm_config_cache": CLAUDE_RUNTIME_NPM_CACHE,
    }
    if include_background:
        env["FORCE_AUTO_BACKGROUND_TASKS"] = "1"
        env["ENABLE_BACKGROUND_TASKS"] = "1"

    cc_oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not cc_oauth_token:
        raise RuntimeError(
            "CLAUDE_CODE_OAUTH_TOKEN is required for Claude Code benchmark runs. "
            "Run `claude setup-token` and export/store that token; Claude Code harnesses "
            "do not accept ANTHROPIC_API_KEY or CLAUDE_JSON_B64."
        )
    env["CLAUDE_CODE_OAUTH_TOKEN"] = cc_oauth_token

    if model:
        env["ANTHROPIC_MODEL"] = model

    return env


def cc_run_command(
    prompt: str,
    *,
    model: str | None = None,
    budget: float | None = None,
    reasoning_effort: str | None = None,
    allowed_tools: list[str] | None = None,
    setting_sources: str = "",
    debug_log_path: str | None = CLAUDE_DEBUG_LOG_PATH,
) -> str:
    claude_parts = ["claude"]
    claude_parts.extend(["--setting-sources", shlex.quote(setting_sources)])
    claude_parts.extend(["--verbose", "--output-format", "stream-json"])
    if debug_log_path:
        claude_parts.extend(["--debug-file", shlex.quote(debug_log_path)])
    if budget is not None:
        claude_parts.extend(["--max-budget-usd", str(budget)])
    if model:
        claude_parts.extend(["--model", shlex.quote(model)])
    if reasoning_effort:
        claude_parts.extend(["--effort", shlex.quote(reasoning_effort)])
    tools = CC_ALLOWED_TOOLS if allowed_tools is None else allowed_tools
    claude_parts.extend(["--allowedTools", " ".join(tools)])
    claude_parts.extend(["-p", shlex.quote(prompt)])
    return "\n".join(
        [
            "set -euo pipefail",
            _claude_runtime_shell_setup(),
            f"mkdir -p {shlex.quote(CLAUDE_RUNTIME_ROOT)}",
            f"rm -f {shlex.quote(debug_log_path)}" if debug_log_path else ":",
            f'command -v claude >/dev/null 2>&1 || {{ echo "{CLAUDE_CLI_MISSING_MARKER}" >&2; exit 1; }}',
            " ".join(claude_parts),
        ]
    )


def cc_missing_cli_error(output: ExecOutput) -> bool:
    text = "\n".join(part for part in (output.stdout, output.stderr) if part)
    return CLAUDE_CLI_MISSING_MARKER in text


def extract_claude_session_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        session_id = event.get("session_id") or event.get("sessionId")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def cc_parse_usage(
    stdout: str,
    *,
    model: str | None = None,
    ledger_texts: list[tuple[str, str]] | None = None,
) -> UsageBreakdown | None:
    events = parse_jsonl_events(stdout)
    if any(event.get("type") == "result" for event in events):
        return claude_code_usage_from_events(events, model=model)

    ledger_usages: list[UsageBreakdown] = []
    for source, text in ledger_texts or []:
        usage = claude_code_usage_from_text(
            text,
            model=model,
            assistant_cost_source=f"estimated_{source}",
        )
        if usage is not None:
            ledger_usages.append(usage)
    if ledger_usages:
        return max(ledger_usages, key=lambda usage: (usage.total_output_tokens, usage.total_input_tokens))

    return claude_code_usage_from_events(events, model=model, assistant_cost_source="estimated_partial")


def _read_claude_usage_ledgers_from_exec(
    exec_fn: Callable[..., ExecOutput],
    session_id: str | None,
) -> list[tuple[str, str]]:
    ledgers: list[tuple[str, str]] = []

    debug_result = exec_fn(
        f"test -s {shlex.quote(CLAUDE_DEBUG_LOG_PATH)} && cat {shlex.quote(CLAUDE_DEBUG_LOG_PATH)}",
        timeout=60,
        workdir="/",
    )
    if debug_result.exit_code == 0 and debug_result.stdout:
        ledgers.append(("debug", debug_result.stdout))

    if session_id:
        session_name = f"{session_id}.jsonl"
        find_cmd = (
            "for base in \"$HOME/.claude/projects\" /root/.claude/projects; do "
            "[ -d \"$base\" ] || continue; "
            f"find \"$base\" -type f -name {shlex.quote(session_name)} 2>/dev/null; "
            "done | head -1"
        )
        find_result = exec_fn(find_cmd, timeout=30, workdir="/")
        session_path = find_result.stdout.strip() if find_result.exit_code == 0 else ""
        if session_path:
            cat_result = exec_fn(f"cat {shlex.quote(session_path)}", timeout=60, workdir="/")
            if cat_result.exit_code == 0 and cat_result.stdout:
                ledgers.append(("ledger", cat_result.stdout))

    return ledgers


def _read_claude_usage_ledgers_from_container(
    container: DockerWorkspaceContainer,
    session_id: str | None,
) -> list[tuple[str, str]]:
    return _read_claude_usage_ledgers_from_exec(container.exec_run, session_id)


def _load_codex_auth_b64() -> str | None:
    if os.environ.get("CODEX_AUTH_JSON_B64"):
        return os.environ["CODEX_AUTH_JSON_B64"]
    path = Path.home() / ".codex" / "auth.json"
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def codex_runtime_env_vars() -> dict[str, str]:
    return {
        "CODEX_RUNTIME_ROOT": CODEX_RUNTIME_ROOT,
        "CODEX_HOME": CODEX_RUNTIME_CODEX_HOME,
        "NVM_DIR": CODEX_RUNTIME_NVM_DIR,
        "NPM_CONFIG_PREFIX": CODEX_RUNTIME_NPM_PREFIX,
        "NPM_CONFIG_CACHE": CODEX_RUNTIME_NPM_CACHE,
        "npm_config_cache": CODEX_RUNTIME_NPM_CACHE,
    }


def _codex_runtime_shell_setup() -> str:
    return f"""
export CODEX_RUNTIME_ROOT="${{CODEX_RUNTIME_ROOT:-{CODEX_RUNTIME_ROOT}}}"
export CODEX_HOME="${{CODEX_HOME:-{CODEX_RUNTIME_CODEX_HOME}}}"
export NVM_DIR="${{NVM_DIR:-{CODEX_RUNTIME_NVM_DIR}}}"
export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CODEX_RUNTIME_NPM_PREFIX}}}"
export NPM_CONFIG_CACHE="${{NPM_CONFIG_CACHE:-{CODEX_RUNTIME_NPM_CACHE}}}"
export npm_config_cache="${{npm_config_cache:-$NPM_CONFIG_CACHE}}"
mkdir -p "$CODEX_HOME" "$NVM_DIR" "$NPM_CONFIG_PREFIX" "$npm_config_cache"
export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
""".strip()


def codex_install_script(*, version: str | None = None) -> str:
    resolved_version = version or "latest"
    return f"""
{_codex_runtime_shell_setup()}
{_nvm_source_with_tmp_npm_prefix()}
npm install -g @openai/codex@{resolved_version}
if [ -n "${{CODEX_AUTH_JSON_B64:-}}" ]; then
  printf '%s' "$CODEX_AUTH_JSON_B64" | base64 -d > "$CODEX_HOME/auth.json"
elif [ -n "${{OPENAI_API_KEY:-}}" ]; then
  echo '{{"OPENAI_API_KEY": "'$OPENAI_API_KEY'"}}' > "$CODEX_HOME/auth.json"
fi
echo "codex installed: $(codex --version 2>/dev/null || echo unknown)"
"""


def codex_env_vars() -> dict[str, str]:
    env = codex_runtime_env_vars()
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    auth_b64 = _load_codex_auth_b64()
    if auth_b64:
        env["CODEX_AUTH_JSON_B64"] = auth_b64
    return env


def resolve_codex_model(model: str | None) -> str:
    return normalize_codex_model(model)


def codex_run_command(
    prompt: str,
    *,
    model: str,
    workdir: str | None = None,
    web_search: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    parts = [_codex_runtime_shell_setup(), "&&", _nvm_source_with_tmp_npm_prefix(), "&&"]
    if workdir:
        parts.extend(["cd", shlex.quote(workdir), "&&"])
    parts.append("codex")
    if web_search:
        parts.append("--search")
    else:
        parts.extend(["-c", shlex.quote('web_search="disabled"')])
    if reasoning_effort:
        parts.extend(["-c", shlex.quote(f'model_reasoning_effort="{reasoning_effort}"')])
    parts.append("exec")
    parts.extend(["--json"])
    parts.extend(["--skip-git-repo-check", "--sandbox", "danger-full-access"])
    parts.extend(["-m", shlex.quote(model), shlex.quote(prompt)])
    return " ".join(parts)


def codex_parse_usage(
    stdout: str,
    model: str | None = None,
    *,
    rollout_text: str | None = None,
) -> UsageBreakdown | None:
    events = parse_jsonl_events(stdout)
    stream_usage = usage_from_events(events, model=model)
    rollout_usage = usage_from_rollout(rollout_text, model=model) if rollout_text else None
    return merge_codex_usage(stream_usage, rollout_usage, model=model)


def filter_codex_raw_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed_types = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error", "session.ended"}
    return [event for event in events if event.get("type") in allowed_types]


def _read_codex_rollout_from_exec(
    exec_fn: Callable[..., ExecOutput],
    thread_id: str | None,
) -> tuple[str | None, str | None]:
    if not thread_id:
        return None, None

    rollout_name = f"rollout-*-{thread_id}.jsonl"
    find_cmd = (
        f"for base in {shlex.quote(CODEX_RUNTIME_CODEX_HOME + '/sessions')} \"$CODEX_HOME/sessions\" \"$HOME/.codex/sessions\" /root/.codex/sessions; do "
        "[ -d \"$base\" ] || continue; "
        f"find \"$base\" -type f -name {shlex.quote(rollout_name)} 2>/dev/null; "
        "done | head -1"
    )
    find_result = exec_fn(find_cmd, timeout=30, workdir="/")
    rollout_path = find_result.stdout.strip() if find_result.exit_code == 0 else ""
    if not rollout_path:
        return None, None

    cat_result = exec_fn(f"cat {shlex.quote(rollout_path)}", timeout=60, workdir="/")
    if cat_result.exit_code != 0:
        return rollout_path, None
    return rollout_path, cat_result.stdout


def _read_codex_rollout_from_container(
    container: DockerWorkspaceContainer,
    thread_id: str | None,
) -> tuple[str | None, str | None]:
    return _read_codex_rollout_from_exec(container.exec_run, thread_id)


def _mini_runtime_package_spec(version: str | None = None) -> str:
    resolved_version = version or os.environ.get("MINI_SWE_AGENT_VERSION")
    if resolved_version and resolved_version != "latest":
        return f"mini-swe-agent=={resolved_version}"
    return "mini-swe-agent"


def _mini_runtime_cache_path(version: str | None = None) -> Path:
    cache_root = Path(os.environ.get("MINI_SWE_AGENT_BUNDLE_CACHE_DIR", "/tmp/bench-agent-runtime/mini-bundles"))
    package_version = (version or os.environ.get("MINI_SWE_AGENT_VERSION") or "latest").replace("/", "_")
    image_key = MINI_RUNTIME_BUNDLE_BASE_IMAGE.replace("/", "_").replace(":", "_")
    return cache_root / f"mini-swe-agent-{package_version}-{image_key}.tar"


def _pty_run_script() -> str:
    return r"""import os
import pty
import sys
import time


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit(1)
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execlp(argv[0], *argv)
    time.sleep(1)
    os.write(master_fd, b"\n\n\n\n\n\n\n\n\n\n")
    while True:
        try:
            data = os.read(master_fd, 4096)
        except OSError:
            break
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    _, status = os.waitpid(pid, 0)
    sys.exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)


main()
"""


def _mini_runtime_build_script(package_spec: str) -> str:
    quoted_package = shlex.quote(package_spec)
    quoted_pty = shlex.quote(_pty_run_script())
    return f"""
set -euo pipefail
rm -rf {MINI_RUNTIME_PREFIX}
mkdir -p {MINI_RUNTIME_PREFIX}/python
cp -a /usr/local/. {MINI_RUNTIME_PREFIX}/python/
mkdir -p {MINI_RUNTIME_PREFIX}/lib
cp -a /usr/lib/*/libssl.so* /usr/lib/*/libcrypto.so* {MINI_RUNTIME_PREFIX}/lib/
export LD_LIBRARY_PATH="{MINI_RUNTIME_PREFIX}/lib:{MINI_RUNTIME_PREFIX}/python/lib:${{LD_LIBRARY_PATH:-}}"
{MINI_RUNTIME_PREFIX}/python/bin/python -m venv --copies {MINI_RUNTIME_PREFIX}/venv
{MINI_RUNTIME_PREFIX}/venv/bin/python -m pip install -q --disable-pip-version-check --upgrade pip
{MINI_RUNTIME_PREFIX}/venv/bin/python -m pip install -q --disable-pip-version-check {quoted_package}
mkdir -p {MINI_RUNTIME_BIN}
cat > {MINI_RUNTIME_BIN}/mini <<'EOF'
#!/bin/sh
export LD_LIBRARY_PATH="/opt/mini/lib:/opt/mini/python/lib:${{LD_LIBRARY_PATH:-}}"
exec /opt/mini/venv/bin/mini "$@"
EOF
printf %s {quoted_pty} > {MINI_RUNTIME_BIN}/pty-run.py
cat > {MINI_RUNTIME_BIN}/pty-run <<'EOF'
#!/bin/sh
export LD_LIBRARY_PATH="/opt/mini/lib:/opt/mini/python/lib:${{LD_LIBRARY_PATH:-}}"
exec /opt/mini/venv/bin/python /opt/mini/bin/pty-run.py "$@"
EOF
chmod +x {MINI_RUNTIME_BIN}/mini {MINI_RUNTIME_BIN}/pty-run
{MINI_RUNTIME_BIN}/mini --help >/dev/null
tar -C /opt -cf /tmp/mini-runtime.tar mini
""".strip()


def swe_agent_build_bundle(version: str | None = None) -> bytes:
    explicit_bundle = os.environ.get("MINI_SWE_AGENT_BUNDLE")
    if explicit_bundle:
        return Path(explicit_bundle).read_bytes()

    cache_path = _mini_runtime_cache_path(version)
    if cache_path.exists():
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    container_id: str | None = None
    try:
        start = subprocess.run(
            ["docker", "run", "-d", "--rm", MINI_RUNTIME_BUNDLE_BASE_IMAGE, "sleep", "infinity"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if start.returncode != 0:
            raise RuntimeError(f"Failed to start mini runtime build container: {start.stderr.strip()}")
        container_id = start.stdout.strip()

        build = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-lc",
                _mini_runtime_build_script(_mini_runtime_package_spec(version)),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if build.returncode != 0:
            raise RuntimeError(
                "Failed to build mini-swe-agent runtime bundle: "
                + _exec_output_excerpt(ExecOutput(build.returncode, build.stdout, build.stderr), limit=2000)
            )

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        cp = subprocess.run(
            ["docker", "cp", f"{container_id}:/tmp/mini-runtime.tar", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if cp.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to copy mini runtime bundle from build container: {cp.stderr.strip()}")
        tmp_path.replace(cache_path)
        return cache_path.read_bytes()
    finally:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, text=True, timeout=60)


def swe_agent_pre_install(
    container: DockerWorkspaceContainer,
    bundle: bytes,
) -> bool:
    prep = container.exec_run(f"mkdir -p /opt && rm -rf {MINI_RUNTIME_PREFIX}", timeout=60, workdir="/")
    if prep.exit_code != 0:
        return False
    return container.put_tar_archive("/opt", bundle)


def swe_agent_install_script() -> str:
    return f"""
set -euo pipefail
export PATH="{MINI_RUNTIME_BIN}:$PATH"
if [ ! -x "{MINI_RUNTIME_BIN}/mini" ]; then
  echo "mini runtime missing at {MINI_RUNTIME_BIN}/mini" >&2
  exit 1
fi
if [ ! -x "{MINI_RUNTIME_BIN}/pty-run" ]; then
  echo "mini pty wrapper missing at {MINI_RUNTIME_BIN}/pty-run" >&2
  exit 1
fi
mini --help >/dev/null
echo "mini runtime verified at {MINI_RUNTIME_PREFIX}"
""".strip()


def swe_agent_env_vars() -> dict[str, str]:
    env: dict[str, str] = {"MSWEA_COST_TRACKING": "ignore_errors"}
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
    fireworks_ai_api_key = os.environ.get("FIREWORKS_AI_API_KEY")
    if fireworks_api_key or fireworks_ai_api_key:
        resolved_fireworks_key = fireworks_api_key or fireworks_ai_api_key or ""
        env["FIREWORKS_API_KEY"] = fireworks_api_key or resolved_fireworks_key
        env["FIREWORKS_AI_API_KEY"] = fireworks_ai_api_key or resolved_fireworks_key
    return env


def swe_agent_run_command(
    prompt: str,
    *,
    model: str,
    budget: float | None = None,
    workdir: str | None = None,
    runtime_bin: str = MINI_RUNTIME_BIN,
    runtime_root: str = MINI_RUNTIME_ROOT,
    runtime_home: str | None = None,
    config_root: str | None = None,
    config_dir: str | None = None,
    cache_root: str | None = None,
    trajectory_dir: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    use_responses_api = model == "openai/gpt-5.4"
    resolved_home = runtime_home or f"{runtime_root}/home"
    resolved_config_root = config_root or f"{runtime_root}/config"
    resolved_config_dir = config_dir or f"{resolved_config_root}/mini-swe-agent"
    resolved_cache_root = cache_root or f"{runtime_root}/cache"
    resolved_trajectory_dir = trajectory_dir or f"{runtime_root}/trajectories"
    trajectory_path = f"{resolved_trajectory_dir}/latest.traj.json"
    setup = (
        f'export MINI_RUNTIME_ROOT={shlex.quote(runtime_root)} && '
        f'export HOME={shlex.quote(resolved_home)} && '
        f'export XDG_CONFIG_HOME={shlex.quote(resolved_config_root)} && '
        f'export XDG_CACHE_HOME={shlex.quote(resolved_cache_root)} && '
        f'export PATH="{runtime_bin}:$PATH" && '
        f'mkdir -p "$HOME" {shlex.quote(resolved_config_dir)} "$XDG_CACHE_HOME" {shlex.quote(resolved_trajectory_dir)} && '
        'printf "MSWEA_MODEL_NAME=%s\\n" ' + shlex.quote(model) + f' > {shlex.quote(resolved_config_dir + "/.env")} && '
        f'printf "MSWEA_CONFIGURED=true\\n" >> {shlex.quote(resolved_config_dir + "/.env")} && '
        'for k in OPENAI_API_KEY ANTHROPIC_API_KEY FIREWORKS_API_KEY FIREWORKS_AI_API_KEY; do '
        f'  [ -n "${{!k:-}}" ] && printf "%s=%s\\n" "$k" "${{!k}}" >> {shlex.quote(resolved_config_dir + "/.env")}; '
        'done; true'
    )
    parts: list[str] = [setup, "&&"]
    if workdir:
        parts.extend(["cd", shlex.quote(workdir), "&&"])
    parts.extend([
        "pty-run",
        "mini",
        "-c", "mini.yaml",
        "-c", "model.model_name=" + shlex.quote(model),
        "-c", "agent.output_path=" + shlex.quote(trajectory_path),
        "-t", shlex.quote(prompt),
        "-m", shlex.quote(model),
        "-y", "--exit-immediately",
    ])
    if use_responses_api:
        parts.extend(["--model-class", "litellm_response"])
    if reasoning_effort:
        reasoning_key = "reasoning.effort" if use_responses_api else "reasoning_effort"
        parts.extend(["-c", shlex.quote(f"model.model_kwargs.{reasoning_key}={reasoning_effort}")])
    if budget is not None:
        parts.extend(["-l", str(budget)])
    return " ".join(parts)


def _extract_swe_agent_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "value", "content", "result", "response", "output", "final_response", "final_message"):
            text = _extract_swe_agent_text(value.get(key))
            if text:
                return text
        if value.get("role") in {"assistant", "model"}:
            for key in ("content", "text", "message"):
                text = _extract_swe_agent_text(value.get(key))
                if text:
                    return text
        return ""
    if isinstance(value, list):
        parts = [_extract_swe_agent_text(item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts).strip()
    return ""


def _lookup_path(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_swe_agent_cost(traj: dict[str, Any]) -> float | None:
    candidate_paths = (
        ("info", "model_stats", "instance_cost"),
        ("info", "model_stats", "cost"),
        ("model_stats", "instance_cost"),
        ("model_stats", "cost"),
        ("info", "cost"),
    )
    for path in candidate_paths:
        value = _lookup_path(traj, path)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue

    for dict_value in iter_dicts(traj):
        for key in ("instance_cost", "cost_usd", "cost"):
            value = dict_value.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
    return None


def _extract_swe_agent_model(traj: dict[str, Any]) -> str | None:
    candidate_paths = (
        ("info", "config", "model", "model_name"),
        ("config", "model", "model_name"),
        ("model", "model_name"),
    )
    for path in candidate_paths:
        value = _lookup_path(traj, path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for dict_value in iter_dicts(traj):
        value = dict_value.get("model")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _swe_agent_pricing_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.strip()
    accounts_idx = normalized.find("accounts/")
    if accounts_idx >= 0:
        return normalized[accounts_idx:]
    if normalized.startswith("fireworks_ai/"):
        return normalized.removeprefix("fireworks_ai/")
    return normalized


def _swe_usage_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _extract_swe_agent_token_usage(traj: dict[str, Any]) -> UsageBreakdown | None:
    total_input_tokens = 0
    total_output_tokens = 0
    reasoning_output_tokens = 0
    cached_input_tokens = 0
    cache_write_input_tokens = 0

    for dict_value in iter_dicts(traj):
        prompt_tokens = _swe_usage_int(dict_value.get("prompt_tokens"))
        completion_tokens = _swe_usage_int(dict_value.get("completion_tokens"))
        completion_details = dict_value.get("completion_tokens_details")
        reasoning_tokens = (
            _swe_usage_int(completion_details.get("reasoning_tokens"))
            if isinstance(completion_details, dict)
            else 0
        )
        if prompt_tokens == 0 and completion_tokens == 0 and reasoning_tokens == 0:
            continue

        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens if completion_tokens else reasoning_tokens
        reasoning_output_tokens += reasoning_tokens

        prompt_details = dict_value.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_input_tokens += _swe_usage_int(prompt_details.get("cached_tokens"))
            cache_write_input_tokens += _swe_usage_int(prompt_details.get("cache_creation_tokens"))

    if total_input_tokens == 0 and total_output_tokens == 0 and reasoning_output_tokens == 0:
        return None

    return usage_from_totals(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )


def _swe_agent_usage_from_trajectory(traj: dict[str, Any], model: str | None = None) -> UsageBreakdown | None:
    reported_cost = _extract_swe_agent_cost(traj)
    token_usage = _extract_swe_agent_token_usage(traj)
    if token_usage is None:
        if reported_cost is None or reported_cost <= 0:
            return None
        return UsageBreakdown(cost_usd=reported_cost, cost_source="reported")

    pricing_model = _swe_agent_pricing_model(_extract_swe_agent_model(traj) or model)
    estimated_cost = estimate_cost(pricing_model, token_usage)
    if reported_cost is not None and reported_cost > 0:
        cost_usd = reported_cost
        cost_source = "reported"
    else:
        cost_usd = estimated_cost
        cost_source = "estimated" if estimated_cost is not None else None

    return UsageBreakdown(
        uncached_input_tokens=token_usage.uncached_input_tokens,
        cached_input_tokens=token_usage.cached_input_tokens,
        cache_write_input_tokens=token_usage.cache_write_input_tokens,
        visible_output_tokens=token_usage.visible_output_tokens,
        reasoning_output_tokens=token_usage.reasoning_output_tokens,
        cost_usd=cost_usd,
        cost_source=cost_source,
    )


def _extract_swe_agent_response(traj: dict[str, Any]) -> str:
    for key in ("final_response", "final_message", "response_text", "result", "answer", "output"):
        text = _extract_swe_agent_text(traj.get(key))
        if text:
            return text

    for key in ("trajectory", "messages", "history", "events", "steps"):
        value = traj.get(key)
        if isinstance(value, list):
            for item in reversed(value):
                text = _extract_swe_agent_text(item)
                if text:
                    return text
    return _extract_swe_agent_text(traj)


def _load_swe_agent_trajectory(container: DockerWorkspaceContainer, stdout: str) -> tuple[Path | None, dict[str, Any] | None]:
    candidate_paths: dict[str, float | None] = {}

    for line in stdout.splitlines():
        match = re.search(r"([^\s'\"`]+\.traj\.json)", line)
        if match:
            candidate_paths.setdefault(match.group(1), None)

    search_roots = " ".join(
        shlex.quote(root)
        for root in [
            MINI_RUNTIME_TRAJECTORY_DIR,
            MINI_RUNTIME_CONFIG_DIR,
        ]
    )
    find_result = container.exec_run(
        "for base in "
        f"{search_roots}; do "
        "[ -d \"$base\" ] || continue; "
        "find \"$base\" -type f -name '*.traj.json' -printf '%T@\\t%p\\n'; "
        "done",
        timeout=60,
        workdir="/",
    )
    if find_result.exit_code == 0:
        for line in find_result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            mtime_str, sep, path_str = line.partition("\t")
            if not sep or not path_str:
                continue
            try:
                candidate_paths[path_str] = float(mtime_str)
            except ValueError:
                candidate_paths.setdefault(path_str, None)

    ordered_candidates = sorted(
        candidate_paths.items(),
        key=lambda item: (item[1] is None, -(item[1] or 0.0), item[0]),
    )

    for path_str, _mtime in ordered_candidates:
        raw = container.read_text(path_str)
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return Path(path_str), parsed

    return None, None


def swe_agent_parse_usage(stdout: str, container: DockerWorkspaceContainer, model: str | None = None) -> UsageBreakdown | None:
    _path, traj = _load_swe_agent_trajectory(container, stdout)
    if not traj:
        return None

    return _swe_agent_usage_from_trajectory(traj, model)


def resolve_spell_model(model_name: str | None) -> str:
    if model_name is None:
        return "anthropic-tc:opus46"
    if any(model_name.startswith(prefix + ":") for prefix in ("anthropic-tc", "anthropic-pf", "fireworks", "fireworks-tc", "openai-tc", "codex-tc")):
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


def _escape_for_spell(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _normalize_spell_trailing(trailing: str) -> str:
    """Normalize shell-escaped bangs outside of Spell string literals."""
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(trailing):
        ch = trailing[i]
        if ch == '"':
            backslashes = 0
            j = i - 1
            while j >= 0 and trailing[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_string = not in_string
            result.append(ch)
        elif ch == "\\" and i + 1 < len(trailing) and trailing[i + 1] == "!" and not in_string:
            result.append("!")
            i += 1
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def build_spell_init(prompt: str, trailing: str | None = None) -> str:
    escaped_prompt = _escape_for_spell(prompt)
    trailing_expr = _normalize_spell_trailing(trailing) if trailing else "'(!extend)"
    return (
        f'(quine completion (eval (do (quine prompt "{escaped_prompt}") '
        f"{trailing_expr})))"
    )


def _validate_spell_bundle_member(arcname: str) -> None:
    parts = set(PurePosixPath(arcname).parts)
    forbidden = sorted(parts & HARBOR_SPELL_BUNDLE_FORBIDDEN_PARTS)
    if forbidden:
        raise RuntimeError(
            f"Refusing to add forbidden path(s) to Spell bundle member {arcname}: "
            f"{', '.join(forbidden)}"
        )


def spell_build_bundle(
    spell_root: Path | None = None,
    *,
    bundle_paths: tuple[str, ...] = SPELL_BUNDLE_PATHS,
    harbor_safe: bool = False,
) -> bytes:
    resolved_root = spell_root or discover_spell_root()
    if resolved_root is None:
        raise RuntimeError(
            "Could not locate Spell source tree. Set SPELL_ROOT to the spell repo root "
            "(must contain deps.edn, src/spell/benchmark_api.clj, and config/agents)."
        )

    missing = [rel for rel in bundle_paths if not (resolved_root / rel).exists()]
    if missing:
        raise RuntimeError(
            f"Spell source tree at {resolved_root} is missing required paths: {', '.join(missing)}"
        )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for rel in bundle_paths:
            arcname = f"spell/{rel}"
            if harbor_safe:
                _validate_spell_bundle_member(arcname)
            tar.add(resolved_root / rel, arcname=arcname)
    return buf.getvalue()


def spell_build_harbor_bundle(spell_root: Path | None = None) -> bytes:
    return spell_build_bundle(
        spell_root,
        bundle_paths=HARBOR_SPELL_BUNDLE_PATHS,
        harbor_safe=True,
    )


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
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FIREWORKS_API_KEY"):
        value = os.environ.get(key)
        if value:
            env[key] = value

    if model_name.startswith("codex-tc:"):
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

    candidate_agent = Path(agent_file).expanduser()
    if not candidate_agent.is_absolute():
        spell_relative = (resolved_root / candidate_agent).resolve()
        if spell_relative.exists():
            candidate_agent = spell_relative
    resolved_agent = candidate_agent.resolve()
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
    usage_log_path: str | None = None,
) -> str:
    from .reasoning_effort import (
        anthropic_thinking_budget,
        fireworks_reasoning_effort,
        is_anthropic_provider,
    )

    thinking = anthropic_thinking_budget(reasoning_effort) if is_anthropic_provider(model_name) else None
    resolved_reasoning_effort = fireworks_reasoning_effort(model_name, reasoning_effort)
    payload: dict[str, Any] = {
        "mode": "spell",
        "model": model_name,
        "budget": budget,
        "agent": _resolve_agent_for_container(agent_file),
        "depth": depth,
        "max_tokens": max_tokens,
        "reasoning_effort": resolved_reasoning_effort,
        "thinking": thinking,
        "trace": trace,
        "trace_dir": trace_dir,
        "usage_log_path": usage_log_path,
    }
    if init is not None:
        payload["init"] = init
    elif prompt is not None:
        payload["prompt"] = prompt
    payload = {key: value for key, value in payload.items() if value is not None}
    payload_json = json.dumps(payload, separators=(",", ":"))
    # Use a heredoc with single-quoted delimiter to prevent all shell
    # interpretation of the JSON payload.  shlex.quote + printf was
    # previously used, but some Docker container bash environments
    # mangle '!' characters inside the payload despite single-quoting.
    return (
        'export PATH="$HOME/.local/bin:$PATH" && '
        'CLOJURE_BIN="$(if command -v clojure >/dev/null 2>&1 && clojure -Sdescribe >/dev/null 2>&1; '
        'then echo clojure; elif command -v clj >/dev/null 2>&1; then echo clj; else echo clojure; fi)" && '
        f"$CLOJURE_BIN -Sdeps {shlex.quote(SPELL_SDEPS)} -M -m spell.benchmark-api "
        "--request - --response - <<'SPELL_PAYLOAD'\n"
        f"{payload_json}\n"
        "SPELL_PAYLOAD"
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


def _spell_error_type(output: ExecOutput, parsed: dict[str, Any] | None) -> str:
    if output.timed_out:
        return (parsed or {}).get("error_type") or "subprocess_timeout"
    return (parsed or {}).get("error_type") if parsed else "parse_error"


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


def spell_usage_from_summary(summary: dict[str, Any] | None) -> UsageBreakdown | None:
    if not isinstance(summary, dict):
        return None
    usage = summary.get("total")
    if not isinstance(usage, dict):
        return None

    def _int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    canonical_token_keys = (
        "uncached_input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "visible_output_tokens",
        "reasoning_output_tokens",
    )
    legacy_token_keys = (
        "input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
    )
    if not any(isinstance(usage.get(key), (int, float)) for key in canonical_token_keys + legacy_token_keys):
        return None

    uncached_input_tokens = _int(usage.get("uncached_input_tokens", 0))
    cached = _int(usage.get("cached_input_tokens", 0))
    cache_write = _int(usage.get("cache_write_input_tokens", 0))
    visible_output_tokens = _int(usage.get("visible_output_tokens", 0))
    reasoning_output_tokens = _int(usage.get("reasoning_output_tokens", 0))

    if not any((uncached_input_tokens, cached, cache_write, visible_output_tokens, reasoning_output_tokens)):
        uncached_input_tokens = _int(usage.get("input_tokens", 0))
        cached = _int(usage.get("cache_read_input_tokens", 0))
        cache_write = _int(usage.get("cache_creation_input_tokens", 0))
        visible_output_tokens = _int(usage.get("output_tokens", 0))

    total_input_tokens = uncached_input_tokens + cached + cache_write
    cost = usage.get("cost")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else None
    return UsageBreakdown(
        uncached_input_tokens=max(total_input_tokens - cached - cache_write, 0),
        cached_input_tokens=cached,
        cache_write_input_tokens=cache_write,
        visible_output_tokens=visible_output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cost_usd=cost_usd,
    )


def spell_parse_usage(stdout: str) -> UsageBreakdown | None:
    parsed = spell_parse_benchmark_response(stdout)
    if not parsed:
        return None
    return spell_usage_from_summary(parsed.get("usage"))


def spell_parse_usage_log(path: Path) -> UsageBreakdown | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(parsed, dict):
        return None
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return None
    return spell_usage_from_summary(usage)


def _install_agent_runtime(
    container: DockerWorkspaceContainer,
    *,
    condition: Literal["spell", "claude_code", "codex", "swe_agent"],
    version: str | None = None,
    model_spec: str | None = None,
) -> bool:
    if condition == "claude_code":
        env = cc_env_vars(model=model_spec)
        if not install_nvm_node(container, env=env):
            return False
        return run_install_script(container, cc_install_script(version=version), env=env)
    if condition == "codex":
        env = codex_env_vars()
        if not install_nvm_node(container, env=env):
            return False
        return run_install_script(container, codex_install_script(version=version), env=env)
    if condition == "swe_agent":
        bundle = swe_agent_build_bundle(version=version)
        if not swe_agent_pre_install(container, bundle):
            return False
        return run_install_script(container, swe_agent_install_script(), env=swe_agent_env_vars())
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
    condition: Literal["spell", "claude_code", "codex", "swe_agent"],
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
    paper_compliant: bool = False,
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
        command = cc_run_command(
            prompt,
            model=resolved_model,
            budget=budget,
            reasoning_effort=reasoning_effort,
            allowed_tools=CC_PAPER_COMPLIANT_ALLOWED_TOOLS if paper_compliant else None,
        )
        output = container.exec_run(
            command,
            timeout=timeout,
            env=env,
        )
        events = parse_jsonl_events(output.stdout)
        session_id = extract_claude_session_id(events)
        missing_cli = cc_missing_cli_error(output)
        has_result_event = any(event.get("type") == "result" for event in events)
        ledger_texts = []
        if not missing_cli and not has_result_event:
            ledger_texts = _read_claude_usage_ledgers_from_container(container, session_id)
        usage = cc_parse_usage(output.stdout, model=resolved_model, ledger_texts=ledger_texts)
        return AgentRunResult(
            output=output,
            usage=usage,
            error_message=output.stderr[:2000] if output.exit_code != 0 else None,
            error_type=("install_error" if missing_cli else "cli_error") if output.exit_code != 0 else None,
            metadata={
                "claude_session_id": session_id,
                "claude_usage_ledger_sources": [source for source, _text in ledger_texts],
            },
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
                reasoning_effort=reasoning_effort,
            ),
            timeout=timeout,
            env=env,
        )
        events = parse_jsonl_events(output.stdout)
        filtered_events = filter_codex_raw_events(events)
        stream_usage = usage_from_events(events, model=resolved_model)
        thread_id = extract_codex_thread_id(events)
        rollout_path, rollout_text = _read_codex_rollout_from_container(container, thread_id)
        rollout_info = parse_codex_rollout(rollout_text) if rollout_text else None
        rollout_usage = usage_from_rollout(rollout_text, model=resolved_model) if rollout_text else None
        merged_usage = merge_codex_usage(stream_usage, rollout_usage, model=resolved_model)
        usage = merged_usage

        if thread_id is None:
            rollout_status = "missing_thread_id"
        elif rollout_path is None:
            rollout_status = "missing"
        elif rollout_info is None:
            rollout_status = "no_token_count_event"
        else:
            rollout_status = "ok"

        metadata = {
            "resolved_model": resolved_model,
            "thread_id": thread_id,
            "rollout_path": rollout_path,
            "rollout_token_count": rollout_info,
            "rollout_status": rollout_status,
            "reasoning_output_tokens": usage.reasoning_output_tokens if usage else None,
        }
        if usage and usage.cost_source:
            metadata["cost_source"] = usage.cost_source
        raw = {
            "events": filtered_events,
            "rollout_token_count": rollout_info,
            "rollout_path": rollout_path,
        }
        raw = {key: value for key, value in raw.items() if value is not None}
        return AgentRunResult(
            output=output,
            usage=usage,
            error_message=output.stderr[:2000] if output.exit_code != 0 else None,
            error_type="codex_cli_error" if output.exit_code != 0 else None,
            metadata=metadata,
            raw=raw or None,
        )

    if condition == "swe_agent":
        from .reasoning_effort import effective_reasoning_effort

        resolved_model = model_spec or "anthropic/claude-opus-4-6"
        resolved_reasoning_effort = effective_reasoning_effort("swe_agent", resolved_model, reasoning_effort)
        env = swe_agent_env_vars()
        output = container.exec_run(
            swe_agent_run_command(
                prompt,
                model=resolved_model,
                budget=budget,
                workdir=container.container_workdir,
                reasoning_effort=resolved_reasoning_effort,
            ),
            timeout=timeout,
            env=env,
        )
        traj_path, traj = _load_swe_agent_trajectory(container, output.stdout)
        usage = swe_agent_parse_usage(output.stdout, container, model=resolved_model)
        response_text = _extract_swe_agent_response(traj) if traj else output.stdout.strip()
        trace_file: str | None = None
        if trace_dir is not None and traj_path is not None:
            dest = Path(trace_dir)
            dest.mkdir(parents=True, exist_ok=True)
            if container.copy_to_host(str(traj_path), dest):
                trace_file = str(dest / traj_path.name)
            else:
                trace_file = str(dest)
        raw = {
            "trajectory_path": str(traj_path) if traj_path else None,
            "trajectory": traj,
        }
        raw = {k: v for k, v in raw.items() if v is not None}
        metadata = {}
        if usage and usage.cost_source:
            metadata["cost_source"] = usage.cost_source
        return AgentRunResult(
            output=output,
            usage=usage,
            response_text=response_text or None,
            error_message=output.stderr[:2000] if output.exit_code != 0 else (None if traj else "mini trajectory not found"),
            error_type="mini_cli_error" if output.exit_code != 0 else ("parse_error" if traj is None else None),
            raw=raw or None,
            trace_dir=trace_file,
            metadata=metadata or None,
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
        error_type=_spell_error_type(output, parsed),
        raw=parsed,
        trace_dir=persisted_trace_dir,
    )
