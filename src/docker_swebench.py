"""Docker container management for SWE-bench agent execution.

Creates lightweight containers from a Python base image with the
SWE-bench workspace mounted, runs agents inside, and extracts patches.

Uses the docker Python SDK (docker-py) for container lifecycle management.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Base image for SWE-bench containers.  Python 3.11 slim has git and
# a working Python, which is sufficient for most SWE-bench repos.
DEFAULT_IMAGE = "python:3.11-slim"

# Idempotent NVM + Node 22 install script (same as terminal_bench/base.py).
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


@dataclass
class ExecResult:
    """Result from a docker exec command."""
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass
class ParsedUsage:
    """Token usage extracted from agent output."""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cached_input_tokens


class DockerContainer:
    """Manages a Docker container for SWE-bench agent execution.

    Creates a container with the workspace bind-mounted at /workspace,
    provides exec_run() for running commands inside, and handles cleanup.

    Uses docker CLI (not docker-py) to avoid an extra dependency.
    """

    def __init__(self, workspace: Path, image: str = DEFAULT_IMAGE):
        self.workspace = workspace
        self.image = image
        self.container_id: str | None = None

    def start(self) -> str:
        """Create and start the container. Returns container ID.

        Copies the workspace into the container via `docker cp` rather than
        bind-mounting. This is necessary because SWE-bench workspaces are git
        worktrees whose `.git` file references the host repo path, which
        doesn't exist inside the container.  Copying + `git init` gives us a
        clean standalone repo.
        """
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--rm",
                "-w", "/workspace",
                self.image,
                "sleep", "infinity",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker run failed: {result.stderr}")
        self.container_id = result.stdout.strip()
        logger.info(f"Started container {self.container_id[:12]}")

        # Install git inside slim image (needed for diff extraction)
        self.exec_run("apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1",
                      timeout=120)

        # Copy workspace contents into container.
        # Use docker cp to transfer, then reinitialize git so diff works.
        cp_result = subprocess.run(
            ["docker", "cp", f"{self.workspace}/.", f"{self.container_id}:/workspace"],
            capture_output=True, text=True, timeout=300,
        )
        if cp_result.returncode != 0:
            raise RuntimeError(f"docker cp failed: {cp_result.stderr}")

        # The copied .git file from the worktree points to a host path.
        # Replace it with a proper standalone repo: init, add all, commit.
        self.exec_run(
            "rm -rf .git && git init -q && "
            "git config user.email 'bench@local' && "
            "git config user.name 'bench' && "
            "git add -A && git commit -q -m 'base'",
            timeout=120,
        )
        logger.info(f"Workspace copied and git initialized in container")
        return self.container_id

    def exec_run(
        self,
        cmd: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        workdir: str = "/workspace",
    ) -> ExecResult:
        """Execute a shell command inside the container."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        docker_cmd = ["docker", "exec"]
        if workdir:
            docker_cmd.extend(["-w", workdir])
        for k, v in (env or {}).items():
            docker_cmd.extend(["-e", f"{k}={v}"])
        docker_cmd.extend([self.container_id, "bash", "-c", cmd])

        timed_out = False
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True, text=True,
                timeout=timeout,
            )
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = 124
            stdout = ""
            stderr = f"Timeout after {timeout}s"

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def diff(self) -> str:
        """Extract git diff HEAD from the container workspace, including new files."""
        result = self.exec_run(
            "git add -A && git diff --cached HEAD; status=$?; git reset -q >/dev/null 2>&1; exit $status"
        )
        if result.exit_code != 0:
            logger.warning(f"git diff failed: {result.stderr[:200]}")
            return ""
        return result.stdout

    def changed_files(self) -> list[str]:
        """List files changed relative to HEAD, including newly created files."""
        result = self.exec_run(
            "git add -A && git diff --name-only --cached HEAD; status=$?; git reset -q >/dev/null 2>&1; exit $status"
        )
        if result.exit_code != 0:
            return []
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container_id:
            subprocess.run(
                ["docker", "stop", self.container_id],
                capture_output=True, timeout=30,
            )
            logger.info(f"Stopped container {self.container_id[:12]}")
            self.container_id = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def install_nvm_node(container: DockerContainer) -> bool:
    """Install NVM and Node 22 inside a container. Returns True on success."""
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Installing NVM + Node in container (attempt {attempt}/{max_attempts})...")
        result = container.exec_run(NVM_NODE_INSTALL, timeout=300)
        if result.exit_code == 0:
            logger.info(result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "Node installed")
            return True

        logger.error(f"NVM/Node install failed: {result.stderr[:500]}")
        if attempt < max_attempts:
            time.sleep(2)

    # Fallback for transient NVM/CDN failures: apt-provided node/npm.
    logger.info("Falling back to apt nodejs/npm install...")
    fallback = container.exec_run(
        "apt-get update -qq && apt-get install -y -qq nodejs npm >/dev/null 2>&1 && "
        "echo \"node installed via apt: $(node --version 2>/dev/null || echo missing)\"",
        timeout=300,
    )
    if fallback.exit_code != 0:
        logger.error(f"apt nodejs/npm install failed: {fallback.stderr[:500]}")
        return False

    logger.info(fallback.stdout.strip().splitlines()[-1] if fallback.stdout.strip() else "node installed via apt")
    return True


def parse_jsonl_events(stdout: str) -> list[dict]:
    """Parse JSONL from stdout, skipping non-JSON lines."""
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                events.append(obj)
        except json.JSONDecodeError:
            continue
    return events


def iter_dicts(v):
    """Recursively yield all dicts nested in a JSON value."""
    if isinstance(v, dict):
        yield v
        for child in v.values():
            yield from iter_dicts(child)
    elif isinstance(v, list):
        for child in v:
            yield from iter_dicts(child)


# ── Claude Code (runs inside container) ──────────────────────────

CC_ALLOWED_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep",
    "WebFetch", "NotebookEdit",
]


def install_claude_code(container: DockerContainer, env: dict[str, str]) -> bool:
    """Install Claude Code CLI inside the container."""
    script = """
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
npm install -g @anthropic-ai/claude-code@latest
echo "claude installed: $(claude --version 2>/dev/null || echo unknown)"
"""
    result = container.exec_run(script, timeout=300, env=env)
    if result.exit_code != 0:
        logger.error(f"Claude Code install failed: {result.stderr[:500]}")
        return False
    logger.info("Claude Code CLI installed")
    return True


def run_claude_code(
    container: DockerContainer,
    prompt: str,
    *,
    model: str,
    budget: float,
    timeout: int,
    env: dict[str, str],
) -> ExecResult:
    """Run Claude Code inside the container."""
    escaped = shlex.quote(prompt)
    cmd = (
        f'[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"; '
        f"claude "
        f"--setting-sources '' "
        f"--verbose "
        f"--output-format stream-json "
        f"--max-budget-usd {budget} "
        f"--model {shlex.quote(model)} "
        f"--allowedTools {' '.join(CC_ALLOWED_TOOLS)} "
        f"-p {escaped}"
    )
    return container.exec_run(cmd, timeout=timeout, env=env)


def parse_claude_code_usage(stdout: str) -> ParsedUsage | None:
    """Parse usage from Claude Code stream-json output."""
    events = parse_jsonl_events(stdout)
    result_event = None
    for event in reversed(events):
        if event.get("type") == "result":
            result_event = event
            break

    if not result_event:
        logger.warning("No result event found in Claude Code output")
        return None

    usage = result_event.get("usage", {})

    def _int(v) -> int:
        return int(v) if isinstance(v, (int, float)) else 0

    input_tokens = _int(usage.get("input_tokens", 0))
    cached = (_int(usage.get("cache_creation_input_tokens", 0))
              + _int(usage.get("cache_read_input_tokens", 0)))
    output_tokens = _int(usage.get("output_tokens", 0))

    cost_usd = result_event.get("total_cost_usd")
    if cost_usd is None:
        cost_usd = result_event.get("cost_usd")
    if cost_usd is not None:
        cost_usd = float(cost_usd)

    return ParsedUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


# ── Codex (runs inside container) ────────────────────────────────

def install_codex(container: DockerContainer, env: dict[str, str]) -> bool:
    """Install Codex CLI inside the container."""
    script = """
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
npm install -g @openai/codex@latest
mkdir -p "$HOME/.codex"
echo '{"OPENAI_API_KEY": "'"$OPENAI_API_KEY"'"}' > "$HOME/.codex/auth.json"
echo "codex installed: $(codex --version 2>/dev/null || echo unknown)"
"""
    result = container.exec_run(script, timeout=300, env=env)
    if result.exit_code != 0:
        logger.error(f"Codex install failed: {result.stderr[:500]}")
        return False
    logger.info("Codex CLI installed")
    return True


def run_codex(
    container: DockerContainer,
    prompt: str,
    *,
    model: str,
    timeout: int,
    env: dict[str, str],
) -> ExecResult:
    """Run Codex inside the container."""
    escaped = shlex.quote(prompt)
    cmd = (
        f'[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"; '
        f"codex exec --json --ephemeral --skip-git-repo-check "
        f"--sandbox danger-full-access "
        f"-m {shlex.quote(model)} "
        f"{escaped}"
    )
    return container.exec_run(cmd, timeout=timeout, env=env)


def parse_codex_usage(stdout: str) -> ParsedUsage | None:
    """Parse usage from Codex JSONL output."""
    events = parse_jsonl_events(stdout)
    usage_candidate = None
    for event in events:
        for d in iter_dicts(event):
            if any(k in d for k in ("input_tokens", "output_tokens")):
                usage_candidate = d

    if not usage_candidate:
        logger.warning("No usage data found in Codex output")
        return None

    def _int(v) -> int:
        return int(v) if isinstance(v, (int, float)) else 0

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


# ── Spell (runs on host, io/sh bridges to container) ────────────

def run_spell_docker(
    container: DockerContainer,
    *,
    payload: dict,
    project_root: Path,
    timeout: int,
) -> tuple[ExecResult, dict | None]:
    """Run Spell on host with io/sh bridged to the Docker container.

    Sets SPELL_DOCKER_CONTAINER and SPELL_DOCKER_WORKDIR env vars so that
    io/sh commands execute inside the container via `docker exec`.

    Returns (ExecResult, parsed_json_response).
    """
    sdeps = '{:deps {local/spell {:local/root "%s"}}}' % project_root
    clj_cmd = ["clj", "-Sdeps", sdeps, "-M", "-m", "spell.benchmark-api",
               "--request", "-", "--response", "-"]

    env = os.environ.copy()
    env["SPELL_DOCKER_CONTAINER"] = container.container_id
    env["SPELL_DOCKER_WORKDIR"] = "/workspace"
    env["SPELL_SH_TIMEOUT"] = "120"

    start = time.time()
    try:
        proc = subprocess.run(
            clj_cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(project_root),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(exit_code=124, stdout="", stderr=f"Timeout ({timeout}s)", timed_out=True), None

    latency = time.time() - start

    stdout = (proc.stdout or "").strip()
    parsed = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            pass

    return ExecResult(
        exit_code=proc.returncode,
        stdout=stdout,
        stderr=proc.stderr or "",
        timed_out=False,
    ), parsed
