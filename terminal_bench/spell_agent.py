"""Spell agent for Terminal-Bench with in-container execution.

Runs Spell's benchmark API inside the task container via DockerExecAgent,
so all benchmarked methods execute in the same container environment.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import tarfile
from pathlib import Path

try:
    from .base import DockerExecAgent, ExecOutput, ParsedUsage
except ImportError:
    from base import DockerExecAgent, ExecOutput, ParsedUsage
from terminal_bench.agents.failure_mode import FailureMode

logger = logging.getLogger(__name__)

# Map litellm model names to Spell benchmark API format (provider:alias)
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

SPELL_AGENT_IN_CONTAINER = "/opt/spell/config/agents/io-tc.agent.edn"
SPELL_SDEPS = '{:deps {local/spell {:local/root "/opt/spell"}}}'
SPELL_BUNDLE_PATHS = ("deps.edn", "src", "config")

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


def _resolve_spell_model(model_name: str | None) -> str:
    """Convert litellm model name to Spell benchmark API format."""
    if model_name is None:
        return "anthropic-tc:opus46"
    if any(model_name.startswith(p + ":") for p in [
        "anthropic-tc", "anthropic-pf", "openai", "codex-msg", "codex-tc",
    ]):
        return model_name
    if model_name in LITELLM_TO_SPELL_MODEL:
        return LITELLM_TO_SPELL_MODEL[model_name]
    if "/" in model_name:
        bare = model_name.split("/", 1)[1]
        return f"anthropic-tc:{bare}"
    return f"anthropic-tc:{model_name}"


def _is_spell_root(path: Path) -> bool:
    return (
        (path / "deps.edn").exists()
        and (path / "src" / "spell" / "benchmark_api.clj").exists()
        and (path / "config" / "agents").exists()
    )


def _discover_spell_root(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())

    from_env = os.environ.get("SPELL_ROOT")
    if from_env:
        candidates.append(Path(from_env).expanduser().resolve())

    # Monorepo default: <spell>/benchmarking/terminal_bench/spell_agent.py
    candidates.append(Path(__file__).resolve().parents[2])

    # Standalone benchmarking checkout with sibling spell repo.
    candidates.append(Path(__file__).resolve().parents[2] / "spell")

    # Last resort: current working directory.
    candidates.append(Path.cwd())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_spell_root(candidate):
            return candidate
    return None


class SpellAgent(DockerExecAgent):
    """Terminal-Bench agent that runs Spell entirely inside the task container."""

    def __init__(
        self,
        model_name: str | None = None,
        budget: float = 5.0,
        depth: int | None = None,
        sh_timeout: int = 120,
        agent_timeout: int = 600,
        agent_file: str | None = None,
        spell_root: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=_resolve_spell_model(model_name), agent_timeout=agent_timeout, **kwargs)
        self._budget = budget
        self._depth = depth
        self._sh_timeout = sh_timeout
        self._agent_file = agent_file or SPELL_AGENT_IN_CONTAINER
        self._spell_root = _discover_spell_root(spell_root)
        self._spell_bundle: bytes | None = None

    @staticmethod
    def name() -> str:
        return "spell"

    def _requires_node(self) -> bool:
        return False

    def _install_script(self) -> str:
        return SPELL_RUNTIME_INSTALL

    def _uses_codex_provider(self) -> bool:
        return self._model_name.startswith("codex-msg:") or self._model_name.startswith("codex-tc:")

    def _load_codex_auth_b64(self) -> str | None:
        # Allow explicit env override for nonstandard auth locations.
        if os.environ.get("CODEX_AUTH_JSON_B64"):
            return os.environ["CODEX_AUTH_JSON_B64"]

        auth_path = Path.home() / ".codex" / "auth.json"
        if not auth_path.exists():
            return None
        return base64.b64encode(auth_path.read_bytes()).decode("ascii")

    def _env_vars(self) -> dict[str, str]:
        env: dict[str, str] = {
            "SPELL_SH_TIMEOUT": str(self._sh_timeout),
        }

        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(key)
            if value:
                env[key] = value

        if self._uses_codex_provider():
            codex_auth_b64 = self._load_codex_auth_b64()
            if codex_auth_b64:
                env["CODEX_AUTH_JSON_B64"] = codex_auth_b64
            else:
                logger.warning(
                    "Spell codex-* model requested but ~/.codex/auth.json was not found. "
                    "In-container run may fail auth unless CODEX_AUTH_JSON_B64 is set."
                )

        return env

    def _build_spell_bundle(self) -> bytes:
        if self._spell_root is None:
            raise RuntimeError(
                "Could not locate Spell source tree. Set SPELL_ROOT to the spell repo root "
                "(must contain deps.edn, src/spell/benchmark_api.clj, and config/agents)."
            )

        missing = [rel for rel in SPELL_BUNDLE_PATHS if not (self._spell_root / rel).exists()]
        if missing:
            raise RuntimeError(
                f"Spell source tree at {self._spell_root} is missing required paths: {', '.join(missing)}"
            )

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel in SPELL_BUNDLE_PATHS:
                src = self._spell_root / rel
                tar.add(src, arcname=f"spell/{rel}")
        return buf.getvalue()

    def _install_agent_cli(self, container) -> bool:
        """Copy Spell runtime into the container, then install Java/Clojure deps."""
        try:
            if self._spell_bundle is None:
                self._spell_bundle = self._build_spell_bundle()
        except Exception as exc:
            logger.error(f"Failed to build Spell bundle: {exc}")
            return False

        prep = self._container_exec(container, "mkdir -p /opt && rm -rf /opt/spell", timeout=60)
        if prep.exit_code != 0:
            logger.error(f"Failed to prepare /opt in container: {prep.stderr[:1000]}")
            return False

        try:
            copied = container.put_archive("/opt", self._spell_bundle)
        except Exception as exc:
            logger.error(f"Failed to copy Spell bundle into container: {exc}")
            return False

        if not copied:
            logger.error("Docker put_archive returned False while copying Spell bundle")
            return False

        logger.info("Installing Spell runtime in container...")
        result = self._container_exec(
            container,
            self._install_script(),
            timeout=600,
            env=self._env_vars(),
        )
        if result.exit_code != 0:
            logger.error(f"Spell runtime install failed:\n{result.stderr[:1000]}")
            return False
        logger.info("Spell runtime installed")
        return True

    def _build_payload(self, instruction: str) -> dict:
        payload: dict = {
            "mode": "spell",
            "model": self._model_name,
            "prompt": instruction,
            "budget": self._budget,
            "agent": self._agent_file,
            "trace": True,
        }
        if self._depth is not None:
            payload["depth"] = self._depth
        return payload

    def _run_command(self, instruction: str, workdir: str) -> str:
        del workdir  # cwd is already controlled by DockerExecAgent._container_exec(workdir=...)
        payload = self._build_payload(instruction)
        payload_json = json.dumps(payload, separators=(",", ":"))
        return (
            'export PATH="$HOME/.local/bin:$PATH" && '
            'CLOJURE_BIN="$(if command -v clojure >/dev/null 2>&1 && clojure -Sdescribe >/dev/null 2>&1; then echo clojure; elif command -v clj >/dev/null 2>&1; then echo clj; else echo clojure; fi)" && '
            f"printf %s {shlex.quote(payload_json)} | "
            f"$CLOJURE_BIN -Sdeps {shlex.quote(SPELL_SDEPS)} -M -m spell.benchmark-api "
            "--request - --response -"
        )

    def _extract_response_json(self, stdout: str) -> dict | None:
        raw = (stdout or "").strip()
        if not raw:
            return None

        # Benchmark API normally prints one JSON object, but keep this resilient.
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

    def _parse_usage(self, stdout: str) -> ParsedUsage | None:
        parsed = self._extract_response_json(stdout)
        if not parsed:
            logger.warning("No benchmark-api JSON response found in Spell stdout")
            return None

        usage = (parsed.get("usage") or {}).get("total", {})

        def _int(v) -> int:
            return int(v) if isinstance(v, (int, float)) else 0

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

    def _determine_failure_mode(self, result: ExecOutput, usage: ParsedUsage | None) -> FailureMode:
        del usage
        if result.timed_out:
            return FailureMode.AGENT_TIMEOUT
        if result.exit_code != 0:
            return FailureMode.UNKNOWN_AGENT_ERROR

        parsed = self._extract_response_json(result.stdout)
        if not parsed:
            return FailureMode.UNKNOWN_AGENT_ERROR
        if parsed.get("ok") is True:
            return FailureMode.NONE

        if parsed.get("error_type") == "timeout":
            return FailureMode.AGENT_TIMEOUT
        return FailureMode.UNKNOWN_AGENT_ERROR
