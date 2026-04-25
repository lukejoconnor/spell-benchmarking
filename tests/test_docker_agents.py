from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from src.docker_agents import (
    CLAUDE_CLI_MISSING_MARKER,
    CLAUDE_RUNTIME_NPM_PREFIX,
    CLAUDE_RUNTIME_ROOT,
    MIN_NODE_MAJOR_VERSION,
    NODE_BINARY_VERSION,
    AgentRunResult,
    CODEX_RUNTIME_CODEX_HOME,
    CODEX_RUNTIME_NPM_CACHE,
    CODEX_RUNTIME_NPM_PREFIX,
    ExecOutput,
    INSTALL_TIMEOUT_RETRY_CLEANUP_SCRIPT,
    cc_env_vars,
    cc_missing_cli_error,
    cc_install_script,
    cc_run_command,
    NVM_NODE_INSTALL,
    UV_BOOTSTRAP_COMMAND,
    _install_agent_runtime,
    _cleanup_nodejs_binary_shell,
    _install_nodejs_binary_shell,
    _install_nodejs_shell,
    _is_retryable_apt_fetch_failure,
    _is_retryable_install_timeout,
    _is_retryable_pip_hash_mismatch,
    _verify_nodejs_shell,
    _read_codex_rollout_from_exec,
    _load_swe_agent_trajectory,
    _normalize_spell_trailing,
    build_spell_init,
    cc_parse_usage,
    codex_env_vars,
    codex_install_script,
    codex_parse_usage,
    codex_run_command,
    ensure_python_packages,
    install_nvm_node_exec,
    ensure_uv_exec,
    run_agent_in_container,
    run_install_script,
    resolve_spell_model,
    spell_env_vars,
    swe_agent_install_script,
)


class FakeInstallContainer:
    def __init__(self, outputs: list[ExecOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def exec_run(self, cmd: str, **kwargs) -> ExecOutput:
        self.calls.append((cmd, dict(kwargs)))
        if not self._outputs:
            raise AssertionError(f"Unexpected exec call: {cmd}")
        return self._outputs.pop(0)


class SequentialExec:
    def __init__(self, outputs: list[ExecOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, cmd: str, **kwargs) -> ExecOutput:
        self.calls.append((cmd, dict(kwargs)))
        if not self._outputs:
            raise AssertionError(f"Unexpected exec call: {cmd}")
        return self._outputs.pop(0)


def _write_executable(path: Path, script: str) -> None:
    path.write_text(script)
    path.chmod(0o755)


class DockerAgentsTests(unittest.TestCase):
    def test_resolve_spell_model_preserves_fireworks_prefix(self) -> None:
        self.assertEqual(resolve_spell_model("fireworks:glm-5"), "fireworks:glm-5")

    def test_build_spell_init_defaults_to_extend(self) -> None:
        init = build_spell_init("Solve it")

        self.assertIn('(quine prompt "Solve it")', init)
        self.assertIn("'(!extend)", init)

    def test_build_spell_init_normalizes_shell_escaped_bang_in_trailing(self) -> None:
        init = build_spell_init("Solve it", trailing="'(\\!describe io)")

        self.assertIn("'(!describe io)", init)
        self.assertNotIn("\\!describe", init)

    def test_normalize_spell_trailing_preserves_backslash_bang_inside_strings(self) -> None:
        trailing = "'(do (think \"literal \\\\! stays\") (!describe io))"

        normalized = _normalize_spell_trailing(trailing)

        self.assertEqual(normalized, trailing)

    def test_codex_parse_usage_estimates_cost_and_counts_cached_tokens(self) -> None:
        usage = codex_parse_usage(
            '\n'.join(
                [
                    '{"type":"thread.started","thread_id":"t1"}',
                    '{"type":"turn.completed","usage":{"input_tokens":6804,"cached_input_tokens":5888,"output_tokens":6347}}',
                ]
            ),
            model="gpt-5-codex",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.input_tokens, 6804)
        self.assertEqual(usage.cached_input_tokens, 5888)
        self.assertEqual(usage.total_input_tokens, 6804)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.065351, places=6)
        self.assertEqual(usage.cost_source, "estimated")
        self.assertEqual(usage.reasoning_output_tokens, 0)

    def test_codex_env_vars_redirect_runtime_to_tmp(self) -> None:
        env = codex_env_vars()

        self.assertEqual(env["CODEX_HOME"], CODEX_RUNTIME_CODEX_HOME)
        self.assertEqual(env["NPM_CONFIG_PREFIX"], CODEX_RUNTIME_NPM_PREFIX)
        self.assertEqual(env["NPM_CONFIG_CACHE"], CODEX_RUNTIME_NPM_CACHE)
        self.assertEqual(env["npm_config_cache"], CODEX_RUNTIME_NPM_CACHE)
        self.assertNotIn("HOME", env)

    def test_cc_parse_usage_preserves_cache_write_and_reprices_underreported_cost(self) -> None:
        usage = cc_parse_usage(
            json.dumps(
                {
                    "type": "result",
                    "result": "ok",
                    "usage": {
                        "input_tokens": 100_000,
                        "cache_creation_input_tokens": 50_000,
                        "cache_read_input_tokens": 25_000,
                        "output_tokens": 10_000,
                    },
                    "total_cost_usd": 0.4575,
                }
            ),
            model="claude-sonnet-4-6",
        )

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.total_input_tokens, 175_000)
        self.assertEqual(usage.cached_input_tokens, 25_000)
        self.assertEqual(usage.cache_write_input_tokens, 50_000)
        self.assertAlmostEqual(usage.cost_usd or 0.0, 0.645, places=6)
        self.assertEqual(usage.cost_source, "estimated")

    def test_codex_install_script_uses_tmp_runtime_paths(self) -> None:
        script = codex_install_script(version="latest")

        self.assertIn(f'export CODEX_HOME="${{CODEX_HOME:-{CODEX_RUNTIME_CODEX_HOME}}}"', script)
        self.assertIn(f'export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CODEX_RUNTIME_NPM_PREFIX}}}"', script)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", script)
        self.assertIn('unset NPM_CONFIG_PREFIX', script)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', script)
        self.assertIn("npm install -g @openai/codex@latest", script)
        self.assertNotIn("$HOME/.nvm/nvm.sh", script)
        self.assertIn('> "$CODEX_HOME/auth.json"', script)

    def test_codex_run_command_sets_tmp_runtime_before_exec_without_overriding_home(self) -> None:
        cmd = codex_run_command("Return 42.", model="gpt-5-codex")

        self.assertIn(f'export CODEX_HOME="${{CODEX_HOME:-{CODEX_RUNTIME_CODEX_HOME}}}"', cmd)
        self.assertNotIn('export HOME="${HOME:-', cmd)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", cmd)
        self.assertIn('unset NPM_CONFIG_PREFIX', cmd)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', cmd)
        self.assertIn("codex -c 'web_search=\"disabled\"' exec", cmd)

    def test_cc_install_script_uses_explicit_runtime_prefix_and_fails_closed(self) -> None:
        script = cc_install_script(version="latest")

        self.assertIn("set -euo pipefail", script)
        self.assertIn(f'export CLAUDE_RUNTIME_ROOT="${{CLAUDE_RUNTIME_ROOT:-{CLAUDE_RUNTIME_ROOT}}}"', script)
        self.assertIn(f'export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CLAUDE_RUNTIME_NPM_PREFIX}}}"', script)
        self.assertIn("export PATH=\"$NPM_CONFIG_PREFIX/bin:$PATH\"", script)
        self.assertIn("npm install -g @anthropic-ai/claude-code@latest", script)
        self.assertIn("command -v claude >/dev/null 2>&1", script)
        self.assertIn(CLAUDE_CLI_MISSING_MARKER, script)

    def test_cc_run_command_checks_for_missing_claude_before_exec(self) -> None:
        cmd = cc_run_command(
            "Return 42.",
            model="anthropic/claude-opus-4-6",
            budget=1.25,
            reasoning_effort="high",
        )

        self.assertIn("set -euo pipefail", cmd)
        self.assertIn("command -v claude >/dev/null 2>&1", cmd)
        self.assertIn(CLAUDE_CLI_MISSING_MARKER, cmd)
        self.assertIn("--setting-sources", cmd)
        self.assertIn("--output-format stream-json", cmd)
        self.assertIn("--max-budget-usd 1.25", cmd)
        self.assertIn("--model anthropic/claude-opus-4-6", cmd)
        self.assertIn("--effort high", cmd)
        self.assertIn("--allowedTools", cmd)
        self.assertIn("-p 'Return 42.'", cmd)

    def test_cc_missing_cli_error_detects_marker(self) -> None:
        self.assertTrue(
            cc_missing_cli_error(
                ExecOutput(exit_code=1, stdout="", stderr=f"oops\n{CLAUDE_CLI_MISSING_MARKER}\n")
            )
        )
        self.assertFalse(cc_missing_cli_error(ExecOutput(exit_code=1, stdout="", stderr="oops")))

    def test_install_agent_runtime_uses_same_env_for_claude_code_node_and_install(self) -> None:
        container = object()
        expected_env = cc_env_vars(model="claude-opus-4-7")

        with (
            patch("src.docker_agents.install_nvm_node", return_value=True) as install_node,
            patch("src.docker_agents.run_install_script", return_value=True) as install_script,
        ):
            ok = _install_agent_runtime(
                container,
                condition="claude_code",
                version="latest",
                model_spec="claude-opus-4-7",
            )

        self.assertTrue(ok)
        install_node.assert_called_once_with(container, env=expected_env)
        install_script.assert_called_once()
        install_container, script = install_script.call_args.args
        self.assertIs(install_container, container)
        self.assertIn("npm install -g @anthropic-ai/claude-code@latest", script)
        self.assertEqual(install_script.call_args.kwargs, {"env": expected_env})

    def test_ensure_uv_exec_uses_shared_uv_bootstrap(self) -> None:
        calls: list[tuple[str, dict[str, str] | None, str | None]] = []

        def fake_exec(cmd: str, **kwargs) -> ExecOutput:
            calls.append((cmd, kwargs.get("env"), kwargs.get("workdir")))
            return ExecOutput(exit_code=0, stdout="", stderr="")

        self.assertTrue(ensure_uv_exec(fake_exec, env={"A": "B"}, workdir="/workspace"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], UV_BOOTSTRAP_COMMAND)
        self.assertEqual(calls[0][1], {"A": "B"})
        self.assertEqual(calls[0][2], "/workspace")

    def test_uv_bootstrap_checks_for_pip_before_using_python(self) -> None:
        self.assertIn("python -m pip --version >/dev/null 2>&1", UV_BOOTSTRAP_COMMAND)
        self.assertIn("python3 -m pip --version >/dev/null 2>&1", UV_BOOTSTRAP_COMMAND)
        self.assertIn("curl -LsSf https://astral.sh/uv/install.sh", UV_BOOTSTRAP_COMMAND)
        self.assertIn("wget -qO- https://astral.sh/uv/install.sh", UV_BOOTSTRAP_COMMAND)
        self.assertIn("Unable to install uv: neither curl, wget, nor pip is available", UV_BOOTSTRAP_COMMAND)

    def test_uv_bootstrap_uses_standalone_installer_when_python3_has_no_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bin_dir = tmp / "bin"
            home_dir = tmp / "home"
            marker_path = tmp / "curl-ran"
            bin_dir.mkdir()
            home_dir.mkdir()
            _write_executable(
                bin_dir / "python3",
                """#!/bin/sh
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
  echo "/usr/bin/python3: No module named pip" >&2
  exit 1
fi
exit 0
""",
            )
            _write_executable(
                bin_dir / "curl",
                """#!/bin/sh
: > "$MARKER_PATH"
cat <<'EOF'
mkdir -p "$UV_UNMANAGED_INSTALL"
cat > "$UV_UNMANAGED_INSTALL/uv" <<'INNER'
#!/bin/sh
exit 0
INNER
chmod +x "$UV_UNMANAGED_INSTALL/uv"
EOF
""",
            )

            result = subprocess.run(
                ["bash", "-c", UV_BOOTSTRAP_COMMAND],
                capture_output=True,
                cwd=tmp,
                env={
                    "HOME": str(home_dir),
                    "MARKER_PATH": str(marker_path),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                },
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertTrue(marker_path.exists(), msg=result.stderr or result.stdout)
            self.assertTrue((home_dir / ".local/bin/uv").exists())
            self.assertTrue((home_dir / ".local/bin/env").exists())

    def test_ensure_uv_exec_retries_once_on_failure(self) -> None:
        attempts = 0

        def fake_exec(cmd: str, **kwargs) -> ExecOutput:
            del cmd, kwargs
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return ExecOutput(exit_code=1, stdout="", stderr="temporary failure")
            return ExecOutput(exit_code=0, stdout="", stderr="")

        self.assertTrue(ensure_uv_exec(fake_exec))
        self.assertEqual(attempts, 2)

    def test_read_codex_rollout_prefers_tmp_runtime_sessions(self) -> None:
        calls: list[str] = []

        def fake_exec(cmd: str, **kwargs) -> ExecOutput:
            del kwargs
            calls.append(cmd)
            if "find \"$base\"" in cmd:
                return ExecOutput(
                    exit_code=0,
                    stdout=(
                        f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/"
                        "rollout-2026-04-06T10-08-30-thread-1.jsonl\n"
                    ),
                    stderr="",
                )
            if cmd.startswith("cat "):
                return ExecOutput(exit_code=0, stdout='{"type":"session_meta"}\n', stderr="")
            raise AssertionError(f"Unexpected command: {cmd}")

        rollout_path, rollout_text = _read_codex_rollout_from_exec(fake_exec, "thread-1")

        self.assertEqual(
            rollout_path,
            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/rollout-2026-04-06T10-08-30-thread-1.jsonl",
        )
        self.assertEqual(rollout_text, '{"type":"session_meta"}\n')
        self.assertIn(f"{CODEX_RUNTIME_CODEX_HOME}/sessions", calls[0])

    def test_nvm_install_exports_profile_for_installer_shell(self) -> None:
        script = NVM_NODE_INSTALL

        self.assertIn("export PROFILE=/dev/null", script)
        self.assertIn("NPM_CONFIG_PREFIX_SAVED", script)
        self.assertIn('unset NPM_CONFIG_PREFIX', script)
        self.assertIn('export NPM_CONFIG_PREFIX="$NPM_CONFIG_PREFIX_SAVED"', script)
        self.assertNotIn("PROFILE=/dev/null curl", script)
        self.assertIn("command -v apk", script)
        self.assertIn("command -v dnf", script)
        self.assertIn("command -v yum", script)

    def test_install_nodejs_shell_supports_multiple_package_managers(self) -> None:
        script = _install_nodejs_shell(log_version=True)

        self.assertIn("command -v apt-get", script)
        self.assertIn("command -v apk", script)
        self.assertIn("command -v dnf", script)
        self.assertIn("command -v yum", script)
        self.assertIn("node installed via package manager", script)
        self.assertNotIn(
            "apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq nodejs npm >/dev/null 2>&1",
            script,
        )
        self.assertIn(
            "apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq nodejs npm >/dev/null",
            script,
        )

    def test_verify_nodejs_shell_uses_claude_runtime_and_requires_npm(self) -> None:
        script = _verify_nodejs_shell()

        self.assertIn(f'export CLAUDE_RUNTIME_ROOT="${{CLAUDE_RUNTIME_ROOT:-{CLAUDE_RUNTIME_ROOT}}}"', script)
        self.assertIn(f'export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CLAUDE_RUNTIME_NPM_PREFIX}}}"', script)
        self.assertIn("command -v node", script)
        self.assertIn("command -v npm", script)
        self.assertIn("npm not available after installation", script)
        self.assertIn(f"need >={MIN_NODE_MAJOR_VERSION}", script)

    def test_verify_nodejs_shell_rejects_node_12(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            prefix = tmp / "npm-global"
            bin_dir = prefix / "bin"
            bin_dir.mkdir(parents=True)
            _write_executable(bin_dir / "node", "#!/bin/sh\necho v12.22.9\n")
            _write_executable(bin_dir / "npm", "#!/bin/sh\necho 8.5.1\n")

            result = subprocess.run(
                ["bash", "-c", _verify_nodejs_shell()],
                capture_output=True,
                env={
                    "PATH": "/bin:/usr/bin",
                    "CLAUDE_RUNTIME_ROOT": str(tmp / "runtime"),
                    "NVM_DIR": str(tmp / "nvm"),
                    "NPM_CONFIG_PREFIX": str(prefix),
                    "NPM_CONFIG_CACHE": str(tmp / "npm-cache"),
                },
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("node too old after installation: v12.22.9", result.stderr)

    def test_install_nodejs_binary_shell_uses_official_tarball_and_runtime_prefix(self) -> None:
        script = _install_nodejs_binary_shell()

        self.assertIn(NODE_BINARY_VERSION, script)
        self.assertIn("https://nodejs.org/dist/${node_version}/${tarball}", script)
        self.assertIn('tar -xzf "$tmpdir/$tarball" -C "$NPM_CONFIG_PREFIX" --strip-components=1', script)
        self.assertIn('"$NPM_CONFIG_PREFIX/bin/node" --version >/dev/null', script)
        self.assertIn('"$NPM_CONFIG_PREFIX/bin/npm" --version >/dev/null', script)
        self.assertIn("node installed via official binary", script)

    def test_cleanup_nodejs_binary_shell_clears_runtime_prefix_contents(self) -> None:
        script = _cleanup_nodejs_binary_shell()

        self.assertIn(f'export NPM_CONFIG_PREFIX="${{NPM_CONFIG_PREFIX:-{CLAUDE_RUNTIME_NPM_PREFIX}}}"', script)
        self.assertIn('find "$NPM_CONFIG_PREFIX" -mindepth 1 -maxdepth 1 -exec rm -rf {} +', script)

    def test_retryable_apt_fetch_failure_matches_transient_mirror_errors(self) -> None:
        self.assertTrue(
            _is_retryable_apt_fetch_failure(
                "",
                "Failed to fetch http://archive.ubuntu.com/ubuntu/pool/main/c/curl.deb: Connection timed out",
            )
        )

    def test_retryable_apt_fetch_failure_matches_hash_mismatch_messages(self) -> None:
        self.assertTrue(
            _is_retryable_apt_fetch_failure(
                "",
                "Err:1 http://archive.ubuntu.com/ubuntu focal/main amd64 curl amd64 7.68.0-1ubuntu2.18\n"
                "  File has unexpected size (1234 != 5678). Mirror sync in progress?",
            )
        )
        self.assertTrue(
            _is_retryable_apt_fetch_failure(
                "",
                "Failed to fetch http://archive.ubuntu.com/ubuntu/dists/focal/InRelease  Hash Sum mismatch",
            )
        )

    def test_retryable_apt_fetch_failure_requires_fetch_context_for_connection_markers(self) -> None:
        self.assertFalse(
            _is_retryable_apt_fetch_failure(
                "",
                "curl: (7) Could not connect to proxy host",
            )
        )

    def test_retryable_apt_fetch_failure_rejects_missing_apt(self) -> None:
        self.assertFalse(_is_retryable_apt_fetch_failure("", "apt-get: command not found"))

    def test_retryable_apt_fetch_failure_rejects_generic_install_error(self) -> None:
        self.assertFalse(
            _is_retryable_apt_fetch_failure(
                "",
                "Unable to install Java runtime: unsupported package manager",
            )
        )

    def test_retryable_pip_hash_mismatch_matches_hash_error_text(self) -> None:
        self.assertTrue(
            _is_retryable_pip_hash_mismatch(
                "",
                "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES from the requirements file.",
            )
        )
        self.assertFalse(_is_retryable_pip_hash_mismatch("", "pip install failed for another reason"))

    def test_install_timeout_is_retryable(self) -> None:
        self.assertTrue(_is_retryable_install_timeout(ExecOutput(exit_code=124, stdout="", stderr="", timed_out=True)))
        self.assertTrue(_is_retryable_install_timeout(ExecOutput(exit_code=124, stdout="", stderr="", timed_out=False)))
        self.assertFalse(_is_retryable_install_timeout(ExecOutput(exit_code=1, stdout="", stderr="", timed_out=False)))

    def test_run_install_script_retries_retryable_apt_failures(self) -> None:
        env = {"TOKEN": "abc"}
        container = FakeInstallContainer(
            [
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch pkg: Connection timed out"),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch pkg: Temporary failure resolving"),
                ExecOutput(exit_code=0, stdout="ready", stderr=""),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = run_install_script(container, "echo install", env=env, timeout=123, workdir="/workspace")

        self.assertTrue(ok)
        self.assertEqual(len(container.calls), 3)
        self.assertEqual([cmd for cmd, _kwargs in container.calls], ["echo install"] * 3)
        for _cmd, kwargs in container.calls:
            self.assertEqual(kwargs, {"timeout": 123, "env": env, "workdir": "/workspace"})
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(5)

    def test_run_install_script_retries_whole_script_on_timeout(self) -> None:
        container = FakeInstallContainer(
            [
                ExecOutput(exit_code=124, stdout="", stderr="Timeout after 300s", timed_out=True),
                ExecOutput(exit_code=0, stdout="", stderr=""),
                ExecOutput(exit_code=0, stdout="ready", stderr=""),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = run_install_script(container, "echo install", timeout=300)

        self.assertTrue(ok)
        self.assertEqual(len(container.calls), 3)
        self.assertEqual(container.calls[0], ("echo install", {"timeout": 300, "env": None, "workdir": None}))
        self.assertEqual(
            container.calls[1],
            (
                INSTALL_TIMEOUT_RETRY_CLEANUP_SCRIPT,
                {"timeout": 30, "env": None, "workdir": "/"},
            ),
        )
        self.assertEqual(container.calls[2], ("echo install", {"timeout": 300, "env": None, "workdir": None}))
        self.assertIn("dpkg --configure -a", container.calls[1][0])
        self.assertEqual(container.calls[0][1], {"timeout": 300, "env": None, "workdir": None})
        self.assertEqual(sleep.call_count, 1)

    def test_run_install_script_retries_hash_mismatch_with_no_cache_env(self) -> None:
        env = {"TOKEN": "abc"}
        container = FakeInstallContainer(
            [
                ExecOutput(
                    exit_code=1,
                    stdout="",
                    stderr="ERROR: THESE PACKAGES DO NOT MATCH THE HASHES from the requirements file.",
                ),
                ExecOutput(exit_code=0, stdout="ready", stderr=""),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = run_install_script(container, "echo install", env=env, timeout=300)

        self.assertTrue(ok)
        self.assertEqual(len(container.calls), 2)
        self.assertEqual(container.calls[0][1], {"timeout": 300, "env": env, "workdir": None})
        self.assertEqual(
            container.calls[1][1],
            {
                "timeout": 300,
                "env": {"TOKEN": "abc", "PIP_NO_CACHE_DIR": "1", "UV_NO_CACHE": "1"},
                "workdir": None,
            },
        )
        self.assertEqual(sleep.call_count, 1)

    def test_run_install_script_fails_fast_on_non_retryable_error(self) -> None:
        container = FakeInstallContainer(
            [ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found")]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = run_install_script(container, "echo install", timeout=321)

        self.assertFalse(ok)
        self.assertEqual(len(container.calls), 1)
        self.assertEqual(container.calls[0][1], {"timeout": 321, "env": None, "workdir": None})
        sleep.assert_not_called()

    def test_run_install_script_stops_after_three_retryable_failures(self) -> None:
        container = FakeInstallContainer(
            [
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch pkg: Connection timed out"),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch pkg: Could not connect"),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch pkg: Network is unreachable"),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = run_install_script(container, "echo install")

        self.assertFalse(ok)
        self.assertEqual(len(container.calls), 3)
        self.assertEqual(sleep.call_count, 2)

    def test_ensure_python_packages_retries_on_hash_mismatch_without_cache(self) -> None:
        container = FakeInstallContainer(
            [
                ExecOutput(exit_code=1, stdout="", stderr="ERROR: THESE PACKAGES DO NOT MATCH THE HASHES from the requirements file."),
                ExecOutput(exit_code=0, stdout="installed", stderr=""),
            ]
        )

        ok = ensure_python_packages(container, ["numpy==1.16.0", "pandas"])

        self.assertTrue(ok)
        self.assertEqual(len(container.calls), 2)
        self.assertEqual(
            container.calls[0][0],
            "python -m pip install -q --disable-pip-version-check numpy==1.16.0 pandas",
        )
        self.assertEqual(
            container.calls[1][0],
            "python -m pip install -q --disable-pip-version-check --no-cache-dir numpy==1.16.0 pandas",
        )
        self.assertEqual(container.calls[0][1], {"timeout": 300})
        self.assertEqual(container.calls[1][1], {"timeout": 300})

    def test_ensure_python_packages_does_not_retry_non_hash_error(self) -> None:
        container = FakeInstallContainer(
            [ExecOutput(exit_code=1, stdout="", stderr="python: command not found")]
        )

        ok = ensure_python_packages(container, ["numpy"])

        self.assertFalse(ok)
        self.assertEqual(len(container.calls), 1)
        self.assertEqual(
            container.calls[0][0],
            "python -m pip install -q --disable-pip-version-check numpy",
        )

    def test_swe_agent_install_script_uses_pip_when_cache_disabled(self) -> None:
        script = swe_agent_install_script()

        self.assertIn('if [ "${PIP_NO_CACHE_DIR:-}" = "1" ]; then', script)
        self.assertIn('python3 -m pip install -q --disable-pip-version-check $pip_cache_flag mini-swe-agent', script)
        self.assertIn('command -v uv >/dev/null 2>&1 && [ -z "$pip_cache_flag" ]', script)

    def test_install_nvm_node_exec_retries_nvm_script_before_succeeding(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch curl: Connection timed out"),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch curl: Temporary failure resolving"),
                ExecOutput(exit_code=0, stdout="node installed: v22.0.0", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual([cmd for cmd, _kwargs in exec_fn.calls], [NVM_NODE_INSTALL] * 3 + [verify_script])
        for _cmd, kwargs in exec_fn.calls[:3]:
            self.assertEqual(kwargs, {"timeout": 300, "env": env, "workdir": None})
        self.assertEqual(exec_fn.calls[3][1], {"timeout": 60, "env": env, "workdir": None})
        self.assertEqual(sleep.call_count, 2)

    def test_nvm_install_preserves_apt_stderr_for_retry_detection(self) -> None:
        self.assertNotIn(
            "apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq curl >/dev/null 2>&1",
            NVM_NODE_INSTALL,
        )
        self.assertIn(
            "apt-get -o Acquire::AllowInsecureRepositories=true --allow-unauthenticated install -y -qq curl >/dev/null",
            NVM_NODE_INSTALL,
        )

    def test_install_nvm_node_exec_falls_back_after_non_retryable_failure(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        binary_script = _install_nodejs_binary_shell()
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found"),
                ExecOutput(exit_code=0, stdout="node installed via official binary: v22.22.2", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual(exec_fn.calls[0], (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}))
        self.assertEqual(exec_fn.calls[1], (binary_script, {"timeout": 300, "env": env, "workdir": "/"}))
        self.assertEqual(exec_fn.calls[2], (verify_script, {"timeout": 60, "env": env, "workdir": None}))

    def test_install_nvm_node_exec_falls_back_when_nvm_success_lacks_npm(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        binary_script = _install_nodejs_binary_shell()
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=0, stdout="node installed: v22.0.0", stderr=""),
                ExecOutput(exit_code=1, stdout="", stderr="npm not available after installation"),
                ExecOutput(exit_code=0, stdout="node installed via official binary: v22.22.2", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual(
            exec_fn.calls,
            [
                (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
                (binary_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
            ],
        )

    def test_install_nvm_node_exec_tries_binary_before_package_manager(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        binary_script = _install_nodejs_binary_shell()
        cleanup_script = _cleanup_nodejs_binary_shell()
        fallback_script = _install_nodejs_shell(log_version=True)
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found"),
                ExecOutput(exit_code=1, stdout="", stderr="curl: (22) The requested URL returned error: 404"),
                ExecOutput(exit_code=0, stdout="", stderr=""),
                ExecOutput(exit_code=0, stdout="node installed via package manager: v22.0.0", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual(
            exec_fn.calls,
            [
                (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}),
                (binary_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (cleanup_script, {"timeout": 60, "env": env, "workdir": None}),
                (fallback_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
            ],
        )

    def test_install_nvm_node_exec_cleans_binary_when_binary_verification_fails_before_package_manager(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        binary_script = _install_nodejs_binary_shell()
        cleanup_script = _cleanup_nodejs_binary_shell()
        fallback_script = _install_nodejs_shell(log_version=True)
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found"),
                ExecOutput(exit_code=0, stdout="node installed via official binary: v22.22.2", stderr=""),
                ExecOutput(exit_code=1, stdout="", stderr="node: not found"),
                ExecOutput(exit_code=0, stdout="", stderr=""),
                ExecOutput(exit_code=0, stdout="node installed via package manager: v22.0.0", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual(
            exec_fn.calls,
            [
                (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}),
                (binary_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
                (cleanup_script, {"timeout": 60, "env": env, "workdir": None}),
                (fallback_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
            ],
        )

    def test_install_nvm_node_exec_fails_when_package_manager_installs_node_12(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        binary_script = _install_nodejs_binary_shell()
        cleanup_script = _cleanup_nodejs_binary_shell()
        fallback_script = _install_nodejs_shell(log_version=True)
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found"),
                ExecOutput(exit_code=1, stdout="", stderr="curl: (22) The requested URL returned error: 404"),
                ExecOutput(exit_code=0, stdout="", stderr=""),
                ExecOutput(exit_code=0, stdout="node installed via package manager: v12.22.9", stderr=""),
                ExecOutput(exit_code=1, stdout="", stderr="node too old after installation: v12.22.9 (need >=18)"),
            ]
        )

        ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertFalse(ok)
        self.assertEqual(
            exec_fn.calls,
            [
                (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}),
                (binary_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (cleanup_script, {"timeout": 60, "env": env, "workdir": None}),
                (fallback_script, {"timeout": 300, "env": env, "workdir": "/"}),
                (verify_script, {"timeout": 60, "env": env, "workdir": None}),
            ],
        )

    def test_install_nvm_node_exec_retries_fallback_on_retryable_apt_failures(self) -> None:
        env = {"NVM_DIR": "/tmp/nvm"}
        fallback_script = _install_nodejs_shell(log_version=True)
        cleanup_script = _cleanup_nodejs_binary_shell()
        verify_script = _verify_nodejs_shell()
        exec_fn = SequentialExec(
            [
                ExecOutput(exit_code=1, stdout="", stderr="apt-get: command not found"),
                ExecOutput(exit_code=1, stdout="", stderr="curl: (22) The requested URL returned error: 404"),
                ExecOutput(exit_code=0, stdout="", stderr=""),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch nodejs: Connection timed out"),
                ExecOutput(exit_code=1, stdout="", stderr="Failed to fetch npm: Could not connect"),
                ExecOutput(exit_code=0, stdout="node installed via package manager: v22.0.0", stderr=""),
                ExecOutput(exit_code=0, stdout="node verified: v22.0.0\nnpm verified: 10.0.0", stderr=""),
            ]
        )

        with patch("src.docker_agents.time.sleep") as sleep:
            ok = install_nvm_node_exec(exec_fn, env=env)

        self.assertTrue(ok)
        self.assertEqual(exec_fn.calls[0], (NVM_NODE_INSTALL, {"timeout": 300, "env": env, "workdir": None}))
        self.assertEqual(exec_fn.calls[1], (_install_nodejs_binary_shell(), {"timeout": 300, "env": env, "workdir": "/"}))
        self.assertEqual(exec_fn.calls[2], (cleanup_script, {"timeout": 60, "env": env, "workdir": None}))
        self.assertEqual([cmd for cmd, _kwargs in exec_fn.calls[3:6]], [fallback_script] * 3)
        for _cmd, kwargs in exec_fn.calls[3:6]:
            self.assertEqual(kwargs, {"timeout": 300, "env": env, "workdir": "/"})
        self.assertEqual(exec_fn.calls[6], (verify_script, {"timeout": 60, "env": env, "workdir": None}))
        self.assertEqual(sleep.call_count, 2)

    def test_spell_env_vars_forwards_fireworks_api_key(self) -> None:
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw-test"}, clear=False):
            env = spell_env_vars("fireworks:glm-5")
        self.assertEqual(env["FIREWORKS_API_KEY"], "fw-test")

    def test_load_swe_agent_trajectory_prefers_newest_parseable_file(self) -> None:
        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del cmd, kwargs
                return ExecOutput(
                    exit_code=0,
                    stdout=(
                        "100.0\t/workspace/older.traj.json\n"
                        "200.0\t/workspace/newer.traj.json\n"
                    ),
                    stderr="",
                )

            def read_text(self, path: str) -> str:
                return {
                    "/workspace/older.traj.json": '{"answer":"old"}',
                    "/workspace/newer.traj.json": '{"answer":"new"}',
                }[path]

        path, traj = _load_swe_agent_trajectory(FakeContainer(), "")

        self.assertEqual(path, Path("/workspace/newer.traj.json"))
        self.assertEqual(traj, {"answer": "new"})

    @patch("src.docker_agents._install_agent_runtime", return_value=True)
    def test_codex_run_in_container_attaches_rollout_metadata(self, _install_runtime) -> None:
        rollout_text = "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "thread-1"}}),
                json.dumps(
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 1_000,
                                    "cached_input_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 350,
                                },
                                "last_token_usage": {
                                    "input_tokens": 1_000,
                                    "cached_input_tokens": 100,
                                    "output_tokens": 200,
                                    "reasoning_output_tokens": 350,
                                },
                                "model_context_window": 258_400,
                            },
                        },
                    }
                ),
            ]
        )

        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del kwargs
                if "codex " in cmd and " exec " in cmd:
                    return ExecOutput(
                        exit_code=0,
                        stdout="\n".join(
                            [
                                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                                json.dumps({"type": "turn.started"}),
                                json.dumps(
                                    {
                                        "type": "turn.completed",
                                        "usage": {
                                            "input_tokens": 1_000,
                                            "cached_input_tokens": 100,
                                            "output_tokens": 200,
                                        },
                                    }
                                ),
                            ]
                        ),
                        stderr="",
                    )
                if "find \"$base\"" in cmd:
                    return ExecOutput(
                        exit_code=0,
                        stdout=(
                            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/"
                            "rollout-2026-04-06T10-08-30-thread-1.jsonl\n"
                        ),
                        stderr="",
                    )
                if cmd.startswith("cat "):
                    return ExecOutput(exit_code=0, stdout=rollout_text, stderr="")
                raise AssertionError(f"Unexpected command: {cmd}")

        result = run_agent_in_container(
            FakeContainer(),
            condition="codex",
            prompt="Return 42.",
            timeout=30,
            model_spec="gpt-5.4",
        )

        self.assertIsInstance(result, AgentRunResult)
        self.assertIsNotNone(result.usage)
        assert result.usage is not None
        self.assertEqual(result.usage.reasoning_output_tokens, 350)
        self.assertEqual(result.metadata["thread_id"], "thread-1")
        self.assertEqual(result.metadata["rollout_status"], "ok")
        self.assertEqual(result.metadata["rollout_token_count"]["total_token_usage"]["reasoning_output_tokens"], 350)
        self.assertEqual(result.metadata["cost_source"], "estimated_with_reasoning")
        self.assertEqual(
            result.raw["rollout_path"],
            f"{CODEX_RUNTIME_CODEX_HOME}/sessions/2026/04/06/rollout-2026-04-06T10-08-30-thread-1.jsonl",
        )

    @patch("src.docker_agents._install_agent_runtime", return_value=True)
    def test_claude_run_in_container_surfaces_missing_cli_as_install_error(self, _install_runtime) -> None:
        class FakeContainer:
            container_workdir = "/workspace"

            def exec_run(self, cmd: str, **kwargs):
                del kwargs
                if "claude" in cmd and "command -v claude" in cmd:
                    return ExecOutput(
                        exit_code=1,
                        stdout="",
                        stderr=f"{CLAUDE_CLI_MISSING_MARKER}\n",
                    )
                raise AssertionError(f"Unexpected command: {cmd}")

        result = run_agent_in_container(
            FakeContainer(),
            condition="claude_code",
            prompt="Return 42.",
            timeout=30,
            model_spec="anthropic/claude-opus-4-6",
        )

        self.assertEqual(result.error_type, "install_error")
        self.assertIn(CLAUDE_CLI_MISSING_MARKER, result.error_message or "")


class SpellRunTimeoutClassificationTest(unittest.TestCase):
    @patch("src.docker_agents._install_agent_runtime", return_value=True)
    def test_spell_timeout_without_payload_is_labelled_subprocess_timeout(self, _install_runtime):
        class FakeContainer:
            def exec_run(self, cmd: str, **kwargs):
                del cmd, kwargs
                return ExecOutput(exit_code=124, stdout="", stderr="Timeout after 30s", timed_out=True)

        result = run_agent_in_container(
            FakeContainer(),
            condition="spell",
            prompt="Return 42.",
            timeout=30,
            model_spec="anthropic-tc:opus46",
        )

        self.assertTrue(result.output.timed_out)
        self.assertEqual(result.error_type, "subprocess_timeout")
        self.assertEqual(result.error_message, "Timeout after 30s")
        self.assertIsNone(result.raw)

    @patch("src.docker_agents._install_agent_runtime", return_value=True)
    def test_spell_timeout_with_partial_payload_preserves_killed_metadata(self, _install_runtime):
        partial_payload = {
            "ok": False,
            "mode": "spell",
            "error": "spell.benchmark-api was terminated before completing",
            "error_type": "killed",
            "patch_on_disk": True,
            "error_data": {"patch_on_disk": True, "partial_work": True},
        }

        class FakeContainer:
            def exec_run(self, cmd: str, **kwargs):
                del cmd, kwargs
                return ExecOutput(
                    exit_code=124,
                    stdout=json.dumps(partial_payload),
                    stderr="Timeout after 30s",
                    timed_out=True,
                )

        result = run_agent_in_container(
            FakeContainer(),
            condition="spell",
            prompt="Return 42.",
            timeout=30,
            model_spec="anthropic-tc:opus46",
        )

        self.assertTrue(result.output.timed_out)
        self.assertEqual(result.error_type, "killed")
        self.assertEqual(result.error_message, partial_payload["error"])
        self.assertEqual(result.raw, partial_payload)
        self.assertTrue(result.raw["patch_on_disk"])
        self.assertTrue(result.raw["error_data"]["partial_work"])


if __name__ == "__main__":
    unittest.main()
