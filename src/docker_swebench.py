"""Compatibility wrappers for the older SWE-bench Docker helpers.

The shared implementation now lives in `src.docker_agents`. This module keeps
lightweight shims for tests and any lingering imports while the benchmark CLI
migrates to the unified layer.
"""

from __future__ import annotations

from .docker_agents import (
    DockerWorkspaceContainer as DockerContainer,
    ExecOutput as ExecResult,
    ParsedUsage,
    cc_parse_usage as parse_claude_code_usage,
    cc_run_command,
    codex_parse_usage as parse_codex_usage,
    codex_run_command,
    install_nvm_node,
)


def run_claude_code(
    container,
    prompt: str,
    *,
    model: str,
    budget: float,
    timeout: int,
    env: dict[str, str],
) -> ExecResult:
    return container.exec_run(
        cc_run_command(prompt, model=model, budget=budget),
        timeout=timeout,
        env=env,
    )


def run_codex(
    container,
    prompt: str,
    *,
    model: str,
    timeout: int,
    env: dict[str, str],
    web_search: bool = False,
) -> ExecResult:
    return container.exec_run(
        codex_run_command(prompt, model=model, web_search=web_search),
        timeout=timeout,
        env=env,
    )


def install_claude_code(container, env: dict[str, str]) -> bool:
    from .docker_agents import cc_install_script, run_install_script

    return run_install_script(container, cc_install_script(), env=env)


def install_codex(container, env: dict[str, str]) -> bool:
    from .docker_agents import codex_install_script, run_install_script

    return run_install_script(container, codex_install_script(), env=env)


def run_spell_docker(*_args, **_kwargs):
    raise RuntimeError("run_spell_docker has been replaced by execute_agent_in_container")
