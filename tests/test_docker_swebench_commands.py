from __future__ import annotations

from pathlib import Path

from src.docker_swebench import claude_allowed_tools, run_claude_code, run_codex


class _FakeContainer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, str]]] = []

    def exec_run(self, cmd: str, timeout: int, env: dict[str, str]):
        self.calls.append((cmd, timeout, env))
        return None


def test_claude_allowed_tools_toggle_web_tools() -> None:
    disabled = claude_allowed_tools(web_search=False)
    enabled = claude_allowed_tools(web_search=True)

    assert "WebSearch" not in disabled
    assert "WebFetch" not in disabled
    assert "WebSearch" in enabled
    assert "WebFetch" in enabled


def test_run_claude_code_disables_web_tools_when_requested() -> None:
    container = _FakeContainer()

    run_claude_code(
        container,
        "Reply with OK only.",
        model="sonnet",
        budget=1.0,
        web_search=False,
        timeout=30,
        env={},
    )

    cmd, timeout, _env = container.calls[0]
    assert timeout == 30
    assert "--allowedTools" in cmd
    assert "WebSearch" not in cmd
    assert "WebFetch" not in cmd


def test_run_codex_disables_web_search_when_requested() -> None:
    container = _FakeContainer()

    run_codex(
        container,
        "Reply with OK only.",
        model="gpt-5-codex",
        web_search=False,
        timeout=30,
        env={},
    )

    cmd, timeout, _env = container.calls[0]
    assert timeout == 30
    assert 'web_search="disabled"' in cmd


def test_run_codex_leaves_web_search_enabled_when_requested() -> None:
    container = _FakeContainer()

    run_codex(
        container,
        "Reply with OK only.",
        model="gpt-5-codex",
        web_search=True,
        timeout=30,
        env={},
    )

    cmd, _timeout, _env = container.calls[0]
    assert 'web_search="disabled"' not in cmd
