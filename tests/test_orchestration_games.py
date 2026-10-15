from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_orchestration_games import ASYNC_RUNTIME_COMMIT, game_request, select_runtime
from src.provenance import benchmark_git_versions


def make_runtime(root: Path, profile: str) -> Path:
    for relative in ("src/spell/benchmark_api.clj", profile):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    return root


def test_async_never_falls_back_to_synchronous_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("SPELL_ROOT", str(tmp_path))
    monkeypatch.delenv("SPELL_ASYNC_ROOT", raising=False)
    with pytest.raises(ValueError, match="SPELL_ASYNC_ROOT"):
        select_runtime("async")


def test_async_checks_revision_and_uses_selected_root_for_provenance(monkeypatch, tmp_path):
    root = make_runtime(tmp_path, "config/agent-profiles/io-tc.agent.edn")
    monkeypatch.setenv("SPELL_ASYNC_ROOT", str(root))
    versions = {"spell": {"commit": ASYNC_RUNTIME_COMMIT, "dirty": False}}
    with patch("scripts.run_orchestration_games.benchmark_git_versions", return_value=versions) as capture:
        selected, recorded = select_runtime("async")
    assert selected == root
    assert recorded == versions
    capture.assert_called_once_with(root)
    with patch("scripts.run_orchestration_games.benchmark_git_versions", return_value={"spell": {"commit": "wrong"}}):
        with pytest.raises(ValueError, match=ASYNC_RUNTIME_COMMIT):
            select_runtime("async")


def test_sync_continues_to_use_spell_root(monkeypatch, tmp_path):
    root = make_runtime(tmp_path, "config/agents/io-tc.agent.edn")
    monkeypatch.setenv("SPELL_ROOT", str(root))
    with patch("scripts.run_orchestration_games.benchmark_git_versions", return_value={}):
        assert select_runtime("sync")[0] == root


def test_async_request_uses_july_profile_and_corrected_toolcall_framing(tmp_path):
    request = game_request("async", "auction", "openai-tc:gpt-5.6-sol", tmp_path, 0, 80)
    assert request["agent_profile"] == "config/agent-profiles/io-tc.agent.edn"
    assert request["prefill"] is False
    assert "agent" not in request
    assert request["reasoning_effort"] == "medium"
    assert request["trace"] is True


def test_explicit_runtime_provenance_is_separate_from_default(monkeypatch, tmp_path):
    primary, alternate = tmp_path / "primary", tmp_path / "alternate"
    monkeypatch.setenv("SPELL_ROOT", str(primary))
    benchmark_git_versions.cache_clear()
    try:
        with patch("src.provenance._git_version_for_repo", side_effect=lambda root: {"source": str(root)}):
            assert benchmark_git_versions()["spell"]["source"] == str(primary)
            assert benchmark_git_versions(alternate)["spell"]["source"] == str(alternate)
    finally:
        benchmark_git_versions.cache_clear()
