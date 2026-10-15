from pathlib import Path
from unittest.mock import patch
import subprocess

from click.testing import CliRunner

from src.docker_agents import cc_install_script, codex_install_script
from src.reasoning_effort import effective_reasoning_effort

ROOT = Path(__file__).resolve().parents[1]


def test_paper_baseline_install_versions():
    assert "@anthropic-ai/claude-code@2.1.107" in cc_install_script()
    assert "@openai/codex@0.120.0" in codex_install_script()
    assert effective_reasoning_effort("spell", "fireworks-tc:qwen3p7-plus", "high") == "high"


def test_multicondition_terminalbench_only_cleans_its_own_runs(tmp_path):
    import bench

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    with patch.object(bench.subprocess, "run", side_effect=run), \
         patch.object(bench, "terminalbench_records_from_run", return_value=[]), \
         patch.object(bench, "prepare_terminalbench_env", return_value={}), \
         patch.object(bench, "_print_summary"):
        result = CliRunner().invoke(bench.cli, [
            "terminalbench", "--condition", "codex", "--condition", "claude_code",
            "--run-id", "cleanup-regression", "--output", str(tmp_path / "result.jsonl"),
        ])
    assert result.exit_code == 0, result.output
    assert len(commands) == 2
    assert all(command[:2] == ["tb", "run"] for command in commands)
    assert all("--cleanup" in command for command in commands)
    assert any("version=0.120.0" in command for command in commands)
    assert any("version=2.1.107" in command for command in commands)


def test_frozen_figure2_task_sets_and_shared_exclusion():
    selected = (ROOT / "paper-repro/figure2/swebench-32.txt").read_text().splitlines()
    evaluated = (ROOT / "paper-repro/figure2/swebench-31-evaluated.txt").read_text().splitlines()
    tb = (ROOT / "paper-repro/figure2/terminalbench-32.txt").read_text().splitlines()
    assert len(set(selected)) == len(selected) == 32
    assert len(set(tb)) == len(tb) == 32
    assert len(evaluated) == 31
    assert set(selected) - set(evaluated) == {"matplotlib__matplotlib-23913"}
    shards = [(ROOT / f"paper-repro/sbl-shards/shard{i}.txt").read_text().splitlines() for i in (1, 2, 3)]
    assert len({item for shard in shards for item in shard}) == 300
    assert set(selected).issubset({item for shard in shards for item in shard})
