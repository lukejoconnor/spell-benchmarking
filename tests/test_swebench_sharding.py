import importlib
from pathlib import Path
import sys
from click.testing import CliRunner
from unittest.mock import patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_ROOT))

bench = importlib.import_module("bench")
from src.datasets import SWEBenchItem


def item(index: int) -> SWEBenchItem:
    return SWEBenchItem(
        instance_id=f"repo__repo-{index}",
        repo="repo/repo",
        base_commit=f"{index:040d}",
        problem_statement=f"problem {index}",
        hints_text=None,
        test_patch="",
        patch="",
    )


def invoke_shard(num_shards: int, shard_index: int):
    with patch.object(bench, "load_swebench_lite", return_value=[item(i) for i in range(10)]):
        return CliRunner().invoke(
            bench.cli,
            [
                "swebench",
                "--dataset",
                "lite",
                "--condition",
                "swe_agent",
                "--swe-agent-model",
                "openai/gpt-5.4",
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
                "--dry-run",
            ],
        )


def test_swebench_contiguous_shards_cover_items_once():
    outputs = [invoke_shard(4, index) for index in range(4)]

    assert all(result.exit_code == 0 for result in outputs)
    assert "items 1-2 of 10 (2 items)" in outputs[0].output
    assert "items 3-5 of 10 (3 items)" in outputs[1].output
    assert "items 6-7 of 10 (2 items)" in outputs[2].output
    assert "items 8-10 of 10 (3 items)" in outputs[3].output


def test_swebench_rejects_out_of_range_shard():
    result = invoke_shard(4, 4)

    assert result.exit_code != 0
    assert "--shard-index must be between 0 and 3" in result.output
