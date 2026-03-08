from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from click.testing import CliRunner

from src.datasets import BenchmarkItem


def _import_bench_with_stubbed_client():
    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - import wiring only
        pass

    stub.SpellBenchmarkClient = _DummyClient
    sys.modules.setdefault("spell_benchmark_client", stub)

    import bench as bench_module  # local import after stubbing

    return bench_module


def test_general_dry_run_reports_web_search_flag(monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    runner = CliRunner()

    monkeypatch.setitem(
        bench.DATASETS,
        "gsm8k",
        lambda: [
            BenchmarkItem(
                id="gsm8k_1",
                dataset="gsm8k",
                context="",
                question="2+2?",
                answer="4",
            )
        ],
    )

    result = runner.invoke(
        bench.cli,
        [
            "general",
            "--dataset",
            "gsm8k",
            "--condition",
            "claude_code",
            "--claude-model",
            "sonnet",
            "--dry-run",
            "--web-search",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"web_search": true' in result.output


def test_swebench_dry_run_reports_web_search_flag(monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    runner = CliRunner()

    monkeypatch.setattr(
        bench,
        "load_swebench_verified_mini",
        lambda _limit: [SimpleNamespace(instance_id="django__django-1")],
    )

    result = runner.invoke(
        bench.cli,
        [
            "swebench",
            "--dataset",
            "mini",
            "--condition",
            "codex",
            "--dry-run",
            "--web-search",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"web_search": true' in result.output
