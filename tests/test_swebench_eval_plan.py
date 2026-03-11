from __future__ import annotations

import importlib
from pathlib import Path
import pytest
import sys
import types


def _import_bench_with_stubbed_client():
    existing = sys.modules.get("bench")
    if existing is not None:
        return existing

    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - only for import wiring
        pass

    stub.SpellBenchmarkClient = _DummyClient
    prior_client = sys.modules.get("spell_benchmark_client")
    sys.modules["spell_benchmark_client"] = stub
    try:
        return importlib.import_module("bench")
    finally:
        if prior_client is None:
            sys.modules.pop("spell_benchmark_client", None)
        else:
            sys.modules["spell_benchmark_client"] = prior_client


def teardown_module(_module) -> None:
    sys.modules.pop("bench", None)
    sys.modules.pop("src.unified_adapters", None)
    stubbed_client = sys.modules.get("spell_benchmark_client")
    if stubbed_client is not None and getattr(stubbed_client, "__file__", None) is None:
        sys.modules.pop("spell_benchmark_client", None)


def _write_pred(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"instance_id":"x","model_name_or_path":"m","model_patch":"diff --git a/a b/a\\n"}\n',
        encoding="utf-8",
    )


def test_build_swebench_eval_plan_routes_pro_to_scale_pipeline(tmp_path) -> None:
    bench = _import_bench_with_stubbed_client()
    run_id = "run123"
    pred_dir = tmp_path / "preds"
    _write_pred(pred_dir / f"spell_pro_{run_id}.jsonl")

    pro = bench.ProEvalConfig(
        eval_repo="data/swebench_pro_eval/SWE-bench_Pro-os",
        output_dir="results/swebench_pro_eval",
        dockerhub_username="jefzda",
    )
    dataset_name, conditions, plan = bench._build_swebench_eval_plan(
        dataset="pro",
        condition="spell",
        run_id=run_id,
        predictions_dir=str(pred_dir),
        dataset_name=None,
        split="test",
        max_workers=2,
        timeout=60,
        namespace="none",
        instance_image_tag="latest",
        env_image_tag="latest",
        final_cache_level="base",
        final_clean=True,
        verify_predictions=True,
        pro=pro,
    )

    assert dataset_name == "ScaleAI/SWE-bench_Pro"
    assert conditions == ["spell"]
    assert len(plan) == 1
    assert plan[0]["mode"] == "scaleai_pro"
    assert "swe_bench_pro_eval.py" in " ".join(plan[0]["command"])


def test_build_swebench_eval_plan_keeps_standard_pipeline_for_verified(tmp_path) -> None:
    bench = _import_bench_with_stubbed_client()
    run_id = "run123"
    pred_dir = tmp_path / "preds"
    _write_pred(pred_dir / f"spell_verified_{run_id}.jsonl")

    pro = bench.ProEvalConfig()
    dataset_name, conditions, plan = bench._build_swebench_eval_plan(
        dataset="verified",
        condition="spell",
        run_id=run_id,
        predictions_dir=str(pred_dir),
        dataset_name=None,
        split="test",
        max_workers=2,
        timeout=60,
        namespace="none",
        instance_image_tag="latest",
        env_image_tag="latest",
        final_cache_level="base",
        final_clean=True,
        verify_predictions=True,
        pro=pro,
    )

    assert dataset_name == "princeton-nlp/SWE-bench_Verified"
    assert conditions == ["spell"]
    assert len(plan) == 1
    assert plan[0]["mode"] == "standard"
    assert "src.swebench_eval_wrapper" in " ".join(plan[0]["command"])


def test_run_swebench_eval_plan_passes_at_threshold(monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    free_bytes = bench.SWEBENCH_MIN_FREE_DISK_GB * (1024**3)
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda _path: (free_bytes * 2, free_bytes, free_bytes))

    bench._run_swebench_eval_plan([])


def test_run_swebench_eval_plan_fails_below_threshold(monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    free_bytes = (bench.SWEBENCH_MIN_FREE_DISK_GB - 1) * (1024**3)
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda _path: (free_bytes * 2, free_bytes, free_bytes))

    with pytest.raises(bench.click.ClickException):
        bench._run_swebench_eval_plan([])


def test_run_swebench_eval_plan_low_disk_message_has_key_numbers(monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    free_bytes = 31 * (1024**3)
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda _path: (100 * (1024**3), 69 * (1024**3), free_bytes))

    with pytest.raises(bench.click.ClickException) as excinfo:
        bench._run_swebench_eval_plan([])

    message = str(excinfo.value)
    assert "Detected free space: 31.0 GB" in message
    assert f"Required minimum: {bench.SWEBENCH_MIN_FREE_DISK_GB} GB" in message
    assert "docker system df" in message
    assert "docker system prune" in message
    assert "Colima disk" in message


def test_relocate_swebench_eval_reports_moves_root_report(tmp_path, monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    monkeypatch.setattr(bench, "BENCHMARK_ROOT", tmp_path)

    src = tmp_path / "codex-tc:gpt-5.3-codex.run123.json"
    src.write_text("{}", encoding="utf-8")

    moved = bench._relocate_swebench_eval_reports("run123")

    assert moved == [tmp_path / "results" / "swebench_eval_reports" / src.name]
    assert moved[0].read_text(encoding="utf-8") == "{}"
    assert not src.exists()


def test_run_swebench_eval_plan_moves_reports_after_standard_eval(tmp_path, monkeypatch) -> None:
    bench = _import_bench_with_stubbed_client()
    free_bytes = bench.SWEBENCH_MIN_FREE_DISK_GB * (1024**3)
    monkeypatch.setattr(bench.shutil, "disk_usage", lambda _path: (free_bytes * 2, free_bytes, free_bytes))
    monkeypatch.setattr(bench, "BENCHMARK_ROOT", tmp_path)

    run_id = "run123"

    def fake_run(command, check, cwd, env):
        assert check is True
        assert cwd == tmp_path
        (tmp_path / f"anthropic-tc:claude-opus-4-6.{run_id}.json").write_text("{}", encoding="utf-8")

    seen = []
    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench.click, "echo", seen.append)

    bench._run_swebench_eval_plan(
        [
            {
                "mode": "standard",
                "condition": "spell",
                "cache_level": "env",
                "clean": False,
                "command": ["python", "-m", "src.swebench_eval_wrapper", "--run_id", run_id],
            }
        ]
    )

    moved = tmp_path / "results" / "swebench_eval_reports" / f"anthropic-tc:claude-opus-4-6.{run_id}.json"
    assert moved.exists()
    assert any("Moved SWE-bench eval report" in line for line in seen)
