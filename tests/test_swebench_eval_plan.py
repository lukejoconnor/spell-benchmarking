from __future__ import annotations

from pathlib import Path
import sys
import types


def _import_bench_with_stubbed_client():
    stub = types.ModuleType("spell_benchmark_client")

    class _DummyClient:  # pragma: no cover - only for import wiring
        pass

    stub.SpellBenchmarkClient = _DummyClient
    sys.modules.setdefault("spell_benchmark_client", stub)

    import bench as bench_module  # local import after stubbing

    return bench_module


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
        pro_eval_repo="data/swebench_pro_eval/SWE-bench_Pro-os",
        pro_raw_sample_path=None,
        pro_output_dir="results/swebench_pro_eval",
        pro_dockerhub_username="jefzda",
        pro_strip_trace_artifacts=True,
        pro_use_local_docker=True,
        pro_block_network=False,
        pro_redo=False,
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
        pro_eval_repo="data/swebench_pro_eval/SWE-bench_Pro-os",
        pro_raw_sample_path=None,
        pro_output_dir="results/swebench_pro_eval",
        pro_dockerhub_username="jefzda",
        pro_strip_trace_artifacts=True,
        pro_use_local_docker=True,
        pro_block_network=False,
        pro_redo=False,
    )

    assert dataset_name == "princeton-nlp/SWE-bench_Verified"
    assert conditions == ["spell"]
    assert len(plan) == 1
    assert plan[0]["mode"] == "standard"
    assert "swebench.harness.run_evaluation" in " ".join(plan[0]["command"])
