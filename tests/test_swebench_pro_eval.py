from __future__ import annotations

import json

from src.swebench_pro_eval import (
    convert_predictions_jsonl_to_scale_json,
    load_raw_samples,
    strip_trace_artifact_sections,
    write_raw_sample_jsonl,
)


def test_strip_trace_artifact_sections_removes_trace_files() -> None:
    patch = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-print("old")
+print("new")
diff --git a/traces/2026-03-02T16-28-24/0000.spl b/traces/2026-03-02T16-28-24/0000.spl
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/traces/2026-03-02T16-28-24/0000.spl
@@ -0,0 +1 @@
+(think ...)
diff --git a/traces/2026-03-02T16-28-24/trace.edn b/traces/2026-03-02T16-28-24/trace.edn
new file mode 100644
index 0000000..4444444
--- /dev/null
+++ b/traces/2026-03-02T16-28-24/trace.edn
@@ -0,0 +1 @@
+{:trace true}
"""

    filtered, removed_paths = strip_trace_artifact_sections(patch)

    assert "src/app.py" in filtered
    assert "traces/2026-03-02T16-28-24/0000.spl" not in filtered
    assert "traces/2026-03-02T16-28-24/trace.edn" not in filtered
    assert removed_paths == [
        "traces/2026-03-02T16-28-24/0000.spl",
        "traces/2026-03-02T16-28-24/trace.edn",
    ]


def test_convert_predictions_jsonl_to_scale_json_strips_trace_artifacts(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "converted.json"

    patch = """diff --git a/src/main.go b/src/main.go
index 1111111..2222222 100644
--- a/src/main.go
+++ b/src/main.go
@@ -1 +1 @@
-old
+new
diff --git a/traces/2026-03-02T16-28-24/tree.txt b/traces/2026-03-02T16-28-24/tree.txt
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/traces/2026-03-02T16-28-24/tree.txt
@@ -0,0 +1 @@
+trace tree
"""
    row = {
        "instance_id": "instance_foo",
        "model_name_or_path": "spell",
        "model_patch": patch,
    }
    predictions_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    stats = convert_predictions_jsonl_to_scale_json(
        predictions_path,
        output_path,
        default_prefix="spell",
        strip_trace_artifacts=True,
    )

    converted = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(converted) == 1
    assert converted[0]["instance_id"] == "instance_foo"
    assert converted[0]["prefix"] == "spell"
    assert "src/main.go" in converted[0]["patch"]
    assert "tree.txt" not in converted[0]["patch"]
    assert stats["count"] == 1
    assert stats["removed_trace_sections"] == 1


def test_write_raw_sample_jsonl_and_load_raw_samples(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "instance_id": "instance_a",
            "dockerhub_tag": "repo.tag-a",
            "before_repo_set_cmd": "echo setup",
            "selected_test_files_to_run": "[]",
            "base_commit": "abc123",
            "fail_to_pass": "[]",
            "pass_to_pass": "[]",
        }
    ]

    def fake_load_dataset(name: str, split: str):
        assert name == "ScaleAI/SWE-bench_Pro"
        assert split == "test"
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    out = tmp_path / "raw.jsonl"
    write_raw_sample_jsonl(out, dataset_name="ScaleAI/SWE-bench_Pro", split="test")
    loaded = load_raw_samples(out)

    assert loaded["instance_a"]["dockerhub_tag"] == "repo.tag-a"
