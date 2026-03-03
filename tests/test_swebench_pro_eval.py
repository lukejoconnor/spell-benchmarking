from __future__ import annotations

import json

import subprocess
from unittest.mock import patch

import pytest

from src.swebench_pro_eval import (
    check_docker_manifest_access,
    convert_predictions_jsonl_to_scale_json,
    dockerhub_image_for_instance,
    load_raw_samples,
    strip_binary_hunks,
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


# --- strip_binary_hunks ---


def test_strip_binary_hunks_removes_binary_files_differ() -> None:
    patch = """diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-old
+new
diff --git a/image.png b/image.png
index 3333333..4444444 100644
Binary files a/image.png and b/image.png differ
diff --git a/README.md b/README.md
index 5555555..6666666 100644
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-old readme
+new readme
"""
    result = strip_binary_hunks(patch)
    assert "src/app.py" in result
    assert "README.md" in result
    assert "image.png" not in result


def test_strip_binary_hunks_removes_git_binary_patch() -> None:
    patch = """diff --git a/font.woff b/font.woff
index 0000000..1111111 100644
GIT binary patch
literal 1234
zcmV;@1030%O;6KS0000000000000
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1 +1 @@
-pass
+return 1
"""
    result = strip_binary_hunks(patch)
    assert "src/main.py" in result
    assert "font.woff" not in result
    assert "GIT binary patch" not in result


def test_strip_binary_hunks_empty_and_none() -> None:
    assert strip_binary_hunks("") == ""
    assert strip_binary_hunks(None) is None


def test_strip_binary_hunks_no_binary_sections() -> None:
    patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+y
"""
    assert strip_binary_hunks(patch) == patch


# --- dockerhub_image_for_instance ---


def test_dockerhub_image_for_instance_normal() -> None:
    samples = {
        "django__django-1234": {"instance_id": "django__django-1234", "dockerhub_tag": "django.1234"},
    }
    uri = dockerhub_image_for_instance(samples, "django__django-1234", dockerhub_username="testuser")
    assert uri == "testuser/sweap-images:django.1234"


def test_dockerhub_image_for_instance_missing_id() -> None:
    with pytest.raises(KeyError, match="not present in raw sample"):
        dockerhub_image_for_instance({}, "nonexistent", dockerhub_username="u")


def test_dockerhub_image_for_instance_missing_tag() -> None:
    samples = {"id1": {"instance_id": "id1", "dockerhub_tag": ""}}
    with pytest.raises(KeyError, match="missing dockerhub_tag"):
        dockerhub_image_for_instance(samples, "id1", dockerhub_username="u")


# --- check_docker_manifest_access ---


def test_check_docker_manifest_access_success() -> None:
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("src.swebench_pro_eval.subprocess.run", return_value=fake_result) as mock_run:
        ok, err = check_docker_manifest_access("user/repo:tag")
    assert ok is True
    assert err == ""
    mock_run.assert_called_once()


def test_check_docker_manifest_access_failure() -> None:
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="unauthorized: access denied")
    with patch("src.swebench_pro_eval.subprocess.run", return_value=fake_result):
        ok, err = check_docker_manifest_access("user/repo:tag")
    assert ok is False
    assert "unauthorized" in err


def test_check_docker_manifest_access_docker_not_found() -> None:
    with patch("src.swebench_pro_eval.subprocess.run", side_effect=FileNotFoundError):
        ok, err = check_docker_manifest_access("user/repo:tag")
    assert ok is False
    assert "docker command not found" in err
