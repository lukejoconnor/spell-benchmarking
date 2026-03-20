from __future__ import annotations

from src.datasets import load_swebench_lite, load_swebench_pro


def _fake_swe_row(**overrides):
    base = {
        "instance_id": "repo__repo-1",
        "repo": "org/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug.",
        "hints_text": "",
        "test_patch": "diff --git a/test.py b/test.py\n",
        "patch": "diff --git a/file.py b/file.py\n",
    }
    base.update(overrides)
    return base


def test_load_swebench_pro_maps_optional_fields(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    rows = [
        _fake_swe_row(
            requirements=["must pass tests", "no API changes"],
            interface={"entrypoint": "solve()"},
            difficulty="medium",
            created_at="2026-01-01",
            version="1.0",
            environment_setup_commit="def456",
        )
    ]

    def fake_load_dataset(name: str, split: str):
        calls.append((name, split))
        return rows

    monkeypatch.setattr("src.datasets.load_dataset", fake_load_dataset)

    items = load_swebench_pro()

    assert calls == [("ScaleAI/SWE-bench_Pro", "test")]
    assert len(items) == 1
    item = items[0]
    assert item.instance_id == "repo__repo-1"
    assert item.repo == "org/repo"
    assert item.raw_instance["instance_id"] == "repo__repo-1"
    assert item.metadata["requirements"] == ["must pass tests", "no API changes"]
    assert item.metadata["interface"] == {"entrypoint": "solve()"}
    assert item.metadata["difficulty"] == "medium"
    assert item.metadata["environment_setup_commit"] == "def456"


def test_load_swebench_lite_omits_empty_optional_metadata(monkeypatch) -> None:
    rows = [
        _fake_swe_row(
            created_at="",
            version="",
            requirements="",
            interface="",
            difficulty=None,
        )
    ]

    monkeypatch.setattr("src.datasets.load_dataset", lambda _name, split=None: rows)

    items = load_swebench_lite()

    assert len(items) == 1
    assert items[0].metadata == {}
    assert items[0].raw_instance["instance_id"] == "repo__repo-1"
