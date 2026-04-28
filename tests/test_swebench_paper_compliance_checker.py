from __future__ import annotations

import json
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from scripts.check_swebench_paper_compliance import check_artifacts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_checker_accepts_clean_metadata(tmp_path: Path) -> None:
    jsonl = tmp_path / "out.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "run_id": "run-1",
                "condition": "codex",
                "item_id": "django__django-1",
                "metadata": {
                    "paper_compliant": True,
                    "hints_included": False,
                    "web_search": False,
                    "paper_compliance_instruction": True,
                },
            }
        ],
    )

    summary = check_artifacts(jsonl, run_id="run-1", condition="codex")

    assert summary["ok"] is True
    assert summary["failures"] == []
    assert summary["warnings"] == []


def test_checker_fails_metadata_and_warns_on_solution_lookup_trace(tmp_path: Path) -> None:
    jsonl = tmp_path / "out.jsonl"
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / "rollout.txt").write_text(
        "codex --search exec\ncurl https://github.com/org/repo/pull/1.patch\n",
        encoding="utf-8",
    )
    _write_jsonl(
        jsonl,
        [
            {
                "run_id": "run-1",
                "condition": "codex",
                "item_id": "django__django-1",
                "metadata": {
                    "paper_compliant": False,
                    "hints_included": True,
                    "web_search": True,
                    "paper_compliance_instruction": False,
                },
            }
        ],
    )

    summary = check_artifacts(jsonl, trace_root=trace_root)

    assert summary["ok"] is False
    assert len(summary["failures"]) == 4
    warning_kinds = {warning["kind"] for warning in summary["warnings"]}
    assert "codex_search" in warning_kinds
    assert "github_url_fetch" in warning_kinds
    assert "patch_url" in warning_kinds


def test_checker_treats_trace_warnings_as_failures_by_default(tmp_path: Path) -> None:
    jsonl = tmp_path / "out.jsonl"
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / "rollout.txt").write_text(
        "curl https://github.com/org/repo/pull/1.patch\n",
        encoding="utf-8",
    )
    _write_jsonl(
        jsonl,
        [
            {
                "run_id": "run-1",
                "condition": "codex",
                "item_id": "django__django-1",
                "metadata": {
                    "paper_compliant": True,
                    "hints_included": False,
                    "web_search": False,
                    "paper_compliance_instruction": True,
                },
            }
        ],
    )

    summary = check_artifacts(jsonl, trace_root=trace_root)
    allowed_summary = check_artifacts(
        jsonl,
        trace_root=trace_root,
        warnings_are_failures=False,
    )

    assert summary["ok"] is False
    assert summary["failures"] == []
    assert summary["warnings"]
    assert allowed_summary["ok"] is True
