from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_reviewer3_mini_pilot.py"
SPEC = importlib.util.spec_from_file_location("analyze_reviewer3_mini_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_mcnemar_p() -> None:
    assert MODULE.exact_mcnemar_p(0, 0) == 1.0
    assert MODULE.exact_mcnemar_p(1, 0) == 1.0
    assert MODULE.exact_mcnemar_p(5, 0) == 0.0625


def test_summarize_builds_paired_cross_tab() -> None:
    mini = {
        "a": {"resolved": True, "error": None, "cost_usd": 0.1, "latency_ms": 1000, "source": "a"},
        "b": {"resolved": False, "error": None, "cost_usd": 0.2, "latency_ms": 3000, "source": "b"},
        "c": {"resolved": True, "error": None, "cost_usd": 0.3, "latency_ms": 2000, "source": "c"},
        "d": {"resolved": False, "error": None, "cost_usd": 0.4, "latency_ms": 4000, "source": "d"},
    }
    summary, paired = MODULE.summarize("test", mini, {"a": True, "b": True, "c": False, "d": False})

    assert len(paired) == 4
    assert summary["spell_resolved"] == 2
    assert summary["mini_resolved"] == 2
    assert (summary["both"], summary["spell_only"], summary["mini_only"], summary["neither"]) == (1, 1, 1, 1)
    assert summary["mini_cost_usd"] == 1.0
    assert summary["mini_median_latency_s"] == 2.5
