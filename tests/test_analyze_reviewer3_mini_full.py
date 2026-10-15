from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_reviewer3_mini_full import exact_mcnemar_p, summarize


def rows(values):
    return {
        item: {
            "resolved": resolved,
            "error": None,
            "cost_usd": cost,
            "latency_ms": latency,
            "source": item,
        }
        for item, resolved, cost, latency in values
    }


def test_summarize_full_exact_item_costs():
    spell = rows([("a", True, 1.0, 1000), ("b", True, 2.0, 2000), ("c", False, 3.0, 3000)])
    mini = rows([("a", True, 0.5, 2000), ("b", False, 1.0, 4000), ("c", True, 1.5, 6000)])

    summary, paired = summarize("test", mini, spell, 3)

    assert len(paired) == 3
    assert summary["spell_resolved"] == 2
    assert summary["mini_resolved"] == 2
    assert summary["both"] == 1
    assert summary["spell_only"] == 1
    assert summary["mini_only"] == 1
    assert summary["neither"] == 0
    assert summary["spell_cost_usd"] == 6.0
    assert summary["spell_cost_n"] == 3
    assert summary["mini_cost_usd"] == 3.0
    assert summary["mini_cost_n"] == 3
    assert summary["mini_cost_ratio_vs_spell"] == 0.5
    assert summary["mini_median_latency_s"] == 4.0


def test_exact_mcnemar_p():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(1, 0) == 1.0
    assert exact_mcnemar_p(5, 0) == 0.0625
