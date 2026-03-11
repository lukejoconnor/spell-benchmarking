"""Benchmark result plotting with consistent color scheme and grouped bar charts.

Provides a canonical color map for all benchmark methods (Spell, Claude Code,
Codex CLI, and one-shot baselines) and horizontal bar/grouped-bar plotting with
optional binomial confidence intervals and manual-vs-harness grading comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Method registry: canonical order, labels, and colors
# ---------------------------------------------------------------------------

METHOD_ORDER = [
    "spell_opus",
    "cc_opus",
    "oneshot_opus",
    "spell_codex",
    "codex_cli",
    "oneshot_codex",
]

DEFAULT_METHOD_LABELS = {
    "spell_opus": "Spell (Opus 4.6)",
    "cc_opus": "Claude Code (Opus 4.6)",
    "oneshot_opus": "One-shot (Opus 4.6)",
    "spell_codex": "Spell (GPT 5.3 Codex)",
    "codex_cli": "Codex CLI (GPT 5.3 Codex)",
    "oneshot_codex": "One-shot (GPT 5.3 Codex)",
}

# Colors from matplotlib's Set2 colormap — muted and visually distinct
_set2 = plt.cm.Set2.colors
METHOD_COLORS = {
    "spell_opus": _set2[0],
    "cc_opus": _set2[1],
    "oneshot_opus": _set2[5],
    "spell_codex": _set2[2],
    "codex_cli": _set2[3],
    "oneshot_codex": _set2[7],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lighter(color, factor: float = 0.55) -> tuple[float, float, float]:
    """Blend *color* toward white.  factor=0 -> white, factor=1 -> original.

    Accepts hex strings or RGB float tuples."""
    if isinstance(color, str):
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
    else:
        r, g, b = color[0], color[1], color[2]
    return (
        r * factor + 1.0 * (1 - factor),
        g * factor + 1.0 * (1 - factor),
        b * factor + 1.0 * (1 - factor),
    )


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score 95 % CI for a binomial proportion, in percentage points."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    lo = max(0.0, center - margin) * 100
    hi = min(1.0, center + margin) * 100
    return lo, hi


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    """One accuracy measurement for a single method."""

    method: str  # key from METHOD_ORDER
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_benchmark(
    results: Sequence[MethodResult],
    *,
    manual_results: Sequence[MethodResult] | None = None,
    title: str | None = None,
    error_bars: bool = False,
    labels: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot benchmark results as a horizontal bar chart.

    Parameters
    ----------
    results
        Harness-graded accuracy for each method.
    manual_results
        Optional manual-graded results (same method keys).  When provided,
        the chart becomes a grouped bar plot with a legend distinguishing
        harness-graded (solid) from manual-graded (lighter shade).
    title
        Optional chart title (displayed as-is).
    error_bars
        Show 95 % Wilson-score binomial confidence intervals.
    labels
        Override display names for methods.  Keys are method identifiers
        (e.g. ``"spell_opus"``), values are display strings.  Falls back
        to ``DEFAULT_METHOD_LABELS`` for any key not provided.
    save_path
        Save figure to this path (PNG, PDF, SVG, ...).
    figsize
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    resolved_labels = dict(DEFAULT_METHOD_LABELS)
    if labels:
        resolved_labels.update(labels)

    # Canonical ordering (reversed for horizontal: top bar = first in order)
    key_rank = {k: i for i, k in enumerate(METHOD_ORDER)}
    results = sorted(results, key=lambda r: key_rank.get(r.method, 99))
    methods = [r.method for r in results]

    grouped = manual_results is not None
    if grouped:
        manual_map = {r.method: r for r in manual_results}

    n = len(methods)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)

    def _xerr(items: Sequence[MethodResult]) -> list[list[float]] | None:
        if not error_bars:
            return None
        lo_err, hi_err = [], []
        for r in items:
            lo, hi = _wilson_ci(r.correct, r.total)
            lo_err.append(r.accuracy - lo)
            hi_err.append(hi - r.accuracy)
        return [lo_err, hi_err]

    bar_kw: dict = dict(capsize=3, error_kw=dict(linewidth=1.2, color="#444444"),
                        edgecolor="white", linewidth=0.5)

    if grouped:
        w = 0.35
        harness_colors = [METHOD_COLORS.get(m, "#888") for m in methods]
        ax.barh(
            y + w / 2,
            [r.accuracy for r in results],
            w,
            color=harness_colors,
            xerr=_xerr(results),
            label="Harness-graded",
            **bar_kw,
        )

        manual_items = [manual_map.get(m, MethodResult(m, 0, 0)) for m in methods]
        manual_colors = [_lighter(METHOD_COLORS.get(m, "#888")) for m in methods]
        ax.barh(
            y - w / 2,
            [r.accuracy for r in manual_items],
            w,
            color=manual_colors,
            xerr=_xerr(manual_items),
            label="Manual-graded",
            **bar_kw,
        )
    else:
        w = 0.6
        colors = [METHOD_COLORS.get(m, "#888") for m in methods]
        ax.barh(
            y,
            [r.accuracy for r in results],
            w,
            color=colors,
            xerr=_xerr(results),
            **bar_kw,
        )

    # Bar labels: "X / Y" next to each bar ---------------------------------
    def _annotate_bars(items: Sequence[MethodResult], y_positions):
        for r, yp in zip(items, y_positions):
            ax.text(
                r.accuracy + 1.2, yp,
                f"{r.correct}/{r.total}",
                va="center", ha="left", fontsize=15, color="#444444",
            )

    if grouped:
        _annotate_bars(results, y + w / 2)
        manual_items_for_label = [manual_map.get(m, MethodResult(m, 0, 0)) for m in methods]
        _annotate_bars(manual_items_for_label, y - w / 2)
    else:
        _annotate_bars(results, y)

    # Axes ----------------------------------------------------------------
    tick_labels = [resolved_labels.get(m, m) for m in methods]
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels, fontsize=17)
    ax.invert_yaxis()  # first method at top
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)", fontsize=20)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    if title:
        ax.set_title(title, fontsize=22, fontweight="bold", pad=14)

    if grouped:
        ax.legend(fontsize=15, loc="lower right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Quick demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    harness = [
        MethodResult("spell_opus", 28, 60),
        MethodResult("cc_opus", 24, 60),
        MethodResult("oneshot_opus", 22, 60),
        MethodResult("spell_codex", 34, 60),
        MethodResult("codex_cli", 30, 60),
        MethodResult("oneshot_codex", 32, 60),
    ]
    manual = [
        MethodResult("spell_opus", 32, 60),
        MethodResult("cc_opus", 26, 60),
        MethodResult("oneshot_opus", 24, 60),
        MethodResult("spell_codex", 38, 60),
        MethodResult("codex_cli", 33, 60),
        MethodResult("oneshot_codex", 34, 60),
    ]

    out = Path(__file__).resolve().parent.parent / "scratch"
    out.mkdir(exist_ok=True)

    # 1. Simple horizontal bar chart
    plot_benchmark(harness, title="Omni-MATH", save_path=out / "demo_simple.png")

    # 2. With error bars
    plot_benchmark(harness, title="Omni-MATH", error_bars=True, save_path=out / "demo_ci.png")

    # 3. Grouped: harness vs manual, with error bars
    plot_benchmark(
        harness,
        manual_results=manual,
        title="Omni-MATH",
        error_bars=True,
        save_path=out / "demo_grouped.png",
    )

    print(f"Saved demo plots to {out}")
