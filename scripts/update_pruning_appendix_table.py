#!/usr/bin/env python3
"""Render the Appendix B pruning-token summary table.

Input is `pruning_summary.tsv` produced by
`scripts/analyze_spell_pruning_tokens.py`. By default this writes a standalone
LaTeX table under `paper-repro/pruning-token-analysis/`. Pass `--paper-tex` to
replace or insert the generated table into the paper's Appendix B.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = BENCH_ROOT / "paper-repro" / "pruning-token-analysis"
BEGIN = "% BEGIN GENERATED PRUNING TOKEN TABLE"
END = "% END GENERATED PRUNING TOKEN TABLE"
EFFORT_LABELS = {"low": "low", "med": "medium", "medium": "medium", "high": "high"}
BENCHMARK_LABELS = {
    "swe-lite": "SWE-bench Lite",
    "terminal-bench": "Terminal-Bench 1.1",
}
ORDER = {
    ("swe-lite", "low"): 0,
    ("swe-lite", "med"): 1,
    ("swe-lite", "medium"): 1,
    ("swe-lite", "high"): 2,
    ("terminal-bench", "low"): 3,
    ("terminal-bench", "med"): 4,
    ("terminal-bench", "medium"): 4,
    ("terminal-bench", "high"): 5,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def token_count(value: str) -> str:
    return f"{float(value):,.0f}"


def render_table(rows: list[dict[str, str]]) -> str:
    ordered = sorted(rows, key=lambda row: ORDER[(row["benchmark"], row["effort"])])
    body = []
    for row in ordered:
        label = f"{BENCHMARK_LABELS[row['benchmark']]} {EFFORT_LABELS[row['effort']]}"
        body.append(
            " & ".join(
                [
                    label,
                    token_count(row["mean_tokens_pruned_per_turn"]),
                    token_count(row["mean_completion_tokens"]),
                    token_count(row["mean_cumulative_tokens_pruned"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            BEGIN,
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{\textbf{Context pruning token accounting.} Mean token counts per traced Spell turn for the GPT-5.4 paper runs. Tokens pruned per turn counts source forms removed by \lstinline[language=spell]|prune| or \lstinline[language=spell]|rethink|, including generated \lstinline[language=spell]|!peek| pruning. Cumulative tokens pruned is the mean number of previously pruned tokens absent from each turn's prompt.}",
            r"\label{tab:pruning-token-summary}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Condition & Mean pruned / turn & Mean completion & Mean cumulative pruned \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            END,
            "",
        ]
    )


def update_paper(path: Path, table: str) -> None:
    text = path.read_text()
    if BEGIN in text and END in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        path.write_text(before + table + after.lstrip("\n"))
        return

    anchor = r"\subsection{Prompt surfaces and prompt-intervention runs}"
    if anchor not in text:
        raise RuntimeError(f"Could not find Appendix B insertion anchor in {path}")
    before, after = text.split(anchor, 1)
    path.write_text(before.rstrip() + "\n\n" + table + "\n" + anchor + after)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_ANALYSIS_DIR / "pruning_summary.tsv")
    parser.add_argument("--out", type=Path, default=DEFAULT_ANALYSIS_DIR / "appendix_b_pruning_summary.tex")
    parser.add_argument("--paper-tex", type=Path, default=None)
    args = parser.parse_args()

    rows = read_rows(args.summary)
    table = render_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"Wrote {args.out}")
    if args.paper_tex is not None:
        update_paper(args.paper_tex, table)
        print(f"Updated {args.paper_tex}")


if __name__ == "__main__":
    main()
