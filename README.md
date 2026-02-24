# Spell Benchmarking

Benchmark infrastructure for evaluating Spell orchestration vs. baseline (single-turn) LLM performance.

## Research Question

Does LLM self-orchestration via Spell improve performance on reasoning tasks, compared to a single monolithic LLM call?

## Datasets

| Dataset | Subset | N | Format | Evaluation |
|---------|--------|---|--------|------------|
| LongBench v2 | Short (<100k chars) | 91 | MCQ (4 choices) | Exact match on A/B/C/D |
| BABILong | 32k context, qa1-qa10 | 1000 | Free-form QA | Exact string match |

## Conditions

1. **Baseline**: Single LLM call with full context + question
2. **Spell**: LLM writes a Spell program that can orchestrate (decompose, delegate, etc.)

## Infrastructure

```
benchmarking/
├── data/                  # Cached datasets (gitignored)
├── results/               # Run outputs (gitignored)
├── bench.py               # Unified Click CLI (general/swebench/featurebench)
├── src/
│   ├── datasets.py        # Dataset loading and formatting
│   ├── unified_adapters.py  # Spell/Baseline/Claude/Codex adapters
│   ├── unified_results.py   # Unified JSONL schema + summaries
│   ├── evaluation.py        # Scoring functions
│   └── featurebench_lib.py  # FeatureBench workspace helpers
├── run_benchmark.py       # Legacy shim -> bench.py general
├── run_swebench.py        # Legacy shim -> bench.py swebench
└── run_featurebench.py    # Legacy shim -> bench.py featurebench
```

## Usage

```bash
# General benchmark
uv run python benchmarking/bench.py general --dataset longbench_short --condition baseline --n 10

# Full comparison
uv run python benchmarking/bench.py general --dataset all --condition all --model anthropic:claude-sonnet-4-5-20250929

# SWE-bench
uv run python benchmarking/bench.py swebench --dataset mini --condition both --n 5

# FeatureBench
uv run python benchmarking/bench.py featurebench --condition spell --task-ids 22 23
```
