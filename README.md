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
├── src/
│   ├── datasets.py        # Dataset loading and formatting
│   ├── runners.py         # Baseline and Spell execution
│   ├── evaluation.py      # Scoring functions
│   └── utils.py           # Shared utilities
├── run_benchmark.py       # Main entry point
└── analyze_results.py     # Results aggregation
```

## Usage

```bash
# Activate venv
source .venv/bin/activate

# Run benchmark (small test)
python run_benchmark.py --dataset longbench_short --condition baseline --n 10

# Run full comparison
python run_benchmark.py --dataset all --condition all --model anthropic:claude-sonnet-4-20250514
```
