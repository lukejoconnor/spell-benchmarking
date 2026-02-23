# Benchmarking

## Workflow

Follow this sequence when running benchmarks:

1. **Pilot a small subset** (3-5 items) to verify the harness works and results look reasonable
2. **Sanity check results** — are accuracy, cost, and error rates in the expected range? Compare against the baseline results below
3. **Branch based on outcome:**
   - If worse than expected: troubleshoot before running full suite (check errors, compare traces)
   - If reasonable: run the full benchmark and display results with comparison to prior runs
4. **Postmortem: what went wrong** — focus on errors first, then wrong answers where another method (baseline, CC, prior Spell run) got it right. Dig into traces for root causes
5. **Postmortem: what went right** — which Spell features were actually used? Did think/rethink, call-now, agents/spawn, or other orchestration primitives contribute to correct answers?

## Reporting Convention

Denominator is always total items, not items-minus-errors. Errors count as wrong. Report as:

```
50% (13/30) — 4 errors, 13 wrong
```

Not: `50% (13/26)`.

## Scientific Neutrality

Maintain a neutral, skeptical stance. Present unfavorable and inconclusive results without spin. Don't emphasize results that favor Spell or soft-pedal unfavorable comparisons.

## Implemented Benchmarks

### Math / Reasoning (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| GSM8K | 50 | Grade-school math | `uv run run_benchmark.py --dataset gsm8k --condition spell` |
| MATH Hard | 20-30 | Competition math (Level 5) | `uv run run_benchmark.py --dataset math_hard --condition spell` |
| AIME 2025 | 30 | AMC/AIME competition | `uv run run_benchmark.py --dataset aime_2025 --condition spell` |
| AIME 2026 | 30 | AMC/AIME competition (newer) | `uv run run_benchmark.py --dataset aime_2026 --condition spell` |
| HMMT Feb 2025 | ~50 | Collegiate competition | `uv run run_benchmark.py --dataset hmmt_feb_2025 --condition spell` |
| Omni-MATH | ~60 | Olympiad-level, difficulty-filtered | `uv run run_benchmark.py --dataset omni_math --condition spell` |

### Long-Context (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| LongBench v2 | 48-91 | Long-context MCQ (8k-128k tokens) | `uv run run_benchmark.py --dataset longbench_short --condition spell` |
| BABILong QA2 | 100 | Multi-hop QA over 32k context | `uv run run_benchmark.py --dataset babilong --condition spell` |

### Coding (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| SWE-bench | 50-500 | Bug fixing in real repos | `uv run run_swebench.py --dataset mini --condition spell` |
| FeatureBench | varies | Feature implementation | `uv run run_featurebench.py --condition spell` |

### Coding (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Exercism Python | ~225 | Programming exercises, two-attempt with test feedback | `clj -M:dev -m exercism-bench run` |

### Orchestration (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Orchestration v1 | 4 | Whether models use Spell primitives | `clj -M:dev -m benchmark run` |
| Orchestration (OpenAI) | 4 | Same, OpenAI models | `clj -M:dev -m benchmark-openai` |

### Common flags (Python harness)

```bash
--condition {baseline,spell,cc,all}   # Which system to run
--n N                                  # Limit to N items
--model anthropic:claude-opus-4-5-20251101   # Model override
--thinking --reasoning-effort high     # Extended thinking
--suffix-grammar                       # Suffix grammar support
--budget N                             # Spell budget in dollars
```

### Common flags (Exercism)

```bash
--slug hello-world    # Single exercise
--difficulty 5-9      # Difficulty range
--limit 10            # Max exercises
--model sonnet        # Model shorthand
```

## Baseline Results

Most recent results for comparison. Check notebook entries for full details.

### Math

| Benchmark | Spell | CC | Baseline | Entry |
|-----------|-------|----|----------|-------|
| GSM8K (30) | 100% | — | — | `post-refactor-regression-test` |
| MATH Hard (20) | 90% | — | — | `post-refactor-regression-test` |
| AIME 2025 (30) | 73.3% | 66.7% | — | `pricing-correction-benchmark-reanalysis` |
| HMMT Feb (30) | 76.7% | 66.7% | — | `pricing-correction-benchmark-reanalysis` |
| Omni-MATH (54) | 51.9% | 38.6% | 40.4% | `omni-math-manual-scoring` |

### Long-Context

| Benchmark | Spell | CC | Entry |
|-----------|-------|----|-------|
| LongBench v2 (48) | 75.0% | 62.5% | `longbench-regression-v2` |
| BABILong QA2 (100) | 65% | 75-85% | `babilong-qa2-opus45` |

### Coding

| Benchmark | Spell | CC | Entry |
|-----------|-------|----|-------|
| Exercism d5-d9 (16) | 93.8% | 100% | `exercism-d5-d9-benchmark` |
| Exercism d4-d5 (30) | 100% | 100% | `exercism-benchmark-pilot` |

## Infrastructure

### Python harness (`benchmarking/`)

```
benchmarking/
├── run_benchmark.py       # Main entry: math, longbench, babilong
├── run_swebench.py        # SWE-bench
├── run_featurebench.py    # FeatureBench
├── src/
│   ├── datasets.py        # Dataset loading
│   ├── runners.py         # BaselineRunner, SpellRunner, ClaudeCodeRunner
│   ├── evaluation.py      # Scoring (exact match, regex extraction)
│   ├── swebench_prep.py   # SWE-bench workspace setup
│   └── swebench_output.py # Results formatting
├── results/               # JSON output files (gitignored)
├── traces/                # Execution traces (gitignored)
└── data/                  # Cached datasets (gitignored)
```

Use `uv run` instead of `python3` directly.

### Clojure harness (`dev/`)

```
dev/
├── benchmark.clj          # Orchestration benchmark (Anthropic)
├── benchmark_openai.clj   # Orchestration benchmark (OpenAI)
└── exercism_bench.clj     # Exercism Python benchmark
```

Results go to `data/exercism-results/` (timestamped directories) and `notebook/entries/orchestration-benchmark-pilot/outputs/`.

## Known Patterns and Gotchas

**Pricing:** Opus 4.5 costs $5/$25 per MTok (input/output). A historical 3x pricing bug was corrected Feb 15 — any Opus 4.5 costs reported before that date should be divided by 3. See `pricing-correction-benchmark-reanalysis`.

**Depth/budget defaults:** Spell defaults to budget $1.00. Most benchmarks need `-b 5` or higher. Depth is unlimited by default.

**(stored ...) problem:** Values too large to inline in the completion become opaque `(stored ...)` refs that the LLM can't read back. Affects ~15% of LongBench items. See `longbench-regression-v2`.

**Zero orchestration usage:** Current benchmarks mostly test sequential tool use, not multi-agent orchestration. Spell's advantage comes from code-first semantics (clean return values, deterministic computation) rather than spawn/ask/plet.

**Think/rethink impact:** Adds ~12pp on olympiad math. Requires structural prefix forcing (opening the completion with `(think "approach" ...)`) — prompt instructions alone have zero effect. See `omni-math-manual-scoring`.
