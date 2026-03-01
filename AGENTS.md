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
| GSM8K | 50 | Grade-school math | `uv run python benchmarking/bench.py general --dataset gsm8k --condition spell` |
| MATH Hard | 20-30 | Competition math (Level 5) | `uv run python benchmarking/bench.py general --dataset math_hard --condition spell` |
| AIME 2025 | 30 | AMC/AIME competition | `uv run python benchmarking/bench.py general --dataset aime_2025 --condition spell` |
| AIME 2026 | 30 | AMC/AIME competition (newer) | `uv run python benchmarking/bench.py general --dataset aime_2026 --condition spell` |
| HMMT Feb 2025 | ~50 | Collegiate competition | `uv run python benchmarking/bench.py general --dataset hmmt_feb_2025 --condition spell` |
| Omni-MATH | ~60 | Olympiad-level, difficulty-filtered | `uv run python benchmarking/bench.py general --dataset omni_math --condition spell` |

### Long-Context (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| LongBench v2 | 48-91 | Long-context MCQ (8k-128k tokens) | `uv run python benchmarking/bench.py general --dataset longbench_short --condition spell` |
| BABILong QA2 | 100 | Multi-hop QA over 32k context | `uv run python benchmarking/bench.py general --dataset babilong_32k_qa2 --condition spell` |

### Coding (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| SWE-bench | 50-500 | Bug fixing in real repos | `uv run python benchmarking/bench.py swebench --dataset mini --condition spell` |
| FeatureBench | varies | Feature implementation | `uv run python benchmarking/bench.py featurebench --condition spell` |
| ScienceAgentBench | 102 | Scientific code generation (data science) | `uv run python benchmarking/bench.py scienceagentbench --condition claude_code` |

### Coding (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Exercism Python | ~225 | Programming exercises, two-attempt with test feedback | `clj -M:dev -m exercism-bench run` |

### Terminal Tasks (Terminal-Bench harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Terminal-Bench 2.0 | ~80 | Shell tasks in Docker containers | `./benchmarking/terminal_bench/run_pilot.sh both` |

Requires Docker (Colima or Docker Desktop). Uses `tb` CLI (`uv tool install terminal-bench`).
Spell agent: `benchmarking/terminal_bench/spell_agent.py` (BaseAgent, bridges io/sh to docker exec).
CC agent: built-in `claude-code` agent (installed inside container).

Individual task runs:
```bash
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"  # if using Colima
export PYTHONPATH="${PWD}/benchmarking/terminal_bench"
tb run --agent-import-path "spell_agent:SpellAgent" --model "anthropic/claude-opus-4-6" \
  --dataset "terminal-bench-core==0.1.1" -t <task-id> --output-path <dir> --run-id <id> \
  --n-concurrent 1 -k "budget=5.0" -k "sh_timeout=120"
```

Run `docker system prune -af --volumes` between tasks to avoid disk exhaustion.

### Orchestration (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Orchestration v1 | 4 | Whether models use Spell primitives | `clj -M:dev -m benchmark run` |
| Orchestration (OpenAI) | 4 | Same, OpenAI models | `clj -M:dev -m benchmark-openai` |

### Common flags (Python harness)

```bash
--condition {baseline,spell,claude_code,codex,all}   # Which system to run
--n N                                  # Limit to N items
--model anthropic:claude-opus-4-5-20251101   # Model override
--thinking --reasoning-effort high     # Extended thinking
--suffix-grammar                       # Suffix grammar support
--budget N                             # Spell budget in dollars
```

`reasoning_effort` defaults to `high` in unified adapters unless explicitly overridden.

Condition semantics:
- `baseline`: single LLM call (text-only). Use any provider/model via `--model` (including Codex-backed single-call models).
- `codex`: Codex CLI harness (agent loop). Default model is `codex` via `--codex-model codex`.
  - In the harness adapter, `codex` resolves to `gpt-5-codex` for stable Codex CLI compatibility.
- `spell` (including SWE-bench workspaces): adapter injects local `-Sdeps` so `spell.benchmark-api` resolves even when `cwd` is a cloned repo workspace.

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

### Scientific Code Generation

| Benchmark | Spell | CC | Entry |
|-----------|-------|----|-------|
| ScienceAgentBench (5 pilot, Opus 4.6) | 100% | 100% | `scienceagentbench-pilot` |

Note: ScienceAgentBench "correct" = program executes and produces expected output file. Full functional correctness requires running eval scripts (not yet integrated).

### Terminal Tasks

| Benchmark | Spell | CC | Entry |
|-----------|-------|----|-------|
| Terminal-Bench (24, Opus 4.6) | 62.5% | 54.2% | `terminal-bench-pilot` |

Breakdown: easy 100%/88%, medium 33%/33%, hard 50%/40%. Paired: 22 concordant, Spell wins 2, CC wins 0. Spell uses tool-call transport (not prefill/thinking).

## Infrastructure

### Python harness (`benchmarking/`)

```
benchmarking/
├── bench.py               # Unified CLI: general/swebench/featurebench/summarize
├── run_benchmark.py       # Legacy shim -> bench.py general
├── run_swebench.py        # Legacy shim -> bench.py swebench
├── run_featurebench.py    # Legacy shim -> bench.py featurebench
├── manual_swebench_high_reasoning.py # Manual SWE-bench runner for focused diagnostics
├── src/
│   ├── datasets.py        # Dataset loading
│   ├── unified_adapters.py # Condition adapters (Spell/Baseline/Claude/Codex)
│   ├── unified_results.py  # Unified JSONL schema + summaries
│   ├── evaluation.py      # Scoring (exact match, regex extraction)
│   ├── swebench_prep.py   # SWE-bench workspace setup
│   └── swebench_output.py # Results formatting
├── results/               # JSON output files (gitignored)
├── traces/                # Execution traces (gitignored)
└── data/                  # Cached datasets (gitignored)
```

Use `uv run` instead of `python3` directly.

### Terminal-Bench harness (`benchmarking/terminal_bench/`)

```
benchmarking/terminal_bench/
├── spell_agent.py         # Spell BaseAgent (bridges io/sh to docker exec)
├── run_pilot.sh           # Runner script for CC vs Spell comparison
└── __init__.py
```

Requires `tb` CLI (`uv tool install terminal-bench`) and Docker. Tasks cached at `~/.cache/terminal-bench/`.

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

**SWE-bench classpath:** `SpellAdapter` injects local `-Sdeps` so `spell.benchmark-api` can run from cloned workspace directories.

**(stored ...) problem:** Values too large to inline in the completion become opaque `(stored ...)` refs that the LLM can't read back. Affects ~15% of LongBench items. See `longbench-regression-v2`.

**Zero orchestration usage:** Current benchmarks mostly test sequential tool use, not multi-agent orchestration. Spell's advantage comes from code-first semantics (clean return values, deterministic computation) rather than spawn/ask/plet.

**Think/rethink impact:** Adds ~12pp on olympiad math. Requires structural prefix forcing (opening the completion with `(think "approach" ...)`) — prompt instructions alone have zero effect. See `omni-math-manual-scoring`.

**ScienceAgentBench program extraction:** Claude Code writes .py files directly to the workspace; Spell sometimes inlines Python in `io/run-command` calls without writing a file. The harness checks workspace for .py files first, then falls back to extracting from markdown code blocks. Spell's extraction rate is lower (20% vs 100% in pilot) because of this difference.

**ScienceAgentBench full evaluation:** Requires downloading the benchmark zip from SharePoint (password: `scienceagentbench`), which needs a browser. The zip contains `datasets/`, `eval_programs/`, `gold_programs/`, `scoring_rubrics/`. Docker-based evaluation then runs each program and checks output correctness. Without the zip, only syntax validity can be checked.

**Terminal-Bench docker exec bridge:** Spell runs on the host, not inside Docker containers. `SPELL_DOCKER_CONTAINER` and `SPELL_DOCKER_WORKDIR` env vars cause `io/sh` to use `docker exec` instead of local shell. The preamble in `spell_agent.py` instructs the LLM to use only `io/sh` (not `io/spit`, `io/read-file` etc.) since those operate on the host filesystem. See `src/spell/io.clj`.

**Terminal-Bench disk exhaustion:** Each task builds a Docker image. Running many tasks without cleanup fills Colima's VM disk. Use `docker system prune -af --volumes` between tasks. Never run more than 2 tasks concurrently on Colima with default disk (20GB). The `run_pilot.sh` script runs tasks sequentially.

**Terminal-Bench empty tool-call error:** Opus 4.6 occasionally returns an empty `spell_suffix` tool_use (`input: {}`), causing a fatal `missing-tool-call` error. This is not currently retried (see TODO #144). Observed on write-compressor.
