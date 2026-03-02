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
| SWE-bench | 50-731 | Bug fixing in real repos | `uv run python benchmarking/bench.py swebench --dataset pro --condition spell` |
| FeatureBench | varies | Feature implementation | `uv run python benchmarking/bench.py featurebench --condition spell` |
| ScienceAgentBench | 102 | Scientific code generation (data science) | `uv run python benchmarking/bench.py scienceagentbench --condition claude_code` |

SWE-bench harness defaults and workflow:
- `bench.py swebench` now defaults to `--condition all` (`spell`, `claude_code`, `codex`) and evaluates by default (`--evaluate`).
- Use `--no-evaluate` to run generation only (write predictions and unified JSONL without running `swebench.harness`).
- Evaluation runs conditions in series with staged cleanup:
  - non-final conditions: `cache_level=env`, `clean=false`
  - final condition: configurable final cleanup (`--eval-final-cache-level`, `--eval-final-clean`)
- For explicit eval-only runs, use `bench.py swebench-eval`.

### Coding (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Exercism Python | ~225 | Programming exercises, two-attempt with test feedback | `clj -M:dev -m exercism-bench run` |

### Terminal Tasks (Terminal-Bench harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Terminal-Bench 2.0 | ~80 | Shell tasks in Docker containers | `./terminal_bench/run_pilot.sh both` |

Requires Docker (Colima or Docker Desktop). Uses `tb` CLI (`uv tool install terminal-bench`).
All three conditions use imported Docker agents under `terminal_bench/`.

Individual task runs:
```bash
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"  # if using Colima
export PYTHONPATH="${PWD}/terminal_bench"
export SPELL_ROOT="/path/to/spell"  # needed when benchmarking is checked out standalone
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
| Omni-MATH (54, Opus 4.5) | 51.9% | 38.6% | 40.4% | `omni-math-manual-scoring` |
| Omni-MATH (60, Opus 4.6) | 33.3%* | 40.0% | — | `omni-math-rerun` |
| Omni-MATH (60, Codex) | 56.7% | — | 53.3%† | `omni-math-rerun` |
| AIME 2026 (30, Opus 4.6) | 93.3% | 96.7%‡ | — | `aime-2026-full` |
| AIME 2026 (30, Codex) | 96.7% | — | 100%† | `aime-2026-full` |

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
| Terminal-Bench (25, Opus 4.6) | 76% | 52% | `terminal-bench-pilot` |
| Terminal-Bench (25, gpt-5.3-codex) | 44% | 56%* | `terminal-bench-codex` |

*CC column for Codex run = Codex CLI (not Claude Code).

Opus 4.6: Spell wins 6, CC wins 0. Codex: Codex CLI wins 5, Spell wins 1 genuine + 1 infra fluke. Model difference dominates framework difference.

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
├── logs/                  # Evaluation/run logs (gitignored)
├── scratch/               # One-off analysis/result batches (gitignored)
├── docs/one-off-organization.md # One-off organization workflow
└── data/                  # Cached datasets (gitignored)
```

Use `uv run` instead of `python3` directly.

### Terminal-Bench harness (`terminal_bench/`)

```
terminal_bench/
├── base.py                # DockerExecAgent base: install-in-docker + token tracking
├── spell_agent.py         # SpellAgent: installs Spell runtime + runs benchmark API in Docker
├── codex_agent.py         # CodexAgent: Codex CLI in Docker, --json JSONL parsing
├── cc_agent.py            # ClaudeCodeAgent: CC in Docker, stream-json parsing
├── run_pilot.sh           # Runner script for CC vs Spell comparison
└── __init__.py
```

Requires `tb` CLI (`uv tool install terminal-bench`) and Docker. Tasks cached at `~/.cache/terminal-bench/`.

**Agent invocations:**
```bash
# All require: PYTHONPATH="${PWD}/terminal_bench:${PYTHONPATH:-}"
# and: DOCKER_HOST="unix:///Users/loconnor/.colima/default/docker.sock"

# Spell (runs inside Docker container with bundled /opt/spell runtime)
# Set SPELL_ROOT when benchmarking repo is not nested under the spell repo root.
tb run --agent-import-path "spell_agent:SpellAgent" \
  --model "anthropic-tc:opus46" -k "budget=5.0" -k "sh_timeout=120"

# Codex CLI (installs in Docker, --json for JSONL token tracking)
tb run --agent-import-path "codex_agent:CodexAgent" \
  --model "openai/gpt-5.3-codex"

# Claude Code (installs in Docker, stream-json for JSONL token tracking)
tb run --agent-import-path "cc_agent:ClaudeCodeAgent" \
  --model "anthropic/claude-opus-4-6" -k "max_budget_usd=5.0"
```

All three agents report `input_tokens` (non-cached), `cached_input_tokens`, `total_input_tokens`, and `output_tokens` in `agent-logs/result.json`. ClaudeCodeAgent also reports `cost_usd`.

**Token semantics differ by provider:**
- **OpenAI** (Codex): `input_tokens` is the TOTAL (includes cached). `cached_input_tokens` is a subset. Non-cached = `input_tokens - cached_input_tokens`.
- **Anthropic** (CC): `input_tokens` is non-cached only. `cache_creation_input_tokens` and `cache_read_input_tokens` are separate additive fields.

The `ParsedUsage` dataclass normalizes both to: `input_tokens` (non-cached), `cached_input_tokens`, `total_input_tokens` (sum).

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

**SWE-bench evaluation defaults:** `bench.py swebench` now runs official harness evaluation by default after prediction generation. Disable via `--no-evaluate` when iterating quickly.

**SWE-bench Docker socket on Colima:** evaluation uses Docker SDK (`docker.from_env`), which depends on `DOCKER_HOST` instead of Docker CLI context. `bench.py` now auto-falls back to `unix://~/.colima/default/docker.sock` when `DOCKER_HOST` is unset and `/var/run/docker.sock` is missing.

**SWE-bench ARM reality check:** x86_64 eval images can run on Apple Silicon via emulation but are slower and may be less stable on heavy builds. Prefer small pilots first, then scale with caching.

**(stored ...) problem:** Values too large to inline in the completion become opaque `(stored ...)` refs that the LLM can't read back. Affects ~15% of LongBench items. See `longbench-regression-v2`.

**Zero orchestration usage:** Current benchmarks mostly test sequential tool use, not multi-agent orchestration. Spell's advantage comes from code-first semantics (clean return values, deterministic computation) rather than spawn/ask/plet.

**Think/rethink impact:** Adds ~12pp on olympiad math. Requires structural prefix forcing (opening the completion with `(think "approach" ...)`) — prompt instructions alone have zero effect. See `omni-math-manual-scoring`.

**ScienceAgentBench program extraction:** Claude Code writes .py files directly to the workspace; Spell sometimes inlines Python in `io/run-command` calls without writing a file. The harness checks workspace for .py files first, then falls back to extracting from markdown code blocks. Spell's extraction rate is lower (20% vs 100% in pilot) because of this difference.

**ScienceAgentBench full evaluation:** Requires downloading the benchmark zip from SharePoint (password: `scienceagentbench`), which needs a browser. The zip contains `datasets/`, `eval_programs/`, `gold_programs/`, `scoring_rubrics/`. Docker-based evaluation then runs each program and checks output correctness. Without the zip, only syntax validity can be checked.

**Terminal-Bench Spell runtime:** SpellAgent now executes inside the task container and copies a minimal Spell runtime bundle (`deps.edn`, `src/`, `config/`) to `/opt/spell` during install. This avoids host-vs-container filesystem asymmetry during scoring.

**Terminal-Bench disk exhaustion:** Each task builds a Docker image. Running many tasks without cleanup fills Colima's VM disk. Use `docker system prune -af --volumes` between tasks. Never run more than 2 tasks concurrently on Colima with default disk (20GB). The `run_pilot.sh` script runs tasks sequentially.

**Terminal-Bench empty tool-call error:** Opus 4.6 occasionally returns an empty `spell_suffix` tool_use (`input: {}`), causing a fatal `missing-tool-call` error. This is not currently retried (see TODO #144). Observed on write-compressor.

**Omni-MATH `missing-tool-call` prevalence:** On Omni-MATH with Opus 4.6 toolcall transport, `missing-tool-call` hits 30% of items (18/60). Much higher than Terminal-Bench. Making this retryable (TODO #144) is critical for math benchmarks on toolcall transport. See `omni-math-rerun`.

**Omni-MATH auto-scorer recall:** `math_equivalent` + `extract_math_answer` has <50% recall on Omni-MATH. Misses: Unicode math symbols (Codex outputs ⌈⌉/√/⟂), yes/no text answers, set ordering, equivalent algebraic expressions, descriptive answers. Manual scoring required for accurate results. *All Omni-MATH Spell/CC/Baseline entries use manual scores. See `omni-math-rerun`.

**Omni-MATH baseline footnotes:** \*Spell/Opus 33.3% includes 19 fatal `missing-tool-call` errors; ex-error accuracy is 48.8% (20/41). †Codex CLI = Codex condition (Codex CLI, not Claude Code). ‡CC/Opus AIME 2026 auto-scored 93.3% but item 7 was a format false negative (`**396**` bold markdown); corrected to 96.7%.
