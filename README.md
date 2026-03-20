# Spell Benchmarking

Benchmark infrastructure for evaluating Spell against one-shot LLM calls, Claude Code, and Codex CLI across math, long-context, and coding tasks.

All commands run from the repository root using `uv run`.

## Quick Start

```bash
# Run 10 GSM8K items with Spell (Opus, toolcall transport)
uv run python benchmarking/bench.py general \
  --dataset gsm8k --condition spell \
  --model anthropic-tc:claude-opus-4-5-20251101 \
  --n 10

# Run 1 SWE-bench item with Spell
uv run python benchmarking/bench.py swebench \
  --dataset mini --condition spell \
  --model anthropic-tc:claude-opus-4-5-20251101 \
  --n 1

# Summarize results
uv run python benchmarking/bench.py summarize --input results/unified/<file>.jsonl

# Organize loose root one-off files into scratch batches
uv run python scripts/organize_one_offs.py --dry-run
uv run python scripts/organize_one_offs.py
```

## Subcommands

### `general` — Math, Long-Context, and Reasoning

Runs a dataset through one or more conditions, evaluates answers automatically, and writes unified JSONL.

```bash
uv run python benchmarking/bench.py general [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `longbench_short` | Dataset to run (see [Datasets](#datasets)) |
| `--condition` | `one_shot` | `one_shot`, `spell`, `claude_code`, `codex`, or `all` |
| `--model` | *(none, required)* | `provider:model` spec for one_shot/spell (see [Model Specs](#model-specs)) |
| `--claude-model` | *(none, required)* | Model name for Claude Code condition |
| `--codex-model` | `codex` | Model name for Codex CLI condition |
| `--n` | all | Limit to first N items |
| `--items` | — | Comma-separated item IDs to run |
| `--parallel` | 5 | Max concurrent runs |
| `--budget` | 1.0 | Spell budget in USD |
| `--depth` | unlimited | Max LLM recursion depth |
| `--max-tokens` | unlimited | Max output tokens per LLM call |
| `--timeout` | 300 | Per-item timeout in seconds |
| `--thinking / --no-thinking` | off | Enable Anthropic extended thinking |
| `--thinking-budget` | 10000 | Token budget for extended thinking |
| `--reasoning-effort` | `high` | `low`, `medium`, or `high` (OpenAI) |
| `--verbosity` | — | `low` or `auto` (OpenAI) |
| `--responses-api / --no-responses-api` | off | Use OpenAI Responses API |
| `--suffix-grammar / --no-suffix-grammar` | off | Enable prefix-aware grammar constraints |
| `--trace / --no-trace` | off | Record execution traces |
| `--agent` | — | Override Spell agent EDN file |
| `--agent-msg` | — | Override agent for message-mode transport |
| `--agent-tc` | — | Override agent for toolcall-mode transport |
| `--dry-run` | — | Print config and exit |
| `--run-id` | auto | Custom run ID |
| `--output` | auto | Output JSONL path |

### `swebench` — Bug Fixing in Real Repositories

Runs the agent inside the official SWE-bench instance image, captures the resulting patch, and records a same-container evaluation result. Use `--clean-eval` when you also want a fresh official evaluation rerun after generation.

```bash
uv run python benchmarking/bench.py swebench [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `mini` | `mini` (50), `lite` (300), `verified` (500), or `pro` (~250) |
| `--condition` | `all` | `spell`, `claude_code`, `codex`, `both`, or `all` |
| `--model` | *(none, required)* | `provider:model` spec for Spell |
| `--claude-model` | *(none, required)* | Model name for Claude Code |
| `--codex-model` | `codex` | Model for Codex CLI |
| `--n` | all | Limit to first N items |
| `--items` | — | Comma-separated SWE-bench instance IDs to run |
| `--seed` | — | Random seed for sampling when using `--n` |
| `--parallel` | 1 | Max concurrent runs |
| `--budget` | 10.0 | Spell budget in USD |
| `--depth` | 50 | Max LLM recursion depth |
| `--max-tokens` | unlimited | Max output tokens per LLM call |
| `--timeout` | 1800 | Per-item timeout in seconds (30 min) |
| `--reasoning-effort` | `high` | `low`, `medium`, or `high` |
| `--clean-eval / --no-clean-eval` | off | Re-run patches in a fresh official eval container after generation |
| `--eval-max-workers` | 4 | Max concurrent evaluation workers |
| `--eval-final-cache-level` | `base` | Cache level used on final condition only |
| `--eval-final-clean / --no-eval-final-clean` | on | Whether the final clean-eval condition performs cleanup |
| `--agent` | — | Override Spell agent EDN file |
| `--agent-msg` | — | Override agent for message-mode |
| `--agent-tc` | — | Override agent for toolcall-mode |
| `--dry-run` | — | Print config and exit |
| `--run-id` | auto | Custom run ID |
| `--output` | auto | Output JSONL path |

Execution details:
- `mini`, `lite`, and `verified` run inside official SWE-bench instance images with the repo mounted at `/testbed`.
- `pro` runs inside the official Docker Hub instance image with the repo at `/app`.
- Same-container eval always runs after generation and is recorded in run metadata.

Clean-eval cleanup sequence for multi-condition runs:
- Non-final conditions: `cache_level=env`, `clean=false` (reuse env images)
- Final condition: `cache_level=<eval-final-cache-level>`, `clean=<eval-final-clean>`

### `swebench-eval` — Official SWE-bench Harness Evaluation

Evaluates prediction JSONL outputs with `swebench.harness.run_evaluation` and
defaults to running all conditions in sequence while preserving env cache until
the final condition.

```bash
uv run python benchmarking/bench.py swebench-eval [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `mini` | `mini`, `lite`, `verified`, or `pro` |
| `--condition` | `all` | `spell`, `claude_code`, `codex`, `both`, or `all` |
| `--run-id` | *(required)* | Run ID produced by `bench.py swebench` |
| `--predictions-dir` | `results/swebench` | Directory containing `<condition>_<dataset>_<run-id>.jsonl` |
| `--namespace` | `none` | Harness namespace (`none` builds/uses local images) |
| `--final-cache-level` | `base` | Cache level used on final condition only |
| `--final-clean / --no-final-clean` | on | Whether final condition performs cleanup |
| `--dry-run` | off | Print command plan without executing |

Default cleanup sequence for multi-condition runs:
- Non-final conditions: `cache_level=env`, `clean=false` (reuse env images)
- Final condition: `cache_level=base`, `clean=true` (prune env + instance images)

### `featurebench` — Feature Implementation

Clones a repository, masks target functions, runs the agent, then executes the project's test suite to verify correctness.

```bash
uv run python benchmarking/bench.py featurebench [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--condition` | `claude_code` | `spell`, `claude_code`, `codex`, `both`, or `all` |
| `--split` | `lite` | Task split |
| `--task-ids` | all | Specific task indices (repeatable) |
| `--model` | *(none, required)* | `provider:model` spec for Spell |
| `--claude-model` | *(none, required)* | Model name for Claude Code |
| `--codex-model` | `codex` | Model for Codex CLI |
| `--parallel` | 1 | Max concurrent runs |
| `--budget` | 10.0 | Spell budget in USD |
| `--depth` | 50 | Max LLM recursion depth |
| `--max-tokens` | unlimited | Max output tokens per LLM call |
| `--timeout` | 1800 | Per-item timeout in seconds (30 min) |
| `--reasoning-effort` | `high` | `low`, `medium`, or `high` |
| `--workspace-root` | `/tmp/fb_ws` | Directory for cloned workspaces |
| `--skip-install` | off | Skip dependency installation |
| `--skip-tests` | off | Skip test verification |
| `--unified` | off | Keep agent execution and verification in the same copied container workspace |
| `--agent` | — | Override Spell agent EDN file |
| `--agent-msg` | — | Override agent for message-mode |
| `--agent-tc` | — | Override agent for toolcall-mode |
| `--run-id` | auto | Custom run ID |
| `--output` | auto | Output JSONL path |

When `--unified` is enabled, prompts are rewritten against `/workspace` and tests run in the same container filesystem where the agent edited the repo.

### `summarize` — Aggregate Results

```bash
uv run python benchmarking/bench.py summarize --input results/unified/<file>.jsonl
```

Optional filters: `--run-id`, `--condition`.

## Datasets

### Math / Reasoning

| Key | Source | Items | Description |
|-----|--------|-------|-------------|
| `gsm8k` | GSM8K test | ~1300 | Grade-school math |
| `math_easy` | MATH Level 1 | ~300 | Competition math (easiest) |
| `math_medium` | MATH Level 3 | ~400 | Competition math (medium) |
| `math_hard` | MATH Level 5 | ~500 | Competition math (hardest) |
| `math_algebra` | MATH algebra | ~600 | Algebra subset |
| `project_euler` | Project Euler | 913 | Integer-answer programming/math problems |
| `project_euler_200` | Project Euler (first 200) | 200 | PE-Bench-compatible subset |
| `aime_2024` | AIME 2024 | 30 | AMC/AIME competition |
| `aime_2025` | AIME 2025 | 30 | AMC/AIME competition |
| `aime_2026` | AIME 2026 | 30 | AMC/AIME competition |
| `hmmt_feb_2025` | HMMT Feb 2025 | ~30 | Collegiate competition |
| `hmmt_nov_2025` | HMMT Nov 2025 | ~30 | Collegiate competition |
| `omni_math` | Omni-MATH | ~4400 | Olympiad-level |
| `omni_math_hard` | Omni-MATH (d>=6) | varies | Hard subset |

### Long-Context

| Key | Source | Items | Description |
|-----|--------|-------|-------------|
| `longbench_short` | LongBench v2 | ~91 | MCQ, context <100k chars |
| `longbench_sampled` | LongBench v2 | ~48 | 8 per domain, stratified |
| `babilong_32k` | BABILong | ~1000 | QA over 32k context |
| `babilong_32k_qa2` | BABILong | ~100 | QA2 (2-hop) only |
| `babilong_32k_hard` | BABILong | ~200 | QA2+QA3 (multi-hop) |
| `babilong_16k` | BABILong | varies | 16k context variant |
| `babilong_8k` | BABILong | varies | 8k context variant |

### Coding (SWE-bench)

| Key | Source | Items | Description |
|-----|--------|-------|-------------|
| `mini` | SWE-bench Verified | 50 | Stratified mini split |
| `lite` | SWE-bench Lite | 300 | SWE-bench Lite |
| `verified` | SWE-bench Verified | 500 | Full verified set |
| `pro` | SWE-bench Pro | ~250 | Harder production-like subset |

### Coding (FeatureBench)

Selected via `--split` and `--task-ids`. Tasks are loaded from the FeatureBench dataset.

## Conditions

| Condition | What it does | Required flags |
|-----------|-------------|----------------|
| `one_shot` | Single LLM call (no orchestration) | `--model` |
| `spell` | Spell agent with self-orchestration | `--model` |
| `claude_code` | Claude Code CLI (`claude -p`) | `--claude-model` |
| `codex` | Codex CLI (`codex exec`) | `--codex-model` (default: `codex`) |
| `all` | Run all conditions | all of the above |

## One-Off Files

Exploratory scripts and ad hoc outputs should live under `scratch/batches/` rather than repo root.

- Workflow reference: `docs/one-off-organization.md`
- Organizer tool: `scripts/organize_one_offs.py`

## Model Specs

The `--model` flag uses a `provider:model` format that determines both the transport mode and the model name.

### Provider Prefixes

| Prefix | Transport | Agent | Example |
|--------|-----------|-------|---------|
| `anthropic-pf` | Anthropic prefill | `io-pf.agent.edn` | `anthropic-pf:claude-opus-4-5-20251101` |
| `anthropic-tc` | Anthropic tool-call | `io-tc.agent.edn` | `anthropic-tc:claude-opus-4-5-20251101` |
| `codex-msg` | Codex messages | `io-msg.agent.edn` | `codex-msg:gpt-5.3-codex` |
| `codex-tc` | Codex tool-call | `io-tc.agent.edn` | `codex-tc:gpt-5.3-codex` |
| `openai` | OpenAI Chat | `io-msg.agent.edn` | `openai:o3` |

### Model Aliases

The Clojure API supports short aliases: `haiku`, `sonnet`, `opus`, `opus46`, `o3`, `o4-mini`, `gpt52`, `gpt53`.

### Claude Code / Codex Models

`--claude-model` accepts a plain model name (no provider prefix): `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, or the aliases `opus`, `sonnet`.

`--codex-model` defaults to `codex`, which resolves to `gpt-5-codex` internally.

## Architecture

```
bench.py (Python CLI)
    │
    ├── SpellAdapter ──→ spell_benchmark_client.py ──→ clj -m spell.benchmark-api
    │                         (subprocess, JSON stdin/stdout)
    │
    ├── OneShotAdapter ──→ (same Clojure API, mode=one_shot)
    │
    ├── ClaudeCodeAdapter ──→ claude CLI (subprocess)
    │
    └── CodexAdapter ──→ codex CLI (subprocess)
```

### Spell Adapter Flow

1. `bench.py` builds a `RunRequest` with model spec, budget, depth, etc.
2. `SpellAdapter` serializes it as JSON and sends to `spell.benchmark-api` via subprocess.
3. The Clojure API:
   - Parses the model spec (`provider:model`) to construct a provider instance
   - Resolves the agent EDN file (from the provider's default, overridable via `--agent`)
   - Calls `spell.api/run` with the provider, agent, and parameters
4. The response (result text, usage, cost, latency) comes back as JSON.

### Agent Resolution

The agent is selected automatically based on the provider prefix:

| Provider prefix | Provider family | Default agent |
|----------------|----------------|---------------|
| `anthropic-pf` | prefill | `config/agents/io-pf.agent.edn` |
| `codex-msg`, `openai` | message | `config/agents/io-msg.agent.edn` |
| `anthropic-tc`, `codex-tc` | toolcall | `config/agents/io-tc.agent.edn` |

Each agent inherits from a base agent (e.g., `base-tc.agent.edn`) and adds the `io` namespace for file access. The agent determines the system prompt and available namespaces but not the model — the model comes from the `--model` flag.

Override with `--agent`, `--agent-msg` (message-mode only), or `--agent-tc` (toolcall-mode only).

## Output Format

Results are written as JSONL with one record per item. Key fields:

```json
{
  "run_id": "general_20260227_120000",
  "benchmark": "general",
  "dataset": "gsm8k",
  "item_id": "gsm8k_test_0",
  "condition": "spell",
  "provider": "anthropic-tc",
  "model": "claude-opus-4-5-20251101",
  "status": "ok",
  "response_text": "...",
  "answer_extracted": "42",
  "correct": true,
  "tokens_in": 15000,
  "tokens_out": 3000,
  "cost_usd": 0.15,
  "latency_ms": 30000,
  "error_type": null,
  "error_message": null,
  "metadata": {}
}
```

Status values: `ok`, `error`, `timeout`, `patch_generated`, `no_patch`, `tests_passed`.

## Reporting Convention

Denominator is always total items. Errors count as wrong.

```
73.3% (22/30) — 2 errors, 6 wrong
```

## Examples

### Pilot a math benchmark (small N, verify config)

```bash
uv run python benchmarking/bench.py general \
  --dataset aime_2025 --condition spell \
  --model anthropic-tc:claude-opus-4-5-20251101 \
  --n 5 --budget 5 --dry-run
```

### Full math comparison: Spell vs Claude Code

```bash
# Spell
uv run python benchmarking/bench.py general \
  --dataset hmmt_feb_2025 --condition spell \
  --model anthropic-tc:claude-opus-4-5-20251101 \
  --budget 5 --output results/unified/hmmt_spell.jsonl

# Claude Code
uv run python benchmarking/bench.py general \
  --dataset hmmt_feb_2025 --condition claude_code \
  --claude-model claude-opus-4-5-20251101 \
  --output results/unified/hmmt_cc.jsonl
```

### SWE-bench with Codex

```bash
uv run python benchmarking/bench.py swebench \
  --dataset mini --condition spell \
  --model codex-tc:gpt-5.3-codex \
  --n 5 --output results/unified/swe_codex.jsonl
```

### FeatureBench: specific tasks

```bash
uv run python benchmarking/bench.py featurebench \
  --condition spell \
  --model anthropic-tc:claude-opus-4-5-20251101 \
  --task-ids 1 --task-ids 2 --task-ids 3
```

### Compare results

```bash
uv run python benchmarking/bench.py summarize --input results/unified/hmmt_spell.jsonl
uv run python benchmarking/bench.py summarize --input results/unified/hmmt_cc.jsonl
```

## Infrastructure

```
benchmarking/
├── bench.py                  # Unified CLI (general / swebench / featurebench / summarize)
├── src/
│   ├── datasets.py           # Dataset loading (HuggingFace, local)
│   ├── unified_adapters.py   # Spell / OneShot / ClaudeCode / Codex adapters
│   ├── unified_results.py    # JSONL schema, summarization
│   ├── evaluation.py         # Scoring (exact match, math equivalence, regex extraction)
│   ├── agent_policy.py       # Agent selection from model/provider transport
│   ├── featurebench_lib.py   # FeatureBench workspace setup and test execution
│   ├── docker_agents.py      # Shared in-container agent execution for software benchmarks
│   ├── swebench_container.py # Official-image SWE-bench generation/eval container wrapper
│   ├── swebench_output.py    # SWE-bench predictions JSONL writer
│   └── workspace.py          # Shared repo cache and worktree management
├── results/                  # JSONL output (gitignored)
├── traces/                   # Execution traces (gitignored)
└── data/                     # Cached datasets (gitignored)
```

The Clojure benchmark API lives at `src/spell/benchmark_api.clj` in the parent repository. The Python client that calls it is at `spell_benchmark_client.py` (also in the parent repository root).
