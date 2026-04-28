# Benchmarking Harness Overview

## Architecture

The harness is a Click CLI (`bench.py`) with subcommands for each benchmark family: `general` (math, long-context, QA), `swebench`, `featurebench`, `terminalbench`, `harbor-terminalbench`, `exercism`, `scienceagentbench`. Every subcommand follows the same execution pipeline:

```
CLI flags → load dataset → build adapter per condition → construct RunRequests → execute → evaluate → write JSONL
```

### Conditions

A **condition** is the system under test. Four are supported:

| Condition | Adapter | What runs |
|-----------|---------|-----------|
| `spell` | `SpellAdapter` | Spell agent via `spell.benchmark-api` (Clojure subprocess) |
| `one_shot` | `OneShotAdapter` | Single LLM call through the same Clojure bridge |
| `claude_code` | `ClaudeCodeAdapter` | Claude Code CLI (`claude -p --output-format stream-json`) |
| `codex` | `CodexAdapter` | Codex CLI (`codex --json`) |
| `swe_agent` | `SweAgentAdapter` | mini-swe-agent CLI (`mini`) |

Each adapter implements a `run(req: RunRequest) -> RunResponse` protocol. The adapters live in `src/unified_adapters.py`. Docker variants (`DockerSpellAdapter`, `DockerSweAgentAdapter`, etc.) in `src/docker_adapters.py` wrap the same logic but execute inside containers.

`claude_code` requires `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. The Claude Code harness strips `ANTHROPIC_API_KEY` and legacy `CLAUDE_JSON_B64` from its process/container environments to keep benchmark runs on Claude Max subscription OAuth.

### The init construction pipeline (Spell only)

Spell conditions don't receive a bare prompt — they receive an **init string**, a Spell program that becomes the starting point of execution. The construction chain is:

1. **`format_benchmark_prompt(item)`** (`src/datasets.py`) — formats the dataset item into a natural-language prompt string, shared across conditions where possible.

2. **`build_spell_init(prompt, trailing)`** (`src/docker_agents.py`) — wraps the prompt in Spell's completion template:
   ```
   (quine completion (eval (do (quine prompt "...escaped prompt...") <trailing>)))
   ```
   If no trailing is provided, defaults to `'(!extend)`.

3. **`_make_spell_init(prompt, trailing)`** (`bench.py`) — thin wrapper that detects raw Spell prefixes (prompts starting with `(`) and passes them through unchanged; otherwise delegates to `build_spell_init`. Used by all benchmarks uniformly.

### Execution modes

Three modes exist for software benchmarks (SWE-bench, FeatureBench, Exercism):

- **Host** (default): Adapter runs locally. For SWE-bench, a git worktree is set up per item; patches are extracted via `git diff`.
- **Docker** (`--docker`): Docker-wrapped adapters run the agent inside a container. Patch returned in response metadata.
- **Unified** (`--unified`): Shared container-agent infrastructure (`src/docker_agents.py`). The agent runs in a `DockerWorkspaceContainer` with the repo pre-cloned. Used by Terminal-Bench and now available for SWE-bench/FeatureBench/Exercism. Calls `run_unified_swebench()` / `run_unified_featurebench()` / `run_unified_exercism()` instead of `adapter.run()`.

Terminal-Bench local/regression runs use `bench.py terminalbench`. That subcommand wraps legacy `tb run`, normalizes per-task `results.json` into `UnifiedRunRecord`s, and copies Spell traces out of per-task `agent-logs/spell-trace/` directories into the standard `traces/<run-id>/spell/...` layout. Its `--dataset 2.0` alias is not public leaderboard-compliant.

Public Terminal-Bench 2.0 Spell runs use `bench.py harbor-terminalbench`, which shells out to `uv tool run --from harbor harbor run -d terminal-bench@2.0` so fresh hosts can run without a separate Harbor CLI install. Submission mode requires `k >= 5`, forbids task filters, rejects the built-in Harbor Codex comparator, and does not expose timeout/resource/local-registry/recovery knobs. `--dev-smoke` is explicitly non-submission and permits `k=1`, `--n`, task selectors, and Harbor's built-in Codex comparator for development comparisons.

In unified mode, `bench.py` still constructs the `RunRequest` (including `init`) identically — the unified runners just receive the pre-built init string.

### Request/Response flow

```
RunRequest                          RunResponse
├── benchmark: str                  ├── status: ok|error|timeout|patch_generated|no_patch
├── condition: str                  ├── response_text: str
├── prompt: str                     ├── tokens_in/out, cost_usd, latency_ms
├── init: str (Spell only)          ├── error_message, error_type
├── model_spec: str                 └── metadata: dict (trace_dir, patch, etc.)
├── agent_file: str (Spell only)
├── timeout, budget, depth, ...
├── workspace: Path (software)
└── trace_dir: Path
```

After execution, responses are evaluated (`_evaluate_general` for math/QA, `git diff` for software), converted to `UnifiedRunRecord`, and appended to a JSONL output file.

### Output format

All results go to `results/unified/<run_id>.jsonl`. Each line is a `UnifiedRunRecord` with: run_id, timestamp, benchmark, dataset, condition, provider, model, status, response_text, answer_extracted, correct, tokens, cost, latency, error info, and freeform metadata. Unified records now always include `metadata.versions`, which captures the git commit, branch, and dirty state for both the main `spell` repo and the `benchmarking` repo.

`summarize_records()` aggregates by condition: total, correct, errors, accuracy, cost.

### SWE-bench variants

Named intervention variants (`SWEBENCH_VARIANTS` dict) configure `trailing` and `agent` overrides for SWE-bench experimentation. Examples: `fix-loop` (uses the fix-loop pattern), `orchestrator` (multi-agent decomposition), `context-mgmt` (aggressive context pruning instructions). Selected via `--variant`.

### Agent resolution

For Spell conditions, `resolve_spell_agent()` (`src/agent_policy.py`) determines which `.agent.edn` file to use based on the benchmark kind, model spec, and CLI overrides (`--agent`, `--agent-msg`, `--agent-tc`). This maps benchmarks to appropriate Spell agent configurations (e.g., tool-call agents for Codex and `openai-tc` models, message agents for Anthropic prefill).

### Evaluation

- **Math/QA** (`general`): Inline in `_evaluate_general` — dataset-specific extractors (numeric, MCQ, freeform, GAIA scorer, math equivalence).
- **SWE-bench**: Post-generation step always runs the official `swebench.harness` Docker evaluator. Supports staged cleanup across conditions.
- **FeatureBench/Exercism**: Test suites run against the modified workspace.
- **ScienceAgentBench**: Program extraction + execution check (full eval requires external benchmark data).
