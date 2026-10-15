# Reproducing the paper

These commands run the experiments in *Self-Programmed Execution for Language Model Agents*. They use the published **Spell v0.1.0** runtime, commit `a3ce161a9f83bcb713310d96c26a95ac60978399`, and this benchmarking repository's paper release. Record the benchmarking commit with `git rev-parse HEAD` before a run; unified results also record both source revisions.

## Setup

Use Python 3.13+, `uv`, Java 11+, the Clojure CLI, and Docker. Linux x86-64 is recommended for the coding benchmarks; official SWE-bench images can require substantial disk space. Clojure's `clojure` command must work. All commands below run from the benchmarking checkout.

```bash
git clone --branch v0.1.0 https://github.com/lukejoconnor/spell.git spell-runtime
export SPELL_ROOT="$(cd spell-runtime && pwd)"
git clone --branch paper https://github.com/lukejoconnor/spell-benchmarking.git
cd spell-benchmarking
uv sync
uv tool install 'terminal-bench==0.2.18'
uv run python bench.py --help
```

The Python client is included here. The Spell checkout supplies the evaluator, benchmark API, profiles, and pricing. `uv sync` pins SWE-bench to 4.1.0, whose API the harness uses. `CLOJURE_COMMAND` can override the default `clojure` launcher. Run `git -C "$SPELL_ROOT" rev-parse HEAD` to check the runtime commit.

Set `OPENAI_API_KEY` for OpenAI models, `ANTHROPIC_API_KEY` for Spell/Anthropic models, and `FIREWORKS_API_KEY` for Fireworks models. Claude Code uses `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`. Credentials remain in environment variables. Hugging Face datasets are downloaded on first use; any required Hub authentication uses the normal local Hugging Face configuration.

Container adapters install **Codex CLI 0.120.0**, **Claude Code 2.1.107**, and **mini-swe-agent 2.4.6**. The mini comparator uses `openai/gpt-5.4`, LiteLLM Responses, and medium reasoning effort. All comparator implementation is included on this branch. For host-side LongBench baselines, install the same CLI versions in a local prefix:

```bash
npm install --prefix .baseline-cli @openai/codex@0.120.0 @anthropic-ai/claude-code@2.1.107
export PATH="$PWD/.baseline-cli/node_modules/.bin:$PATH"
```

The benchmarking adapters load `scripts/paper-model-pricing.edn` for GPT-5.6 Sol, Fable 5, and Qwen3.7 Plus, which postdate the v0.1.0 pricing table. These explicit per-million-token rates are sourced from the public Spell pricing table at commit `29cc920158124747022a6e5bc607c59ffd1e89ff`. Host and container runs apply the same overlay; evaluator source remains v0.1.0. The overlay provides non-missing cost estimates and enables dollar-budget accounting for those models. Treat these as the specified reproduction rates, rather than a claim about current provider invoices.

## Coding comparisons: Figures 3 and 4

Terminal-Bench 1.1 denotes `terminal-bench-core==0.1.1` (80 tasks). SWE-bench Lite uses the official 300-item test split, omits hints, and evaluates patches using the official harness in each task container. Errors remain in the accuracy denominator. Web search is disabled by the benchmark policy.

```bash
for effort in low medium high; do
  uv run python bench.py terminalbench --dataset old-core --condition spell \
    --model openai-tc:gpt-5.4 --reasoning-effort "$effort" --n-concurrent 4 \
    --agent-timeout-sec 3600 --test-timeout-sec 600 --budget 5 \
    --run-id "tb-spell-$effort" --output "results/paper/tb-spell-$effort.jsonl"
  uv run python bench.py terminalbench --dataset old-core --condition codex \
    --codex-model gpt-5.4 --reasoning-effort "$effort" --n-concurrent 4 \
    --agent-timeout-sec 3600 --test-timeout-sec 600 \
    --run-id "tb-codex-$effort" --output "results/paper/tb-codex-$effort.jsonl"

  uv run python bench.py swebench --dataset lite --condition spell \
    --model openai-tc:gpt-5.4 --reasoning-effort "$effort" \
    --parallel 4 --timeout 3600 --budget 10 --prewarm-envs --paper-compliant \
    --run-id "swe-spell-$effort" --output "results/paper/swe-spell-$effort.jsonl"
  uv run python bench.py swebench --dataset lite --condition codex \
    --codex-model gpt-5.4 --reasoning-effort "$effort" \
    --parallel 4 --timeout 3600 --prewarm-envs --paper-compliant \
    --run-id "swe-codex-$effort" --output "results/paper/swe-codex-$effort.jsonl"
done
```

For the mini-swe-agent comparison, use `--condition swe_agent --swe-agent-model openai/gpt-5.4 --reasoning-effort medium` with the same task sets and caps; supply `--budget 5` for Terminal-Bench and `--budget 10` for SWE. The original full SWE mini run included ten Matplotlib setup repairs; the corresponding environment repair is included here.

For the Opus 4.6 Terminal-Bench comparison, run the same 80-task command twice at medium effort: `--condition spell --model anthropic-tc:claude-opus-4-6` and `--condition claude_code --claude-model claude-opus-4-6`, both with `--budget 5`. For the coding-prompt intervention, add `--trailing "'(!describe reminders :coding)"` to each Spell Terminal-Bench effort condition. Default runs use `'(!extend)` without that intervention.

The optional cloud dispatchers are in this repository's `scripts/` and the existing `sbl-fig3/` and `appworld-obs5/` directories. They call the GCP helper in `$SPELL_ROOT/scripts/gcp-benchmark.sh`; local reproduction does not require GCP. Spell defaults to `v0.1.0`. Set `BENCH_REF` (or `BENCHMARKING_REF` for LongBench) to a published benchmarking revision before dispatch. Cloud credentials, secret provisioning, and VM costs are separate from model API costs. Run-scoped Docker cleanup is enabled; the wrapper does not perform daemon-wide pruning.

## Model comparison: Figure 2

The exact 32 Terminal-Bench task IDs and the seeded SWE sample are in [figure2/](figure2/terminalbench-32.txt). Use these explicit lists to preserve the paper task set independently of upstream ordering. SWE's sample used `random.Random(20260725).sample(dataset_order, 32)`; `matplotlib__matplotlib-23913` was a shared container-build failure, leaving 31 evaluated tasks. The original 32 and the 31 evaluated IDs are both included.

[Model specifications and effort settings](figure2/models.json) include the six plotted models and the additional Fable 5 appendix row. Use each model's exact spec and effort for both task sets; Kimi-K2.6 does not use a reasoning parameter.

```bash
# Example cell: GPT-5.4 medium; repeat for the specs in models.json.
uv run python bench.py terminalbench --dataset old-core --condition spell \
  --model openai-tc:gpt-5.4 --reasoning-effort medium --n-concurrent 4 \
  --agent-timeout-sec 3600 --test-timeout-sec 600 --budget 5 \
  $(sed 's/^/-t /' paper-repro/figure2/terminalbench-32.txt) \
  --run-id fig2-tb-gpt54 --output results/paper/fig2-tb-gpt54.jsonl

uv run python bench.py swebench --dataset lite --condition spell \
  --model openai-tc:gpt-5.4 --reasoning-effort medium --parallel 4 \
  --timeout 1800 --budget 10 --prewarm-envs --paper-compliant \
  --items "$(paste -sd, paper-repro/figure2/swebench-31-evaluated.txt)" \
  --run-id fig2-swe-gpt54 --output results/paper/fig2-swe-gpt54.jsonl
```

Using the 31-item list reproduces the evaluated task population without rerunning the known shared setup failure. Use the 32-item list to recheck that infrastructure failure and report its outcome separately. Other execution errors count as unsuccessful tasks. Model endpoints and provider behavior can change; record refusals, unavailable endpoints, and errors rather than replacing a model or silently dropping rows.

## LongBench and AppWorld: Figure 4

```bash
# Full LongBench v2 in published order, first 200 records; not longbench_short.
uv run python bench.py general --dataset longbench --n 200 --condition spell \
  --model openai-tc:gpt-5.4 --reasoning-effort medium --parallel 4 \
  --budget 3 --timeout 1200 --run-id lb-spell --output results/paper/lb-spell.jsonl
uv run python bench.py general --dataset longbench --n 200 --condition codex \
  --codex-model gpt-5.4 --reasoning-effort medium --parallel 4 --timeout 300 \
  --run-id lb-codex --output results/paper/lb-codex.jsonl

# All 57 AppWorld dev tasks; repeat with condition codex / codex-model gpt-5.4.
uv run python bench.py terminalbench --dataset appworld-dev --condition spell \
  --model openai-tc:gpt-5.4 --reasoning-effort medium --n-concurrent 4 \
  --agent-timeout-sec 3600 --test-timeout-sec 600 --budget 5 \
  --run-id appworld-spell --output results/paper/appworld-spell.jsonl
```

LongBench documents are supplied on disk to both agents. AppWorld uses the Terminal-Bench dataset adapter; its 57 IDs are retained in `appworld-obs5/appworld-dev-items.txt`. Record the upstream dataset revision supplied by the harness; matching task IDs alone does not guarantee identical content after upstream changes.

## Orchestration games

```bash
uv run python scripts/run_orchestration_games.py --variant sync \
  --model openai-tc:gpt-5.4 --replicates 8 --run-id sync-gpt54
uv run python scripts/run_orchestration_games.py --variant async \
  --model openai-tc:gpt-5.4 --replicates 8 --run-id async-gpt54
uv run python scripts/run_orchestration_games.py --variant async \
  --model openai-tc:gpt-5.6-sol --replicates 8 --run-id async-gpt56sol
```

Each command runs all three games, uses medium effort, the `io-tc` profile, prompt-only initialization, depth 80, and a 900-second per-trial timeout. Budget zero means no dollar cap. Request, result, provenance, and traces are saved under `results/games/`. Add `--dry-run` to inspect requests without model calls. These commands run trials serially; scheduling is independent of the within-game orchestration tested.

Use the [trace-audit rubric](games/SCORING.md) to score the actual orchestration. The original asynchronous experiment used v0.1.1; these default commands use the published v0.1.0 runtime and should be reported as such. The synchronous appendix specifies its task prompts and trace-audit criteria but does not document a full original cap configuration; the explicit 900-second/depth-80 caps above define a fresh reproduction setup.

## Numerical summaries and trace analyses

```bash
uv run python bench.py summarize --input results/paper/tb-spell-medium.jsonl
uv run python bench.py summarize --input results/paper/swe-spell-medium.jsonl

# Pool the three effort conditions separately for each benchmark.
uv run python scripts/analyze_spell_features.py \
  traces/tb-spell-low traces/tb-spell-medium traces/tb-spell-high \
  --label 'Terminal-Bench 1.1' --output-dir results/features-tb
uv run python scripts/analyze_spell_features.py \
  traces/swe-spell-low traces/swe-spell-medium traces/swe-spell-high \
  --label 'SWE-bench Lite' --output-dir results/features-swe

uv run --with tiktoken python scripts/analyze_spell_pruning_tokens.py \
  --trace-root traces/tb-spell-medium --benchmark terminal-bench --effort medium \
  --output-dir results/pruning-tb-medium
```

The summary reads TB's correctness and SWE's official `same_container_resolved`, retaining all selected tasks in the denominator. A SWE patch can pass after generation timed out; use its evaluator result. Costs sum the stored per-item `cost_usd`, including failures. The included mini comparison analyzers provide paired tests and cost totals from supplied JSONL paths; run their `--help` for input flags.

Feature output includes call counts, traces with at least one use, per-trace metrics, and per-turn IO batching. Only `:response` suffix strings are counted; inherited prompts and replayed `.spl` completions are excluded. As in the paper's trace scripts, matching uses textual call patterns, so code examples in emitted strings/comments can match them. `!peek-now` is pooled with `!peek`; `mapv`, `get-in`, and `if-let` are distinct from `map`, `get`, and `if`. Select only the intended run roots. Overlapping roots are deduplicated. Traces without readable string responses are reported in `feature_usage.json` and excluded from trace denominators; this eligibility rule differs from structural pruning's parser requirements.

Pruning uses the Spell AST walker and `o200k_base` tokenization of complete programs and deleted forms. Repeat for each effort/benchmark. The checked-in six pruning aggregates and per-trace metrics remain available for comparison. Fresh runs are stochastic and need not reproduce identical counts; regenerate measurements from their own scored records and traces.
