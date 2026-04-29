# Benchmarking

## Workflow

Follow this sequence when running benchmarks:

1. **Pilot a small subset** (3-5 items) to verify the harness works and results look reasonable
2. **Sanity check results** — are accuracy, cost, and error rates in the expected range? Compare against the one-shot results below
3. **Branch based on outcome:**
   - If worse than expected: troubleshoot before running full suite (check errors, compare traces)
   - If reasonable: run the full benchmark and display results with comparison to prior runs
4. **Postmortem: what went wrong** — focus on errors first, then wrong answers where another method (one-shot, CC, prior Spell run) got it right. Dig into traces for root causes
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
| BrowseComp-Plus | 150 | Multi-hop deep-research QA over 1000-doc pack (gold + evidence + random); RLMs protocol (Sun et al. 2025) | `uv run python benchmarking/bench.py general --dataset browsecomp_plus --condition spell` |
| BrowseComp-Plus pilot | 5 | Smoke-test variant of BrowseComp-Plus | `uv run python benchmarking/bench.py general --dataset browsecomp_plus_pilot --condition spell` |

**BrowseComp-Plus notes:** docs are XOR/canary-decrypted on first load and cached to `data/browsecomp_plus/queries_decrypted.jsonl` (~2.1 GB; first run ~5 min, subsequent ~15 s). Each query gets 1000 docs (all gold + evidence + random fill from the 100K corpus) written to a single `data/contexts/<id>.txt` file; per-doc text is truncated to 1500 chars by default. Scoring uses LLM-as-judge with `gpt-4.1-mini` and the upstream GRADER_TEMPLATE; override the judge model with `BROWSECOMP_PLUS_JUDGE_MODEL`. The original benchmark uses Qwen3-32B (~1pp delta vs GPT-4.1 per upstream docs).

### Coding (Python harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| SWE-bench | 50-731 | Bug fixing in real repos | `uv run python benchmarking/bench.py swebench --dataset pro --condition spell` |
| FeatureBench | varies | Feature implementation | `uv run python benchmarking/bench.py featurebench --condition spell` |
| ScienceAgentBench | 102 | Scientific code generation (data science) | `uv run python benchmarking/bench.py scienceagentbench --condition claude_code` |

Software-benchmark container infra:
- `bench.py swebench`, `bench.py featurebench`, and `bench.py exercism` now run on the shared in-container agent layer by default.
- SWE-bench generation now uses official per-instance images: `/testbed` for `mini`/`lite`/`verified`, `/app` for `pro`.
- SWE-bench Pro clean eval still runs through the separate ScaleAI Docker pipeline when `--clean-eval` is requested.
- Exercism `one_shot` is the lone exception: it stays on the local single-call path because generation does not need a mutable workspace.
- ScienceAgentBench is intentionally not on the shared software-benchmark container infra in this pass.

SWE-bench harness defaults and workflow:
- `bench.py swebench` now defaults to `--condition all` (`spell`, `claude_code`, `codex`, `swe_agent`) and always runs same-container evaluation after generation.
- Use `--name <label>` to tag a run variant across JSONL records, traces, and SWE-bench prediction filenames.
- Use `--trailing <expr>` on Spell runs to override the default trailing policy directly instead of relying on preset variants.
- `bench.py swebench` uses `src/swebench_container.py` to start official instance images, then runs agents through the shared `src/docker_agents.py` runtime layer.
- Use `--clean-eval` to rerun generated patches in a fresh official evaluation container.
- Clean eval runs conditions in series with staged cleanup:
  - non-final conditions: `cache_level=env`, `clean=false`
  - final condition: configurable final cleanup (`--eval-final-cache-level`, `--eval-final-clean`)
- For explicit eval-only runs, use `bench.py swebench-eval` and pass the same `--name` you used for generation when predictions were label-tagged.
- For remote SWE-bench image caching on GCP, set `SPELL_GCP_ENV_IMAGE_CACHE_REPOSITORY` to an Artifact Registry repo. The startup script logs Docker into that repo with the VM service account token and exports `SPELL_SWEBENCH_ENV_IMAGE_CACHE_REPOSITORY` so `bench.py swebench` can pull/push both shared `sweb.env.*` images and per-instance `sweb.eval.*` images across fresh VMs.
- GPT-5.4 and GPT-5.4-pro switch to the long-context pricing tier once total input exceeds 272k tokens; benchmark-side cost estimates apply the session-level 2x input / 1.5x output rates, including the benchmark's cache-write input bucket as input-side usage.

### Coding (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Exercism Python | ~225 | Programming exercises, two-attempt with test feedback | `clj -M:dev -m exercism-bench run` |

Software-benchmark container infra:
- `src/docker_agents.py` is the shared container-agent layer for Terminal-Bench plus the Python-harness software benchmarks.
- `bench.py exercism` and `bench.py featurebench` keep edits and verification in the same copied container workspace.
- Exercism prompts use stable in-container paths instead of host temp paths.

### Terminal Tasks (Terminal-Bench harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Terminal-Bench old core | 80 | Shell tasks in Docker containers | `uv run python bench.py terminalbench --dataset old-core --condition all` |
| Terminal-Bench 2.x pinned | ~94 | Local/core shell tasks in Docker containers | `uv run python bench.py terminalbench --dataset 2.0 --condition all` |
| Harbor Terminal-Bench 2.0 | 89 | Public TB2 Spell/Codex/Claude Code submission path | `uv run python bench.py harbor-terminalbench --condition spell` |

Requires Docker (Colima or Docker Desktop). The legacy rows use `tb` CLI (`uv tool install terminal-bench`).
The Harbor row uses Harbor CLI via `uv tool run --from harbor harbor` plus local installed-agent adapters: `harbor_spell_agent.py` for Spell and `harbor_codex_agent.py` for Codex, so fresh benchmark VMs do not need a separate manual Harbor install. Harbor Claude Code uses Harbor's built-in `claude-code` agent, but the benchmark wrapper launches Harbor with an OAuth-only environment: it requires `CLAUDE_CODE_OAUTH_TOKEN` and strips `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, and legacy Claude JSON auth variables before Harbor starts.
Legacy `terminalbench` conditions use imported Docker agents under `terminal_bench/`, orchestrated through `bench.py terminalbench`.
If `benchmarking/` is checked out as a standalone worktree instead of nested under the main Spell repo, export `SPELL_ROOT=/path/to/spell` before running either Terminal-Bench path.
The default dataset remains old core (`terminal-bench-core==0.1.1`) for comparability with recent local runs. Dataset aliases resolve as follows:

- `old-core`, `core-0.1.1`, `0.1.1`, and `1.1` -> `terminal-bench-core==0.1.1`
- `2.0`, `head`, `current`, and `latest` -> `terminal-bench-core==head` using the bundled local registry entry pinned to upstream `dataset/terminal-bench-core/v0.2.x` commit `76de0a03069c2a1101d8e0845daedb5065da87ea`

Raw `tb --dataset` strings still pass through without the bundled registry override, so use the registry string directly when you need upstream registry behavior. Note that the upstream default registry may treat raw `terminal-bench-core` as a moving head entry; prefer the aliases above for reproducible local CLI behavior.

`bench.py terminalbench --dataset 2.0` is **not** a public leaderboard-compliant Terminal-Bench 2.0 run. It is the local scientific/regression harness: it uses `tb run`, a local registry path for the pinned 2.x task set, local timeout/cleanup knobs, and the test-patch-collision recovery wrapper.

Public Terminal-Bench 2.0 Spell submission runs must use Harbor:

- Submission mode: `uv run python bench.py harbor-terminalbench --condition spell --model openai-tc:gpt-5.4 --reasoning-effort medium -k 5`
- Codex submission mode: `uv run python bench.py harbor-terminalbench --condition codex --codex-model gpt-5.4 --reasoning-effort medium -k 5`
- Claude Code submission mode: `uv run python bench.py harbor-terminalbench --condition claude_code --claude-model anthropic/claude-opus-4-6 --reasoning-effort medium -k 5`
- Development Codex comparison: `uv run python bench.py harbor-terminalbench --condition codex --codex-model gpt-5.4 --reasoning-effort medium --dev-smoke -k 1`
- Development Claude Code comparison: `uv run python bench.py harbor-terminalbench --condition claude_code --claude-model anthropic/claude-opus-4-6 --reasoning-effort medium --dev-smoke -k 1`
- Development smoke only: add `--dev-smoke` to allow `-k 1`, `--n`, `--task`, or task-name filters.
- Timeout-expanded dev smoke only: add `--dev-smoke --agent-timeout-multiplier <N>` with `N > 1`. This is non-submission only.

Submission mode is guarded to use `uv tool run --from harbor harbor run -d terminal-bench@2.0`, require `k >= 5`, run the full dataset, and avoid timeout/resource/local-registry/recovery flags. Codex submission runs use the local `harbor_codex_agent.py` adapter rather than Harbor's built-in `codex` agent so web search is explicitly disabled and GPT-5.4 cost is computed per model call. `reasoning_output_tokens` is recorded as a sub-bucket of `output_tokens`, not added again for billing. Claude Code submission runs intentionally use Max-subscription OAuth rather than Anthropic API billing; if `CLAUDE_CODE_OAUTH_TOKEN` is absent, the wrapper fails before launching Harbor.

Harbor Spell writes live cumulative usage snapshots to `/logs/agent/spell-usage.json`; this file is the source of truth for Spell token/cost accounting on success, error, and timeout rows. `spell-output.txt` may contain final-response metadata, but Harbor Spell must not use it as a cost/token fallback.

For timeout-expanded dev-smoke runs, the wrapper writes `default-timeout-summary.json` under each Harbor job directory. The `resolved_within_default_timeout` field is a post-hoc diagnostic for whether a resolved trial also finished within the default 900s TB2 agent budget; it is not leaderboard scoring and must not be included in submission artifacts.

Examples:
```bash
# Spell vs Codex CLI on two tasks with an explicit io reminder
uv run python bench.py terminalbench \
  --condition spell \
  --condition codex \
  --model codex-tc:gpt-5.3-codex \
  --agent config/agents/io-tc.agent.edn \
  --trailing "'(!describe io)" \
  -t hello-world \
  -t csv-to-parquet

# Dry-run old-core/default commands without executing them
uv run python bench.py terminalbench \
  --condition spell \
  --dataset old-core \
  -t hello-world \
  --dry-run

# Dry-run the pinned 2.x Terminal-Bench task set without executing it
uv run python bench.py terminalbench \
  --condition spell \
  --dataset 2.0 \
  -t hello-world \
  --dry-run
```

Direct `tb run` still works for one-off debugging when you need raw harness behavior, but batch runs and unified JSONL collection should go through `bench.py terminalbench`.

Terminal-Bench scoring caveat:
- If an agent touches test-like files and the official evaluator patch later collides with those edits, the harness attempts to restore only those touched test files, retry the evaluator patch, and run the tests.
- Unified records for recovered rows include `metadata.test_patch_collision_recovered=true` and `metadata.score_caveat`; display such rows with an asterisk in paper/report tables.
- For old artifacts, use `uv run python scripts/scan_terminalbench_patch_collisions.py [results-root] --format table` to identify unrechecked collision rows for annotation.

### Orchestration (Clojure harness)

| Benchmark | Items | What it tests | Run command |
|-----------|-------|---------------|-------------|
| Orchestration v1 | 4 | Whether models use Spell primitives | `clj -M:dev -m benchmark run` |
| Orchestration (OpenAI) | 4 | Same, OpenAI models | `clj -M:dev -m benchmark-openai` |

### Common flags (Python harness)

```bash
--condition {one_shot,spell,claude_code,codex,swe_agent,all}   # Which system to run
--n N                                  # Limit to N items
--model anthropic-tc:claude-opus-4-6         # Model override
--name trial-label                     # Optional run label in JSONL/traces/predictions
--trailing "'(!extend)"               # Spell trailing override  ⚠️ see note below
--agent path/to/agent.edn             # Primary Spell configuration override
--thinking --reasoning-effort high     # Extended thinking
--suffix-grammar                       # Suffix grammar support
--budget N                             # Spell budget in dollars
```

`reasoning_effort` defaults to `high` when the benchmark runner/provider supports it. Records should only carry a `reasoning_effort` value when the selected path actually forwards it.

**`--trailing` quoting — do not backslash-escape the leading quote.** The trailing expression starts with `'` (Clojure's quote reader macro). In shell, pass it inside double quotes with NO backslash: `--trailing "'(patterns/fix-loop prompt)"`. Using `"\'(...)"` produces a Clojure character literal `\'` instead of `(quote ...)`, silently breaking the trailing expression. The model recovers as a plain sequential agent and results look plausible — the only way to detect the bug is checking trace node 0 for the expected init structure.

Condition semantics:
- `one_shot`: single LLM call (text-only). Use any provider/model via `--model` (including Codex-backed single-call models).
  - `reasoning_effort` currently applies to `openai-tc`, `codex-tc`, `anthropic-pf`, and `anthropic-tc`.
- `codex`: Codex CLI harness (agent loop). Default model is `codex` via `--codex-model codex`.
  - In the harness adapter, `codex` resolves to `gpt-5-codex` for stable Codex CLI compatibility.
- `claude_code`: Claude Code CLI harness. `--reasoning-effort` maps to the CLI's `--effort` flag.
- `swe_agent`: mini-swe-agent comparator. It does not currently expose a benchmark-wired reasoning-effort control, so records should leave this field empty.
- `spell` (including SWE-bench workspaces): adapter injects local `-Sdeps` so `spell.benchmark-api` resolves even when `cwd` is a cloned repo workspace.
  - `reasoning_effort` currently applies to `openai-tc`, `codex-tc`, `anthropic-pf`, and `anthropic-tc`.

Spell `--model` specs (provider prefix + model name from `.provider.edn`):

| Provider | Model spec | Default model |
|----------|-----------|---------------|
| Anthropic (toolcall) | `anthropic-tc:claude-opus-4-6` | `claude-opus-4-6` |
| Anthropic (prefill) | `anthropic-pf:claude-opus-4-6` | `claude-opus-4-6` |
| Codex (toolcall) | `codex-tc:gpt-5.3-codex` | `gpt-5.3-codex` |

Model names must match the provider's API exactly. The provider `.edn` files in `config/providers/` define defaults.

### Common flags (Exercism)

```bash
--slug hello-world    # Single exercise
--difficulty 5-9      # Difficulty range
--limit 10            # Max exercises
--model sonnet        # Model shorthand
```

## One-Shot Results

Most recent results for comparison. Check notebook entries for full details.

### Math

| Benchmark | Spell | CC | One-Shot | Entry |
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
├── src/
│   ├── datasets.py        # Dataset loading
│   ├── docker_agents.py   # Shared in-container agent execution
│   ├── swebench_container.py # Official-image SWE-bench container backend
│   ├── unified_adapters.py # Condition adapters (Spell/OneShot/Claude/Codex)
│   ├── unified_results.py  # Unified JSONL schema + summaries
│   ├── evaluation.py      # Scoring (exact match, regex extraction)
│   └── swebench_output.py # Results formatting
├── results/               # JSON output files (gitignored)
├── traces/                # Execution traces (gitignored)
├── logs/                  # Evaluation/run logs (gitignored)
├── scratch/               # One-off analysis/result batches (gitignored)
├── docs/one-off-organization.md # One-off organization workflow
└── data/                  # Cached datasets (gitignored)
```

Use `uv run` instead of `python3` directly.

Prompt layering rules for the Python harness:
- Put benchmark-specific prompt text in `src/datasets.py`, shared across conditions whenever the transport permits it.
- Do not add method-specific capability text such as Python or web-search hints to benchmark prompts.
- Put Spell-only orchestration behavior in `bench.py` as a trailing policy that feeds `build_spell_init(prompt, trailing)`.

### Terminal-Bench harness (`terminal_bench/`)

```
terminal_bench/
├── base.py                # DockerExecAgent base: install-in-docker + token tracking
├── spell_agent.py         # SpellAgent: installs Spell runtime + runs benchmark API in Docker
├── codex_agent.py         # CodexAgent: Codex CLI in Docker, --json JSONL parsing
├── cc_agent.py            # ClaudeCodeAgent: CC in Docker, stream-json parsing
└── __init__.py
```

Requires `tb` CLI (`uv tool install terminal-bench`) and Docker. Tasks cached at `~/.cache/terminal-bench/`.

**Direct agent invocations (for one-off debugging):**
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
# Requires CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`; does not use ANTHROPIC_API_KEY.
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

**Non-software default output format (Spell):** The Spell benchmark adapter sends `format={"required":["result"]}` by default for non-software benchmarks (`general`/math/long-context). Software benchmarks (`swebench`, `featurebench`, `scienceagentbench`, `exercism`) do not get this default. Override per request with `RunRequest.format_spec`.

**Result-map unwrapping:** When format-constrained runs return maps like `{:result ...}`, `spell.benchmark-api` unwraps that value before writing JSON `result`, so existing answer extractors continue to consume plain answer text.

**SWE-bench evaluation:** `bench.py swebench` always runs official harness evaluation after prediction generation. For eval-only runs, use `bench.py swebench-eval`.

**SWE-bench Docker socket on Colima:** evaluation uses Docker SDK (`docker.from_env`), which depends on `DOCKER_HOST` instead of Docker CLI context. `bench.py` now auto-falls back to `unix://~/.colima/default/docker.sock` when `DOCKER_HOST` is unset and `/var/run/docker.sock` is missing.

**SWE-bench disk cleanup:** generation now removes the per-item `sweb.eval.*` image and stopped container automatically after each run. Shared `sweb.env.*` images remain cached. Pass `--keep-images` when you need to inspect a specific instance image after generation.

**SWE-bench ARM reality check:** x86_64 eval images can run on Apple Silicon via emulation but are slower and may be less stable on heavy builds. Prefer small pilots first, then scale with caching.

**(stored ...) problem:** Values too large to inline in the completion become opaque `(stored ...)` refs that the LLM can't read back. Affects ~15% of LongBench items. See `longbench-regression-v2`.

**Zero orchestration usage:** Current benchmarks mostly test sequential tool use, not multi-agent orchestration. Spell's advantage comes from code-first semantics (clean return values, deterministic computation) rather than spawn/ask/plet.

**Think/rethink impact:** Adds ~12pp on olympiad math. Requires structural prefix forcing (opening the completion with `(think "approach" ...)`) — prompt instructions alone have zero effect. See `omni-math-manual-scoring`.

**ScienceAgentBench program extraction:** Claude Code writes .py files directly to the workspace; Spell sometimes inlines Python in `io/run-command` calls without writing a file. The harness checks workspace for .py files first, then falls back to extracting from markdown code blocks. Spell's extraction rate is lower (20% vs 100% in pilot) because of this difference.

**ScienceAgentBench full evaluation:** Requires downloading the benchmark zip from SharePoint (password: `scienceagentbench`), which needs a browser. The zip contains `datasets/`, `eval_programs/`, `gold_programs/`, `scoring_rubrics/`. Docker-based evaluation then runs each program and checks output correctness. Without the zip, only syntax validity can be checked.

**Terminal-Bench Spell runtime:** SpellAgent now executes inside the task container and copies a minimal Spell runtime bundle (`deps.edn`, `src/`, `config/`) to `/opt/spell` during install. This avoids host-vs-container filesystem asymmetry during scoring.

**Terminal-Bench disk exhaustion:** Each task builds a Docker image. Running many tasks without cleanup fills Colima's VM disk. `bench.py terminalbench --cleanup` threads `tb` cleanup through each run and also prunes Docker between conditions. Never run more than 2 tasks concurrently on Colima with default disk (20GB).

**Terminal-Bench empty tool-call error:** Opus 4.6 occasionally returns an empty `spell_suffix` tool_use (`input: {}`), causing a fatal `missing-tool-call` error. This is not currently retried (see TODO #144). Observed on write-compressor.

**Omni-MATH `missing-tool-call` prevalence:** On Omni-MATH with Opus 4.6 toolcall transport, `missing-tool-call` hits 30% of items (18/60). Much higher than Terminal-Bench. Making this retryable (TODO #144) is critical for math benchmarks on toolcall transport. See `omni-math-rerun`.

**Omni-MATH auto-scorer recall:** `math_equivalent` + `extract_math_answer` has <50% recall on Omni-MATH. Misses: Unicode math symbols (Codex outputs ⌈⌉/√/⟂), yes/no text answers, set ordering, equivalent algebraic expressions, descriptive answers. Manual scoring required for accurate results. *All Omni-MATH Spell/CC/One-Shot entries use manual scores. See `omni-math-rerun`.

**Omni-MATH one-shot footnotes:** \*Spell/Opus 33.3% includes 19 fatal `missing-tool-call` errors; ex-error accuracy is 48.8% (20/41). †Codex CLI = Codex condition (Codex CLI, not Claude Code). ‡CC/Opus AIME 2026 auto-scored 93.3% but item 7 was a format false negative (`**396**` bold markdown); corrected to 96.7%.
