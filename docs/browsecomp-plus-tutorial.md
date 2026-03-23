# Tutorial: Running the BrowseComp-Plus Benchmark

**Date:** 2026-03-23
**Author:** Claude
**User:** ambershen

## Summary

How to run Spell's context-folding agent on the BrowseComp-Plus dataset — both a single test question and the full 150-item pilot. Covers prerequisites, the full pipeline, known gotchas, and how to interpret results.

---

## Background

**BrowseComp-Plus** is an 830-item web browsing benchmark (`Tevatron/browsecomp-plus` on HuggingFace). Questions require multi-hop web search to answer. Fields are XOR-encrypted; the harness decrypts them automatically.

**FoldAgent baseline** (the paper we're replicating): 0.420 pass@1 on 150 items using Seed-OSS-36B + local Qwen3 search server on O2. We run the Spell context-folding implementation as a drop-in.

**Evaluation**: LLM-as-judge (Claude Opus) — the model's final answer is compared to the expected answer by an LLM that returns CORRECT/INCORRECT.

---

## Prerequisites

### 1. Python environment

```bash
cd spell-benchmarking/
uv sync   # first time only
```

### 2. API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The key is already in `~/.zshrc`. Either `source ~/.zshrc` or export manually. Must be set **before** running — the Clojure subprocess inherits it.

### 3. Java (macOS)

Java is required for `clj`. The harness auto-detects it via `/usr/libexec/java_home` and Homebrew fallbacks — no manual setup needed as of commit `d9a1c97`.

If you ever see `"Unable to locate a Java Runtime"`, install via Homebrew:
```bash
brew install openjdk
```

---

## Running a single question

```bash
cd spell-benchmarking/
export ANTHROPIC_API_KEY=sk-ant-...

uv run python bench.py general \
  --dataset browsecomp_plus \
  --condition context_folding \
  --model anthropic-tc:claude-opus-4-6 \
  --budget 5 \
  --n 1 \
  --parallel 1
```

Output goes to `results/unified/<run_id>.jsonl`. Check the result:

```bash
cat results/unified/<run_id>.jsonl | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print('status:', d['status'])
print('correct:', d.get('correct'))
print('response:', d.get('response_text', '')[:300])
print('error:', d.get('error_message', '')[:300])
"
```

---

## Running the 150-item pilot

```bash
uv run python bench.py general \
  --dataset browsecomp_plus \
  --condition context_folding \
  --model anthropic-tc:claude-opus-4-6 \
  --budget 5 \
  --n 150 \
  --parallel 5
```

`--parallel 5` runs 5 items concurrently. Lower if you hit rate limits.

Expected runtime: ~2–4 hours (each item takes 1–10 min depending on branching depth).

---

## Summarizing results

```bash
uv run python bench.py summarize results/unified/<run_id>.jsonl
```

Reports accuracy, error rate, cost, latency. The denominator is always total items (errors count as wrong).

---

## Flags reference

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | — | `browsecomp_plus`, `browsecomp_plus_easy`, `browsecomp_plus_medium`, `browsecomp_plus_hard` |
| `--condition` | — | `context_folding` for this benchmark |
| `--model` | — | `anthropic-tc:claude-opus-4-6` (tool-call transport required) |
| `--budget` | 1.0 | USD per item |
| `--n` | all | Limit item count |
| `--parallel` | 5 | Concurrent items |
| `--timeout` | 300 | Per-item timeout in seconds |
| `--search-url` | — | Local search server URL (for O2 replication with Qwen3 search) |
| `--cf-agent` | — | Override agent .edn file (for Seed-OSS-36B on O2) |
| `--trace` | off | Write execution traces to `traces/<run_id>/` |
| `--name` | — | Label tag in JSONL and trace filenames |

---

## Evaluation

The harness calls `evaluate_browsecomp_judge(item, response_text)` which sends the model's answer and the expected answer to Claude Opus and asks it to return CORRECT or INCORRECT.

The judge prompt compares semantically (not exact string match), so paraphrases count as correct.

---

## Known issues / gotchas

**API overload (529)**: Anthropic Claude Opus can be overloaded during peak hours. The harness logs the error and marks the item as an error (counts as wrong). Retry later, or use `--n 5` for a quick pilot first.

**`ANTHROPIC_API_KEY` not inherited**: The variable must be set in the shell before running `uv run`. It's in `~/.zshrc` but won't be auto-loaded in all contexts. Always `export ANTHROPIC_API_KEY=...` first.

**Difficulty labels**: The dataset (`Tevatron/browsecomp-plus`) doesn't include pre-classified difficulty labels — all 830 items have `difficulty="unknown"`. FoldAgent's 150-item easy/medium/hard split was pre-classified separately and is not reproduced here. Use `--n 150` to match the paper's item count.

**Budget**: Context-folding uses many LLM calls (main loop + per-branch sub-loops). At `--budget 5` per item, Opus runs will cost ~$0.50–$3 per item. Budget 5 gives headroom for complex questions.

**SPL parse errors**: As of commit `996c299`, `context_folding.spl` is correctly balanced. If you see "Unmatched delimiter" after editing the SPL file, check paren balance with:
```bash
python3 -c "
content = open('spell/config/spl-lib/context_folding.spl').read()
# ... (use a paren counter script)
"
```

---

## O2 full replication (advanced)

For the full FoldAgent replication with Seed-OSS-36B + Qwen3 search server:

```bash
uv run python bench.py general \
  --dataset browsecomp_plus \
  --condition context_folding \
  --model "seed-oss-36b:ByteDance-Seed/Seed-OSS-36B-Instruct" \
  --cf-agent ../spell/config/agents/context-folding-seed36b.agent.edn \
  --search-url http://localhost:8000 \
  --budget 5 \
  --n 150 \
  --parallel 5
```

Requires:
- vLLM serving Seed-OSS-36B on port 8001 (O2 SLURM job via `slurm/start-vllm.sh`)
- Qwen3 search server on port 8000 (O2 SLURM job via `slurm/start-search-server.sh`)
- SSH port-forwarding (see `slurm/port-forward.sh`)
- `VLLM_API_KEY=dummy` (vLLM doesn't require a real key)

See `browsecomp-plus-replication` notebook entry for the full O2/SLURM setup.

---

## References

- `browsecomp-plus-replication`: full implementation details (datasets, evaluation, adapters, SLURM scripts)
- `react-context-folding-tutorial`: how to run context-folding standalone (CLI, not via harness)
- FoldAgent paper: arxiv 2510.11967
