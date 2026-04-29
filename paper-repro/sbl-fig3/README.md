# Paper Figure 3 — SWE-bench Lite Replication

Reproducibility artifacts for the Figure 3 SWE-bench Lite full-300 reruns.

## Conditions

- 6 method × effort cells:
  - Spell × {low, medium, high} (Spell paper branch, `openai-tc:gpt-5.4`)
  - Codex CLI × {low, medium, high} (`--codex-model gpt-5.4`)
- Sharding: 3 shards × 100 SWE-bench Lite items, deterministic split of the
  alphabetically sorted official SBL instance_ids (see `../sbl-shards/`).
- Per-VM concurrency: `--parallel 4`, per-item timeout 3600 s.
- SWE-bench harness: `--prewarm-envs --paper-compliant` (no hints text in prompt).
- Total: 18 GCP VMs.

## Repro

```bash
cd $SPELL_REPO/benchmarking/paper-repro/sbl-fig3
./dispatch.sh full          # 18 VMs

# monitor
$SPELL_REPO/scripts/gcp-benchmark.sh wait --run-group paper-fig3-sbl --finish
```

VM names: `sbl-{spell,codex}-{low,medium,high}-s{1,2,3}`.

Run names (`--name` flag inside bench.py): `fig3-sbl-{spell,codex}-gpt54-{low,medium,high}-s{1,2,3}`.

Run group: `paper-fig3-sbl` (pilot uses `paper-fig3-sbl-pilot`).

## Refs

- Spell branch: `paper` (head `ae74e5a` at dispatch time).
- Benchmarking branch: `paper` (head `2ef2551` at dispatch time).

Both branches must be pushed to origin before launching, since disposable VMs
clone them at startup.

## Scoring

Score from each VM's per-run `same_container_resolved` field in the unified
JSONL written under `results/swebench/<run-name>.jsonl`. Costs come from the
`cost_usd` field in the unified JSONL (do NOT recompute from raw token fields).
