# Paper Figure 3 — SWE-bench Lite Replication

Reproducibility artifacts for the Figure 3 SWE-bench Lite full-300 reruns.

## Conditions

- 6 method × effort cells:
  - Spell × {low, medium, high} (Spell v0.1.0, `openai-tc:gpt-5.4`)
  - Codex CLI × {low, medium, high} (`--codex-model gpt-5.4`)
- Sharding: 3 shards × 100 SWE-bench Lite items, deterministic split of the
  alphabetically sorted official SBL instance_ids (see `../sbl-shards/`).
- Per-VM concurrency: `--parallel 4`, per-item timeout 3600 s.
- SWE-bench harness: `--prewarm-envs --paper-compliant` (no hints text in prompt).
- Total: 18 GCP VMs.

## Repro

```bash
./paper-repro/sbl-fig3/dispatch.sh full          # 18 VMs

# monitor
$SPELL_ROOT/scripts/gcp-benchmark.sh wait --run-group paper-fig3-sbl --finish
```

VM names: `sbl-{spell,codex}-{low,medium,high}-s{1,2,3}`.

Run names (`--name` flag inside bench.py): `fig3-sbl-{spell,codex}-gpt54-{low,medium,high}-s{1,2,3}`.

Run group: `paper-fig3-sbl` (pilot uses `paper-fig3-sbl-pilot`).

## Refs

- Spell reference: `v0.1.0` (`a3ce161a9f83bcb713310d96c26a95ac60978399`).
- Benchmarking branch: `paper`; record its commit before dispatch.

The benchmarking revision must be available on origin before dispatch. Spell v0.1.0 is a published tag. Both refs can be overridden explicitly.

## Scoring

Score from each VM's per-run `same_container_resolved` field in the unified
JSONL written under `results/swebench/<run-name>.jsonl`. Costs come from the
`cost_usd` field in the unified JSONL (do NOT recompute from raw token fields).
