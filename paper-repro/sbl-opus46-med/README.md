# SWE-bench Lite — Opus 4.6 / medium — Spell vs Claude Code

Paper-branch dispatch for an Opus 4.6 / medium-effort SWE-bench Lite comparison
between Spell and Claude Code, mirroring the GPT-5.4 Figure 3 SBL setup with
two adjustments: (1) `--parallel 2` instead of 4 per VM, (2) Claude Code
replaces the Codex CLI comparator.

## Configuration

| Setting              | Value                                                    |
|----------------------|----------------------------------------------------------|
| Dataset              | SWE-bench Lite, full 300 (3 shards × 100 items)          |
| Conditions           | `spell`, `claude_code`                                   |
| Model                | `claude-opus-4-6` (Spell via `anthropic-tc:`)            |
| Reasoning effort     | medium                                                   |
| Per-VM parallelism   | `--parallel 2`                                           |
| Per-item timeout     | 3600 s                                                   |
| Compliance           | `--prewarm-envs --paper-compliant`                       |
| VMs                  | 6 (3 shards × 2 conditions)                              |
| Machine type         | `e2-standard-16`, 300 GB boot disk                       |
| Refs                 | spell `v0.1.0`, benchmarking `paper`                      |
| Run group            | `paper-sbl-opus46-med` (pilot: `paper-sbl-opus46-med-pilot`) |

Shards reuse `paper-repro/sbl-shards/shard{1,2,3}.txt` (deterministic split of
sorted instance_ids) so item subsets are identical to the GPT-5.4 SBL Fig3 run.

Claude Code uses the Max-subscription OAUTH token (`CLAUDE_CODE_OAUTH_TOKEN`
secret) populated by `scripts/gcp-startup.sh`. CC-timeout cost recovery is
handled by benchmarking commit `0b074fa` (PR #68).

## Usage

```bash
./dispatch.sh pilot                    # 2 items, both conditions
./dispatch.sh full                     # full 6-VM sweep
./dispatch.sh full-cond spell          # one condition's 3 shards
./dispatch.sh full-cond claude_code
```

Monitor:

```bash
./scripts/gcp-benchmark.sh wait --run-group paper-sbl-opus46-med --finish
./scripts/gcp-benchmark.sh pull-all --run-group paper-sbl-opus46-med
```
