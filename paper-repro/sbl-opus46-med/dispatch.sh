#!/usr/bin/env bash
# SWE-bench Lite paper-branch run: Spell vs Claude Code on Opus 4.6 / medium effort.
#
# Conditions: spell × medium and claude_code × medium, model claude-opus-4-6.
# Sharding: 3 shards × 100 items, reusing the deterministic split at
# paper-repro/sbl-shards/shard{1,2,3}.txt (sorted instance_ids, same shards
# used for the GPT-5.4 Figure 3 SBL run).
#
# Each VM runs --parallel 2 with a per-item timeout of 3600s. Total: 6 VMs
# (2 conditions × 3 shards). VMs use Spell paper branch + Benchmarking paper
# branch. Apples-to-apples settings vs the GPT-5.4 SBL Fig3 dispatch except
# parallel=2 (vs 4) per user spec.
#
# Claude Code routes through the Max-subscription OAUTH token from Secret
# Manager (`CLAUDE_CODE_OAUTH_TOKEN`); the GCP startup script populates it
# in the VM environment. Cost on CC timeout rows is recovered from
# per-message stream usage by benchmarking commit 0b074fa (PR #68).
#
# Usage:
#   ./dispatch.sh pilot                 # 2-item dry-run, spell-med + cc-med
#   ./dispatch.sh full                  # full 6-VM sweep (3 shards × 2 conds)
#   ./dispatch.sh full-cond spell       # one condition's 3 shards
#   ./dispatch.sh full-cond claude_code
#
# Monitoring after dispatch:
#   ./scripts/gcp-benchmark.sh wait --run-group paper-sbl-opus46-med --finish
#   ./scripts/gcp-benchmark.sh pull-all --run-group paper-sbl-opus46-med

set -euo pipefail

SPELL_REPO="${SPELL_REPO:-${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}}"
GCP="$SPELL_REPO/scripts/gcp-benchmark.sh"
SHARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sbl-shards" && pwd)"

SPELL_REF="${SPELL_REF:-v0.1.0}"
BENCH_REF="${BENCH_REF:-paper}"
RUN_GROUP="${RUN_GROUP:-paper-sbl-opus46-med}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-16}"
BOOT_DISK="${BOOT_DISK:-300}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-3600}"
EFFORT="${EFFORT:-medium}"
SPELL_MODEL="${SPELL_MODEL:-anthropic-tc:claude-opus-4-6}"
CC_MODEL="${CC_MODEL:-claude-opus-4-6}"
ENV_CACHE="${ENV_CACHE:-}"  # Optional Artifact Registry repo for env image cache

shard_csv() {
  local shard="$1"
  paste -sd, "$SHARD_DIR/shard${shard}.txt"
}

cache_flag() {
  if [[ -n "$ENV_CACHE" ]]; then
    echo "--env-image-cache-repository ${ENV_CACHE}"
  fi
}

dispatch_spell() {
  local shard="$1"
  local items vm_name run_name
  items="$(shard_csv "$shard")"
  vm_name="sbl-opus46-spell-med-s${shard}"
  run_name="paper-sbl-opus46-spell-med-s${shard}"
  echo "==> Launching $vm_name"
  "$GCP" run \
    --name "$vm_name" \
    --run-group "$RUN_GROUP" \
    --machine-type "$MACHINE_TYPE" \
    --disk-size-gb "$BOOT_DISK" \
    --spell-ref "$SPELL_REF" \
    --benchmarking-ref "$BENCH_REF" \
    --command "uv run python bench.py swebench \
      --dataset lite --condition spell --model ${SPELL_MODEL} \
      --reasoning-effort ${EFFORT} \
      --items '${items}' \
      --parallel ${PARALLEL} --timeout ${TIMEOUT} \
      --prewarm-envs --paper-compliant \
      $(cache_flag) \
      --name ${run_name}"
}

dispatch_cc() {
  local shard="$1"
  local items vm_name run_name
  items="$(shard_csv "$shard")"
  vm_name="sbl-opus46-cc-med-s${shard}"
  run_name="paper-sbl-opus46-cc-med-s${shard}"
  echo "==> Launching $vm_name"
  "$GCP" run \
    --name "$vm_name" \
    --run-group "$RUN_GROUP" \
    --machine-type "$MACHINE_TYPE" \
    --disk-size-gb "$BOOT_DISK" \
    --spell-ref "$SPELL_REF" \
    --benchmarking-ref "$BENCH_REF" \
    --command "uv run python bench.py swebench \
      --dataset lite --condition claude_code --claude-model ${CC_MODEL} \
      --reasoning-effort ${EFFORT} \
      --items '${items}' \
      --parallel ${PARALLEL} --timeout ${TIMEOUT} \
      --prewarm-envs --paper-compliant \
      $(cache_flag) \
      --name ${run_name}"
}

cmd_pilot() {
  # 2-item smoke: head of shard1, spell + claude_code, medium effort.
  local pilot_items
  pilot_items="$(head -n 2 "$SHARD_DIR/shard1.txt" | paste -sd, -)"

  for cond in spell claude_code; do
    local vm_name="pilot-sbl-opus46-${cond//_/-}-med"
    local run_name="pilot-sbl-opus46-${cond//_/-}-med"
    local cli="--condition ${cond}"
    if [[ "$cond" == spell ]]; then
      cli+=" --model ${SPELL_MODEL}"
    else
      cli+=" --claude-model ${CC_MODEL}"
    fi
    echo "==> Pilot $vm_name"
    "$GCP" run \
      --name "$vm_name" \
      --run-group "${RUN_GROUP}-pilot" \
      --machine-type "$MACHINE_TYPE" \
      --disk-size-gb "$BOOT_DISK" \
      --spell-ref "$SPELL_REF" \
      --benchmarking-ref "$BENCH_REF" \
      --command "uv run python bench.py swebench \
        --dataset lite ${cli} \
        --reasoning-effort ${EFFORT} \
        --items '${pilot_items}' \
        --parallel ${PARALLEL} --timeout ${TIMEOUT} \
        --prewarm-envs --paper-compliant \
        $(cache_flag) \
        --name ${run_name}"
  done
}

cmd_full() {
  for shard in 1 2 3; do
    dispatch_spell "$shard"
    dispatch_cc "$shard"
  done
}

cmd_full_cond() {
  local cond="$1"
  for shard in 1 2 3; do
    if [[ "$cond" == spell ]]; then dispatch_spell "$shard"
    elif [[ "$cond" == claude_code || "$cond" == cc ]]; then dispatch_cc "$shard"
    else
      echo "Unknown condition: $cond (expected spell|claude_code)" >&2
      exit 2
    fi
  done
}

case "${1:-}" in
  pilot)      cmd_pilot ;;
  full)       cmd_full ;;
  full-cond)  cmd_full_cond "$2" ;;
  *)
    echo "Usage: $0 {pilot|full|full-cond <spell|claude_code>}" >&2
    exit 2
    ;;
esac
