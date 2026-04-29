#!/usr/bin/env bash
# Reproducibility dispatch for paper Figure 3 — SWE-bench Lite full 300.
#
# Conditions: spell × {low, med, high} and codex CLI × {low, med, high}, model gpt-5.4.
# Sharding: 3 shards × 100 items (deterministic split of sorted instance_ids;
# shard files live at paper-repro/sbl-shards/shard{1,2,3}.txt).
#
# Each VM runs --parallel 4 with a per-item timeout of 3600s. Total: 18 VMs
# (6 conditions × 3 shards). VMs use Spell paper branch + Benchmarking paper
# branch; both pushed to origin so disposable VMs can clone them.
#
# Usage:
#   ./dispatch.sh pilot        # 2-item dry-run, spell-med + codex-med
#   ./dispatch.sh full         # full 18-VM sweep
#   ./dispatch.sh full-cond spell low   # one condition's 3 shards
#
# Monitoring after dispatch:
#   ./scripts/gcp-benchmark.sh wait --run-group paper-fig3-sbl --finish
#   ./scripts/gcp-benchmark.sh pull-all --run-group paper-fig3-sbl

set -euo pipefail

SPELL_REPO="${SPELL_REPO:-$HOME/Dropbox/GitHub/spell}"
GCP="$SPELL_REPO/scripts/gcp-benchmark.sh"
SHARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sbl-shards" && pwd)"

SPELL_REF="${SPELL_REF:-paper}"
BENCH_REF="${BENCH_REF:-paper}"
RUN_GROUP="${RUN_GROUP:-paper-fig3-sbl}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-16}"
BOOT_DISK="${BOOT_DISK:-300}"
PARALLEL="${PARALLEL:-4}"
TIMEOUT="${TIMEOUT:-3600}"
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
  local effort="$1" shard="$2"
  local items vm_name run_name
  items="$(shard_csv "$shard")"
  vm_name="sbl-spell-${effort}-s${shard}"
  run_name="fig3-sbl-spell-gpt54-${effort}-s${shard}"
  echo "==> Launching $vm_name"
  "$GCP" run \
    --name "$vm_name" \
    --run-group "$RUN_GROUP" \
    --machine-type "$MACHINE_TYPE" \
    --disk-size-gb "$BOOT_DISK" \
    --spell-ref "$SPELL_REF" \
    --benchmarking-ref "$BENCH_REF" \
    --command "uv run python bench.py swebench \
      --dataset lite --condition spell --model openai-tc:gpt-5.4 \
      --reasoning-effort ${effort} \
      --items '${items}' \
      --parallel ${PARALLEL} --timeout ${TIMEOUT} \
      --prewarm-envs --paper-compliant \
      $(cache_flag) \
      --name ${run_name}"
}

dispatch_codex() {
  local effort="$1" shard="$2"
  local items vm_name run_name
  items="$(shard_csv "$shard")"
  vm_name="sbl-codex-${effort}-s${shard}"
  run_name="fig3-sbl-codex-gpt54-${effort}-s${shard}"
  echo "==> Launching $vm_name"
  "$GCP" run \
    --name "$vm_name" \
    --run-group "$RUN_GROUP" \
    --machine-type "$MACHINE_TYPE" \
    --disk-size-gb "$BOOT_DISK" \
    --spell-ref "$SPELL_REF" \
    --benchmarking-ref "$BENCH_REF" \
    --command "uv run python bench.py swebench \
      --dataset lite --condition codex --codex-model gpt-5.4 \
      --reasoning-effort ${effort} \
      --items '${items}' \
      --parallel ${PARALLEL} --timeout ${TIMEOUT} \
      --prewarm-envs --paper-compliant \
      $(cache_flag) \
      --name ${run_name}"
}

cmd_pilot() {
  # 2-item smoke: head of shard1, spell + codex, medium effort only.
  local pilot_items
  pilot_items="$(head -n 2 "$SHARD_DIR/shard1.txt" | paste -sd,)"

  for cond in spell codex; do
    local vm_name="pilot-sbl-${cond}-med"
    local run_name="pilot-sbl-${cond}-gpt54-med"
    local cli="--condition ${cond}"
    if [[ "$cond" == spell ]]; then
      cli+=" --model openai-tc:gpt-5.4"
    else
      cli+=" --codex-model gpt-5.4"
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
        --reasoning-effort medium \
        --items '${pilot_items}' \
        --parallel 2 --timeout ${TIMEOUT} \
        --prewarm-envs --paper-compliant \
        $(cache_flag) \
        --name ${run_name}"
  done
}

cmd_full() {
  for effort in low medium high; do
    for shard in 1 2 3; do
      dispatch_spell "$effort" "$shard"
      dispatch_codex "$effort" "$shard"
    done
  done
}

cmd_full_cond() {
  local cond="$1" effort="$2"
  for shard in 1 2 3; do
    if [[ "$cond" == spell ]]; then dispatch_spell "$effort" "$shard"
    else dispatch_codex "$effort" "$shard"
    fi
  done
}

case "${1:-}" in
  pilot)      cmd_pilot ;;
  full)       cmd_full ;;
  full-cond)  cmd_full_cond "$2" "$3" ;;
  *)
    echo "Usage: $0 {pilot|full|full-cond <spell|codex> <low|medium|high>}" >&2
    exit 2
    ;;
esac
