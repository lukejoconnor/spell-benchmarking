#!/usr/bin/env bash
# Paper Observation 5 AppWorld dispatch: Spell vs Codex CLI, GPT-5.4 medium.
#
# Dataset: Terminal-Bench adapter `appworld-dev`, explicit 57-item task list in
# appworld-dev-items.txt. Pilot uses the original 4 AppWorld smoke items from
# notebook entry `2026-04-15-appworld-pilot-tb-adapter`.
#
# Conditions:
#   Spell: openai-tc:gpt-5.4, config/agents/io-tc.agent.edn, medium effort,
#          trailing `'(!extend)`.
#   Codex: Codex CLI gpt-5.4, medium effort.
#
# Usage from benchmarking/:
#   paper-repro/appworld-obs5/dispatch.sh pilot
#   paper-repro/appworld-obs5/dispatch.sh pilot-spell
#   paper-repro/appworld-obs5/dispatch.sh full
#   paper-repro/appworld-obs5/dispatch.sh full-spell
#   paper-repro/appworld-obs5/dispatch.sh full-cond spell
#   paper-repro/appworld-obs5/dispatch.sh full-cond codex
#
# Monitor:
#   ../scripts/gcp-benchmark.sh wait --run-group <group> --finish

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BENCH_ROOT="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
SPELL_REPO="${SPELL_REPO:-$(cd -- "$BENCH_ROOT/.." >/dev/null 2>&1 && pwd)}"
GCP="${GCP:-$SPELL_REPO/scripts/gcp-benchmark.sh}"

SPELL_REF="${SPELL_REF:-paper}"
BENCH_REF="${BENCH_REF:-paper}"
RUN_GROUP_OVERRIDE="${RUN_GROUP:-}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-16}"
BOOT_DISK="${BOOT_DISK:-200}"
NETWORK="${SPELL_GCP_NETWORK:-spell-benchmark-vpc}"
N_CONCURRENT="${N_CONCURRENT:-4}"
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-600}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-3600}"
SPELL_MODEL="${SPELL_MODEL:-openai-tc:gpt-5.4}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"
EFFORT="${EFFORT:-medium}"
DATASET="${DATASET:-appworld-dev}"
SPELL_TRAILING="${SPELL_TRAILING:-}"
if [[ -z "$SPELL_TRAILING" ]]; then
  SPELL_TRAILING="'(!extend)"
fi
SPELL_AGENT="${SPELL_AGENT:-config/agents/io-tc.agent.edn}"

task_flags() {
  local item_file="$1"
  while IFS= read -r item; do
    [[ -z "$item" || "$item" =~ ^# ]] && continue
    printf ' -t %q' "$item"
  done < "$item_file"
}

bench_command() {
  local condition="$1" item_file="$2" run_name="$3"
  local items
  items="$(task_flags "$item_file")"

  if [[ "$condition" == "spell" ]]; then
    printf 'uv run python bench.py terminalbench --dataset %q --condition spell --model %q --agent %q --trailing %q --reasoning-effort %q --n-concurrent %q --agent-timeout-sec %q --test-timeout-sec %q --name %q%s' \
      "$DATASET" "$SPELL_MODEL" "$SPELL_AGENT" "$SPELL_TRAILING" "$EFFORT" "$N_CONCURRENT" "$AGENT_TIMEOUT_SEC" "$TEST_TIMEOUT_SEC" "$run_name" "$items"
  elif [[ "$condition" == "codex" ]]; then
    printf 'uv run python bench.py terminalbench --dataset %q --condition codex --codex-model %q --reasoning-effort %q --n-concurrent %q --agent-timeout-sec %q --test-timeout-sec %q --name %q%s' \
      "$DATASET" "$CODEX_MODEL" "$EFFORT" "$N_CONCURRENT" "$AGENT_TIMEOUT_SEC" "$TEST_TIMEOUT_SEC" "$run_name" "$items"
  else
    echo "unknown condition: $condition" >&2
    return 2
  fi
}

dispatch_condition() {
  local condition="$1" item_file="$2" label="$3"
  local vm_name run_name command
  vm_name="appworld-${label}-${condition}"
  run_name="paper-appworld-${label}-${condition}-gpt54-med"
  command="$(bench_command "$condition" "$item_file" "$run_name")"

  echo "==> Launching ${vm_name}"
  "$GCP" run \
    --name "$vm_name" \
    --run-group "$RUN_GROUP" \
    --machine-type "$MACHINE_TYPE" \
    --disk-size-gb "$BOOT_DISK" \
    --network "$NETWORK" \
    --spell-ref "$SPELL_REF" \
    --benchmarking-ref "$BENCH_REF" \
    --command "$command"
}

cmd_pilot() {
  RUN_GROUP="${RUN_GROUP_OVERRIDE:-paper-appworld-gpt54-med-simple-trailing-pilot-$(date -u +%Y%m%d)}"
  dispatch_condition spell "$SCRIPT_DIR/pilot-items.txt" pilot
  dispatch_condition codex "$SCRIPT_DIR/pilot-items.txt" pilot
  echo
  echo "Run group: ${RUN_GROUP}"
  echo "Monitor:   ${GCP} wait --run-group ${RUN_GROUP} --finish"
}

cmd_pilot_spell() {
  RUN_GROUP="${RUN_GROUP_OVERRIDE:-paper-appworld-gpt54-med-simple-trailing-pilot-$(date -u +%Y%m%d)}"
  dispatch_condition spell "$SCRIPT_DIR/pilot-items.txt" pilot
  echo
  echo "Run group: ${RUN_GROUP}"
  echo "Monitor:   ${GCP} wait --run-group ${RUN_GROUP} --finish"
}

cmd_full() {
  RUN_GROUP="${RUN_GROUP_OVERRIDE:-paper-appworld-gpt54-med-simple-trailing-full-$(date -u +%Y%m%d)}"
  dispatch_condition spell "$SCRIPT_DIR/appworld-dev-items.txt" full
  dispatch_condition codex "$SCRIPT_DIR/appworld-dev-items.txt" full
  echo
  echo "Run group: ${RUN_GROUP}"
  echo "Monitor:   ${GCP} wait --run-group ${RUN_GROUP} --finish"
}

cmd_full_cond() {
  local condition="$1"
  RUN_GROUP="${RUN_GROUP_OVERRIDE:-paper-appworld-gpt54-med-simple-trailing-full-$(date -u +%Y%m%d)}"
  dispatch_condition "$condition" "$SCRIPT_DIR/appworld-dev-items.txt" full
  echo
  echo "Run group: ${RUN_GROUP}"
  echo "Monitor:   ${GCP} wait --run-group ${RUN_GROUP} --finish"
}

case "${1:-}" in
  pilot)
    cmd_pilot
    ;;
  pilot-spell)
    cmd_pilot_spell
    ;;
  full)
    cmd_full
    ;;
  full-spell)
    cmd_full_cond spell
    ;;
  full-cond)
    [[ $# -eq 2 ]] || { echo "Usage: $0 full-cond <spell|codex>" >&2; exit 2; }
    cmd_full_cond "$2"
    ;;
  *)
    echo "Usage: $0 {pilot|pilot-spell|full|full-spell|full-cond <spell|codex>}" >&2
    exit 2
    ;;
esac
