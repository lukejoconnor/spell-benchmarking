#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
BENCHMARKING_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
SPELL_ROOT="${SPELL_ROOT:-$(cd -- "$BENCHMARKING_ROOT/.." >/dev/null 2>&1 && pwd)}"
GCP_LAUNCHER="$SPELL_ROOT/scripts/gcp-benchmark.sh"

DATASET="${DATASET:-longbench}"
MODEL="${MODEL:-openai-tc:gpt-5.4}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.4}"
SPELL_TRAILING="${SPELL_TRAILING:-}"
if [[ -z "$SPELL_TRAILING" ]]; then
  SPELL_TRAILING="'(!extend)"
fi
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
PARALLEL="${PARALLEL:-4}"
PILOT_N="${PILOT_N:-4}"
PAPER_N="${PAPER_N:-200}"
REST_START="${REST_START:-4}"
SPELL_REF="${SPELL_REF:-v0.1.0}"
BENCHMARKING_REF="${BENCHMARKING_REF:-paper}"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
RUN_GROUP="${RUN_GROUP:-longbench-paper-gpt54-med-simple-trailing-${DATE_TAG}}"
VM_PREFIX="${VM_PREFIX:-lbp54m-simple-${DATE_TAG}}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
DISK_SIZE_GB="${DISK_SIZE_GB:-100}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-168h}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_longbench_paper_gcp.sh pilot
  scripts/run_longbench_paper_gcp.sh pilot-spell
  scripts/run_longbench_paper_gcp.sh print-pilot
  scripts/run_longbench_paper_gcp.sh print-pilot-spell
  scripts/run_longbench_paper_gcp.sh wait-pilot
  scripts/run_longbench_paper_gcp.sh rest
  scripts/run_longbench_paper_gcp.sh rest-spell
  scripts/run_longbench_paper_gcp.sh print-rest
  scripts/run_longbench_paper_gcp.sh print-rest-spell
  scripts/run_longbench_paper_gcp.sh wait-rest
  scripts/run_longbench_paper_gcp.sh status
  scripts/run_longbench_paper_gcp.sh pull
  scripts/run_longbench_paper_gcp.sh finish

Runs the LongBench v2 Spell vs Codex GPT-5.4/medium paper replication on GCP.
Spell defaults to v0.1.0; benchmarking defaults to paper. The pilot runs the first 4 items.
The rest phase runs items 5-200 from the same ordered LongBench v2 subset used
for the paper-facing 200-item result.
Use pilot-spell/rest-spell for the simple-trailing Spell-only paper rerun.

Useful overrides:
  RUN_GROUP=... VM_PREFIX=... PARALLEL=4 SPELL_REF=v0.1.0 BENCHMARKING_REF=paper
  MODEL=openai-tc:gpt-5.4 CODEX_MODEL=gpt-5.4 SPELL_TRAILING="'(!extend)"
  MACHINE_TYPE=e2-standard-4
EOF
}

die() {
  printf '[longbench-paper-gcp] error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '[longbench-paper-gcp] %s\n' "$*"
}

shell_quote() {
  printf '%q' "$1"
}

require_launcher() {
  [[ -x "$GCP_LAUNCHER" ]] || die "missing executable GCP launcher at $GCP_LAUNCHER"
}

longbench_rest_items_prefix() {
  local item_expr
  item_expr=$(
    cat <<PY
from src.datasets import DATASETS
items = DATASETS["${DATASET}"]()
print(",".join(item.id for item in items[${REST_START}:${PAPER_N}]))
PY
  )
  printf 'ITEMS=$(uv run python -c %s) && ' "$(shell_quote "$item_expr")"
}

bench_command() {
  local condition="$1"
  local phase="$2"
  local name="$3"
  local -a args

  args=(uv run python bench.py general
    --dataset "$DATASET"
    --condition "$condition"
    --reasoning-effort "$REASONING_EFFORT"
    --parallel "$PARALLEL"
    --name "$name")

  case "$condition" in
    spell)
      args+=(--model "$MODEL"
        --agent config/agents/io-tc.agent.edn
        --trailing "$SPELL_TRAILING"
        --timeout 1200
        --budget 3)
      ;;
    codex)
      args+=(--codex-model "$CODEX_MODEL")
      ;;
    *)
      die "unsupported condition: $condition"
      ;;
  esac

  case "$phase" in
    pilot)
      args+=(--n "$PILOT_N")
      ;;
    rest)
      ;;
    *)
      die "unsupported phase: $phase"
      ;;
  esac

  local rendered=""
  local arg
  for arg in "${args[@]}"; do
    rendered+=" $(shell_quote "$arg")"
  done
  rendered="${rendered# }"

  if [[ "$phase" == "rest" ]]; then
    rendered+=' --items "$ITEMS"'
    printf '%s%s' "$(longbench_rest_items_prefix)" "$rendered"
  else
    printf '%s' "$rendered"
  fi
}

dispatch_one() {
  local condition="$1"
  local phase="$2"
  local vm_name="${VM_PREFIX}-${phase}-${condition}"
  local run_name="lb-paper-${phase}-${condition}"
  local command
  local -a gcp_args
  command="$(bench_command "$condition" "$phase" "$run_name")"
  gcp_args=(
    --run-group "$RUN_GROUP"
    --spell-ref "$SPELL_REF"
    --benchmarking-ref "$BENCHMARKING_REF"
    --machine-type "$MACHINE_TYPE"
    --disk-size-gb "$DISK_SIZE_GB"
    --max-run-duration "$MAX_RUN_DURATION"
  )

  log "dispatching ${condition}/${phase} on ${vm_name}"
  "$GCP_LAUNCHER" run \
    --name "$vm_name" \
    "${gcp_args[@]}" \
    --command "$command"
}

wait_group() {
  "$GCP_LAUNCHER" wait --run-group "$RUN_GROUP" --finish
}

case "${1:-}" in
  pilot)
    require_launcher
    dispatch_one spell pilot
    dispatch_one codex pilot
    ;;
  pilot-spell)
    require_launcher
    dispatch_one spell pilot
    ;;
  print-pilot)
    printf 'spell pilot:\n%s\n\n' "$(bench_command spell pilot lb-paper-pilot-spell)"
    printf 'codex pilot:\n%s\n' "$(bench_command codex pilot lb-paper-pilot-codex)"
    ;;
  print-pilot-spell)
    printf 'spell pilot:\n%s\n' "$(bench_command spell pilot lb-paper-pilot-spell)"
    ;;
  wait-pilot)
    require_launcher
    wait_group
    ;;
  rest)
    require_launcher
    dispatch_one spell rest
    dispatch_one codex rest
    ;;
  rest-spell)
    require_launcher
    dispatch_one spell rest
    ;;
  print-rest)
    printf 'spell rest:\n%s\n\n' "$(bench_command spell rest lb-paper-rest-spell)"
    printf 'codex rest:\n%s\n' "$(bench_command codex rest lb-paper-rest-codex)"
    ;;
  print-rest-spell)
    printf 'spell rest:\n%s\n' "$(bench_command spell rest lb-paper-rest-spell)"
    ;;
  wait-rest)
    require_launcher
    wait_group
    ;;
  status)
    require_launcher
    "$GCP_LAUNCHER" status-all --run-group "$RUN_GROUP"
    ;;
  pull)
    require_launcher
    "$GCP_LAUNCHER" pull-all --run-group "$RUN_GROUP"
    ;;
  finish)
    require_launcher
    "$GCP_LAUNCHER" finish-all --run-group "$RUN_GROUP"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
