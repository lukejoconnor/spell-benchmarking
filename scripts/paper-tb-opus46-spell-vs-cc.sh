#!/usr/bin/env bash
# Run the Opus 4.6 Spell and Claude Code Terminal-Bench comparison.
# Optional GCP dispatch; see paper-repro/README.md for local commands and setup.

set -euo pipefail

SUFFIX="${1:-$(date -u +%Y%m%d)}"
RUN_GROUP="paper-tb-opus46-${SUFFIX}"
SPELL_REF="${SPELL_REF:-v0.1.0}"
BENCH_REF="${BENCH_REF:-paper}"
SPELL_MODEL="anthropic-tc:claude-opus-4-6"
CC_MODEL="claude-opus-4-6"
DATASET="old-core"
EFFORT="medium"
N_CONCURRENT=4
TEST_TIMEOUT_SEC=600
MACHINE_TYPE="${SPELL_GCP_MACHINE_TYPE:-e2-standard-8}"
NETWORK="${SPELL_GCP_NETWORK:-spell-benchmark-vpc}"

cd "$(dirname "$0")/.."

# Spell shard: default trailing expression ('(!extend)) — do not pass --trailing.
SPELL_VM="paper-tb-opus46-spell"
SPELL_RUN_NAME="${SPELL_VM}"
SPELL_COMMAND="uv run python bench.py terminalbench \
  --dataset ${DATASET} \
  --condition spell \
  --model ${SPELL_MODEL} \
  --reasoning-effort ${EFFORT} \
  --n-concurrent ${N_CONCURRENT} \
  --test-timeout-sec ${TEST_TIMEOUT_SEC} \
  --name ${SPELL_RUN_NAME}"

echo "==> Launching ${SPELL_VM}"
"${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}/scripts/gcp-benchmark.sh" run \
  --name "${SPELL_VM}" \
  --run-group "${RUN_GROUP}" \
  --machine-type "${MACHINE_TYPE}" \
  --network "${NETWORK}" \
  --spell-ref "${SPELL_REF}" \
  --benchmarking-ref "${BENCH_REF}" \
  --command "${SPELL_COMMAND}"

# Claude Code shard: routes through OAUTH token populated by gcp-startup.sh
# from Secret Manager `CLAUDE_CODE_OAUTH_TOKEN`.
CC_VM="paper-tb-opus46-cc"
CC_RUN_NAME="${CC_VM}"
CC_COMMAND="uv run python bench.py terminalbench \
  --dataset ${DATASET} \
  --condition claude_code \
  --claude-model ${CC_MODEL} \
  --reasoning-effort ${EFFORT} \
  --n-concurrent ${N_CONCURRENT} \
  --test-timeout-sec ${TEST_TIMEOUT_SEC} \
  --name ${CC_RUN_NAME}"

echo "==> Launching ${CC_VM}"
"${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}/scripts/gcp-benchmark.sh" run \
  --name "${CC_VM}" \
  --run-group "${RUN_GROUP}" \
  --machine-type "${MACHINE_TYPE}" \
  --network "${NETWORK}" \
  --spell-ref "${SPELL_REF}" \
  --benchmarking-ref "${BENCH_REF}" \
  --command "${CC_COMMAND}"

echo
echo "Run group: ${RUN_GROUP}"
echo "Monitor:   "${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}/scripts/gcp-benchmark.sh" wait --run-group ${RUN_GROUP} --finish"
