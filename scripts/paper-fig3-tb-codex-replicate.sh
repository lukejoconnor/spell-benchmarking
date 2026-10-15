#!/usr/bin/env bash
# Run the three GPT-5.4 Codex CLI Terminal-Bench effort conditions.
# Optional GCP dispatch; see paper-repro/README.md for local commands and setup.

set -euo pipefail

SUFFIX="${1:-$(date -u +%Y%m%d)}"
RUN_GROUP="paper-fig3-tb-codex-${SUFFIX}"
SPELL_REF="${SPELL_REF:-v0.1.0}"
BENCH_REF="${BENCH_REF:-paper}"
CODEX_MODEL="gpt-5.4"
DATASET="old-core"
N_CONCURRENT=4
TEST_TIMEOUT_SEC=600
MACHINE_TYPE="${SPELL_GCP_MACHINE_TYPE:-e2-standard-8}"
NETWORK="${SPELL_GCP_NETWORK:-spell-benchmark-vpc}"

cd "$(dirname "$0")/.."

for EFFORT in low medium high; do
  VM_NAME="paper-fig3-tb-codex-gpt54-${EFFORT}"
  RUN_NAME="paper-fig3-tb-codex-gpt54-${EFFORT}"
  COMMAND="uv run python bench.py terminalbench \
    --dataset ${DATASET} \
    --condition codex \
    --codex-model ${CODEX_MODEL} \
    --reasoning-effort ${EFFORT} \
    --n-concurrent ${N_CONCURRENT} \
    --test-timeout-sec ${TEST_TIMEOUT_SEC} \
    --name ${RUN_NAME}"

  echo "==> Launching ${VM_NAME}"
  "${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}/scripts/gcp-benchmark.sh" run \
    --name "${VM_NAME}" \
    --run-group "${RUN_GROUP}" \
    --machine-type "${MACHINE_TYPE}" \
    --network "${NETWORK}" \
    --spell-ref "${SPELL_REF}" \
    --benchmarking-ref "${BENCH_REF}" \
    --command "${COMMAND}"
done

echo
echo "Run group: ${RUN_GROUP}"
echo "Monitor:   "${SPELL_ROOT:?Set SPELL_ROOT to the Spell v0.1.0 checkout}/scripts/gcp-benchmark.sh" wait --run-group ${RUN_GROUP} --finish"
