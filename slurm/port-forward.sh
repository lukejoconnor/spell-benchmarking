#!/bin/bash
# Port-forward vLLM and search server from O2 to localhost.
# Usage: VLLM_JOB=<jobid> SEARCH_JOB=<jobid> O2_USER=<username> ./port-forward.sh

set -euo pipefail

O2_USER="${O2_USER:?Set O2_USER to your O2 username}"
VLLM_JOB="${VLLM_JOB:?Set VLLM_JOB to the SLURM job ID of the vLLM job}"
SEARCH_JOB="${SEARCH_JOB:?Set SEARCH_JOB to the SLURM job ID of the search server job}"
O2_HOST="${O2_HOST:-o2.hms.harvard.edu}"

VLLM_NODE=$(ssh "${O2_USER}@${O2_HOST}" "squeue -j ${VLLM_JOB} -h -o %N")
SEARCH_NODE=$(ssh "${O2_USER}@${O2_HOST}" "squeue -j ${SEARCH_JOB} -h -o %N")

echo "vLLM node:    ${VLLM_NODE}"
echo "Search node:  ${SEARCH_NODE}"
echo "Forwarding:   localhost:8001 -> ${VLLM_NODE}:8001"
echo "              localhost:8000 -> ${SEARCH_NODE}:8000"
echo ""
echo "Press Ctrl-C to stop forwarding."

ssh -N \
  -L "8001:${VLLM_NODE}:8001" \
  -L "8000:${SEARCH_NODE}:8000" \
  "${O2_USER}@${O2_HOST}"
