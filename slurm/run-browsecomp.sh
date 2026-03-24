#!/bin/bash
#SBATCH -p gpu_quad
#SBATCH -t 1-00:00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -J browsecomp-cf
#SBATCH -o /n/data1/hms/dbmi/oconnor/lab/amber/tmp/browsecomp-cf-%j.out

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
FOLDAGENT_DIR="/n/data1/hms/dbmi/oconnor/lab/amber/software/FoldAgent"
SPELL_DIR="/n/data1/hms/dbmi/oconnor/lab/amber/spell"
BENCH_DIR="${SPELL_DIR}/benchmarking"
CONDA_ENV="foldagent"
SEARCH_PORT=8000

# Benchmark parameters (override via environment)
MODEL="${MODEL:-anthropic-tc:claude-sonnet-4-6}"
N="${N:-5}"
PARALLEL="${PARALLEL:-1}"
BUDGET="${BUDGET:-10}"
TIMEOUT="${TIMEOUT:-600}"
MAX_BRANCH_STEPS="${MAX_BRANCH_STEPS:-15}"
NAME="${NAME:-cf-browsecomp}"

# ── Validate ───────────────────────────────────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY must be set"
    exit 1
fi

# ── Start search server in background ──────────────────────────────────────────
echo "=== Starting search server on port ${SEARCH_PORT} ==="
module load conda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$FOLDAGENT_DIR"
python envs/search_server.py \
    --model Qwen/Qwen3-Embedding-8B \
    --corpus Tevatron/browsecomp-plus-corpus \
    --corpus-embedding-dataset miaolu3/browsecomp-plus \
    --port "$SEARCH_PORT" &
SEARCH_PID=$!

# Wait for search server to be ready
echo "Waiting for search server (PID ${SEARCH_PID})..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${SEARCH_PORT}/health" >/dev/null 2>&1 || \
       curl -s -X POST -H 'Content-Type: application/json' \
            -d '{"query":"test","k":1}' \
            "http://localhost:${SEARCH_PORT}/search" >/dev/null 2>&1; then
        echo "Search server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
        echo "ERROR: Search server died"
        exit 1
    fi
    sleep 1
done

# ── Run benchmark ──────────────────────────────────────────────────────────────
echo "=== Running benchmark ==="
cd "$BENCH_DIR"

# Ensure Java is available for clj
export JAVA_HOME="${JAVA_HOME:-$(dirname $(dirname $(readlink -f $(which java))))}"

uv run python bench.py general \
    --dataset browsecomp_plus \
    --condition context_folding \
    --model "$MODEL" \
    --search-url "http://localhost:${SEARCH_PORT}" \
    --budget "$BUDGET" \
    --n "$N" \
    --parallel "$PARALLEL" \
    --timeout "$TIMEOUT" \
    --cf-max-branch-steps "$MAX_BRANCH_STEPS" \
    --trace \
    --name "$NAME"

echo "=== Benchmark complete ==="

# ── Cleanup ────────────────────────────────────────────────────────────────────
kill "$SEARCH_PID" 2>/dev/null || true
wait "$SEARCH_PID" 2>/dev/null || true
echo "Done."
