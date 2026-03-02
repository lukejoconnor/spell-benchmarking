#!/bin/bash
# Terminal-Bench 2.0 Pilot: CC vs Spell (Opus 4.6)
#
# Usage:
#   ./terminal_bench/run_pilot.sh [cc|spell|both] [task-ids...]
#
# Examples:
#   ./terminal_bench/run_pilot.sh both                    # run all pilot tasks
#   ./terminal_bench/run_pilot.sh cc hello-world          # CC only, one task
#   ./terminal_bench/run_pilot.sh spell csv-to-parquet    # Spell only, one task

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colima Docker socket (if Docker Desktop is not installed)
if [ -S "${HOME}/.colima/default/docker.sock" ] && [ -z "${DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
fi

CONDITION="${1:-both}"
shift 2>/dev/null || true

# Default pilot tasks if none specified
if [ $# -eq 0 ]; then
    TASKS=(hello-world csv-to-parquet fix-permissions openssl-selfsigned-cert)
else
    TASKS=("$@")
fi

MODEL="anthropic/claude-opus-4-6"
DATASET="terminal-bench-core==0.1.1"
OUTPUT_DIR="results/terminal-bench"
TIMESTAMP=$(date +%Y-%m-%d__%H-%M-%S)

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${REPO_ROOT}/terminal_bench:${PYTHONPATH:-}"
if [ -z "${SPELL_ROOT:-}" ] && [ -f "$REPO_ROOT/../deps.edn" ] && [ -f "$REPO_ROOT/../src/spell/benchmark_api.clj" ]; then
    export SPELL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

# Build task-id flags
TASK_FLAGS=""
for task in "${TASKS[@]}"; do
    TASK_FLAGS="$TASK_FLAGS -t $task"
done

echo "=== Terminal-Bench 2.0 Pilot ==="
echo "Tasks: ${TASKS[*]}"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo ""

run_cc() {
    echo "--- Running Claude Code ---"
    tb run \
        --agent-import-path "cc_agent:ClaudeCodeAgent" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        $TASK_FLAGS \
        --output-path "$OUTPUT_DIR/cc_${TIMESTAMP}" \
        --run-id "cc_${TIMESTAMP}" \
        --n-concurrent 1 \
        --livestream \
        --log-level info
    echo "CC done. Results: $OUTPUT_DIR/cc_${TIMESTAMP}"
}

run_spell() {
    echo "--- Running Spell ---"
    tb run \
        --agent-import-path "spell_agent:SpellAgent" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        $TASK_FLAGS \
        --output-path "$OUTPUT_DIR/spell_${TIMESTAMP}" \
        --run-id "spell_${TIMESTAMP}" \
        --n-concurrent 1 \
        --livestream \
        --log-level info \
        -k "budget=5.0" \
        -k "sh_timeout=120"
    echo "Spell done. Results: $OUTPUT_DIR/spell_${TIMESTAMP}"
}

case "$CONDITION" in
    cc)    run_cc ;;
    spell) run_spell ;;
    both)  run_cc; echo ""; run_spell ;;
    *)     echo "Usage: $0 [cc|spell|both] [task-ids...]"; exit 1 ;;
esac

echo ""
echo "=== Pilot Complete ==="
echo "Compare results in $OUTPUT_DIR/"
