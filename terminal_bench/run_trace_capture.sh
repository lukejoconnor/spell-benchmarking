#!/bin/bash
# Re-run select Spell/Opus tasks to capture execution traces.
# Runs WITHOUT --cleanup, then docker cp traces out before pruning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -S "${HOME}/.colima/default/docker.sock" ] && [ -z "${DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
fi

export PYTHONPATH="${REPO_ROOT}/terminal_bench:${PYTHONPATH:-}"
if [ -z "${SPELL_ROOT:-}" ] && [ -f "$REPO_ROOT/../deps.edn" ]; then
    export SPELL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

DATASET="terminal-bench-core==0.1.1"
OUTPUT_DIR="results/spell-trace-capture"
TRACE_DIR="results/spell-traces"
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"

# Tasks to capture traces for
TASKS=(oom write-compressor extract-safely intrusion-detection git-workflow-hack openssl-selfsigned-cert heterogeneous-dates processing-pipeline)

for task in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  TRACE CAPTURE: $task | $(date +%H:%M:%S)"
    echo "=========================================="

    run_id="trace_${task}_${TIMESTAMP}"
    out_path="$OUTPUT_DIR/${run_id}"

    # SpellAgent now extracts traces automatically via perform_task override
    tb run \
        --agent-import-path "spell_agent:SpellAgent" \
        --model "anthropic-tc:opus46" \
        --dataset "$DATASET" \
        -t "$task" \
        --output-path "$out_path" \
        --run-id "$run_id" \
        --n-concurrent 1 \
        --cleanup \
        --log-level info \
        -k "budget=5.0" \
        -k "sh_timeout=120" \
        -k "agent_timeout=600"

    echo "  Run complete. Checking for traces..."

    # Find extracted trace files
    trace_found=$(find "$out_path" -path "*/spell-trace/*" -name "tree.txt" 2>/dev/null | head -1)
    if [ -n "$trace_found" ]; then
        echo "  Traces found at $(dirname "$trace_found")"
    else
        echo "  WARNING: No trace files extracted"
    fi

    # Clean up Docker
    echo "  Pruning Docker..."
    docker system prune -af --volumes 2>/dev/null | tail -1

    echo "  Done: $task"
done

echo ""
echo "ALL TRACE CAPTURES COMPLETE: $(date)"
echo "Traces in: $TRACE_DIR"
