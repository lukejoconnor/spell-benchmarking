#!/bin/bash
# Terminal-Bench 2.0: Full 25-task rerun across all 4 conditions
# Batches of 5, with Docker cleanup between conditions.
#
# Usage:
#   ./terminal_bench/run_full_rerun.sh [batch_num]   # run specific batch (1-5)
#   ./terminal_bench/run_full_rerun.sh all            # run all batches
#   ./terminal_bench/run_full_rerun.sh                # defaults to "all"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colima Docker socket
if [ -S "${HOME}/.colima/default/docker.sock" ] && [ -z "${DOCKER_HOST:-}" ]; then
    export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
fi

export PYTHONPATH="${REPO_ROOT}/terminal_bench:${PYTHONPATH:-}"
if [ -z "${SPELL_ROOT:-}" ] && [ -f "$REPO_ROOT/../deps.edn" ]; then
    export SPELL_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
fi

DATASET="terminal-bench-core==0.1.1"
OUTPUT_DIR="results/terminal-bench-rerun"
TIMESTAMP=$(date +%Y-%m-%d)

mkdir -p "$OUTPUT_DIR"

# --- Batch definitions ---
BATCH1=(hello-world csv-to-parquet fix-permissions openssl-selfsigned-cert grid-pattern-transform)
BATCH2=(extract-safely processing-pipeline organization-json-generator password-recovery write-compressor)
BATCH3=(polyglot-rust-c gpt2-codegolf prove-plus-comm pytorch-model-cli intrusion-detection)
BATCH4=(modernize-fortran-build git-workflow-hack fix-git heterogeneous-dates chess-best-move)
BATCH5=(sqlite-db-truncate polyglot-c-py oom path-tracing configure-git-webserver)

# --- Helper: build -t flags ---
build_task_flags() {
    local flags=""
    for task in "$@"; do
        flags="$flags -t $task"
    done
    echo "$flags"
}

# --- Helper: run a single condition on a set of tasks ---
run_condition() {
    local cond_name="$1"
    local batch_num="$2"
    shift 2
    local tasks=("$@")
    local task_flags=$(build_task_flags "${tasks[@]}")
    local run_id="${cond_name}_b${batch_num}_${TIMESTAMP}"
    local out_path="$OUTPUT_DIR/${run_id}"

    echo ""
    echo "=========================================="
    echo "  ${cond_name} | Batch ${batch_num} | $(date +%H:%M:%S)"
    echo "  Tasks: ${tasks[*]}"
    echo "=========================================="

    case "$cond_name" in
        spell-opus)
            tb run \
                --agent-import-path "spell_agent:SpellAgent" \
                --model "anthropic-tc:opus46" \
                --dataset "$DATASET" \
                $task_flags \
                --output-path "$out_path" \
                --run-id "$run_id" \
                --n-concurrent 1 \
                --cleanup \
                --log-level info \
                -k "budget=5.0" \
                -k "sh_timeout=120" \
                -k "agent_timeout=600"
            ;;
        cc-opus)
            tb run \
                --agent-import-path "cc_agent:ClaudeCodeAgent" \
                --model "anthropic/claude-opus-4-6" \
                --dataset "$DATASET" \
                $task_flags \
                --output-path "$out_path" \
                --run-id "$run_id" \
                --n-concurrent 1 \
                --cleanup \
                --log-level info \
                -k "max_budget_usd=5.0"
            ;;
        spell-codex)
            tb run \
                --agent-import-path "spell_agent:SpellAgent" \
                --model "codex-tc:gpt-5.3-codex" \
                --dataset "$DATASET" \
                $task_flags \
                --output-path "$out_path" \
                --run-id "$run_id" \
                --n-concurrent 1 \
                --cleanup \
                --log-level info \
                -k "budget=5.0" \
                -k "sh_timeout=120" \
                -k "agent_timeout=600"
            ;;
        codex-cli)
            tb run \
                --agent-import-path "codex_agent:CodexAgent" \
                --model "openai/gpt-5.3-codex" \
                --dataset "$DATASET" \
                $task_flags \
                --output-path "$out_path" \
                --run-id "$run_id" \
                --n-concurrent 1 \
                --cleanup \
                --log-level info \
                -k "agent_timeout=600"
            ;;
    esac

    echo "  Done: ${cond_name} batch ${batch_num} -> $out_path"
}

# --- Helper: run all 4 conditions on a batch ---
run_batch() {
    local batch_num="$1"
    shift
    local tasks=("$@")

    echo ""
    echo "############################################"
    echo "  BATCH ${batch_num}: ${tasks[*]}"
    echo "  Started: $(date)"
    echo "############################################"

    for cond in spell-opus cc-opus spell-codex codex-cli; do
        run_condition "$cond" "$batch_num" "${tasks[@]}"
        echo "  Pruning Docker..."
        docker system prune -af --volumes 2>/dev/null | tail -1
    done

    echo ""
    echo "  BATCH ${batch_num} COMPLETE: $(date)"
    echo "############################################"
}

# --- Main ---
TARGET="${1:-all}"

case "$TARGET" in
    1) run_batch 1 "${BATCH1[@]}" ;;
    2) run_batch 2 "${BATCH2[@]}" ;;
    3) run_batch 3 "${BATCH3[@]}" ;;
    4) run_batch 4 "${BATCH4[@]}" ;;
    5) run_batch 5 "${BATCH5[@]}" ;;
    all)
        for b in 1 2 3 4 5; do
            eval "run_batch $b \"\${BATCH${b}[@]}\""
        done
        echo ""
        echo "ALL BATCHES COMPLETE: $(date)"
        ;;
    *) echo "Usage: $0 [1|2|3|4|5|all]"; exit 1 ;;
esac
