#!/bin/bash
# Terminal-Bench 2.0: Codex CLI vs Spell/Codex-tc (gpt-5.3-codex)
#
# Runs all 25 tasks sequentially with docker prune between each task.
# Each task runs both conditions before moving to the next.
#
# Usage:
#   ./benchmarking/terminal_bench/run_codex_comparison.sh [task-ids...]

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to spell project root

# Colima Docker socket
DOCKER_HOST="unix:///Users/loconnor/.colima/default/docker.sock"
export DOCKER_HOST

DATASET="terminal-bench-core==0.1.1"
OUTPUT_DIR="benchmarking/results/terminal-bench"
TIMESTAMP=$(date +%Y-%m-%d__%H-%M-%S)
SPELL_OUTPUT="$OUTPUT_DIR/spell_codex_${TIMESTAMP}"
CODEX_OUTPUT="$OUTPUT_DIR/codex_cli_${TIMESTAMP}"

# All 25 tasks from the prior Opus 4.6 run
ALL_TASKS=(
    hello-world csv-to-parquet fix-permissions openssl-selfsigned-cert
    grid-pattern-transform extract-safely processing-pipeline
    organization-json-generator password-recovery write-compressor
    polyglot-rust-c gpt2-codegolf prove-plus-comm pytorch-model-cli
    intrusion-detection modernize-fortran-build git-workflow-hack
    fix-git heterogeneous-dates chess-best-move sqlite-db-truncate
    polyglot-c-py oom path-tracing configure-git-webserver
)

# Use provided tasks or default to all
if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=("${ALL_TASKS[@]}")
fi

mkdir -p "$SPELL_OUTPUT" "$CODEX_OUTPUT"
export PYTHONPATH="${PWD}/benchmarking/terminal_bench:${PYTHONPATH:-}"

echo "=== Terminal-Bench 2.0: Codex CLI vs Spell/Codex-tc ==="
echo "Tasks: ${#TASKS[@]}"
echo "Spell output: $SPELL_OUTPUT"
echo "Codex output: $CODEX_OUTPUT"
echo ""

SPELL_PASS=0
SPELL_FAIL=0
CODEX_PASS=0
CODEX_FAIL=0

for task in "${TASKS[@]}"; do
    echo ""
    echo "========== Task: $task =========="

    # --- Spell/Codex-tc ---
    echo "--- Spell/Codex-tc: $task ---"
    docker system prune -af --volumes > /dev/null 2>&1
    if tb run \
        --agent-import-path "spell_agent:SpellAgent" \
        --model "codex-tc:gpt-5.3-codex" \
        --dataset "$DATASET" \
        -t "$task" \
        --output-path "$SPELL_OUTPUT" \
        --run-id "spell_codex_${TIMESTAMP}" \
        --n-concurrent 1 \
        --log-level warning \
        -k "budget=5.0" \
        -k "sh_timeout=120" \
        2>&1 | tee -a "$SPELL_OUTPUT/run.log"; then
        # Check result
        RESULT_FILE="$SPELL_OUTPUT/spell_codex_${TIMESTAMP}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            RESOLVED=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('n_resolved',0))")
            if [ "$RESOLVED" -gt 0 ]; then
                SPELL_PASS=$((SPELL_PASS + 1))
                echo "  -> Spell: PASS"
            else
                SPELL_FAIL=$((SPELL_FAIL + 1))
                echo "  -> Spell: FAIL"
            fi
        else
            SPELL_FAIL=$((SPELL_FAIL + 1))
            echo "  -> Spell: FAIL (no result file)"
        fi
    else
        SPELL_FAIL=$((SPELL_FAIL + 1))
        echo "  -> Spell: ERROR"
    fi

    # --- Codex CLI ---
    echo "--- Codex CLI: $task ---"
    docker system prune -af --volumes > /dev/null 2>&1
    if tb run \
        --agent codex \
        --model "openai/gpt-5.3-codex" \
        --dataset "$DATASET" \
        -t "$task" \
        --output-path "$CODEX_OUTPUT" \
        --run-id "codex_cli_${TIMESTAMP}" \
        --n-concurrent 1 \
        --log-level warning \
        2>&1 | tee -a "$CODEX_OUTPUT/run.log"; then
        RESULT_FILE="$CODEX_OUTPUT/codex_cli_${TIMESTAMP}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            RESOLVED=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('n_resolved',0))")
            if [ "$RESOLVED" -gt 0 ]; then
                CODEX_PASS=$((CODEX_PASS + 1))
                echo "  -> Codex: PASS"
            else
                CODEX_FAIL=$((CODEX_FAIL + 1))
                echo "  -> Codex: FAIL"
            fi
        else
            CODEX_FAIL=$((CODEX_FAIL + 1))
            echo "  -> Codex: FAIL (no result file)"
        fi
    else
        CODEX_FAIL=$((CODEX_FAIL + 1))
        echo "  -> Codex: ERROR"
    fi

    TOTAL=$((SPELL_PASS + SPELL_FAIL))
    echo ""
    echo "  Running totals ($TOTAL/$((${#TASKS[@]}))):"
    echo "    Spell: $SPELL_PASS/$TOTAL ($(python3 -c "print(f'{$SPELL_PASS/$TOTAL*100:.0f}%')" 2>/dev/null || echo '?%'))"
    echo "    Codex: $CODEX_PASS/$TOTAL ($(python3 -c "print(f'{$CODEX_PASS/$TOTAL*100:.0f}%')" 2>/dev/null || echo '?%'))"
done

echo ""
echo "=== Final Results ==="
echo "Spell/Codex-tc: $SPELL_PASS/${#TASKS[@]}"
echo "Codex CLI:      $CODEX_PASS/${#TASKS[@]}"
echo ""
echo "Results in:"
echo "  Spell: $SPELL_OUTPUT"
echo "  Codex: $CODEX_OUTPUT"
