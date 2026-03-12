#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLEAN_HOME="$ROOT_DIR/.codex-clean-home"
INSTR_FILE="$ROOT_DIR/.codex-empty-instructions.md"

mkdir -p "$CLEAN_HOME"

# Reuse existing auth so you do not need to log in again.
if [ -f "$HOME/.codex/auth.json" ]; then
  ln -sf "$HOME/.codex/auth.json" "$CLEAN_HOME/auth.json"
fi

exec env CODEX_HOME="$CLEAN_HOME" codex \
  -c "model_instructions_file=\"$INSTR_FILE\"" \
  -c "project_doc_max_bytes=0" \
  "$@"
