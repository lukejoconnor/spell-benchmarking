#!/bin/bash
#SBATCH -p gpu_quad
#SBATCH -t 2-00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -J browsecomp-search
#SBATCH -o /tmp/browsecomp-search-%j.out

set -euo pipefail

# Path to FoldAgent environment with search_server.py
# Adjust FOLDAGENT_DIR to match your O2 setup
FOLDAGENT_DIR="${FOLDAGENT_DIR:-/path/to/FoldAgent}"

cd "$FOLDAGENT_DIR"

python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --port 8000
