#!/bin/bash
#SBATCH -p gpu_quad
#SBATCH -t 2-00:00
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH -J vllm-seed36b
#SBATCH -o /tmp/vllm-seed36b-%j.out

set -euo pipefail

# Load any required modules (adjust for your O2 environment)
# module load gcc/9.2.0 cuda/11.7

vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct \
  --port 8001 \
  --max-model-len 32768 \
  --dtype bfloat16
