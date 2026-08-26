#!/usr/bin/env bash
# ==============================================================================
# SciForma — Docker inference launcher
# Runs generate_benchmark.py inside the container with GPU + data mounts
#
# Usage:
#   export SCIFORMA_DATA_ROOT="/path/to/data"   # contains models/ + experiments/
#   export SCIFORMA_MODEL_PATH="/path/to/local/base/pipeline"
#   export SCIFORMA_EMA_WEIGHTS="/path/to/local/ema_weights.pt"
#   bash docker_run_inference.sh [GPU_IDS]
# ==============================================================================
set -euo pipefail

IMAGE="${SCIFORMA_IMAGE:-sciforma:latest}"
GPU_IDS="${1:-0}"
DATA_ROOT="${SCIFORMA_DATA_ROOT:?Set SCIFORMA_DATA_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sciforma_output}"
MODEL_PATH="${SCIFORMA_MODEL_PATH:?Set SCIFORMA_MODEL_PATH to a local pipeline directory}"
EMA_WEIGHTS="${SCIFORMA_EMA_WEIGHTS:?Set SCIFORMA_EMA_WEIGHTS to a local ema_weights.pt}"
mkdir -p "$OUTPUT_DIR"

docker run --rm -it \
    --gpus "device=$GPU_IDS" \
    --shm-size 32gb \
    --volume "$DATA_ROOT:/workspace/data:ro" \
    --volume "$MODEL_PATH:/workspace/model:ro" \
    --volume "$EMA_WEIGHTS:/workspace/ema_weights.pt:ro" \
    --volume "$OUTPUT_DIR:/workspace/output:rw" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env SCIFORMA_DATA_ROOT="/workspace/data" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    "$IMAGE" \
    python generate/benchmark.py \
        --model_path /workspace/model \
        --ema_weights /workspace/ema_weights.pt \
        --split simple medium hard \
        --output_dir /workspace/output/sciforma-9b \
        --cfg 4.0 --steps 50 --max_seq_len 2048 \
        --resume

echo "Done. Results: $OUTPUT_DIR"
