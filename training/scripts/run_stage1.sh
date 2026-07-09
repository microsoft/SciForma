#!/bin/bash
# =============================================================================
# SciForma Stage 1 SFT — Full Fine-tuning on B200 8-GPU
# =============================================================================
#
# Trains SciForma-Base Stage 1 SFT on SciFormaData-700K generation set (~661K samples).
#
# Hardware: 8× NVIDIA B200
# Config: training/configs/stage1_sft.py
# Accelerate: training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml
# Duration: 200K steps (~5 epochs)
#
# Usage:
#   export SCIFORMA_DATA_ROOT=/path/to/data
#   export HF_TOKEN="hf_..."
#   bash training/scripts/run_stage1.sh
# =============================================================================

set -euo pipefail

# Navigate to project root
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Check required env vars ────────────────────────────────────────────────────
if [[ -z "${SCIFORMA_DATA_ROOT:-}" ]]; then
    echo "[ERROR] SCIFORMA_DATA_ROOT is not set."
    echo "  export SCIFORMA_DATA_ROOT=/path/to/data"
    exit 1
fi

# ── Environment ────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_PROJECT="${WANDB_PROJECT:-SciForma-SFT}"
export HF_HOME="${HF_HOME:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"

echo "============================================================"
echo "SciForma Stage 1 SFT — B200 8-GPU Training"
echo "============================================================"
echo "SCIFORMA_DATA_ROOT: $SCIFORMA_DATA_ROOT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Config: training/configs/stage1_sft.py"
echo "Accelerate: training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml"
echo "============================================================"

accelerate launch \
    --config_file training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
    training/scripts/train_sft.py \
    training/configs/stage1_sft.py

echo "============================================================"
echo "Stage 1 training completed!"
echo "Checkpoint saved to: ${SCIFORMA_DATA_ROOT}/experiments/sciforma-base-stage1/"
echo "============================================================"
