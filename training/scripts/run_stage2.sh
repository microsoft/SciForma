#!/bin/bash
# =============================================================================
# SciForma Stage 2: Mixed Gen + Edit SFT (B200 8-GPU)
# Continues from Stage 1 checkpoint-140000/ema_weights.pt
# Config: training/configs/stage2_sft.py
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-${SCIFORMA_DATA_ROOT:-/tmp}/models/hf_cache}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-SciForma-Stage2}"
[[ -n "${WANDB_API_KEY:-}" ]] && wandb login "${WANDB_API_KEY}" 2>/dev/null || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is not set. Run: export HF_TOKEN=hf_..."
  exit 1
fi

if [[ -z "${SCIFORMA_DATA_ROOT:-}" ]]; then
  echo "[ERROR] SCIFORMA_DATA_ROOT is not set."
  exit 1
fi

CONFIG="training/configs/stage2_sft.py"
ACCEL="training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml"

echo "=============================================="
echo "  SciForma Stage 2 SFT (Mixed Gen + Edit)"
echo "  Config:     $CONFIG"
echo "  Accelerate: $ACCEL"
echo "  GPUs:       $CUDA_VISIBLE_DEVICES"
echo "  NOTE: Set init_weights_path in $CONFIG"
echo "        to Stage 1 checkpoint-140000/ema_weights.pt"
echo "=============================================="

accelerate launch \
    --config_file "$ACCEL" \
    training/scripts/train_sft.py "$CONFIG"

echo "Training complete."
