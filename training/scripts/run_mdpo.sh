#!/bin/bash
# =============================================================================
# SciForma-9B: M-DPO training (contrastive logsumexp, M-DPO training for SciForma-9B)
# Config: training/configs/mdpo/mdpo.py
# Hardware: 4× B200, batch=1/GPU, GA=3 → eff_batch=12
# Usage:
#   export SCIFORMA_DATA_ROOT=/path/to/data
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...    # optional
#   bash training/scripts/run_mdpo.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-${SCIFORMA_DATA_ROOT:-/tmp}/models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-SciForma-MDPO}"
[[ -n "${WANDB_API_KEY:-}" ]] && wandb login "${WANDB_API_KEY}" 2>/dev/null || true

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HF_TOKEN is not set. Run: export HF_TOKEN=hf_..."
  exit 1
fi

CONFIG="training/configs/mdpo/mdpo.py"
ACCEL="training/accelerate_cfg/b200_zero2_bf16_4gpu_dpo.yaml"

echo "=============================================="
echo "  SciForma-9B M-DPO Training (4-GPU B200)"
echo "  Config:     $CONFIG"
echo "  Accelerate: $ACCEL"
echo "  GPUs:       $CUDA_VISIBLE_DEVICES"
echo "=============================================="

accelerate launch --config_file "$ACCEL" training/scripts/train_dpo.py "$CONFIG"

echo "Training complete."
