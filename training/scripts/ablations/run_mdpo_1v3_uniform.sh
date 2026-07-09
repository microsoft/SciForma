#!/bin/bash
# =============================================================================
# MD3PO 1v3 (uniform alpha) on B200 8-GPU
# =============================================================================
# Usage:
#   conda activate flowfactoryflux
#   export HF_TOKEN="<your_hf_token>"
#   bash b200_experiment/run_md3po_1v3_uniform_b200_8gpu.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_FILE="configs/mdpo/ablations/mdpo_1v3_uniform.py"
ACCEL_FILE="training/accelerate_cfg/b200_zero2_bf16_8gpu_dpo.yaml"

export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-SciForma-MDPO}"
  if [[ -n "${WANDB_KEY:-}" ]]; then
    export WANDB_API_KEY="$WANDB_KEY"
    wandb login "$WANDB_KEY" 2>/dev/null || true
  fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] Missing config: $CONFIG_FILE"
  exit 1
fi
if [[ ! -f "$ACCEL_FILE" ]]; then
  echo "[ERROR] Missing accelerate config: $ACCEL_FILE"
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN not set. Set it if model download is needed."
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
  else
    CUDA_VISIBLE_DEVICES="0"
  fi
fi
export CUDA_VISIBLE_DEVICES
export HF_HOME="${HF_HOME:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=============================================="
echo "MD3PO 1v3 uniform alpha - B200 8-GPU"
echo "=============================================="
echo "Repo:       $REPO_ROOT"
echo "GPUs:       $CUDA_VISIBLE_DEVICES"
echo "Config:     $CONFIG_FILE"
echo "Accelerate: $ACCEL_FILE"
echo "=============================================="

accelerate launch \
  --config_file "$ACCEL_FILE" \
  training/scripts/train_dpo.py \
  "$CONFIG_FILE"

echo "=============================================="
echo "Training completed"
echo "=============================================="
