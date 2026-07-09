#!/bin/bash
# MD3PO Contrastive + Global Worst — B200 4-GPU (0,1,2,3)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCIFORMA_DATA_ROOT}/models/hf_cache}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
export WANDB_PROJECT="${WANDB_PROJECT:-SciForma-MDPO}"
  [[ -n "${WANDB_KEY:-}" ]] && export WANDB_API_KEY="$WANDB_KEY" && wandb login "$WANDB_KEY" 2>/dev/null || true
fi

CONFIG="configs/mdpo/ablations/mdpo_1v2ct_global_contrastive.py"
ACCEL="training/accelerate_cfg/b200_zero2_bf16_4gpu_dpo.yaml"

echo "=== MD3PO Contrastive + Global Worst === GPUs: $CUDA_VISIBLE_DEVICES"
accelerate launch --config_file "$ACCEL" training/scripts/train_dpo.py "$CONFIG"
echo "=== Contrastive + Global done ==="
