#!/usr/bin/env bash

# 1. 创建环境 (命名为 flux2)
conda create -n flux2 python=3.10 -y
conda activate flux2

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers accelerate protobuf wandb
pip install opencv-python pandas pyarrow datasets ftfy sentencepiece einops scikit-learn timm mmengine tqdm
pip install bitsandbytes>=0.43.0 peft>=0.11.1 pydantic ray[train] prodigyopt

# 7. (强烈推荐) 安装 Flash Attention 2
# Flux 模型很大，Flash Attention 能显著减少显存占用并加速训练
# 这一步可能会编译较慢，需要系统中有 CUDA 编译器 (nvcc)
# pip install flash-attn --no-build-isolation

# ── Environment variables (add to ~/.bashrc or ~/.zshrc) ──────────────────────
# export HF_TOKEN="hf_..."                 # Required: HuggingFace access token
# export SCIFORMA_DATA_ROOT="/your/data"   # Training data + checkpoint root
# export SCIFORMA_GT_BASE="$SCIFORMA_DATA_ROOT/SciFormaData-700K/generation/images_1024/1024_pretrain"
# export WANDB_API_KEY="..."               # Optional: WandB monitoring

# ── Verify installation ──────────────────────────────────────────────────────
# python training/scripts/verify_setup.py         # Basic: 10 passed
# python training/scripts/verify_setup.py --check-data  # Full: 18 passed
