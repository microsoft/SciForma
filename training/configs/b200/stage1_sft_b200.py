"""
SciForma Stage 1 SFT — B200 8-GPU Config (HF Dataset)
=======================================================
Full fine-tuning on all 661K 768px generation images.
Produces the Stage1 checkpoint used to initialize Stage2.

Hardware: B200 × 8 (192 GB each)
NOTE: 原始脚本文件名叫 run_flux2klein_2gpu.sh 但实际用了 8 卡！
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7，accelerate_cfg/b200_deepspeed_zero2_bf16_8gpu.yaml

Effective batch = 2 × 8GPU × 1GA = 16

Dataset: SciFormaData-700K, split='gen_768', all quality tiers (661K)
Steps per epoch = 661,274 / 16 = 41,329
Total steps: 200,000 ≈ 4.8 epochs (与原始 B200 完全一致)

Data root:
  /data/sciforma/SciFormaData-700K/
Unified parquet:
  /data/sciforma/SciFormaData-700K/metadata.parquet

To train:
    SCIFORMA_DATA_ROOT=/data/sciforma \\
    WANDB_API_KEY=xxx HF_TOKEN=xxx \\
    accelerate launch --config_file accelerate_cfg/b200_deepspeed_zero2_bf16_8gpu.yaml \\
        scripts/train_sft.py configs/b200/stage1_sft_b200.py
"""
import os

_base_ = ['../base_config.py']

_SCIFORMA_ROOT = os.environ.get("SCIFORMA_DATA_ROOT", "/data/sciforma")
_HF_DATA_ROOT = os.path.join(_SCIFORMA_ROOT, "SciFormaData-700K")

# ── Model ──────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Dataset ────────────────────────────────────────────────────────────────────
# 统一 parquet: /data/sciforma/SciFormaData-700K/metadata.parquet
use_parquet_dataset = True
dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_768'],      # Stage1: 661K 768px images, all quality tiers
    quality_filter=None,
    num_workers=8,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=2,      # bs=2 per GPU，8GPU × 2 × 1GA = eff_batch 16
    num_replicas=1,    # MUST be 1 — accelerate 处理分布式
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

train_batch_size           = 2    # 2 × 8GPU × 1GA = eff_batch 16
gradient_accumulation_steps = 1
max_train_steps            = 200000
checkpointing_steps        = 5000
validation_steps           = 2500
gradient_checkpointing     = False  # B200 192GB 不需要
max_grad_norm              = 1.0

# ── Optimizer ──────────────────────────────────────────────────────────────────
optimizer       = "adamw"
learning_rate   = 1e-5
lr_scheduler    = 'constant_with_warmup'
lr_warmup_steps = 1000

adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema      = True
ema_decay    = 0.9999
ema_on_gpu   = False
ema_steps    = 100
ema_update_after_step = 0

# ── Precision ──────────────────────────────────────────────────────────────────
mixed_precision = "bf16"
allow_tf32      = True

# ── Data ───────────────────────────────────────────────────────────────────────
max_sequence_length    = 1024
dataloader_num_workers = 8
pin_memory             = True

# ── Output ─────────────────────────────────────────────────────────────────────
model_output_dir = os.environ.get("SCIFORMA_OUTPUT_DIR",
    "/data/sciforma/experiments/sciforma_stage1_b200")
output_dir       = model_output_dir
logging_dir      = "logs"
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma"
tracker_run_name     = "sciforma_stage1_b200_8gpu_bs16"

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 3.5
num_inference_steps       = 28

validation_prompts = [
    "The figure illustrates a transformer architecture with encoder and decoder. Multi-head self-attention blocks are blue rectangles with arrows for query, key, value. Feed-forward network layers are shown as orange rectangles. Layer normalization boxes are yellow, positioned after each sub-layer.",
    "The figure presents a bar chart comparing accuracy of five methods on ImageNet. X-axis shows method names, Y-axis shows top-1 accuracy from 70% to 90%. Bars are colored: blue for baselines, red for proposed method which achieves highest accuracy at 87.3%.",
    "The diagram shows a convolutional neural network architecture. Input image on the left passes through alternating convolutional layers (green) and pooling layers (orange). Feature maps decrease in spatial size but increase in channel depth. Final fully-connected layers (blue) output class probabilities.",
]
resolution_list = [[576, 1024], [576, 1024], [576, 1024]]

seed = 42
