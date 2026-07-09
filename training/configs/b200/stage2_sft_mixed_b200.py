"""
SciForma Stage 2 SFT Mixed Gen+Edit — B200 8-GPU Config (HF Dataset)
=====================================================================
Joint fine-tuning on High quality 1024px generation + editing pairs.
Produces SciForma-Base. Init from Stage1 EMA checkpoint.

Hardware: B200 × 8 (192 GB each)
Effective batch = 1 × 8GPU × 2GA = 16  (与原始 B200 Stage2 完全一致)

Dataset: SciFormaData-700K, split='gen_1024' (High, 241K) + split='edit_1024' (70K)
Steps per epoch = 312,222 / 16 = 19,514
Total steps: 120,000 ≈ 6.1 epochs

Data root (B200 挂载路径):
  /data/sciforma/SciFormaData-700K/
Unified parquet:
  /data/sciforma/SciFormaData-700K/metadata.parquet

To train:
    SCIFORMA_DATA_ROOT=/data/sciforma \\
    SCIFORMA_STAGE1_CKPT=/path/to/stage1/checkpoint-140000/ema_weights.pt \\
    WANDB_API_KEY=xxx HF_TOKEN=xxx \\
    accelerate launch --config_file accelerate_cfg/b200_deepspeed_zero2_bf16.yaml \\
        --num_processes 8 \\
        scripts/train_sft.py configs/b200/stage2_sft_mixed_b200.py
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

# ── Init from Stage 1 ──────────────────────────────────────────────────────────
init_weights_path = os.environ.get(
    "SCIFORMA_STAGE1_CKPT",
    os.path.join(_SCIFORMA_ROOT, "experiments/sciforma_stage1_b200/checkpoint-140000/ema_weights.pt")
)

# ── Dataset ────────────────────────────────────────────────────────────────────
# 统一 parquet: /data/sciforma/SciFormaData-700K/metadata.parquet
use_parquet_dataset = True
dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_1024', 'edit_1024'],  # Stage2 mixed
    quality_filter='High',             # gen_1024 → 241K High; edit 不受影响
    num_workers=8,
)

# SamplerV3 按 (bucket_h, bucket_w, data_type) 分组 → 每 batch 纯 gen 或纯 edit
sampler_cfg = dict(
    type='DistributedBucketSamplerV3',
    dataset=None,
    batch_size=1,      # bs=1 per GPU
    num_replicas=1,    # MUST be 1 — accelerate 处理分布式
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_mixed_edit_train_iteration'

train_batch_size           = 1    # 1 × 8GPU × 2GA = eff_batch 16，与原始完全一致
gradient_accumulation_steps = 2
max_train_steps            = 120000
checkpointing_steps        = 5000
validation_steps           = 2500
gradient_checkpointing     = False  # B200 192GB 不需要
max_grad_norm              = 1.0

# loss 配置（与原始 stage2 保持一致）
gen_loss_weight  = 1.0
edit_loss_weight = 1.0
edit_loss_mode   = "uniform"

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
    "/data/sciforma/experiments/sciforma_stage2_b200")
output_dir       = model_output_dir
logging_dir      = "logs"
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma"
tracker_run_name     = "sciforma_stage2_b200_8gpu_bs16"

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 4.0
num_inference_steps       = 28

validation_prompts = [
    "The figure illustrates a transformer architecture with encoder and decoder. Multi-head self-attention blocks are blue rectangles with arrows for query, key, value. Feed-forward network layers are shown as orange rectangles. Layer normalization boxes are yellow, positioned after each sub-layer.",
    "The figure presents a bar chart comparing accuracy of five methods on ImageNet. X-axis shows method names, Y-axis shows top-1 accuracy from 70% to 90%. Bars are colored: blue for baselines, red for proposed method which achieves highest accuracy at 87.3%.",
    "The diagram shows a convolutional neural network architecture. Input image on the left passes through alternating convolutional layers (green) and pooling layers (orange). Feature maps decrease in spatial size but increase in channel depth. Final fully-connected layers (blue) output class probabilities.",
]
resolution_list = [[576, 1024], [576, 1024], [576, 1024]]

seed = 42
