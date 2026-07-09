"""
SciForma Stage 1 SFT — Generation Pretraining
===============================================
Full-parameter fine-tuning on ~655K arXiv generation pairs.

Hardware: 8× B200, DeepSpeed ZeRO-2, BF16
Effective batch: 2/GPU × 8 GPU × GA=1 = 16
Duration: 200K steps (~5 epochs over 655K samples)
"""
import os

_base_ = ['./base_config.py']

# ── Model ─────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV3',
    base_dir=os.environ.get("SCIFORMA_DATA_ROOT", ""),
    parquet_base_path='ArXiV_parquet/Flux2Klein9BParquet_0201_NEW',
    num_workers=8,
    num_train_examples=None,   # use all ~655K samples
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=2,
    num_replicas=1,   # Accelerate handles GPU distribution; sampler stays at 1
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

use_lora                    = False   # full fine-tuning
train_batch_size            = 2       # per GPU
gradient_accumulation_steps = 1
max_train_steps             = 200000
checkpointing_steps         = 1000
checkpoints_total_limit     = None
validation_steps            = 500
gradient_checkpointing      = False   # B200 192GB VRAM — not needed

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema               = True
ema_decay             = 0.9999
ema_update_after_step = 0
ema_steps             = 100

# ── Optimizer ──────────────────────────────────────────────────────────────────
# AdamW: Prodigy incompatible with 9B+ models (torch.dot limit)
optimizer         = "adamw"
learning_rate     = 1e-5
adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── LR Scheduler ───────────────────────────────────────────────────────────────
lr_scheduler    = "constant_with_warmup"
lr_warmup_steps = 1000

# ── Precision ──────────────────────────────────────────────────────────────────
mixed_precision = "bf16"
allow_tf32      = True

# ── Data processing ────────────────────────────────────────────────────────────
max_sequence_length    = 1024
dataloader_num_workers = 8
pin_memory             = True

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 3.5
num_inference_steps       = 28

# ── Output ─────────────────────────────────────────────────────────────────────
model_output_dir       = os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/sciforma-base-stage1"
output_dir             = model_output_dir
resume_from_checkpoint = "latest"

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma-SFT"
tracker_run_name     = "sciforma-base-stage1"

seed = 42
