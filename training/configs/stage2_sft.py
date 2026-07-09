"""
SciForma Stage 2 SFT — Mixed Generation + Editing
===================================================
Joint fine-tuning on high-quality generation pairs and editing triplets.
Produces SciForma-Base.

Hardware: 8× B200, DeepSpeed ZeRO-2, BF16
Effective batch: 1/GPU × 8 GPU × GA=2 = 16
Duration: 120K steps
Init: Stage 1 checkpoint-140000/ema_weights.pt

Data:
  Gen:  ~244K high-quality generation pairs (quality_filter='high')
  Edit: ~70K axis-specific editing triplets
  Sampler groups by (resolution, data_type) → each batch is pure-gen or pure-edit.
"""
import os

_base_ = ['./base_config.py']

# ── Model ─────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Init from Stage 1 ──────────────────────────────────────────────────────────
init_weights_path = (
    os.environ.get("SCIFORMA_DATA_ROOT", "")
    + "/experiments/sciforma-base-stage1/checkpoint-140000/ema_weights.pt"
)

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetV4',
    base_dir=os.environ.get("SCIFORMA_DATA_ROOT", ""),
    gen_parquet_path='ArXiV_parquet/Flux2Klein9B_1024_pretrain',
    edit_parquet_path='ArXiV_editing_parquet/mask_unit_test/260209_1024',
    quality_filter='high',   # ~244K high-quality gen samples (~38% of full set)
    num_workers=8,
    num_train_examples=None,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV3',
    dataset=None,
    batch_size=1,
    num_replicas=1,
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_mixed_edit_train_iteration'

use_lora                    = False
train_batch_size            = 1
gradient_accumulation_steps = 2
max_train_steps             = 120000
checkpointing_steps         = 500
checkpoints_total_limit     = None
validation_steps            = 500
gradient_checkpointing      = False   # B200 192GB VRAM — not needed

# ── Edit loss ──────────────────────────────────────────────────────────────────
gen_loss_weight  = 1.0
edit_loss_weight = 1.0
edit_loss_mode   = "uniform"   # standard flow-matching MSE over full target sequence

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema               = True
ema_on_gpu            = False   # CPU EMA frees ~36GB VRAM; overhead negligible
ema_decay             = 0.9999
ema_update_after_step = 0
ema_steps             = 100

# ── Optimizer ──────────────────────────────────────────────────────────────────
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
validation_guidance_scale = 4.0
num_inference_steps       = 50

# ── Output ─────────────────────────────────────────────────────────────────────
model_output_dir       = os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/sciforma-base"
output_dir             = model_output_dir
resume_from_checkpoint = "latest"

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 5
log_with             = "wandb"
tracker_project_name = "SciForma-SFT"
tracker_run_name     = "sciforma-base-stage2"

seed = 42
