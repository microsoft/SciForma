"""
SciForma Stage 1 SFT — HuggingFace Dataset Variant
====================================================
Same as stage1_sft.py but adapted for users who downloaded SciFormaData-700K
from HuggingFace.

Expected directory structure after HF download:
  $SCIFORMA_DATA_ROOT/
    SciFormaData-700K/
      generation/
        metadata.parquet          ← image_path, caption, quality_tier
        images_768/768_pretrain/  ← raw PNG images (768px bucket)
        images_1024/1024_pretrain/← raw PNG images (1024px bucket)
      editing/
        metadata.parquet
        images/260209_1024/       ← source.png + target.png

Before training, generate NPZ latents:
    python scripts/cache_latents.py \\
        --parquet_dir $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation \\
        --image_base $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation \\
        --base_model black-forest-labs/FLUX.2-klein-base-9B
"""
import os

_base_ = ['../base_config.py']

_SCIFORMA_ROOT = os.environ.get("SCIFORMA_DATA_ROOT", "")
_HF_DATA_ROOT = os.path.join(_SCIFORMA_ROOT, "SciFormaData-700K")

# ── Model ─────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_768'],      # Stage1: all 661K images at 768px
    quality_filter=None,
    num_workers=8,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=2,
    num_replicas=1,
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

train_batch_size           = 2
gradient_accumulation_steps = 1
max_train_steps            = 200000
checkpointing_steps        = 5000
validation_steps           = 5000
gradient_checkpointing     = False

max_grad_norm              = 1.0

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer       = "adamw"
learning_rate   = 1e-5
lr_scheduler    = 'cosine_with_restarts'
lr_warmup_steps = 500

adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── EMA ───────────────────────────────────────────────────────────────────────
use_ema      = True
ema_decay    = 0.9999
ema_on_gpu   = False
ema_steps    = 100
ema_update_after_step = 0

# ── Precision ────────────────────────────────────────────────────────────────
mixed_precision = "bf16"
allow_tf32      = True

# ── Data processing ───────────────────────────────────────────────────────────
max_sequence_length   = 1024
dataloader_num_workers = 8
pin_memory            = True

# ── Output ───────────────────────────────────────────────────────────────────
model_output_dir = os.environ.get("SCIFORMA_OUTPUT_DIR",
                                   os.path.join(_SCIFORMA_ROOT, "experiments/sciforma-base-stage1"))
output_dir       = model_output_dir
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ──────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma-SFT"
tracker_run_name     = "sciforma-base-stage1-hf"

# ── Validation ───────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 4.0
num_inference_steps       = 28


# validation_prompts REQUIRED — without it train_sft.py skips validation entirely
validation_prompts = [
    "The figure illustrates a transformer architecture. The global layout is vertical, with encoder on the left and decoder on the right. Multi-head attention blocks are shown as blue rectangles with arrows indicating query, key, and value inputs. Feed-forward networks are orange rectangles. Layer normalization is depicted as small yellow boxes between components.",
    "The figure shows a line chart comparing training loss curves for three methods over 100 epochs. The x-axis shows epochs from 0 to 100, and the y-axis shows loss from 0 to 2.0. Method A (blue solid line) decreases steeply then plateaus at 0.3. Method B (orange dashed line) decreases gradually reaching 0.5. Baseline (green dotted line) stays high at 1.8.",
]
resolution_list = [[576, 1024], [576, 1024]]

seed = 42
