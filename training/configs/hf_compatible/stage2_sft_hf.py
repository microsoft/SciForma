"""
SciForma Stage 2 SFT — HuggingFace Dataset Variant
====================================================
Uses ArXiVHFDatasetV1 for HF-format SciFormaData-700K.
Only uses HIGH quality subset (~244K samples) as in original Stage2 training.

Before training, run scripts/cache_latents.py to pre-compute NPZ latents.
"""
import os

_base_ = ['../base_config.py']

_SCIFORMA_ROOT = os.environ.get("SCIFORMA_DATA_ROOT", "")
_HF_DATA_ROOT = os.path.join(_SCIFORMA_ROOT, "SciFormaData-700K")
_HF_GEN_ROOT = os.path.join(_HF_DATA_ROOT, "generation")

# ── Model ─────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Checkpoint init ────────────────────────────────────────────────────────────
# Set this to your Stage1 checkpoint if continuing from Stage1
# init_weights_path = os.path.join(_SCIFORMA_ROOT, "experiments/sciforma-base-stage1/checkpoint-140000/ema_weights.pt")

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=os.path.join(os.environ.get('SCIFORMA_DATA_ROOT',''), 'SciFormaData-700K'),
    splits=['gen_1024'],
    quality_filter='High',   # 244K High quality
    num_workers=8,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=1,
    num_replicas=1,
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ────────────────────────────────────────────────────────────────
# Use fulltune iteration for gen-only HF Stage2 (mixed_edit requires bucket_size
# from V4 dataset's collate_fn; ArXiVHFDatasetV1 doesn't provide this field).
# For mixed gen+edit training with HF data, use stage2_sft_mixed_hf.py instead.
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

train_batch_size           = 1
gradient_accumulation_steps = 2
max_train_steps            = 120000
checkpointing_steps        = 5000
validation_steps           = 5000
gradient_checkpointing     = False
max_grad_norm              = 1.0

# ── Optimizer ────────────────────────────────────────────────────────────────
optimizer       = "adamw"
learning_rate   = 5e-6
lr_scheduler    = 'constant_with_warmup'
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
                                   os.path.join(_SCIFORMA_ROOT, "experiments/sciforma-base-stage2"))
output_dir       = model_output_dir
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ──────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma-SFT"
tracker_run_name     = "sciforma-base-stage2-hf"

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
