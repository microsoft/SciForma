"""
SciForma Stage 1 SFT — A100 4-GPU Config
==========================================
Reproduces the original B200 Stage1 training on 4x A100 80GB.

Hardware: A100 80GB PCIe × 4
Effective batch = 1 × 4GPU × 4GA = 16  (same as B200: bs=2×2GPU×GA=1=4... but B200 comment says 16)

Dataset: SciFormaData-700K generation, ALL quality tiers (661,660 samples at 768px)
Steps per epoch = 661,660 / 16 = 41,354
Total steps: 200,000 ≈ 4.84 epochs

To train:
    SCIFORMA_DATA_ROOT=/data/sciforma \\
    WANDB_API_KEY=xxx HF_TOKEN=xxx \\
    accelerate launch --config_file accelerate_cfg/deepspeed_zero2_bf16.yaml \\
        --num_processes 4 \\
        scripts/train_sft.py configs/a100/stage1_sft_a100.py
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
use_parquet_dataset = True
dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_768'],      # Stage1: all 661K images at 768px
    quality_filter=None,     # all quality tiers
    num_workers=8,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None,
    batch_size=1,      # bs=1 per GPU, A100 80GB
    num_replicas=1,
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

train_batch_size           = 1
gradient_accumulation_steps = 4   # 1×4GPU×4GA = eff_batch 16
max_train_steps            = 200000   # 661K / 16 = 41K steps/epoch × ~5 epochs
checkpointing_steps        = 5000
validation_steps           = 2500
gradient_checkpointing     = False    # A100 80GB has enough VRAM with bs=1
max_grad_norm              = 1.0

# ── Optimizer ──────────────────────────────────────────────────────────────────
optimizer       = "adamw"
learning_rate   = 1e-5            # matches B200 Stage1
lr_scheduler    = 'constant_with_warmup'
lr_warmup_steps = 1000

adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema      = True
ema_decay    = 0.9999
ema_on_gpu   = False   # CPU EMA frees ~18 GB; essential for A100 stability
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
    "/data/sciforma/experiments/sciforma_stage1_a100")
output_dir       = model_output_dir
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma"
tracker_run_name     = "sciforma_stage1_a100_4gpu_bs16"

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 3.5
num_inference_steps       = 28


# validation_prompts REQUIRED — without it train_sft.py skips validation entirely
validation_prompts = [
    "The figure illustrates a transformer architecture. The global layout is vertical, with encoder on the left and decoder on the right. Multi-head attention blocks are shown as blue rectangles with arrows indicating query, key, and value inputs. Feed-forward networks are orange rectangles. Layer normalization is depicted as small yellow boxes between components.",
    "The figure shows a line chart comparing training loss curves for three methods over 100 epochs. The x-axis shows epochs from 0 to 100, and the y-axis shows loss from 0 to 2.0. Method A (blue solid line) decreases steeply then plateaus at 0.3. Method B (orange dashed line) decreases gradually reaching 0.5. Baseline (green dotted line) stays high at 1.8.",
]
resolution_list = [[576, 1024], [576, 1024]]

seed = 42
