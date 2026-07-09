"""
SciForma Stage 2 SFT Mixed Gen+Edit — A100 4-GPU Config
=========================================================
Reproduces the original B200 Stage2 training on 4x A100 80GB.

Hardware: A100 80GB PCIe × 4
Effective batch = 1 × 4GPU × 2GA = 8  (B200 used 1×8GPU×2GA = 16)
→ To cover same epochs, need 2× steps: 120K × 2 = 240,000 steps

Dataset:
  Gen:  SciFormaData-700K generation, High quality (244,304 samples at 1024px)
  Edit: SciFormaData-700K editing (70,866 pairs at 1024px)
  Total: ~315K samples/epoch
  Steps/epoch = 315,170 / 8 = 39,396
  Total steps: 240,000 ≈ 6.1 epochs (matches B200 ~7.86 epochs at eff_batch=16)

Init from Stage 1 EMA checkpoint.

To train:
    SCIFORMA_DATA_ROOT=/data/sciforma \\
    SCIFORMA_STAGE1_CKPT=/path/to/stage1/checkpoint-140000/ema_weights.pt \\
    WANDB_API_KEY=xxx HF_TOKEN=xxx \\
    accelerate launch --config_file accelerate_cfg/deepspeed_zero2_bf16.yaml \\
        --num_processes 4 \\
        scripts/train_sft.py configs/a100/stage2_sft_mixed_a100.py
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
    "/data/sciforma/experiments/sciforma_stage1_a100/checkpoint-140000/ema_weights.pt"
)

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True
dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_1024', 'edit_1024'],  # Stage2: High quality gen + editing pairs
    quality_filter='High',             # gen_1024 → 244K High only; edit rows unaffected
    num_workers=8,
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

train_batch_size           = 1
gradient_accumulation_steps = 2   # 1×4GPU×2GA = eff_batch 8
max_train_steps            = 240000  # 2× B200 steps to match same epochs (eff_batch halved)
checkpointing_steps        = 5000
validation_steps           = 2500
gradient_checkpointing     = False
max_grad_norm              = 1.0

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
    "/data/sciforma/experiments/sciforma_stage2_a100")
output_dir       = model_output_dir
resume_from_checkpoint = 'latest'
checkpoints_total_limit = 3

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma"
tracker_run_name     = "sciforma_stage2_mixed_a100_4gpu_bs8"

# ── Validation ─────────────────────────────────────────────────────────────────
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
