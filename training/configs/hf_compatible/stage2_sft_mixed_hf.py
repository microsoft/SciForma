"""
SciForma Stage 2 SFT — HuggingFace Dataset Variant (Mixed Gen + Edit)
=======================================================================
Joint fine-tuning on high-quality generation pairs AND editing triplets.
Produces SciForma-Base.

Architecture (mirrors original Stage2 on B200):
  ArXiVHFDatasetV4  ← single dataset, gen+edit unified with data_type column
    └─ DistributedBucketSamplerV3  ← groups by (bucket_h, bucket_w, data_type)
         └─ Flux2Klein_mixed_edit_train_iteration  ← dispatches by batch_mode

Data:
  - Gen (High quality): 244,304 samples @ 1024px
  - Edit: 70,866 pairs @ 1024px
  Total: ~315K samples per epoch

Expected structure:
  $SCIFORMA_DATA_ROOT/SciFormaData-700K/
    generation/
      metadata.parquet      (quality_filter='High' → 244,304 samples)
      images_1024/1024_pretrain/{year}/{paper_id}/{img}.png
      images_1024/...{img}_flux_h.npz   ← after cache_latents.py
    editing/
      metadata.parquet      (70,866 pairs)
      images/260209_1024/{year}/{paper_id}/{hash}/source.png
      images/260209_1024/{year}/{paper_id}/{hash}/target.png
      images/...source_flux_h.npz, target_flux_h.npz  ← after cache_latents.py

Before training:
1. Cache generation latents (High quality only):
   python scripts/cache_latents.py \\
       --parquet_dir $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation \\
       --image_base $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation

2. Cache editing latents:
   python scripts/cache_latents.py \\
       --parquet_dir $SCIFORMA_DATA_ROOT/SciFormaData-700K/editing \\
       --image_base $SCIFORMA_DATA_ROOT/SciFormaData-700K/editing

3. Train:
   accelerate launch scripts/train_sft.py configs/hf_compatible/stage2_sft_mixed_hf.py
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

# ── Init from Stage 1 ──────────────────────────────────────────────────────────
init_weights_path = os.environ.get(
    "SCIFORMA_STAGE1_CKPT",
    os.path.join(_SCIFORMA_ROOT, "experiments/sciforma-base-stage1/checkpoint-140000/ema_weights.pt")
)

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

# Single combined dataset: gen (244K High) + edit (70K) unified with data_type column
dataset_cfg = dict(
    type='ArXiVHFDatasetUnified',
    data_root=_HF_DATA_ROOT,
    splits=['gen_1024', 'edit_1024'],  # Stage2 mixed
    quality_filter='High',             # gen_1024 → 244K High; edit rows unaffected
    num_workers=8,
)

# SamplerV3 groups by (bucket_h, bucket_w, data_type) → pure-gen or pure-edit batches
sampler_cfg = dict(
    type='DistributedBucketSamplerV3',
    dataset=None,
    batch_size=1,
    num_replicas=1,    # MUST be 1 — accelerator.prepare() handles GPU distribution
    rank=0,
    drop_last=True,
    shuffle=True,
)

# ── Training ────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_mixed_edit_train_iteration'

train_batch_size           = 1
gradient_accumulation_steps = 2
max_train_steps            = 120000
checkpointing_steps        = 5000
validation_steps           = 5000
gradient_checkpointing     = False
max_grad_norm              = 1.0

# ── Loss ─────────────────────────────────────────────────────────────────────
gen_loss_weight  = 1.0
edit_loss_weight = 1.0
edit_loss_mode   = "uniform"   # Standard flow-matching MSE (best from ablation)

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
tracker_run_name     = "sciforma-base-stage2-mixed-hf"

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
