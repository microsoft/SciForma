"""
SciForma M-DPO Training Configuration
======================================
Multi-Dimensional Preference Optimization for SciForma-9B.

Trains the M-DPO stage starting from SciForma-Base (Stage 2 SFT).

Key settings:
  Loss:     contrastive (logsumexp Bradley-Terry, see M-DPO paper section)
  Losers:   3 axis-specific (Component, Text) + 1 global worst
  Hardware: 4× B200, batch=1/GPU, GA=3  →  effective batch = 12
  Dataset:  ~16.5K scored preference triples

Reference model = frozen copy of policy init (SciForma-Base checkpoint).
"""
import os

_base_ = ['../base_config.py']

# ── Model ─────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None

transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Checkpoint init ────────────────────────────────────────────────────────────
# Init policy and reference from SciForma-Base
# Option A (HuggingFace, recommended): loads transformer weights directly
#   SCIFORMA_STAGE2_CKPT not set → falls back to HF model
# Option B (local EMA weights): set SCIFORMA_STAGE2_CKPT=/path/to/ema_weights.pt
_base_ckpt = os.environ.get(
    "SCIFORMA_STAGE2_CKPT",
    # local fallback path (B200 experiment)
    os.path.join(
        os.environ.get("SCIFORMA_DATA_ROOT", ""),
        "experiments/260216_stage2_mixed_gen_edit_b200_uniform_12wstep"
        "/checkpoint-90000/ema_weights.pt"
    )
)
policy_init_path = _base_ckpt
ref_init_path    = _base_ckpt   # reference = same init as policy (frozen)
ref_on_cpu       = False        # keep reference on GPU (B200 has 192GB VRAM)

use_lora    = False             # full fine-tuning
lora_layers = None

# ── Dataset ────────────────────────────────────────────────────────────────────
use_parquet_dataset = True

dataset_cfg = dict(
    type='ArXiVParquetDatasetMD3PO',
    base_dir=os.environ.get("SCIFORMA_DATA_ROOT", ""),
    parquet_base_path='ArXiV_parquet/0407_longshort_gdro_vae',  # SciFormaData preference pairs
    parquet_glob='gdro_rank_*.parquet',
    num_workers=8,
    path_remapping={'/mnt/data/': os.environ.get('SCIFORMA_DATA_ROOT', '') + '/'},
    deterministic_latents=True,

    # Preference pair selection
    target_dims=('component_score', 'text_score'),  # axes to optimize
    winner_key='reward',
    min_winner_score=0.70,
    target_min_gap=0.30,
    other_max_gap=0.50,
    loser_balance_lambda=0.50,
    min_total_gap=0.20,
    strict_all_dims=True,
    require_distinct_losers=True,
    min_group_images=6,

    # Global worst loser (additional negative beyond axis-specific)
    inject_global_worst=True,
    global_worst_min_gap=0.20,
)

sampler_cfg = dict(
    type='DistributedBucketSamplerV2',
    dataset=None, batch_size=1,
    num_replicas=1, rank=0,
    drop_last=True, shuffle=True,
)

# ── M-DPO loss ─────────────────────────────────────────────────────────────────
# Contrastive logsumexp: L = log(1 + Σ_d exp(-β·Δ_d))
# This enforces conjunctive correctness: all structural axes must improve.
train_iteration_func = 'Flux2Klein_md3po_train_iteration'
md3po_agg_mode       = 'contrastive'   # logsumexp multi-choice Bradley-Terry
md3po_alpha_mode     = 'uniform'
md3po_global_loss_weight = 0.0

dpo_beta       = 2000.0   # temperature
dpo_sft_weight = 0.0      # no SFT regularization

# ── Training ───────────────────────────────────────────────────────────────────
train_batch_size           = 1
gradient_accumulation_steps = 3    # 4 GPU × 1 × 3 = eff_batch 12
max_train_steps            = 10000
checkpointing_steps        = 500
validation_steps           = 500
gradient_checkpointing     = True
max_grad_norm              = 1.5   # compensate multi-loser gradient magnitude

# ── Optimizer ──────────────────────────────────────────────────────────────────
optimizer       = "adamw"
learning_rate   = 1e-6
lr_scheduler    = 'constant_with_warmup'
lr_warmup_steps = 50

adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema      = True
ema_decay    = 0.9999
ema_on_gpu   = True     # B200 has enough VRAM
ema_steps    = 100
ema_update_after_step = 0

# ── Precision ──────────────────────────────────────────────────────────────────
mixed_precision = "bf16"
allow_tf32      = True

# ── Data processing ────────────────────────────────────────────────────────────
max_sequence_length   = 1024
dataloader_num_workers = 8
pin_memory            = True

# ── Output ─────────────────────────────────────────────────────────────────────
model_output_dir = os.environ.get("SCIFORMA_DATA_ROOT", "") + "/experiments/sciforma-9b-mdpo"
output_dir       = model_output_dir
resume_from_checkpoint = 'latest'
checkpoints_total_limit = None

# ── Logging ────────────────────────────────────────────────────────────────────
logging_steps        = 5
log_with             = "wandb"
tracker_project_name = "SciForma-MDPO"
tracker_run_name     = "sciforma-9b-mdpo"

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 4.0
num_inference_steps       = 28

seed = 42
