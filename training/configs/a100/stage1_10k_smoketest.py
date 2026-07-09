"""
SciForma Stage 1 — 10K Step Smoke Test on A100 4-GPU
======================================================
Reproduces the original ArXivQwenImage Stage1 training for 10K steps
to verify the full pipeline works end-to-end on A100.

Data: Local 2015 subset (~5,086 samples, pre-computed NPZ latents)
      /data/sciforma/ArXiV_parquet/flux_latents_saveh/

Hardware: A100 80GB PCIe × 4 (GPU 0,1 free; avoid GPU 2)
Effective batch = 1 × 4GPU × 4GA = 16  (matches B200 Stage1 eff batch)

WandB: logs to wandb.ai project SciForma

Init: from SciForma-Base (our trained Stage1 EMA weights) for faster convergence check.

To run:
    
    
    
    export CUDA_VISIBLE_DEVICES=0,1,3,4   # skip GPU 2 if occupied

    conda activate flux2
    cd /path/to/SciForma
    accelerate launch --config_file accelerate_cfg/deepspeed_zero2_bf16.yaml \\
        --num_processes 4 \\
        scripts/train_sft.py configs/a100/stage1_10k_smoketest.py
"""
import os

_base_ = ['../base_config.py']

# ── Model ──────────────────────────────────────────────────────────────────────
model_type = 'Flux2Klein'
pretrained_model_name_or_path = "black-forest-labs/FLUX.2-klein-base-9B"
huggingface_token = None
transformer_cfg = dict(type='Flux2Transformer2DModel')

# ── Full fine-tuning (not LoRA) ────────────────────────────────────────────────
use_lora = False

# Init from SciForma-Base to verify weight loading pipeline
init_weights_path = "/data/sciforma/sciforma_base_ema.pt"

# ── Dataset: Local 2015 subset with pre-computed NPZ latents ───────────────────
use_parquet_dataset = True
dataset_cfg = dict(
    type='ArXiVParquetDatasetV3',
    base_dir='/data/sciforma',
    parquet_base_path='ArXiV_parquet/flux_latents_saveh',
    num_workers=4,
    num_train_examples=None,  # all 5,086 samples from 2015
    debug_mode=False,
    is_main_process=True,
    stat_data=True,
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

# ── Training ───────────────────────────────────────────────────────────────────
train_iteration_func = 'Flux2Klein_fulltune_train_iteration'

train_batch_size           = 1
gradient_accumulation_steps = 4   # eff_batch = 1×4GPU×4GA = 16
max_train_steps            = 10000
checkpointing_steps        = 2000
validation_steps           = 1000
gradient_checkpointing     = True    # recompute activations: ~30GB → ~5GB, enables single A100
max_grad_norm              = 1.0

# ── Optimizer ──────────────────────────────────────────────────────────────────
optimizer       = "adamw"
use_8bit_adam  = True   # 8-bit Adam: optimizer states ~9 GB (vs fp32 72 GB) → fits single A100
learning_rate   = 1e-5
lr_scheduler    = 'constant_with_warmup'
lr_warmup_steps = 100    # short warmup for smoke test

adam_beta1        = 0.9
adam_beta2        = 0.999
adam_weight_decay = 0.01
adam_epsilon      = 1e-8

# ── EMA ────────────────────────────────────────────────────────────────────────
use_ema      = True
ema_decay    = 0.9999
ema_on_gpu   = False   # CPU EMA — essential for A100 80GB
ema_steps    = 100
ema_update_after_step = 0

# ── Precision ──────────────────────────────────────────────────────────────────
mixed_precision = "bf16"
allow_tf32      = True

# ── Data ───────────────────────────────────────────────────────────────────────
max_sequence_length    = 1024
dataloader_num_workers = 4
pin_memory             = True

# ── Output ─────────────────────────────────────────────────────────────────────
model_output_dir = "/data/sciforma/experiments/sciforma_smoketest_10k"
output_dir       = model_output_dir
logging_dir      = "logs"
resume_from_checkpoint = None    # fresh start for smoke test
checkpoints_total_limit = 2

# ── WandB ──────────────────────────────────────────────────────────────────────
logging_steps        = 10
log_with             = "wandb"
tracker_project_name = "SciForma"
tracker_run_name     = "sciforma_stage1_smoketest_a100_10ksteps"

# ── Validation ─────────────────────────────────────────────────────────────────
validation_func           = 'Flux2Klein_fulltune_validation_func_parquet'
validation_guidance_scale = 3.5
num_inference_steps       = 20   # fewer steps for quick validation

# validation_prompts is REQUIRED — without it train_sft.py skips validation entirely
validation_prompts = [
    "The figure illustrates a neural network architecture with three layers: input, hidden, and output. Blue rectangles represent neurons connected by weighted edges shown as arrows. The input layer has four nodes receiving feature vectors, the hidden layer has three nodes with ReLU activation, and the output layer has two nodes with softmax for classification.",
    "The figure shows a bar chart comparing model performance across five datasets. The x-axis lists dataset names (MNIST, CIFAR-10, ImageNet, COCO, VOC) and the y-axis shows accuracy percentage from 0 to 100. Three colored bars per dataset represent three methods: blue for baseline, orange for proposed method, and green for state-of-the-art.",
]
resolution_list = [
    [576, 1024],
    [576, 1024],
]

seed = 42
