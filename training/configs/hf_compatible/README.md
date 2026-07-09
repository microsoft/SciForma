# SciForma HuggingFace-Compatible Training Configs

These configs enable training directly on data downloaded from HuggingFace.

## Directory Structure Required

```
$SCIFORMA_DATA_ROOT/SciFormaData-700K/
├── generation/
│   ├── metadata.parquet              ← 661,660 rows, columns: paper_id, image_path, caption, ...
│   ├── images_768/768_pretrain/      ← 661,660 PNGs (Low/Medium quality, 155 GB)
│   └── images_1024/1024_pretrain/    ← 651,860 PNGs (High quality, 241 GB)
└── editing/
    ├── metadata.parquet              ← 70,866 pairs, columns: source/target_image_path, caption, ...
    └── images/260209_1024/           ← source.png + target.png pairs (110 GB)
```

## Step 1: Download Dataset

```bash
export SCIFORMA_DATA_ROOT=/your/data/path

# Download from HuggingFace (when published)
huggingface-cli download microsoft/SciFormaData-700K \
    --local-dir $SCIFORMA_DATA_ROOT/SciFormaData-700K
```

## Step 2: Pre-compute VAE Latents

Required before training. Creates `*_flux_h.npz` alongside each PNG.

```bash
# Stage 1 + Stage 2 generation latents (~6-8 hours on A100)
python scripts/cache_latents.py \
    --parquet_dir $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation \
    --image_base $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation \
    --base_model black-forest-labs/FLUX.2-klein-base-9B

# Editing latents (for Stage 2 mixed training, ~2-3 hours)
python scripts/cache_latents.py \
    --parquet_dir $SCIFORMA_DATA_ROOT/SciFormaData-700K/editing \
    --image_base $SCIFORMA_DATA_ROOT/SciFormaData-700K/editing \
    --base_model black-forest-labs/FLUX.2-klein-base-9B
```

## Step 3: Train

```bash
export SCIFORMA_DATA_ROOT=/your/data/path

# Stage 1 SFT — all quality, 661,274 samples
accelerate launch --config_file accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
    scripts/train_sft.py configs/hf_compatible/stage1_sft_hf.py

# Stage 2 SFT — High quality only, 244,090 samples (generation only)
accelerate launch --config_file accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
    scripts/train_sft.py configs/hf_compatible/stage2_sft_hf.py

# Stage 2 SFT — Mixed: High quality gen (244K) + editing triplets (70K)
# This is the config that produces SciForma-Base
export SCIFORMA_STAGE1_CKPT=$SCIFORMA_DATA_ROOT/experiments/sciforma-base-stage1/checkpoint-140000/ema_weights.pt
accelerate launch --config_file accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
    scripts/train_sft.py configs/hf_compatible/stage2_sft_mixed_hf.py
```

## Dataset Details

| Config | Dataset | Rows | Description |
|--------|---------|------|-------------|
| `stage1_sft_hf.py` | `ArXiVHFDatasetV1` | **661,274** | All quality (2015-2025) |
| `stage2_sft_hf.py` | `ArXiVHFDatasetV1(High)` | **244,090** | High quality only |
| `stage2_sft_mixed_hf.py` | Gen(244K) + Edit(70K) | **314,953** | Mixed gen+edit → SciForma-Base |

Note: A small number of empty-caption rows are auto-filtered (gen: 386, edit: 3).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCIFORMA_DATA_ROOT` | (required) | Root directory of HF dataset |
| `SCIFORMA_OUTPUT_DIR` | `$SCIFORMA_DATA_ROOT/experiments/...` | Where to save checkpoints |
| `SCIFORMA_STAGE1_CKPT` | `$SCIFORMA_DATA_ROOT/experiments/sciforma-base-stage1/checkpoint-140000/ema_weights.pt` | Stage 1 checkpoint for Stage 2 init |
| `SCIFORMA_GT_BASE` | `$SCIFORMA_DATA_ROOT/SciFormaData-700K/generation/images_1024/1024_pretrain` | GT images for benchmark eval |

## Benchmark Evaluation

```bash
export SCIFORMA_GT_BASE=$SCIFORMA_DATA_ROOT/SciFormaData-700K/generation/images_1024/1024_pretrain

python benchmark/evaluate.py \
    --gen_dir ./outputs/sciforma-9b \
    --output_dir ./eval_results/sciforma-9b
```
