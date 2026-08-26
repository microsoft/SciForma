<div align="center">

# SciForma: Structure-Faithful Generation of Scientific Diagrams

<p>
  <a href="https://arxiv.org/abs/2607.18091"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2607.18091-b31b1b?logo=arxiv&logoColor=white" height="22" /></a>
  &nbsp;
  <a href="https://microsoft.github.io/SciForma/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2ea44f?logo=githubpages&logoColor=white" height="22" /></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/microsoft/SciFormaData-700K"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-green" height="22" /></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/microsoft/SciFormaBench"><img alt="Benchmark" src="https://img.shields.io/badge/%F0%9F%A4%97-Benchmark-blue" height="22" /></a>
  &nbsp;
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg" height="22" /></a>
</p>

**Yuxuan Luo<sup>1</sup> · Peng Zhang<sup>2</sup> · Xinjie Zhang<sup>3✉</sup> · Xun Guo<sup>3</sup> · Zhouhui Lian<sup>1✉</sup> · Yan Lu<sup>3</sup>**

<sup>1</sup>Peking University &nbsp;&nbsp; <sup>2</sup>Zhejiang University &nbsp;&nbsp; <sup>3</sup>Microsoft Research Asia

<a href="assets/teaser.pdf"><img src="assets/teaser.png" width="100%" alt="SciForma Teaser"></a>

</div>

---

**SciForma** is a 9B-parameter framework for the *structure-faithful* generation of scientific methodology diagrams. SciForma decomposes diagram quality into three independently verifiable axes — **Component**, **Arrow**, and **Text** — guided by a *structural inventory*. Built on FLUX.2-klein-base-9B, it is fine-tuned in two SFT stages then post-trained with **Multi-Dimensional Conjunctive Preference Optimization (M-DPO)**, which enforces simultaneous correctness across all axes and adaptively routes gradients toward the most deficient axis.

## Highlights

- **Structural inventory.** Per-axis (Component / Arrow / Text) verification with critical/moderate error severity instead of a single holistic score.
- **M-DPO.** Multi-way Bradley–Terry objective with dimension-anchored preference construction and adaptive gradient reweighting — breaks the SFT plateau where scalar DPO / GDRO / GRPO stagnate.
- **LaTeX-to-Diagram Agent.** Generate figures directly from your paper's LaTeX source via a GPT planner and a locally trained SciForma checkpoint.
- **Data & benchmark.** `SciFormaData-700K` (661K generation pairs + 70K editing triplets) and `SciFormaBench-2K` (Simple 500 / Medium 900 / Hard 600) with human-verified inventories.

---

## Results

On **SciFormaBench-2K** (GPT-5.4 judge):

| Method                          | Average | Component | Arrow | Text |
| :------------------------------ | :-----: | :-------: | :---: | :---: |
| GPT-Image-2                     |  85.62  |   83.34   | 89.61 | 83.53 |
| Nano Banana Pro                 |  81.34  |   81.10   | 83.60 | 78.70 |
| GPT-Image-1.5                   |  68.96  |   75.70   | 62.50 | 68.20 |
| **SciForma-9B (+ M-DPO)** |  69.51  |   74.49   | 66.46 | 67.00 |
| **SciForma-Base (SFT)**   |  67.59  |   73.52   | 64.64 | 63.84 |
| Wan2.7-Image                    |  64.71  |   73.90   | 60.30 | 61.10 |
| FLUX.2-klein-base-9B            |  33.87  |   51.50   | 25.20 | 23.60 |

On **AIBench**, SciForma-9B reaches **70.29**, edging human-drawn originals (70.09) with the largest margin on Topology (+6.19). As a drop-in Visualizer inside PaperBanana, SciForma achieves a **30.7%** pairwise win rate against human-drawn originals.

## 📦 Installation

### Requirements

- **Python 3.10** (mmengine 0.10.7 requires 3.10; Python 3.11+ not tested), CUDA 12.1
- GPU: ≥ 24 GB VRAM for inference (one A100/H100/B200); 8× B200 for training

### Setup

```bash
# Clone
git clone https://github.com/microsoft/SciForma.git
cd SciForma

# Create conda environment
conda create -n sciforma python=3.10 -y
conda activate sciforma

# Install PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from git (required for Flux2Klein / Flux2Transformer2DModel support)
pip install git+https://github.com/huggingface/diffusers.git

# Install all other dependencies
pip install -r requirements.txt

# (Optional, recommended for training) Flash Attention 2 — requires nvcc
# pip install flash-attn --no-build-isolation
```

Or use the provided setup script:

```bash
bash env_setup/init_flux2_env.sh
```

### Set credentials

```bash
export SCIFORMA_DATA_ROOT="/path/to/data"   # root for training data & checkpoints
export WANDB_API_KEY="..."         # optional, for training monitoring
```

### Verify Installation

After setup, run the verification script to confirm everything is working:

```bash
# Basic check (imports + configs, ~10 seconds): 10 passed expected
python training/scripts/verify_setup.py

# Full check including dataset + benchmark GT + scripts: 26 passed expected
export SCIFORMA_DATA_ROOT=/your/data/path
export SCIFORMA_GT_BASE=$SCIFORMA_DATA_ROOT/SciFormaData-700K/generation/images_1024/1024_pretrain
python training/scripts/verify_setup.py --check-data --check-scripts
```

Expected output: `✅ All critical checks passed — ready to train!`

---

## 💡 Usage

### Benchmark Evaluation (SciFormaBench-2K)

Prompts, GT images, and rubrics are loaded automatically from [`microsoft/SciFormaBench`](https://huggingface.co/datasets/microsoft/SciFormaBench) — no local data needed.

**Step 1 — Generate images:**

```bash
# From a local EMA checkpoint (your own trained model)
python generate/benchmark.py \
    --model_path /path/to/local/FLUX.2-klein-base-9B \
    --ema_weights /path/to/checkpoint-90000/ema_weights.pt \
    --split simple medium hard \
    --output_dir ./results/my-model
```

Generated images are saved as `promptNNNN_<slug>.png` in `<output_dir>/<split>/cfg_4.0/`.

**Step 2 — Score with GPT (Component / Arrow / Text axes):**

```bash
# GT images + rubrics downloaded from microsoft/SciFormaBench automatically
python eval/eval_benchmark.py \
    --gen_dir  ./results/my-model \
    --output_dir ./eval_results/my-model \
    --deployment_name gpt-4o
```

Results in `eval_results/my-model/eval_summary.json`.

> **Judge model**: Paper used `gpt-5.4`. `gpt-4o` is the publicly available alternative with comparable scores (±1%). Set `AZURE_OPENAI_ENDPOINT` in `.env` for Azure, or `OPENAI_API_KEY` for standard OpenAI.

---

### LaTeX-to-Diagram Agent

Generate a diagram directly from your LaTeX source code — no manual prompt writing needed.

```bash
# List all figures in your paper
python generate/latex_to_diagram.py --latex paper.tex --list_captions

# Generate a specific figure (LLM + local SciForma checkpoint)
python generate/latex_to_diagram.py \
    --latex paper.tex \
    --caption "Overview of the proposed method." \
    --model_path /path/to/local/pipeline \
    --output figure.png
```

The pipeline runs: LaTeX parser → Planner (GPT) → Condense → local SciForma checkpoint.
Requires `AZURE_OPENAI_ENDPOINT` or `OPENAI_API_KEY` in `.env` for the planning step.

---

## 📚 Dataset

**SciFormaData-700K** — [🤗 microsoft/SciFormaData-700K](https://huggingface.co/datasets/microsoft/SciFormaData-700K) · License: CC BY-NC 4.0

| Split         | Rows    | Resolution   | Description                                    |
| ------------- | ------- | ------------ | ---------------------------------------------- |
| `gen_768`   | 661,660 | 768px-equiv  | Generation pairs, all quality tiers            |
| `gen_1024`  | 651,860 | 1024px-equiv | Generation pairs, High+Medium quality          |
| `edit_1024` | 70,866  | 1024px-equiv | Editing triplets (source, instruction, target) |

All three splits are stored in a **single unified parquet** (`metadata.parquet`) with a `split` column for easy filtering.

### Loading the dataset

```python
import pandas as pd

# HuggingFace (after release)
# from datasets import load_dataset
# ds = load_dataset("microsoft/SciFormaData-700K")

# Local / after download
df = pd.read_parquet("SciFormaData-700K/metadata.parquet")

# --- Generation 768px (Stage 1 training) ---
gen_768 = df[df["split"] == "gen_768"]
# columns: paper_id, caption, image_path, bucket_w, bucket_h, quality_tier, is_tikz

# --- Generation 1024px, High quality only (Stage 2 training) ---
gen_1024_high = df[(df["split"] == "gen_1024") & (df["quality_tier"] == "High")]
# 244,304 rows

# --- Editing triplets (Stage 2 mixed training) ---
edit = df[df["split"] == "edit_1024"]
# columns: paper_id, caption (edit instruction), source_image_path, target_image_path
```

**Key fields**:

- `caption` — structured text prompt (avg. 538 tokens, axis-decomposed: Components / Arrows / Text)
- `image_path` — relative path to PNG (generation splits only)
- `source_image_path` / `target_image_path` — relative paths to source and target PNGs (edit split)
- `quality_tier` — `"High"` / `"Medium"` / `"Low"` (generation splits)
- `bucket_w`, `bucket_h` — quantized training resolution in pixels
- `is_tikz` — whether the diagram was rendered from TikZ source (2.6% of generation)

### SciFormaBench-2K

Evaluation benchmark — [`microsoft/SciFormaBench`](https://huggingface.co/datasets/microsoft/SciFormaBench):

| Split         | Size |
| ------------- | ---- |
| Simple        | 500  |
| Medium        | 900  |
| Hard          | 600  |

Prompts, GT images, and structural rubrics are in [`microsoft/SciFormaBench`](https://huggingface.co/datasets/microsoft/SciFormaBench).

---

## 🏋️ Training

SciForma is trained in three stages on NVIDIA B200 GPUs.

| Stage       | Config                             | Data                   | Steps | Output                  |
| ----------- | ---------------------------------- | ---------------------- | ----- | ----------------------- |
| Stage 1 SFT | `training/configs/stage1_sft.py` | 661K gen pairs         | 200K  | —                      |
| Stage 2 SFT | `training/configs/stage2_sft.py` | 244K gen + 70K edit    | 120K  | **SciForma-Base** |
| M-DPO       | `training/configs/mdpo/mdpo.py`  | ~42K preference groups | 5K    | **SciForma-9B**   |

### Stage 1: SFT Pretraining

```bash
bash training/scripts/run_stage1.sh
```

Config: `training/configs/stage1_sft.py` · Accelerate: `training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml`

Key hyperparameters: AdamW `lr=1e-5`, batch=2/GPU × 8 GPUs × GA=1 → effective batch 16, `ema_decay=0.9999`.

---

### Stage 2: Joint Generation + Editing

```bash
# Set init_weights_path → Stage 1 checkpoint-140000/ema_weights.pt
accelerate launch \
    --config_file training/accelerate_cfg/b200_zero2_bf16_8gpu.yaml \
    training/scripts/run_stage2.sh
```

Config: `training/configs/stage2_sft.py`

Adds inventory-aligned editing triplets co-trained with generation. Source tokens receive RoPE temporal offset T=10; target T=0. Loss mode: `uniform` (standard flow-matching MSE across full target).

---

### M-DPO: Multi-Dimensional Conjunctive Preference Optimization

```bash
bash training/scripts/run_mdpo.sh
```

Config: `training/configs/mdpo/mdpo.py`

**M-DPO loss** — for each group, the winner `w` is contrasted against three axis-specific losers `{l_C, l_A, l_T}` using a multi-way Bradley–Terry objective:

```
logits_d = (L_ref(w) − L_π(w)) − (L_ref(l_d) − L_π(l_d))   for d ∈ {C, A, T}
```

Three aggregation modes (`md3po_agg_mode`):

| Mode            | Formula                                               | Use                                              |
| --------------- | ----------------------------------------------------- | ------------------------------------------------ |
| `contrastive` | `logsumexp(cat([0, −β·logits]))`                 | Multi-choice Bradley-Terry; losers coupled       |
| `adaptive`    | `sum(softmax(−β·logits, detach) · per_dim_DPO)` | Softmax as gradient router; each dim independent |
| `mean`        | Hardness-based alpha weighting                        | + optional global worst loser                    |

Key hyperparameters: `dpo_beta=2000`, `sft_weight=0`, `lr=1e-6`, logit-normal timestep weighting.

---

## License

The code in this repository is released under the [MIT License](LICENSE). Users are responsible for complying with the licenses of any base models and checkpoints they use.

## 🙏 Acknowledgements

SciForma builds on [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) (Black Forest Labs), [diffusers](https://github.com/huggingface/diffusers) (HuggingFace), and [mmengine](https://github.com/open-mmlab/mmengine) (OpenMMLab). Editing triplets were constructed with [SAM3](https://github.com/ysun822/SAM3).
