#!/usr/bin/env python3
"""
SciForma — Pre-compute VAE Latents from Raw Images
====================================================
Generates NPZ cached latents from raw PNG/JPEG images.
Required before training with SciForma SFT scripts if you downloaded
raw images from HuggingFace rather than NPZ latents directly.

Each NPZ contains:
  - vae_h:         [2*C, H', W'] float32  — VAE encoder output (mean+logvar)
  - prompt_embeds: [SeqLen, 12288] float16 — Text embeddings from Qwen3 encoder
  - attention_mask:[SeqLen] bool           — Attention mask
  (text_ids are NOT stored — they are computed dynamically by _prepare_text_ids()
   in the training iteration function, with correct [SeqLen, 4] format)

NPZ path convention:
  {SCIFORMA_DATA_ROOT}/{parquet_base_path}/{year}/{paper_id}/{image_name}_flux_h.npz

This matches the `cache_path` column in the metadata parquet.

Usage:
    # Cache latents for Stage 1 parquet
    python scripts/cache_latents.py \
        --parquet_dir $SCIFORMA_DATA_ROOT/ArXiV_parquet/Flux2Klein9BParquet_0201_NEW \
        --image_base $SCIFORMA_DATA_ROOT/ArXiV_filtered_stages/stage4_aspect_quantize_filter \
        --base_model black-forest-labs/FLUX.2-klein-base-9B \
        --num_workers 4 \
        --batch_size 8

    # Or use the environment variable:
    export SCIFORMA_DATA_ROOT=/path/to/sciforma_data
    python scripts/cache_latents.py --parquet_dir $SCIFORMA_DATA_ROOT/ArXiV_parquet/Flux2Klein9BParquet_0201_NEW

Requirements:
    pip install diffusers transformers torch torchvision pandas pyarrow
    pip install git+https://github.com/huggingface/diffusers.git  # pinned: 3996788b

Hardware:
    Recommended: 1 GPU (A100/H100/B200) with at least 16 GB VRAM
    Encoding 655K images at batch=8 takes approximately 6-8 hours on A100.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_vae_encoder(base_model: str, device: str, dtype=torch.float32):
    """Load FLUX VAE encoder."""
    from diffusers import AutoencoderKL
    hf_token = os.environ.get("HF_TOKEN")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", token=hf_token)
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def load_text_encoder(base_model: str, device: str):
    """Load Qwen3 text encoder for prompt embeddings."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    hf_token = os.environ.get("HF_TOKEN")
    # FLUX.2-klein uses Qwen3 text encoder (stored as subfolder "text_encoder")
    # Check diffusers config for exact subfolder name
    tok = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer", token=hf_token)
    enc = AutoModelForCausalLM.from_pretrained(
        base_model, subfolder="text_encoder",
        torch_dtype=torch.float16, token=hf_token
    ).to(device)
    enc.eval()
    enc.requires_grad_(False)
    return tok, enc


def encode_image(vae, image_path: str, bucket_w: int, bucket_h: int, device: str, dtype) -> np.ndarray:
    """Encode a single image to VAE h (mean+logvar concatenated)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((bucket_w, bucket_h), Image.LANCZOS)
    img_tensor = torch.tensor(np.array(img)).float() / 127.5 - 1.0  # [-1, 1]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        dist = vae.encode(img_tensor).latent_dist
        # Return mean+logvar concatenated along channel dim
        h = torch.cat([dist.mean, dist.logvar], dim=1)  # [1, 2*C, H', W']
    return h.squeeze(0).float().cpu().numpy()  # [2*C, H', W']


def encode_text(tokenizer, text_encoder, caption: str, max_length: int, device: str) -> tuple:
    """Encode caption text to prompt_embeds and attention_mask."""
    inputs = tokenizer(
        caption,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(**inputs, output_hidden_states=True)
        embeds = outputs.hidden_states[-2]  # second-to-last hidden state for FLUX
    return (
        embeds.squeeze(0).half().cpu().numpy(),        # [SeqLen, Hidden]
        inputs.attention_mask.squeeze(0).bool().cpu().numpy(),  # [SeqLen]
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-compute VAE latents for SciForma training")
    parser.add_argument("--parquet_dir", required=True,
                        help="Directory containing the training parquet files (with cache_path column)")
    parser.add_argument("--image_base", default=None,
                        help="Root dir for raw images (image_path in parquet is relative to this)")
    parser.add_argument("--base_model", default="black-forest-labs/FLUX.2-klein-base-9B",
                        help="HuggingFace model ID for VAE and text encoder")
    parser.add_argument("--output_base", default=None,
                        help="Where to write NPZ files (default: same as parquet_dir, next to parquet files)")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Maximum text sequence length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per GPU batch (for memory efficiency)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip NPZ files that already exist")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Loading VAE from: {args.base_model}")
    vae = load_vae_encoder(args.base_model, args.device)
    print(f"Loading text encoder from: {args.base_model}")
    tokenizer, text_encoder = load_text_encoder(args.base_model, args.device)

    # Find all parquet files
    files = sorted(glob.glob(str(Path(args.parquet_dir) / "**/*.parquet"), recursive=True))
    print(f"Found {len(files)} parquet files")

    total_encoded = 0
    total_skipped = 0

    for parquet_file in tqdm(files, desc="Parquet files"):
        # Support both old format (with cache_path) and HF format (image_path only)
        df = pd.read_parquet(parquet_file)
        has_cache_path = 'cache_path' in df.columns
        parquet_dir = Path(parquet_file).parent

        # Resolve width/height column names (old: bucket_w/bucket_h, HF: width/height)
        w_col = 'bucket_w' if 'bucket_w' in df.columns else 'width'
        h_col = 'bucket_h' if 'bucket_h' in df.columns else 'height'

        # Detect image path column(s): generation has 'image_path', editing has source/target
        is_editing = 'source_image_path' in df.columns and 'target_image_path' in df.columns
        if not is_editing and 'image_path' not in df.columns and not has_cache_path:
            print(f"  WARN: no image_path or source_image_path in {parquet_file}, skipping")
            continue

        # Build list of (img_path, caption) to encode
        def get_image_rows(row):
            """Returns list of (img_path, caption) to encode for this row."""
            caption = row.get('caption', '')
            if is_editing:
                return [(row['source_image_path'], caption), (row['target_image_path'], caption)]
            else:
                return [(row.get('image_path', ''), caption)]

        for _, row in df.iterrows():
            for img_path, caption in get_image_rows(row):
                if not img_path:
                    continue

                # Derive NPZ path from image path (replace .png with _flux_h.npz)
                if has_cache_path and not is_editing:
                    # Old format: explicit cache_path field
                    npz_rel = row["cache_path"]
                    npz_out = Path(args.output_base or parquet_dir) / npz_rel
                else:
                    # HF format: derive NPZ path alongside image
                    img_rel = img_path
                    npz_rel = img_rel.replace('.png', '_flux_h.npz').replace('.jpg', '_flux_h.npz')
                    img_base = Path(args.image_base) if args.image_base else parquet_dir
                    npz_out = img_base / npz_rel

                if args.skip_existing and npz_out.exists():
                    total_skipped += 1
                    continue

                image_full = (Path(args.image_base) / img_path
                              if args.image_base else parquet_dir / img_path)

                if not image_full.exists():
                    print(f"  WARN: image not found: {image_full}")
                    continue

                try:
                    # Encode image
                    w = int(row.get(w_col, 1024))
                    h = int(row.get(h_col, 512))
                    vae_h = encode_image(vae, str(image_full), w, h,
                                         args.device, torch.float32)

                    # Encode text
                    prompt_embeds, attention_mask = encode_text(
                        tokenizer, text_encoder, caption, args.max_seq_len, args.device
                    )

                    # Save NPZ
                    npz_out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        str(npz_out),
                        vae_h=vae_h,
                        prompt_embeds=prompt_embeds,
                        attention_mask=attention_mask,
                    )
                    total_encoded += 1

                except Exception as e:
                    print(f"  ERROR encoding {image_full}: {e}")

    print(f"\nDone. Encoded: {total_encoded:,}, Skipped (existing): {total_skipped:,}")


if __name__ == "__main__":
    main()
