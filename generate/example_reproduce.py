#!/usr/bin/env python3
"""
SciForma — Minimal Reproducible Example
=========================================
Reproduces the specific benchmark image:
  hard split, local index 494 (old_local_idx=640)
  "The figure illustrates a three-stage pipeline for efficient 3D scene rendering..."

Expected output: prompt0494_The_figure_illustrates_a_three-stage_pipeline_for.png
Resolution: 1728 × 576

⚠️  REPRODUCIBILITY NOTE (复现性说明)
torch.Generator(device="cuda") is REQUIRED to reproduce paper results.
torch.Generator("cpu") produces completely different noise even with the same seed.
必须使用 torch.Generator(device="cuda")，CPU generator 即使相同 seed 也产生不同噪声。

Usage:
    python generate/example_reproduce.py \
        --ema_weights /path/to/checkpoint-4000/ema_weights.pt \
        --output_dir ./outputs/example
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

# ── Paper-fixed constants ─────────────────────────────────────────────────────
CFG       = 4.0
NUM_STEPS = 50
MAX_SEQ   = 2048
SEED_BASE = 42
DTYPE     = torch.bfloat16

# ── Target sample ─────────────────────────────────────────────────────────────
SPLIT       = "hard"
LOCAL_IDX   = 494      # index in eval/prompts/hard.json (current 600-prompt set)
OLD_LOCAL_IDX = 640    # index in original 800-prompt generation set (used for paper seed)

# ⚠️  CRITICAL: Use old_local_idx for seed to reproduce paper result.
# The paper generated images using seed = SEED_BASE + old_local_idx (i.e. 42+640=682).
# Using seed = SEED_BASE + LOCAL_IDX (42+494=536) produces a DIFFERENT image.
REPRODUCTION_SEED = SEED_BASE + OLD_LOCAL_IDX  # = 682, matches paper

# File name: prompt0494_The_figure_illustrates_a_three-stage_pipeline_for.png


def load_ema_weights(ema_path: str, transformer) -> None:
    """Load EMA shadow_params into transformer (required for paper-quality output)."""
    print(f"  Loading EMA weights: {ema_path}")
    ema = torch.load(ema_path, map_location="cpu", weights_only=False)
    shadow = ema["shadow_params"]
    names = [n for n, _ in transformer.named_parameters()]
    assert len(shadow) == len(names), f"Mismatch: shadow={len(shadow)} model={len(names)}"
    state = OrderedDict((n, s.to(DTYPE)) for n, s in zip(names, shadow))
    transformer.load_state_dict(state, strict=False)
    print(f"  Loaded {len(state)} EMA parameters ✓")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Reproduce a single SciForma benchmark image")
    parser.add_argument("--model_path", default="LoYuXrqw/SciForma-9B",
                        help="HuggingFace repo ID or base model path")
    parser.add_argument("--ema_weights", default=None,
                        help="Path to ema_weights.pt (overrides model_path weights)")
    parser.add_argument("--output_dir", default="./outputs/example",
                        help="Where to save the generated image")
    args = parser.parse_args()

    # ── Load benchmark prompt ─────────────────────────────────────────────────
    prompt_file = repo_root / "benchmark" / "prompts" / f"{SPLIT}.json"
    bench  = json.loads(prompt_file.read_text())
    prompt = bench["validation_prompts"][LOCAL_IDX]
    width, height = bench["resolution_list"][LOCAL_IDX]  # [1728, 576]

    print(f"\nTarget: {SPLIT} split, index={LOCAL_IDX} (original index={OLD_LOCAL_IDX})")
    print(f"Resolution: {width}×{height}")
    print(f"Seed: {SEED_BASE} + {OLD_LOCAL_IDX} (old_local_idx) = {REPRODUCTION_SEED}")
    print(f"Prompt: {prompt[:80]}...\n")

    # ── Load model ────────────────────────────────────────────────────────────
    hf_token = __import__("os").environ.get("HF_TOKEN")
    print(f"Loading pipeline: {args.model_path}")
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_path, torch_dtype=DTYPE, token=hf_token,
    )
    if args.ema_weights:
        load_ema_weights(args.ema_weights, pipe.transformer)

    pipe = pipe.to("cuda")   # full GPU — matches original inference setup
    pipe.transformer.eval()
    print("Pipeline ready.\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    # ⚠️  CRITICAL: must use Generator(device="cuda"), NOT Generator("cpu")
    # Verified: cuda generator reproduces ~9px mean diff vs original;
    #           cpu generator produces ~43px mean diff (4.9x worse)
    # ⚠️  CRITICAL: use REPRODUCTION_SEED (=682) not SEED_BASE+LOCAL_IDX (=536)
    # Paper generated with seed = 42 + old_local_idx = 42 + 640 = 682
    generator = torch.Generator(device="cuda").manual_seed(REPRODUCTION_SEED)

    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG,
            max_sequence_length=MAX_SEQ,
            generator=generator,
            output_type="pil",
        ).images[0]

    # ── Save ──────────────────────────────────────────────────────────────────
    import os, re
    os.makedirs(args.output_dir, exist_ok=True)
    slug = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt[:50]).strip().replace(" ", "_")
    fname = f"prompt{LOCAL_IDX:04d}_{slug}.png"
    out_path = os.path.join(args.output_dir, fname)
    image.save(out_path)

    print(f"✓ Saved: {out_path}")
    print(f"  Size: {image.size}")
    print()
    print("To verify reproduction quality vs. original:")
    print("  import numpy as np; from PIL import Image")
    print(f"  orig = Image.open('/path/to/original/prompt0640_....png')")
    print(f"  gen  = Image.open('{out_path}')")
    print("  diff = np.abs(np.array(orig).astype(int) - np.array(gen).astype(int)).mean()")
    print("  # Expected: ~9px with cuda generator | ~43px with cpu generator")


if __name__ == "__main__":
    main()
