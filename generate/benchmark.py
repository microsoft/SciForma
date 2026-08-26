#!/usr/bin/env python3
"""
SciForma — Benchmark Image Generation

Generates images for SciFormaBench-2K (simple / medium / hard splits).
Loads models from local directories and supports local EMA checkpoints
(`ema_weights.pt`).

Usage — local EMA checkpoint (e.g., your own trained model):
    python generate/benchmark.py \
        --model_path /path/to/local/FLUX.2-klein-base-9B \
        --ema_weights /path/to/checkpoint-90000/ema_weights.pt \
        --split simple medium hard \
        --output_dir ./benchmark_outputs/sciforma-base

Output layout:
    <output_dir>/
      simple/cfg_4.0/prompt0000_<slug>.png
      medium/cfg_4.0/prompt0000_<slug>.png
      hard/cfg_4.0/prompt0000_<slug>.png
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline


CFG = 4.0
NUM_STEPS = 50
MAX_SEQ_LEN = 2048
SEED_BASE = 42
DTYPE = torch.bfloat16

SPLIT_MAP = {
    "simple": "prompts/simple.json",
    "medium": "prompts/medium.json",
    "hard":   "prompts/hard.json",
}

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "eval"


def load_ema_into_transformer(ema_weights_path: str, transformer) -> None:
    """
    Load shadow_params from ema_weights.pt directly into the transformer.

    ema_weights.pt format (saved by EMAModel during training):
        {
          "shadow_params": [tensor, tensor, ...],  # one per transformer parameter
          "decay": float,
          "optimization_step": int,
        }

    The shadow_params list is ordered identically to transformer.parameters(),
    so we map by position, then load via load_state_dict.
    """
    print(f"  Loading EMA weights from: {ema_weights_path}")
    ema_state = torch.load(ema_weights_path, map_location="cpu", weights_only=False)
    shadow_params = ema_state.get("shadow_params")

    if shadow_params is None:
        raise ValueError(f"'shadow_params' not found in {ema_weights_path}")

    # Build a state dict: {param_name: shadow_tensor}
    param_names = [name for name, _ in transformer.named_parameters()]
    if len(shadow_params) != len(param_names):
        raise ValueError(
            f"Param count mismatch: shadow_params={len(shadow_params)}, "
            f"transformer={len(param_names)}"
        )

    state_dict = OrderedDict()
    for name, shadow in zip(param_names, shadow_params):
        state_dict[name] = shadow.to(dtype=DTYPE)

    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys in EMA load")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys in EMA load")
    print(f"  Loaded {len(state_dict)} EMA parameters ✓")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        required=True,
        help="Local full-pipeline directory. Remote model IDs are not accepted.",
    )
    parser.add_argument(
        "--ema_weights", type=str, default=None,
        help="Path to ema_weights.pt. When set, shadow_params override model_path weights.",
    )
    parser.add_argument(
        "--split", type=str, nargs="+",
        choices=["simple", "medium", "hard"],
        default=["simple", "medium", "hard"],
        help="Which benchmark splits to run",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_outputs/sciforma",
        help="Root directory for generated images",
    )
    parser.add_argument(
        "--benchmark_dir", type=str, default=None,
        help="Local directory containing benchmark JSON files. "
             "If not set, prompts are loaded from --hf_benchmark automatically.",
    )
    parser.add_argument(
        "--hf_benchmark", type=str, default="microsoft/SciFormaBench",
        help="HuggingFace dataset ID for benchmark prompts (default: microsoft/SciFormaBench). "
             "Used when --benchmark_dir is not set.",
    )
    parser.add_argument("--cfg",        type=float, default=CFG)
    parser.add_argument("--steps",      type=int,   default=NUM_STEPS)
    parser.add_argument("--max_seq_len",type=int,   default=MAX_SEQ_LEN)
    parser.add_argument("--seed",       type=int,   default=SEED_BASE)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-generated images (resume interrupted run)",
    )
    parser.add_argument(
        "--max_images", type=int, default=None,
        help="Limit number of images per split (useful for quick tests)",
    )
    return parser.parse_args()


def get_existing_indices(output_dir: str) -> set:
    existing = set()
    if not os.path.isdir(output_dir):
        return existing
    pat = re.compile(r"^prompt(\d+)_.*\.png$")
    for fname in os.listdir(output_dir):
        m = pat.match(fname)
        if m:
            existing.add(int(m.group(1)))
    return existing


def slug(text: str, max_len: int = 50) -> str:
    s = "".join(c if c.isalnum() or c in " -_" else "_" for c in text[:max_len])
    return s.strip().replace(" ", "_")


def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"--model_path must be a local directory: {model_path}")

    print(f"\n{'='*60}")
    print(f"Loading local pipeline: {model_path}")
    pipe = Flux2KleinPipeline.from_pretrained(
        str(model_path), torch_dtype=DTYPE, local_files_only=True
    )

    if args.ema_weights:
        print(f"\nApplying EMA weights: {args.ema_weights}")
        load_ema_into_transformer(args.ema_weights, pipe.transformer)

    pipe.enable_model_cpu_offload()
    pipe.transformer.eval()
    print("Pipeline ready.\n")

    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = args.hf_benchmark

    for split in args.split:
        if hf_repo and not args.benchmark_dir:
            from datasets import load_dataset
            print(f"Loading {split} prompts from {hf_repo} ...")
            ds = load_dataset(hf_repo, split, split="test", token=hf_token)
            if args.max_images:
                ds = ds.select(range(min(args.max_images, len(ds))))
            prompts     = [row["prompt"] for row in ds]
            resolutions = [[row["width"], row["height"]] for row in ds]
        else:
            bench_path = os.path.join(args.benchmark_dir, SPLIT_MAP[split])
            with open(bench_path) as f:
                bench = json.load(f)
            prompts     = bench["validation_prompts"]
            resolutions = bench["resolution_list"]
        output_dir = os.path.join(args.output_dir, split, f"cfg_{args.cfg}")
        os.makedirs(output_dir, exist_ok=True)

        existing = get_existing_indices(output_dir) if args.resume else set()
        total = len(prompts)
        if args.max_images:
            total = min(total, args.max_images)
        skip = len(existing)
        print(f"{'─'*60}")
        print(f"Split: {split}  |  {total} prompts  |  skip={skip}")
        print(f"Output: {output_dir}")
        print(f"{'─'*60}")

        for idx in range(total):
            if idx in existing:
                continue

            prompt = prompts[idx]
            w, h = resolutions[idx][0], resolutions[idx][1]
            fname = f"prompt{idx:04d}_{slug(prompt)}.png"
            fpath = os.path.join(output_dir, fname)

            # ⚠️ CRITICAL: must use device="cuda", NOT "cpu"
            # CPU generator produces different noise even with same seed (~43px diff)
            # CUDA generator reproduces paper results (~9px diff)
            generator = torch.Generator(device="cuda").manual_seed(args.seed + idx)
            print(f"  [{idx+1}/{total}] {w}×{h}  {prompt[:70]}...")

            with torch.no_grad():
                out = pipe(
                    prompt=prompt,
                    width=w,
                    height=h,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    max_sequence_length=args.max_seq_len,
                    generator=generator,
                    output_type="pil",
                )
            out.images[0].save(fpath)

        print(f"✓ {split} done → {output_dir}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print("\nAll splits complete.")


if __name__ == "__main__":
    main()
