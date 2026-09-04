#!/usr/bin/env python3
"""Generate a CSSL scientific methodology diagram with SciForma-9B."""

import argparse
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

MODEL_ID = "LoYuXrqw/SciForma-9B"
BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
WIDTH, HEIGHT = 1600, 640
CFG = 4.0
NUM_STEPS = 50
MAX_SEQ_LEN = 2048
SEED = 42
DTYPE = torch.bfloat16

PROMPT = """Create a clean scientific diagram of the Context-aware Sparse Spatiotemporal Learning (CSSL) framework for event-based vision.

LAYOUT:
Use a single left-to-right pipeline on a white background. Place the main components in this order: Event flow, Convolution, Dense output feature, Threshold operator, Sparse feature, and two output tasks stacked vertically on the far right.

COMPONENTS:
- Event flow: a 3D x-y-t cube containing red and blue event points.
- Convolution: one light-blue rectangular module labeled "Convolution".
- Dense output feature: a stack of gray grid feature maps labeled "Dense output feature".
- Context-aware Threshold: one peach grid below the dense feature maps, labeled "Context-aware Threshold".
- Threshold operator: one gray circle containing a step-function symbol.
- Sparse feature: a stack of white grid maps with a few red active cells, labeled "Sparse feature".
- Event-based object detection: a grayscale street image with colored bounding boxes, placed at the upper right.
- Event-based optical flow: a colorful optical-flow street image, placed at the lower right.

CONNECTIONS:
Event flow feeds Convolution. Convolution has two outgoing paths: the main horizontal path goes to Dense output feature, while one short downward path goes to Context-aware Threshold. Dense output feature and Context-aware Threshold each feed the same Threshold operator. The Threshold operator feeds Sparse feature. From the right edge of Sparse feature, draw two completely separate outgoing arrows: one direct diagonal arrow to the upper Event-based object detection panel, and one direct diagonal arrow to the lower Event-based optical flow panel.

Draw each connection once. Use only short, straight or right-angle, solid blue arrows with clear arrowheads. The two output arrows must remain visually separate from start to finish: do not merge them into a shared trunk, vertical bus, bracket, fork node, or intermediate junction. Do not show edge numbers or edge labels. Do not draw curved arrows, feedback loops, bidirectional arrows, crossing lines, direct arrows from Event flow to Context-aware Threshold, or direct arrows from Dense output feature to Sparse feature. Keep labels readable and do not add extra modules or connections."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=MODEL_ID,
                        help="Hugging Face transformer repo ID or local directory")
    parser.add_argument("--base_model_path", default=BASE_MODEL_ID,
                        help="Hugging Face base pipeline ID or local directory")
    parser.add_argument("--output", default="cssl_showcase.png",
                        help="Output PNG path")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Use CPU offload if the full pipeline does not fit on the GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=DTYPE,
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        torch_dtype=DTYPE,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.transformer.eval()

    with torch.no_grad():
        image = pipe(
            prompt=PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=CFG,
            max_sequence_length=MAX_SEQ_LEN,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output} ({WIDTH}x{HEIGHT}, seed={SEED}, cfg={CFG})")


if __name__ == "__main__":
    main()
