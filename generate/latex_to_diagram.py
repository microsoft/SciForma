"""
latex_to_diagram.py — One-command LaTeX → scientific diagram generation.

Pipeline: LaTeX → [Planner] → [Condense] → SciForma-9B → PNG

Usage:
    python generate/latex_to_diagram.py \\
        --latex path/to/paper.tex \\
        --caption "Overview of the proposed method." \\
        --output output.png

    # List all detected figure captions:
    python generate/latex_to_diagram.py \\
        --latex path/to/paper.tex \\
        --list_captions

    # Custom model (default: LoYuXrqw/SciForma-9B):
    python generate/latex_to_diagram.py \\
        --latex paper.tex \\
        --caption "..." \\
        --model_path LoYuXrqw/SciForma-Base \\
        --output figure.png

Environment variables (set ONE of):
    OPENAI_API_KEY=sk-xxx                         (standard OpenAI)
    AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT  (Azure OpenAI)

    HF_TOKEN=hf_xxx    (if SciForma model repo is private)
"""
from __future__ import annotations
import argparse, asyncio, os, sys
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        for line in _env.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate.latex_parser import LatexParser
from generate.agents.planner import plan
from generate.agents.condense import condense



def generate_image(
    prompt: str,
    model_path: str = "LoYuXrqw/SciForma-9B",
    output: str = "output.png",
    width: int = 1008,
    height: int = 576,
    cfg: float = 4.0,
    steps: int = 50,
    seed: int = 42,
):
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

    hf_token = os.environ.get("HF_TOKEN")
    print(f"\nLoading SciForma transformer from {model_path} ...")

    try:
        from huggingface_hub import model_info
        info = model_info(model_path, token=hf_token)
        files = [f.rfilename for f in info.siblings]
        has_full = any("scheduler_config.json" in f for f in files)
    except Exception:
        has_full = os.path.isdir(model_path)  # local path

    if has_full:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, token=hf_token
        )
    else:
        BASE = "black-forest-labs/FLUX.2-klein-base-9B"
        print(f"  Transformer-only repo → loading base pipeline from {BASE}")
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer",
            torch_dtype=torch.bfloat16, token=hf_token,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            BASE, transformer=transformer,
            torch_dtype=torch.bfloat16, token=hf_token,
        )

    pipe.enable_model_cpu_offload()
    pipe.transformer.eval()
    print("Pipeline ready. Generating image...")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            max_sequence_length=2048,
            generator=generator,
        ).images[0]

    image.save(output)
    print(f"Saved → {output}")
    return output



async def run_pipeline(args):
    print(f"Parsing LaTeX: {args.latex}")
    parser = LatexParser(args.latex)

    if args.list_captions:
        caps = parser.get_figure_captions()
        print(f"\nFound {len(caps)} figure captions:\n")
        for i, c in enumerate(caps):
            print(f"  [{i}] {c[:120]}{'...' if len(c)>120 else ''}")
        return

    methodology = parser.get_methodology_text(max_chars=8000)
    caption = args.caption
    print(f"  Methodology: {len(methodology)} chars")
    print(f"  Caption: {caption[:80]}...")

    llm_model = args.llm_model

    print(f"\n[1/2] Planning diagram description (model={llm_model})...")
    raw_desc = await plan(methodology, caption, model=llm_model)
    print(f"  Planner output: {len(raw_desc)} chars")

    if args.verbose:
        print(f"\n--- Planner output ---\n{raw_desc}\n---\n")

    print(f"[2/2] Condensing to FLUX prompt style...")
    flux_prompt = await condense(raw_desc, model=llm_model)
    print(f"  Final prompt: {len(flux_prompt)} chars")

    if args.verbose or args.print_prompt:
        print(f"\n--- Final FLUX prompt ---\n{flux_prompt}\n---\n")

    if args.prompt_only:
        out_txt = Path(args.output).with_suffix(".txt")
        out_txt.write_text(flux_prompt)
        print(f"Prompt saved → {out_txt}")
        return

    generate_image(
        prompt=flux_prompt,
        model_path=args.model_path,
        output=args.output,
        width=args.width,
        height=args.height,
        cfg=args.cfg,
        steps=args.steps,
        seed=args.seed,
    )



def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a scientific diagram from LaTeX source using SciForma-9B"
    )
    p.add_argument("--latex",        required=True,
                   help="Path to .tex file, directory, or .tar.gz")
    p.add_argument("--caption",      default="",
                   help="Figure caption (\\caption{...} text to target)")
    p.add_argument("--output",       default="output.png",
                   help="Output PNG path")
    p.add_argument("--model_path",   default="LoYuXrqw/SciForma-9B",
                   help="HuggingFace model ID or local path")
    p.add_argument("--llm_model",    default="gpt-4o",
                   help="LLM for planning/condensing (gpt-4o, o3, etc.)")
    p.add_argument("--width",        type=int, default=1008)
    p.add_argument("--height",       type=int, default=576)
    p.add_argument("--cfg",          type=float, default=4.0)
    p.add_argument("--steps",        type=int, default=50)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--list_captions", action="store_true",
                   help="Print all detected figure captions and exit")
    p.add_argument("--prompt_only",  action="store_true",
                   help="Only generate the FLUX prompt (no image generation)")
    p.add_argument("--print_prompt", action="store_true",
                   help="Print the final FLUX prompt")
    p.add_argument("--verbose",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(args))
