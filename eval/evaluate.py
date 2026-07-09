#!/usr/bin/env python3
"""
SciForma Benchmark Evaluation
================================
Scores generated images against SciFormaBench-2K using GPT-5.4.

Evaluation settings are FIXED to match the official paper protocol:
  • split_dims = True   — Component / Arrow / Text each get a dedicated GPT call.
                          Without this, Arrow scores are inflated by ~0.25.
  • rubrics = rubrics/  — Uses benchmark/rubrics/{easy,medium,hard}.json,
                          which is the exact inventory used in the paper eval.

Usage:
    python eval/evaluate.py \
        --gen_dir  ./outputs/sciforma-9b \
        --output_dir ./eval_results/sciforma-9b

    # Evaluate only specific splits:
    python eval/evaluate.py \
        --gen_dir  ./outputs/my-model \
        --output_dir ./eval_results/my-model \
        --split easy medium

Expected gen_dir layout (matches generate.py output):
    <gen_dir>/
        easy/   promptNNNN_<slug>.png  ...
        medium/ promptNNNN_<slug>.png  ...
        hard/   promptNNNN_<slug>.png  ...

Reference scores (SciFormaBench-2K, GPT-5.4, split_dims=True):
    SciForma-Base  67.59%  (Comp 73.52  Arrow 64.64  Text 63.84)
    SciForma-9B    69.51%  (Comp 74.49  Arrow 66.46  Text 67.00)
    GPT-Image-1.5  68.96%

Reproduction note (2026-06-26):
    Two independent runs with gpt-5.4_2026-03-05 gave 68.59% and 68.65%.
    Difference (~1%) vs paper is due to gpt-5.4 model version drift (March vs April).
    Scores are stable across runs (±0.06% variance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── FIXED evaluation parameters ──────────────────────────────────────────────
# These MUST match the paper setup. Do not add CLI flags for these.

# TRAPI_INSTANCE removed — use AZURE_OPENAI_ENDPOINT env var
DEPLOYMENT_NAME = "gpt-5.4"
NUM_RETEST      = 2        # average over 2 independent scoring rounds
MAX_TOKENS      = 16384    # enough headroom for dense diagrams
TEMPERATURE     = 0.0      # deterministic
WORKERS         = 4        # concurrent outer workers; each spawns 3 dim threads
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(
    gen_dir: str,
    output_dir: str,
    splits: list[str],
    gt_base: str,
    rubrics_dir: str,
    benchmark_dir: str,
    workers: int,
    skip_existing: bool,
) -> None:
    """Drive eval_benchmark.py with the correct fixed parameters."""
    import subprocess

    eval_script = Path(__file__).parent / "eval_benchmark.py"
    if not eval_script.exists():
        sys.exit(f"[ERROR] eval_benchmark.py not found at {eval_script}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check for existing results
    summary = Path(output_dir) / "eval_summary.json"
    if skip_existing and summary.exists():
        d = json.loads(summary.read_text())
        print(f"[SKIP] already done: {output_dir}")
        print(f"       overall={d['overall_mean']:.4f}")
        return

    cmd = [
        sys.executable, str(eval_script),
        "--gen_dir",             gen_dir,
        "--gt_base",             gt_base,
        "--benchmark_dir",       benchmark_dir,
        "--output_dir",          output_dir,
        "--deployment_name",     DEPLOYMENT_NAME,
        "--auth",                "cli",   # AzureCliCredential (az login)
        "--workers",             str(workers),
        "--num_retest",          str(NUM_RETEST),
        "--max_completion_tokens", str(MAX_TOKENS),
        "--temperature",         str(TEMPERATURE),
        "--split_dims",          # always on — required for paper-consistent scores
        "--rubrics_suffix",      "",   # loads rubrics/{easy,medium,hard}.json
        "--levels", *splits,
    ]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {Path(gen_dir).name}")
    print(f"  Splits: {splits}")
    print(f"  Output: {output_dir}")
    print(f"  Model:  {DEPLOYMENT_NAME} @ Azure OpenAI")
    print(f"  Params: retest={NUM_RETEST}  tokens={MAX_TOKENS}  split_dims=ON")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[ERROR] eval failed (rc={result.returncode})")

    # Print summary
    if summary.exists():
        d = json.loads(summary.read_text())
        am = d.get("aspect_means", {})
        lm = d.get("level_means", {})
        ref = {"easy": 0.7601, "medium": 0.6743, "hard": 0.6083}
        print(f"\n{'='*60}")
        print(f"  RESULTS: {Path(gen_dir).name}")
        print(f"  Overall: {d['overall_mean']:.4f}  "
              f"(SciForma-9B ref: 0.6951)")
        print(f"  Comp:  {am.get('component_score',0):.4f}  "
              f"Arrow: {am.get('arrow_score',0):.4f}  "
              f"Text:  {am.get('text_score',0):.4f}")
        for lvl in ["easy", "medium", "hard"]:
            if lvl in lm:
                diff = lm[lvl]["mean"] - ref.get(lvl, 0)
                print(f"  {lvl:7s}: {lm[lvl]['mean']:.4f}  "
                      f"(ref {ref.get(lvl,0):.4f}  diff={diff:+.4f})  "
                      f"n={lm[lvl]['count']}")
        print(f"{'='*60}\n")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Default GT image location — override with $SCIFORMA_GT_BASE or $SCIFORMA_DATA_ROOT
    default_gt = os.environ.get(
        "SCIFORMA_GT_BASE",
        os.path.join(
            os.environ.get("SCIFORMA_DATA_ROOT", ""),
            "ArXiV_filtered_stages/stage4_aspect_quantize_filter/1024_pretrain",
        )
    )

    parser = argparse.ArgumentParser(
        description="Evaluate generated images on SciFormaBench-2K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gen_dir", required=True,
        help="Root dir of generated images (expects easy/ medium/ hard/ sub-dirs)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write per-sample JSON files and eval_summary.json",
    )
    parser.add_argument(
        "--split", nargs="+", choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
        dest="splits",
        help="Which splits to evaluate (default: all three)",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS,
        help=f"Concurrent outer workers (default: {WORKERS}). "
             "Each worker spawns 3 dim threads for split_dims.",
    )
    parser.add_argument(
        "--gt_base", type=str, default=default_gt,
        help="Root dir of GT reference images",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip if eval_summary.json already exists in output_dir",
    )
    args = parser.parse_args()

    rubrics_dir    = str(repo_root / "eval" / "rubrics")
    benchmark_dir  = str(repo_root / "eval")

    run_eval(
        gen_dir      = args.gen_dir,
        output_dir   = args.output_dir,
        splits       = args.splits,
        gt_base      = args.gt_base,
        rubrics_dir  = rubrics_dir,
        benchmark_dir= benchmark_dir,
        workers      = args.workers,
        skip_existing= args.skip_existing,
    )


if __name__ == "__main__":
    main()
