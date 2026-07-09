#!/usr/bin/env python3
"""
SciForma Setup Verification Script
====================================
Runs a comprehensive smoke test to verify that the SciForma training environment
is correctly set up before launching any training jobs.

⚠️  IMPORTANT: Run this with Python 3.10 (flux2 conda env), NOT system Python 3.13+.
   mmengine 0.10.7 requires Python 3.10 for _base_ config syntax.
   conda activate flux2 && python training/scripts/verify_setup.py

Usage:
    # Basic check (imports + configs):
    python training/scripts/verify_setup.py

    # Full check with dataset init (requires SCIFORMA_DATA_ROOT):
    python training/scripts/verify_setup.py --check-data

    # Check with specific config:
    python training/scripts/verify_setup.py --config training/configs/stage1_sft.py

Exit codes: 0 = all passed, 1 = failures found
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "✅"
FAIL = "✗"
WARN = "⚠️"

results = []


def check(name, fn, warn_only=False):
    """Run a check and record result."""
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
        return True
    except Exception as e:
        tag = WARN if warn_only else FAIL
        results.append((tag, f"{name}: {e}"))
        print(f"  {tag} {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Config file to test")
    parser.add_argument("--check-data", action="store_true", help="Also initialize dataset")
    parser.add_argument("--check-scripts", action="store_true", help="Validate all script config paths")
    args = parser.parse_args()

    print("=" * 65)
    print("SciForma Setup Verification")
    print("=" * 65)

    # ── 1. Core imports ────────────────────────────────────────────────────────
    print("\n1. Core imports")

    def check_torch():
        import torch
        print(f"     torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    check("torch", check_torch)

    def check_diffusers():
        import diffusers
        print(f"     diffusers {diffusers.__version__}")
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    check("diffusers (Flux2Klein)", check_diffusers)

    def check_accelerate():
        import accelerate
        print(f"     accelerate {accelerate.__version__}")
    check("accelerate", check_accelerate)

    def check_mmengine():
        import mmengine
        print(f"     mmengine {mmengine.__version__}")
    check("mmengine", check_mmengine)

    def check_sciforma():
        import sciforma
        import sciforma.datasets
        import sciforma.train_iteration_funcs
        from sciforma.registry import DATASETS, TRAIN_ITERATION_FUNCS
        n_ds = len(DATASETS._module_dict)
        n_ti = len(TRAIN_ITERATION_FUNCS._module_dict)
        print(f"     {n_ds} datasets, {n_ti} train_iteration_funcs registered")
        assert n_ds >= 8, f"Expected ≥8 datasets, got {n_ds}"
        assert n_ti >= 6, f"Expected ≥6 train iter funcs, got {n_ti}"
        # Critical: DistributedBucketSamplerV2 required for Stage1 training
        assert 'DistributedBucketSamplerV2' in DATASETS._module_dict, \
            "DistributedBucketSamplerV2 not registered! Stage1 training will fail."
        # Critical: HF editing dataset must return batch_mode='edit'
        import inspect
        from sciforma.datasets import ArXiVHFEditingDatasetV1, ArXiVHFDatasetV1
        edit_src = inspect.getsource(ArXiVHFEditingDatasetV1.__getitem__)
        gen_src = inspect.getsource(ArXiVHFDatasetV1.__getitem__)
        assert "'edit'" in edit_src or '"edit"' in edit_src, \
            "ArXiVHFEditingDatasetV1 missing batch_mode='edit'! Stage2 mixed training will silently fail."
    check("sciforma (imports + registry)", check_sciforma)

    # ── 2. Config loading ──────────────────────────────────────────────────────
    print("\n2. Config loading (mmengine)")
    from mmengine import Config

    configs_to_test = [
        "training/configs/stage1_sft.py",
        "training/configs/stage2_sft.py",
        "training/configs/mdpo/mdpo.py",
        "training/configs/hf_compatible/stage1_sft_hf.py",
        "training/configs/hf_compatible/stage2_sft_hf.py",
    ]
    if args.config:
        configs_to_test = [args.config]

    for cfg_path in configs_to_test:
        full_path = PROJECT_ROOT / cfg_path
        def _load(p=full_path, n=cfg_path):
            cfg = Config.fromfile(str(p))
            print(f"     dataset_type={cfg.dataset_cfg['type']}, train_func={cfg.train_iteration_func}")
        check(cfg_path, _load)

    # ── 3. Dataset init (optional) ─────────────────────────────────────────────
    if args.check_data:
        print("\n3. Dataset initialization (debug mode)")
        data_root = os.environ.get("SCIFORMA_DATA_ROOT", "")
        if not data_root:
            print(f"  {WARN} SCIFORMA_DATA_ROOT not set, skipping data checks")
        else:
            from sciforma.datasets import ArXiVHFDatasetV1

            def check_gen_dataset():
                ds = ArXiVHFDatasetV1(
                    data_root=str(Path(data_root) / "SciFormaData-700K/generation"),
                    quality_filter=None,
                    debug_mode=True
                )
                print(f"     {len(ds):,} samples (debug), columns={ds.meta_df.columns.tolist()}")
            check("ArXiVHFDatasetV1 (stage1)", check_gen_dataset)

            def check_high_dataset():
                ds = ArXiVHFDatasetV1(
                    data_root=str(Path(data_root) / "SciFormaData-700K/generation"),
                    quality_filter="High",
                    debug_mode=True
                )
                print(f"     {len(ds):,} High quality samples")
            check("ArXiVHFDatasetV1 (stage2 High)", check_high_dataset)

            # Test editing dataset if available
            edit_root = str(Path(data_root) / "SciFormaData-700K/editing")
            if (Path(edit_root) / "metadata.parquet").exists():
                from sciforma.datasets import ArXiVHFEditingDatasetV1
                def check_edit_dataset():
                    ds = ArXiVHFEditingDatasetV1(
                        data_root=edit_root,
                        debug_mode=True
                    )
                    print(f"     {len(ds):,} editing pairs (debug)")
                check("ArXiVHFEditingDatasetV1", check_edit_dataset)

    # ── 4. Benchmark paths (optional, with SCIFORMA_GT_BASE) ──────────────────
    gt_base = os.environ.get("SCIFORMA_GT_BASE", "")
    if gt_base:
        print("\n4. Benchmark GT image paths")
        bench_dir = PROJECT_ROOT / "benchmark"
        split_file_map = {"easy": "easy", "medium": "medium", "hard": "hard"}
        for split, fname in split_file_map.items():
            def _check_gt(s=split, f=fname, gb=gt_base):
                import json
                data = json.loads((bench_dir/"prompts"/f"{f}.json").read_text())
                gt_images = data.get("gt_images", [])
                found = sum(1 for gt in gt_images
                            if (Path(gb)/str(gt.get("year",""))/gt.get("image_path","")).exists())
                assert found == len(gt_images), f"GT missing: {len(gt_images)-found}/{len(gt_images)}"
                print(f"     {s}: {found}/{len(gt_images)} GT images ✅")
            check(f"eval/{split} GT ({gt_base.split('/')[-1]})", _check_gt)
    else:
        print(f"\n4. Benchmark GT (set SCIFORMA_GT_BASE to check)")

    # ── 5. Script path validation (optional) ─────────────────────────────────
    if args.check_scripts:
        print("\n5. Script config paths")
        import re
        scripts_checked = 0
        for sh in sorted((PROJECT_ROOT / "scripts").rglob("*.sh")):
            def _check_sh(f=sh):
                content = f.read_text()
                refs = re.findall(r'configs/[^\s"\']+\.py', content)
                for ref in refs:
                    assert (PROJECT_ROOT / ref).exists(), f"Missing: {ref}"
                if refs:
                    print(f"     {f.relative_to(PROJECT_ROOT)}: {len(refs)} config ref(s) ✅")
            check(f"script/{sh.name}", _check_sh)

    # ── 6. Environment ─────────────────────────────────────────────────────────
    print("\n6. Environment variables")
    env_vars = ["SCIFORMA_DATA_ROOT", "HF_TOKEN", "WANDB_API_KEY"]
    for var in env_vars:
        val = os.environ.get(var, "")
        status = PASS if val else WARN
        display = f"set ({len(val)} chars)" if val else "not set"
        results.append((status, f"${var}"))
        print(f"  {status} ${var}: {display}")

    # ── Summary ────────────────────────────────────────────────────────────────
    n_pass = sum(1 for s, _ in results if s == PASS)
    n_warn = sum(1 for s, _ in results if s == WARN)
    n_fail = sum(1 for s, _ in results if s == FAIL)

    print(f"\n{'=' * 65}")
    print(f"Results: {n_pass} passed, {n_warn} warnings, {n_fail} failed")
    if n_fail == 0:
        print("✅ All critical checks passed — ready to train!")
    else:
        print("✗ Some checks failed — see above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
