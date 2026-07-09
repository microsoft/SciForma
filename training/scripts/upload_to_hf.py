import os
#!/usr/bin/env python3
"""
SciForma — HuggingFace Upload Script
=======================================
Uploads the staged SciForma assets to HuggingFace.

⚠️  DO NOT RUN until the team has decided to publish.
    HF repo IDs: microsoft/SciFormaData-700K, microsoft/SciForma-Base, microsoft/SciForma-9B

Usage (when ready to publish):
    # Upload models
    python training/scripts/upload_to_hf.py --what models --hf_user microsoft --dry_run

    # Upload dataset
    python training/scripts/upload_to_hf.py --what dataset --hf_user microsoft --dry_run

    # Actually upload (remove --dry_run)
    python training/scripts/upload_to_hf.py --what all --hf_user microsoft

Requires:
    pip install huggingface_hub
    export HF_TOKEN="hf_..."
"""

import argparse
import os
from pathlib import Path

STAGING = Path(os.environ.get("SCIFORMA_STAGING_DIR", "/data/sciforma/hf_staging"))


def upload_model(local_dir: Path, repo_id: str, token: str, dry_run: bool):
    """Upload a model directory to HuggingFace."""
    print(f"\nUploading {local_dir.name} → {repo_id}")
    print(f"  Local: {local_dir}")
    print(f"  Size: {sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file()) / 1e9:.1f} GB")

    if dry_run:
        print(f"  [DRY RUN] Would upload to: {repo_id}")
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)

    # Create repo if needed
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)

    # Upload all files
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        ignore_patterns=["*.log", "__pycache__", ".DS_Store"],
    )
    print(f"  ✅ Uploaded to https://huggingface.co/{repo_id}")


def upload_dataset(local_dir: Path, repo_id: str, token: str, dry_run: bool):
    """Upload dataset to HuggingFace."""
    print(f"\nUploading dataset → {repo_id}")
    print(f"  Local: {local_dir}")

    # Compute sizes
    for subdir in ["generation", "editing", "benchmark"]:
        p = local_dir / subdir
        if p.exists():
            size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
            print(f"  {subdir}/: {size_gb:.1f} GB")

    if dry_run:
        print(f"  [DRY RUN] Would upload to: {repo_id}")
        print("  Note: Images will upload in parallel using LFS")
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)

    create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

    # Upload metadata files first (fast)
    for meta_file in local_dir.rglob("*.parquet"):
        rel = meta_file.relative_to(local_dir)
        api.upload_file(
            path_or_fileobj=str(meta_file),
            path_in_repo=str(rel),
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    # Upload README
    readme = local_dir / "README.md"
    if readme.exists():
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

    # Note: Images (503 GB) are uploaded separately via azcopy or HF large file upload
    print(f"  ✅ Metadata uploaded to https://huggingface.co/datasets/{repo_id}")
    print("  ⚠️  Images (503 GB) need separate upload via 'huggingface-cli upload'")


def main():
    parser = argparse.ArgumentParser(description="Upload SciForma assets to HuggingFace")
    parser.add_argument("--what", choices=["models", "dataset", "all"], default="all")
    parser.add_argument("--hf_user, default="microsoft",
                        help="HuggingFace username/org (e.g. 'microsoft')")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be uploaded without actually doing it")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return

    if args.hf_user, default="microsoft":
        print("ERROR: Please set --hf_user to the actual HuggingFace username/org")
        print("  e.g.: python training/scripts/upload_to_hf.py --hf_user microsoft")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}HF Upload to user: {args.hf_user}")
    print(f"Staging: {STAGING}")

    if args.what in ("models", "all"):
        upload_model(
            STAGING / "SciForma-Base",
            f"{args.hf_user}/SciForma-Base",
            token, args.dry_run
        )
        upload_model(
            STAGING / "SciForma-9B",
            f"{args.hf_user}/SciForma-9B",
            token, args.dry_run
        )

    if args.what in ("dataset", "all"):
        upload_dataset(
            STAGING / "SciFormaData-700K",
            f"{args.hf_user}/SciFormaData-700K",
            token, args.dry_run
        )

    print("\n✅ Done")


if __name__ == "__main__":
    main()
