"""
ArXiV HF Dataset V1 — For training from HuggingFace-format SciFormaData-700K.

This dataset loader reads from the HF-style metadata.parquet and raw PNG images.
It supports two modes:
  1. NPZ latent cache mode (fast): reads pre-computed NPZ files alongside images
  2. Raw image mode (slow): encodes images on-the-fly via VAE

HF Dataset structure expected:
  $SCIFORMA_DATA_ROOT/SciFormaData-700K/generation/
    metadata.parquet          ← paper_id, image_path, caption, width, height, quality_tier
    images_768/768_pretrain/{year}/{paper_id}/{image}.png
    images_1024/1024_pretrain/{year}/{paper_id}/{image}.png
    # After running scripts/cache_latents.py:
    images_768/768_pretrain/{year}/{paper_id}/{image}_flux_h.npz  (optional)
    images_1024/1024_pretrain/{year}/{paper_id}/{image}_flux_h.npz (optional)

Usage in config:
    dataset_cfg = dict(
        type='ArXiVHFDatasetV1',
        data_root='/path/to/SciFormaData-700K/generation',
        parquet_file='metadata.parquet',  # relative to data_root
        quality_filter=None,              # None = all, 'High' = Stage2-style subset
        num_workers=8,
    )
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sciforma.registry import DATASETS
from torch.utils.data import Dataset
import math


@DATASETS.register_module()
class ArXiVHFDatasetV1(Dataset):
    """
    Dataset loader for HuggingFace-format SciFormaData-700K.
    Reads metadata.parquet + raw PNG images (with optional NPZ cache).
    """

    # Substitution map for force_resolution: replaces src prefix with dst prefix in image_path
    _RESOLUTION_REMAP = {
        '768':  ('images_1024/1024_pretrain', 'images_768/768_pretrain'),
        '1024': ('images_768/768_pretrain',   'images_1024/1024_pretrain'),
    }
    # Scale factors for bucket_w/bucket_h when force_resolution remaps paths
    # 768px and 1024px images of the same paper differ by exactly 4/3 in each dimension
    _BUCKET_SCALE = {
        '768':  3 / 4,   # 1024px bucket → 768px bucket: multiply by 3/4
        '1024': 4 / 3,   # 768px bucket → 1024px bucket: multiply by 4/3
    }

    def __init__(
        self,
        data_root: str,
        parquet_file: str = "metadata.parquet",
        quality_filter: str = None,    # None=all, 'High'=high quality only
        force_resolution: str = None,  # '768' or '1024': override image_path resolution
        num_workers: int = 8,
        max_samples: int = None,
        debug_mode: bool = False,
    ):
        self.data_root = Path(data_root)
        parquet_path = self.data_root / parquet_file

        print(f"Loading metadata from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df):,} rows")

        # Drop empty captions (0.06% of data, not useful for training)
        empty_mask = df['caption'].isna() | (df['caption'].str.len() < 10)
        if empty_mask.sum() > 0:
            df = df[~empty_mask].reset_index(drop=True)
            print(f"  Dropped {empty_mask.sum()} empty captions: {len(df):,} rows")

        # Apply quality filter
        if quality_filter:
            df = df[df['quality_tier'] == quality_filter].reset_index(drop=True)
            print(f"  After quality_filter='{quality_filter}': {len(df):,} rows")

        # Force resolution: remap image_path so ALL images use the specified resolution.
        # Stage1 should use force_resolution='768' to get all 661K images at 768px,
        # since the parquet stores High-quality rows with images_1024 paths by default.
        # Also scales bucket_w/bucket_h to match the target resolution (4/3 ratio).
        if force_resolution and force_resolution in self._RESOLUTION_REMAP:
            src, dst = self._RESOLUTION_REMAP[force_resolution]
            scale = self._BUCKET_SCALE[force_resolution]
            mask = df['image_path'].str.startswith(src)
            n_remapped = mask.sum()
            if n_remapped > 0:
                df.loc[mask, 'image_path'] = df.loc[mask, 'image_path'].str.replace(src, dst, regex=False)
                df.loc[mask, 'bucket_w'] = (df.loc[mask, 'bucket_w'] * scale).round().astype(int)
                df.loc[mask, 'bucket_h'] = (df.loc[mask, 'bucket_h'] * scale).round().astype(int)
                print(f"  force_resolution='{force_resolution}': remapped {n_remapped:,} paths+buckets")
        elif force_resolution:
            print(f"  WARN: force_resolution='{force_resolution}' unknown, ignored")

        if debug_mode:
            df = df.head(500)

        if max_samples:
            df = df.head(max_samples)

        self.meta_df = df
        self._vae = None  # lazy-loaded if NPZ not available

        # Add bucket_w and bucket_h columns for sampler compatibility
        if 'width' in df.columns and 'bucket_w' not in df.columns:
            self.meta_df = df.rename(columns={'width': 'bucket_w', 'height': 'bucket_h'})

        print(f"  Final dataset: {len(self.meta_df):,} samples")

    def __len__(self):
        return len(self.meta_df)

    def _get_npz_path(self, image_path: str) -> Path:
        """Get NPZ cache path for an image (same dir, _flux_h.npz suffix)."""
        img_p = self.data_root / image_path
        # image.png → image_flux_h.npz
        npz_name = img_p.stem + "_flux_h.npz"
        return img_p.parent / npz_name

    def _load_from_npz(self, npz_path: Path):
        """Load pre-computed latents from NPZ file."""
        with np.load(npz_path, allow_pickle=True) as data:
            vae_h = data['vae_h'].astype(np.float32)
            c = vae_h.shape[0] // 2
            mean, logvar = vae_h[:c], vae_h[c:]
            logvar = np.clip(logvar, -30.0, 20.0)
            latents = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape).astype(np.float32)

            if 'prompt_embeds' in data:
                text_embeds = data['prompt_embeds'].astype(np.float16)
            else:
                text_embeds = data['text_embeds'].astype(np.float16)

            if 'attention_mask' in data:
                text_mask = data['attention_mask'].astype(np.int8)
            else:
                text_mask = np.ones(text_embeds.shape[0], dtype=np.int8)

        return latents, text_embeds, text_mask

    def _load_from_image(self, image_path: str, caption: str, width: int, height: int):
        """
        Encode raw image via VAE and text via text encoder.
        Falls back to this if NPZ not available.
        NOTE: Requires VAE and text encoder to be available.
        """
        raise NotImplementedError(
            "Raw image encoding not yet supported in training mode. "
            "Please run scripts/cache_latents.py first to pre-compute NPZ latents.\n"
            f"Image: {self.data_root / image_path}"
        )

    def __getitem__(self, index):
        row = self.meta_df.iloc[index % len(self.meta_df)]
        image_path = row['image_path']
        npz_path = self._get_npz_path(image_path)

        if npz_path.exists():
            latents, text_embeds, text_mask = self._load_from_npz(npz_path)
        else:
            # Fallback: raw image encoding (slow, requires VAE)
            caption = row.get('caption', '')
            w = int(row.get('bucket_w', row.get('width', 1024)))
            h = int(row.get('bucket_h', row.get('height', 1024)))
            latents, text_embeds, text_mask = self._load_from_image(image_path, caption, w, h)

        # NOTE: text_ids=None forces _prepare_text_ids() in iteration function
        # (parquet/NPZ has [SeqLen,3] but transformer expects [SeqLen,4] T,H,W,L)
        seq_len = text_embeds.shape[0]

        # bucket_size: (H, W) tuple required by mixed_edit_train_iteration
        bw = int(row.get('bucket_w', row.get('width', 1024)))
        bh = int(row.get('bucket_h', row.get('height', 512)))

        return {
            'latents': torch.from_numpy(latents),
            'text_embeds': torch.from_numpy(text_embeds.astype(np.float32)),
            'text_ids': None,  # computed dynamically by _prepare_text_ids() — DO NOT change
            'text_mask': torch.from_numpy(text_mask.astype(np.int8)),
            'bucket_size': (bh, bw),
            'batch_mode': 'gen',  # dispatches to generation path in mixed_edit_train_iteration
        }
