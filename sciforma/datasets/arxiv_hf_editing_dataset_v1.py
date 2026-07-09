"""
ArXiV HF Editing Dataset V1 — For training Stage 2 mixed gen+edit from HuggingFace format.

Reads from editing/metadata.parquet + source/target PNG images.
Works alongside ArXiVHFDatasetV1 for the mixed generation+editing Stage 2 training.

Expected structure (relative to data_root = editing/):
  metadata.parquet    ← source_image_path, target_image_path, caption (edit instruction)
  images/260209_1024/{year}/{paper_id}/{hash}/source.png
  images/260209_1024/{year}/{paper_id}/{hash}/target.png
  # After cache_latents.py:
  images/260209_1024/{year}/{paper_id}/{hash}/source_flux_h.npz
  images/260209_1024/{year}/{paper_id}/{hash}/target_flux_h.npz

Returns per-sample dict with:
  source_latents, target_latents, text_embeds, text_mask, batch_mode='edit'
  (compatible with Flux2Klein_mixed_edit_train_iteration → _process_edit_batch)

Note: edit_bboxes column is available in metadata.parquet but not converted to
  edit_mask here. The edit_mask (for bbox-weighted loss) is optional; without it,
  the loss falls back to standard flow matching on the full target image.

Usage in Stage 2 config (mixed gen+edit):
    # In train_sft.py, the mixed edit loader uses source+target pairs.
    # See sciforma/datasets/arxiv_parquet_dataset_v4.py for the original implementation.
    dataset_cfg = dict(
        type='ArXiVHFEditingDatasetV1',
        data_root='/path/to/SciFormaData-700K/editing',
        parquet_file='metadata.parquet',
        num_workers=8,
    )
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sciforma.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class ArXiVHFEditingDatasetV1(Dataset):
    """
    Dataset loader for HF-format SciFormaData-700K editing triplets.
    Each sample is a (source_image, edit_instruction, target_image) triplet.
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str = "metadata.parquet",
        num_workers: int = 8,
        max_samples: int = None,
        debug_mode: bool = False,
    ):
        self.data_root = Path(data_root)
        parquet_path = self.data_root / parquet_file

        print(f"Loading editing metadata from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df):,} editing pairs")

        # Filter empty captions
        empty_mask = df['caption'].isna() | (df['caption'].str.len() < 5)
        if empty_mask.sum():
            df = df[~empty_mask].reset_index(drop=True)
            print(f"  Dropped {empty_mask.sum()} empty captions: {len(df):,} pairs")

        if debug_mode:
            df = df.head(200)
        if max_samples:
            df = df.head(max_samples)

        self.meta_df = df
        print(f"  Final dataset: {len(self.meta_df):,} editing pairs")

        # Add bucket columns if missing (for bucket sampler compatibility)
        if 'bucket_w' not in df.columns and 'width' in df.columns:
            self.meta_df = df.rename(columns={'width': 'bucket_w', 'height': 'bucket_h'})

    def __len__(self):
        return len(self.meta_df)

    def _get_npz_path(self, img_path: str, suffix: str = "_flux_h.npz") -> Path:
        """Derive NPZ cache path from image path."""
        p = self.data_root / img_path
        return p.parent / (p.stem + suffix)

    def _load_npz(self, npz_path: Path):
        """Load pre-computed latents from NPZ."""
        with np.load(npz_path, allow_pickle=True) as data:
            vae_h = data['vae_h'].astype(np.float32)
            c = vae_h.shape[0] // 2
            mean, logvar = vae_h[:c], vae_h[c:]
            logvar = np.clip(logvar, -30.0, 20.0)
            latents = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape).astype(np.float32)

            text_embeds = data.get('prompt_embeds', data.get('text_embeds', np.zeros((1, 12288), dtype=np.float16))).astype(np.float16)
            text_mask = data.get('attention_mask', np.ones(text_embeds.shape[0], dtype=np.int8)).astype(np.int8)
        return latents, text_embeds, text_mask

    def __getitem__(self, index):
        row = self.meta_df.iloc[index % len(self.meta_df)]

        src_npz = self._get_npz_path(row['source_image_path'])
        tgt_npz = self._get_npz_path(row['target_image_path'])

        if src_npz.exists() and tgt_npz.exists():
            src_latents, src_embeds, src_mask = self._load_npz(src_npz)
            tgt_latents, tgt_embeds, tgt_mask = self._load_npz(tgt_npz)
        else:
            raise NotImplementedError(
                "Raw image editing not yet supported. "
                "Please run scripts/cache_latents.py first.\n"
                f"Source: {self.data_root / row['source_image_path']}\n"
                f"Target: {self.data_root / row['target_image_path']}"
            )

        seq_len = tgt_embeds.shape[0]
        # (text_ids NOT computed here — set to None, computed dynamically in iteration func)

        # bucket_size: (H, W) required by mixed_edit_train_iteration
        bw = int(row.get('bucket_w', row.get('width', 1024)))
        bh = int(row.get('bucket_h', row.get('height', 512)))

        # NOTE: text_ids=None forces _prepare_text_ids() in iteration function
        # (parquet has [SeqLen,3] but transformer expects [SeqLen,4])

        return {
            'source_latents': torch.from_numpy(src_latents),
            'target_latents': torch.from_numpy(tgt_latents),  # matches V4 format
            'latents': torch.from_numpy(tgt_latents),          # also as 'latents' for fulltune compatibility
            'text_embeds': torch.from_numpy(tgt_embeds.astype(np.float32)),
            'text_ids': None,  # computed dynamically by _prepare_text_ids() — DO NOT change
            'text_mask': torch.from_numpy(tgt_mask.astype(np.int8)),
            'source_embeds': torch.from_numpy(src_embeds.astype(np.float32)),
            'bucket_size': (bh, bw),
            'batch_mode': 'edit',  # required by Flux2Klein_mixed_edit_train_iteration
        }
