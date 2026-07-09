"""
ArXiV HF Dataset Unified — Single parquet, split-based training for all stages.

Reads SciFormaData-700K/metadata.parquet (unified, 1.38M rows) and filters
by the `split` column. This replaces all previous separate dataset classes
(ArXiVHFDatasetV1, ArXiVHFEditingDatasetV1, ArXiVHFDatasetV4) with one unified
interface.

Split values:
  'gen_768'   — 661,660 rows, all images at 768px (Stage 1)
  'gen_1024'  — 651,860 rows, all images at 1024px (Stage 2 gen)
  'edit_1024' — 70,866 rows, source+target pairs at 1024px (Stage 2 edit)

Usage:
  # Stage 1: all 768px generation data
  dataset_cfg = dict(
      type='ArXiVHFDatasetUnified',
      data_root='/path/SciFormaData-700K',
      splits=['gen_768'],
  )

  # Stage 2 gen-only: High quality 1024px
  dataset_cfg = dict(
      type='ArXiVHFDatasetUnified',
      data_root='/path/SciFormaData-700K',
      splits=['gen_1024'],
      quality_filter='High',   # 244,304 rows
  )

  # Stage 2 mixed gen+edit: requires DistributedBucketSamplerV3
  dataset_cfg = dict(
      type='ArXiVHFDatasetUnified',
      data_root='/path/SciFormaData-700K',
      splits=['gen_1024', 'edit_1024'],
      quality_filter='High',
  )
  sampler_cfg = dict(type='DistributedBucketSamplerV3', ...)
  train_iteration_func = 'Flux2Klein_mixed_edit_train_iteration'
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from sciforma.registry import DATASETS


@DATASETS.register_module()
class ArXiVHFDatasetUnified(Dataset):
    """
    Unified dataset for all SciForma training stages.
    Reads a single metadata.parquet and filters by `split` column.

    Works as drop-in replacement for:
      - ArXiVHFDatasetV1 (splits=['gen_768'] or splits=['gen_1024'])
      - ArXiVHFEditingDatasetV1 (splits=['edit_1024'])
      - ArXiVHFDatasetV4 (splits=['gen_1024', 'edit_1024'])
    """

    def __init__(
        self,
        data_root: str,
        parquet_file: str = "metadata.parquet",
        splits: list = None,        # e.g. ['gen_768'] | ['gen_1024', 'edit_1024']
        quality_filter: str = None, # 'High'|'Medium'|'Low' — filters gen rows only
        num_workers: int = 8,
        max_samples: int = None,
        debug_mode: bool = False,
    ):
        if splits is None:
            splits = ['gen_768']

        self.data_root = Path(data_root)
        pq = self.data_root / parquet_file

        print(f"Loading unified metadata: {pq}")
        df = pd.read_parquet(pq)
        print(f"  Total rows: {len(df):,}")

        # Filter by requested splits
        df = df[df['split'].isin(splits)].reset_index(drop=True)
        print(f"  After splits={splits}: {len(df):,}")

        # Quality filter applies to gen rows only
        if quality_filter:
            gen_mask = df['split'].isin(['gen_768', 'gen_1024'])
            keep = (~gen_mask) | (df['quality_tier'] == quality_filter)
            df = df[keep].reset_index(drop=True)
            print(f"  After quality_filter='{quality_filter}': {len(df):,}")

        # Drop empty captions
        empty = df['caption'].isna() | (df['caption'].str.len() < 5)
        if empty.sum():
            df = df[~empty].reset_index(drop=True)
            print(f"  Dropped {empty.sum()} empty captions → {len(df):,}")

        if debug_mode:
            df = df.head(400)
        if max_samples:
            df = df.head(max_samples)

        self.meta_df = df
        n_gen = (df['split'].isin(['gen_768', 'gen_1024'])).sum()
        n_edit = (df['split'] == 'edit_1024').sum()
        print(f"  Final: {len(df):,}  (gen={n_gen:,}, edit={n_edit:,})")

    def __len__(self):
        return len(self.meta_df)

    def _npz_path(self, img_path: str) -> Path:
        p = self.data_root / img_path
        return p.parent / (p.stem + "_flux_h.npz")

    def _load_npz(self, npz_path: Path):
        with np.load(npz_path, allow_pickle=True) as d:
            vae_h = d['vae_h'].astype(np.float32)
            c = vae_h.shape[0] // 2
            mean, logvar = vae_h[:c], vae_h[c:]
            logvar = np.clip(logvar, -30.0, 20.0)
            latents = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape).astype(np.float32)
            text_embeds = d.get('prompt_embeds', d.get('text_embeds',
                np.zeros((1, 12288), dtype=np.float16))).astype(np.float16)
            text_mask = d.get('attention_mask',
                np.ones(text_embeds.shape[0], dtype=np.int8)).astype(np.int8)
        return latents, text_embeds, text_mask

    def __getitem__(self, index):
        row = self.meta_df.iloc[index % len(self.meta_df)]
        split = row['split']
        bw = int(row['bucket_w'])
        bh = int(row['bucket_h'])

        if split in ('gen_768', 'gen_1024'):
            npz = self._npz_path(row['image_path'])
            if not npz.exists():
                raise FileNotFoundError(
                    f"NPZ not found: {npz}\nRun: python scripts/cache_latents.py")
            latents, text_embeds, text_mask = self._load_npz(npz)
            return {
                'latents': torch.from_numpy(latents),
                'text_embeds': torch.from_numpy(text_embeds.astype(np.float32)),
                'text_ids': None,
                'text_mask': torch.from_numpy(text_mask.astype(np.int8)),
                'bucket_size': (bh, bw),
                'caption': str(row.get('caption', '')),
                'aspect_ratio': float(row.get('aspect_ratio', bw / max(bh, 1))),
                'batch_mode': 'gen',
                'data_type': 'gen',
            }

        elif split == 'edit_1024':
            src_npz = self._npz_path(row['source_image_path'])
            tgt_npz = self._npz_path(row['target_image_path'])
            if not src_npz.exists() or not tgt_npz.exists():
                raise FileNotFoundError(
                    f"Edit NPZ not found:\n  {src_npz}\n  {tgt_npz}\n"
                    "Run: python scripts/cache_latents.py for editing data")
            src_lat, src_emb, src_mask = self._load_npz(src_npz)
            tgt_lat, tgt_emb, tgt_mask = self._load_npz(tgt_npz)
            return {
                'source_latents': torch.from_numpy(src_lat),
                'target_latents': torch.from_numpy(tgt_lat),
                'latents': torch.from_numpy(tgt_lat),
                'text_embeds': torch.from_numpy(tgt_emb.astype(np.float32)),
                'text_ids': None,
                'text_mask': torch.from_numpy(tgt_mask.astype(np.int8)),
                'source_embeds': torch.from_numpy(src_emb.astype(np.float32)),
                'bucket_size': (bh, bw),
                'caption': str(row.get('caption', '')),
                'aspect_ratio': float(row.get('aspect_ratio', bw / max(bh, 1))),
                'batch_mode': 'edit',
                'data_type': 'edit',
            }
        else:
            raise ValueError(f"Unknown split: {split}")

    def collate_fn(self, batch):
        """Collate for mixed gen+edit batches (mirrors ArXiVHFDatasetV4)."""
        data_types = [x['data_type'] for x in batch]
        unique = set(data_types)

        embeds = pad_sequence([x['text_embeds'] for x in batch], batch_first=True)
        masks = pad_sequence([x['text_mask'] for x in batch], batch_first=True)

        result = {
            'text_embeds': embeds,
            'text_mask': masks,
            'text_ids': None,
            'captions': [x['caption'] for x in batch],
            'bucket_size': batch[0]['bucket_size'],
            'aspect_ratio': batch[0]['aspect_ratio'],
            'data_types': data_types,
        }

        if unique == {'gen'}:
            result['batch_mode'] = 'gen'
            result['latents'] = torch.stack([x['latents'] for x in batch])
        elif unique == {'edit'}:
            result['batch_mode'] = 'edit'
            result['source_latents'] = torch.stack([x['source_latents'] for x in batch])
            result['target_latents'] = torch.stack([x['target_latents'] for x in batch])
            result['latents'] = result['target_latents']
        else:
            result['batch_mode'] = 'mixed'
            gi = [i for i, t in enumerate(data_types) if t == 'gen']
            ei = [i for i, t in enumerate(data_types) if t == 'edit']
            result['gen_indices'] = gi
            result['edit_indices'] = ei
            if gi:
                result['latents'] = torch.stack([batch[i]['latents'] for i in gi])
            if ei:
                result['source_latents'] = torch.stack([batch[i]['source_latents'] for i in ei])
                result['target_latents'] = torch.stack([batch[i]['target_latents'] for i in ei])

        return result
