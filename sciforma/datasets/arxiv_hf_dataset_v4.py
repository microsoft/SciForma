"""
ArXiV HF Dataset V4 — Combined Gen+Edit for HuggingFace-format SciFormaData-700K.

HF-format equivalent of ArXiVParquetDatasetV4. Loads both generation and editing
data into a single unified DataFrame with a 'data_type' column, enabling
DistributedBucketSamplerV3 to group batches as pure-gen or pure-edit.

Architecture (matches original Stage2 training setup):
  ArXiVHFDatasetV4 (single dataset, gen+edit unified)
    └─ DistributedBucketSamplerV3 (groups by (bucket_h, bucket_w, data_type))
         └─ DataLoader (batches are always pure-gen OR pure-edit)
              └─ Flux2Klein_mixed_edit_train_iteration (dispatches by batch_mode)

Expected structure:
  $SCIFORMA_DATA_ROOT/SciFormaData-700K/
    generation/
      metadata.parquet      ← 661,660 rows; quality_tier='High'|'Other'
      images_768/768_pretrain/{year}/{paper_id}/{img}.png
      images_1024/1024_pretrain/{year}/{paper_id}/{img}.png
      # After cache_latents.py:
      images_*/..._flux_h.npz
    editing/
      metadata.parquet      ← 70,866 pairs; has width/height
      images/260209_1024/{year}/{paper_id}/{hash}/source.png
      images/260209_1024/{year}/{paper_id}/{hash}/target.png
      # After cache_latents.py:
      images/.../source_flux_h.npz, target_flux_h.npz

Usage in stage2_sft_mixed_hf.py config:
    dataset_cfg = dict(
        type='ArXiVHFDatasetV4',
        gen_data_root='/path/SciFormaData-700K/generation',
        edit_data_root='/path/SciFormaData-700K/editing',
        quality_filter='High',   # Gen data: only High quality (244K)
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
class ArXiVHFDatasetV4(Dataset):
    """
    Combined generation + editing dataset for Stage2 mixed HF-format training.

    Loads both gen and edit parquets into a single DataFrame with a 'data_type'
    column ('gen' or 'edit'). Works with DistributedBucketSamplerV3 which groups
    by (bucket_h, bucket_w, data_type), guaranteeing pure-gen or pure-edit batches.
    """

    def __init__(
        self,
        gen_data_root: str = None,
        edit_data_root: str = None,
        gen_parquet_file: str = "metadata.parquet",
        edit_parquet_file: str = "metadata.parquet",
        quality_filter: str = None,   # 'High' for Stage2, None for Stage1
        num_workers: int = 8,
        max_samples: int = None,
        debug_mode: bool = False,
    ):
        assert gen_data_root or edit_data_root, \
            "At least one of gen_data_root or edit_data_root must be provided"

        frames = []

        # ── Load Generation Data ───────────────────────────────────────────────
        if gen_data_root:
            gen_root = Path(gen_data_root)
            gen_pq = gen_root / gen_parquet_file
            print(f"Loading gen metadata: {gen_pq}")
            gdf = pd.read_parquet(gen_pq)
            print(f"  Loaded {len(gdf):,} gen rows")

            # Drop empty captions
            empty = gdf['caption'].isna() | (gdf['caption'].str.len() < 10)
            if empty.sum():
                gdf = gdf[~empty].reset_index(drop=True)
                print(f"  Dropped {empty.sum()} empty captions → {len(gdf):,}")

            # Quality filter: HF parquet uses 'quality_tier' ('High'/'Other')
            if quality_filter:
                gdf = gdf[gdf['quality_tier'] == quality_filter].reset_index(drop=True)
                print(f"  After quality_filter='{quality_filter}': {len(gdf):,}")

            # HF gen parquet already has bucket_w/bucket_h columns
            gdf['data_type'] = 'gen'
            gdf['_data_root'] = str(gen_root)
            frames.append(gdf)
            print(f"  Gen data ready: {len(gdf):,} samples")

        # ── Load Editing Data ──────────────────────────────────────────────────
        if edit_data_root:
            edit_root = Path(edit_data_root)
            edit_pq = edit_root / edit_parquet_file
            print(f"Loading edit metadata: {edit_pq}")
            edf = pd.read_parquet(edit_pq)
            print(f"  Loaded {len(edf):,} edit pairs")

            # Drop empty captions
            empty = edf['caption'].isna() | (edf['caption'].str.len() < 5)
            if empty.sum():
                edf = edf[~empty].reset_index(drop=True)
                print(f"  Dropped {empty.sum()} empty captions → {len(edf):,}")

            # HF edit parquet uses 'width'/'height'; rename for sampler compatibility
            if 'bucket_w' not in edf.columns and 'width' in edf.columns:
                edf = edf.rename(columns={'width': 'bucket_w', 'height': 'bucket_h'})

            edf['data_type'] = 'edit'
            edf['_data_root'] = str(edit_root)
            frames.append(edf)
            print(f"  Edit data ready: {len(edf):,} pairs")

        # ── Combine ────────────────────────────────────────────────────────────
        self.meta_df = pd.concat(frames, ignore_index=True)

        if debug_mode:
            self.meta_df = self.meta_df.head(400)
        if max_samples:
            self.meta_df = self.meta_df.head(max_samples)

        n_gen = (self.meta_df['data_type'] == 'gen').sum()
        n_edit = (self.meta_df['data_type'] == 'edit').sum()
        print(f"  Combined: {len(self.meta_df):,} total (gen={n_gen:,}, edit={n_edit:,})")

    def __len__(self):
        return len(self.meta_df)

    # ── NPZ Loaders ───────────────────────────────────────────────────────────

    def _npz_path(self, data_root: str, img_path: str) -> Path:
        """Derive NPZ cache path: {img_stem}_flux_h.npz alongside image."""
        p = Path(data_root) / img_path
        return p.parent / (p.stem + "_flux_h.npz")

    def _load_npz(self, npz_path: Path):
        """Load latents + text embeddings from NPZ file."""
        with np.load(npz_path, allow_pickle=True) as data:
            vae_h = data['vae_h'].astype(np.float32)
            c = vae_h.shape[0] // 2
            mean, logvar = vae_h[:c], vae_h[c:]
            logvar = np.clip(logvar, -30.0, 20.0)
            latents = mean + np.exp(0.5 * logvar) * np.random.randn(*mean.shape).astype(np.float32)

            text_embeds = data.get(
                'prompt_embeds', data.get('text_embeds', np.zeros((1, 12288), dtype=np.float16))
            ).astype(np.float16)
            text_mask = data.get(
                'attention_mask', np.ones(text_embeds.shape[0], dtype=np.int8)
            ).astype(np.int8)
        return latents, text_embeds, text_mask

    # ── __getitem__ ────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        row = self.meta_df.iloc[index % len(self.meta_df)]
        data_type = row['data_type']
        data_root = row['_data_root']

        bw = int(row.get('bucket_w', row.get('width', 1024)))
        bh = int(row.get('bucket_h', row.get('height', 512)))

        if data_type == 'gen':
            npz = self._npz_path(data_root, row['image_path'])
            if not npz.exists():
                raise FileNotFoundError(
                    f"NPZ not found: {npz}\nRun: python scripts/cache_latents.py"
                )
            latents, text_embeds, text_mask = self._load_npz(npz)
            return {
                'latents': torch.from_numpy(latents),
                'text_embeds': torch.from_numpy(text_embeds.astype(np.float32)),
                'text_ids': None,
                'text_mask': torch.from_numpy(text_mask.astype(np.int8)),
                'bucket_size': (bh, bw),
                'caption': str(row.get('caption', '')),
                'aspect_ratio': float(row.get('aspect_ratio', bw / max(bh, 1))),
                'data_type': 'gen',
            }

        elif data_type == 'edit':
            src_npz = self._npz_path(data_root, row['source_image_path'])
            tgt_npz = self._npz_path(data_root, row['target_image_path'])
            if not src_npz.exists() or not tgt_npz.exists():
                raise FileNotFoundError(
                    f"Edit NPZ not found:\n  src={src_npz}\n  tgt={tgt_npz}\n"
                    "Run: python scripts/cache_latents.py for editing data"
                )
            src_latents, src_embeds, src_mask = self._load_npz(src_npz)
            tgt_latents, tgt_embeds, tgt_mask = self._load_npz(tgt_npz)
            return {
                'source_latents': torch.from_numpy(src_latents),
                'target_latents': torch.from_numpy(tgt_latents),
                'latents': torch.from_numpy(tgt_latents),  # fulltune compat
                'text_embeds': torch.from_numpy(tgt_embeds.astype(np.float32)),
                'text_ids': None,
                'text_mask': torch.from_numpy(tgt_mask.astype(np.int8)),
                'source_embeds': torch.from_numpy(src_embeds.astype(np.float32)),
                'bucket_size': (bh, bw),
                'caption': str(row.get('caption', '')),
                'aspect_ratio': float(row.get('aspect_ratio', bw / max(bh, 1))),
                'data_type': 'edit',
            }

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    # ── Collate Function (mirrors ArXiVParquetDatasetV4.collate_fn) ───────────

    def collate_fn(self, batch):
        """
        Collate mixed gen/edit batches.
        DistributedBucketSamplerV3 groups by data_type, so batches are typically
        pure-gen or pure-edit. The 'mixed' branch handles edge cases.
        """
        data_types = [x['data_type'] for x in batch]
        unique_types = set(data_types)

        embeds_list = [x['text_embeds'] for x in batch]
        masks_list = [x['text_mask'] for x in batch]
        padded_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks_list, batch_first=True, padding_value=0)

        result = {
            'text_embeds': padded_embeds,
            'text_mask': padded_masks,
            'text_ids': None,   # computed dynamically by _prepare_text_ids()
            'captions': [x['caption'] for x in batch],
            'bucket_size': batch[0]['bucket_size'],
            'aspect_ratio': batch[0]['aspect_ratio'],
            'data_types': data_types,
        }

        if unique_types == {'gen'}:
            result['batch_mode'] = 'gen'
            result['latents'] = torch.stack([x['latents'] for x in batch])

        elif unique_types == {'edit'}:
            result['batch_mode'] = 'edit'
            result['source_latents'] = torch.stack([x['source_latents'] for x in batch])
            result['target_latents'] = torch.stack([x['target_latents'] for x in batch])
            result['latents'] = result['target_latents']   # alias for fulltune compat

        else:
            # Mixed batch (should be rare with SamplerV3)
            result['batch_mode'] = 'mixed'
            gen_idx = [i for i, t in enumerate(data_types) if t == 'gen']
            edit_idx = [i for i, t in enumerate(data_types) if t == 'edit']
            result['gen_indices'] = gen_idx
            result['edit_indices'] = edit_idx
            if gen_idx:
                result['latents'] = torch.stack([batch[i]['latents'] for i in gen_idx])
            if edit_idx:
                result['source_latents'] = torch.stack([batch[i]['source_latents'] for i in edit_idx])
                result['target_latents'] = torch.stack([batch[i]['target_latents'] for i in edit_idx])

        return result
