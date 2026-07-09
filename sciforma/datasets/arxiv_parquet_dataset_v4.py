"""
ArXiV Parquet Dataset V4 - Mixed Generation + Editing Training Dataset

Supports two data modes simultaneously:
1. Generation (train_gen): Standard single-image generation data
   - NPZ contains: vae_h, prompt_embeds, attention_mask
   - Same format as V3 dataset (save_h format)
   
2. Editing (train_edit): Source→Target image editing data
   - NPZ contains: source_vae_h, target_vae_h, prompt_embeds, attention_mask
   - The prompt describes the editing instruction

Both modes share the same bucket resolution system.
The dataset does NOT do distributed sharding internally - accelerate.prepare handles that.

Usage:
    # Generation only
    dataset = ArXiVParquetDatasetV4(
        base_dir="/data/",
        gen_parquet_path="ArXiV_parquet/Flux2Klein9B_1024_pretrain",
        edit_parquet_path=None,
    )
    
    # Edit only
    dataset = ArXiVParquetDatasetV4(
        base_dir="/data/",
        gen_parquet_path=None,
        edit_parquet_path="ArXiV_parquet/Flux2Klein9B_1024_edit",
    )
    
    # Mixed
    dataset = ArXiVParquetDatasetV4(
        base_dir="/data/",
        gen_parquet_path="ArXiV_parquet/Flux2Klein9B_1024_pretrain",
        edit_parquet_path="ArXiV_parquet/Flux2Klein9B_1024_edit",
    )
"""

import os
import json
import math
import time
import random
import itertools
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
from sciforma.registry import DATASETS
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist


@DATASETS.register_module()
class ArXiVParquetDatasetV4(Dataset):
    """
    Mixed Generation + Editing Dataset for Flux2Klein.
    
    Loads both gen and edit parquet data into a unified DataFrame with a 
    'data_type' column ('gen' or 'edit') so the training loop can branch 
    accordingly.
    
    Key design decisions:
    - No distributed sharding inside dataset (accelerate.prepare handles it)
    - Bucket-aware: both gen/edit share the same bucket_h/bucket_w grid
    - Interleaved sampling via shared DistributedBucketSamplerV2
    - Collate function returns a 'data_type' field per sample
    """
    
    # ==================== Default parquet columns ====================
    GEN_COLUMNS = [
        "caption", "cache_path", "vae_h_shape", "prompt_embeds_shape",
        "bucket_w", "bucket_h", "aspect_ratio"
    ]
    EDIT_COLUMNS = [
        "caption", "cache_path", 
        "source_vae_h_shape", "target_vae_h_shape", "prompt_embeds_shape",
        "bucket_w", "bucket_h", "aspect_ratio",
        "edit_bboxes",  # Optional: JSON string of [[x1,y1,x2,y2], ...] for ground truth edit mask
    ]
    
    def __init__(
        self,
        base_dir: str,
        gen_parquet_path: str = None,
        edit_parquet_path: str = None,
        num_workers: int = 16,
        num_train_examples: int = None,
        debug_mode: bool = False,
        is_main_process: bool = False,
        stat_data: bool = False,
        gen_weight: float = 1.0,
        edit_weight: float = 1.0,
        quality_filter: str = None,
    ):
        """
        Args:
            base_dir: Root directory for data (e.g., /data/yuxuanluo/ or /mnt/data/)
            gen_parquet_path: Relative path under base_dir for generation parquet data.
                              Set None to disable generation data loading.
            edit_parquet_path: Relative path under base_dir for editing parquet data.
                               Set None to disable editing data loading.
            num_workers: Number of parallel workers for loading parquet metadata.
            num_train_examples: Max total samples to use (None = use all).
            debug_mode: If True, only load first 200 parquet files per source.
            is_main_process: Whether this is the main process (for logging stats).
            stat_data: Whether to print bucket statistics.
            gen_weight: Sampling weight for generation data (for future weighted sampling).
            edit_weight: Sampling weight for editing data (for future weighted sampling).
            quality_filter: Filter generation data by quality grade from parquet.
                            Options: 'high', 'high+medium', 'all', None (no filtering).
                            Requires 'quality_grade' column in *_train.parquet.
        """
        assert gen_parquet_path is not None or edit_parquet_path is not None, \
            "At least one of gen_parquet_path or edit_parquet_path must be provided!"
        
        self.base_path = Path(base_dir)
        self.gen_weight = gen_weight
        self.edit_weight = edit_weight
        self.quality_filter = quality_filter
        
        meta_frames = []
        
        # ==================== Load Generation Data ====================
        if gen_parquet_path is not None:
            gen_base = self.base_path / gen_parquet_path
            print(f"🔍 [V4-Gen] Building metadata from *_train.parquet in {gen_base}...")
            gen_paths = self._collect_parquet_files(gen_base, pattern="*_train.parquet", debug_mode=debug_mode)
            if not gen_paths:
                # Fallback: try all *.parquet if no *_train.parquet found
                print(f"⚠️  No *_train.parquet found, falling back to *.parquet")
                gen_paths = self._collect_parquet_files(gen_base, pattern="*.parquet", debug_mode=debug_mode)
            
            print(f"⏳ Loading {len(gen_paths)} gen parquet files...")
            gen_df = self._parallel_load_parquet(
                gen_paths, max_workers=num_workers, 
                default_key=self.GEN_COLUMNS, data_type="gen"
            )
            # Store the gen data_base_path for resolving NPZ paths
            gen_df['_data_base'] = str(gen_base)
            
            # Apply quality filtering if enabled and column exists
            if quality_filter and 'quality_grade' in gen_df.columns:
                before_count = len(gen_df)
                if quality_filter.lower() == 'high':
                    gen_df = gen_df[gen_df['quality_grade'].str.upper() == 'HIGH'].reset_index(drop=True)
                elif quality_filter.lower() in ('high+medium', 'high_medium'):
                    gen_df = gen_df[gen_df['quality_grade'].str.upper().isin(['HIGH', 'MEDIUM'])].reset_index(drop=True)
                elif quality_filter.lower() == 'all':
                    # Keep everything, including UNKNOWN/ERROR
                    pass
                else:
                    print(f"⚠️  Unknown quality_filter='{quality_filter}', keeping all")
                after_count = len(gen_df)
                print(f"🔍 [V4-Gen] Quality filter '{quality_filter}': {before_count} -> {after_count} "
                      f"(removed {before_count - after_count})")
            elif quality_filter and 'quality_grade' not in gen_df.columns:
                print(f"⚠️  quality_filter='{quality_filter}' requested but no 'quality_grade' column in parquet. "
                      f"Run build_train_parquet_with_quality.py first.")
            
            meta_frames.append(gen_df)
            print(f"✅ Gen data: {len(gen_df)} samples")
        
        # ==================== Load Editing Data ====================
        if edit_parquet_path is not None:
            edit_base = self.base_path / edit_parquet_path
            print(f"🔍 [V4-Edit] Building metadata from *_train.parquet in {edit_base}...")
            edit_paths = self._collect_parquet_files(edit_base, pattern="*_train.parquet", debug_mode=debug_mode)
            if not edit_paths:
                print(f"⚠️  No *_train.parquet found, falling back to *.parquet")
                edit_paths = self._collect_parquet_files(edit_base, pattern="*.parquet", debug_mode=debug_mode)
            
            print(f"⏳ Loading {len(edit_paths)} edit parquet files...")
            edit_df = self._parallel_load_parquet(
                edit_paths, max_workers=num_workers,
                default_key=self.EDIT_COLUMNS, data_type="edit"
            )
            edit_df['_data_base'] = str(edit_base)
            meta_frames.append(edit_df)
            print(f"✅ Edit data: {len(edit_df)} samples")
        
        # ==================== Merge ====================
        self.meta_df = pd.concat(meta_frames, ignore_index=True)
        
        # Limit total examples
        if num_train_examples is not None and len(self.meta_df) > num_train_examples:
            self.meta_df = self.meta_df.iloc[:num_train_examples]
        
        print(f"✅ [V4] Total loaded: {len(self.meta_df)} samples "
              f"(gen={len(self.meta_df[self.meta_df['data_type'] == 'gen'])}, "
              f"edit={len(self.meta_df[self.meta_df['data_type'] == 'edit'])})")
        
        # Filter small buckets
        self._filter_small_buckets(batch_size=8, num_replicas=4)
        
        if stat_data and is_main_process:
            self._print_stats()
    
    # =====================================================================
    # Parquet File Discovery
    # =====================================================================
    
    def _collect_parquet_files(self, base: Path, pattern: str = "*.parquet", debug_mode: bool = False):
        """Collect parquet files from year subdirectories."""
        year_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        all_paths = []
        for y_dir in year_dirs:
            all_paths.extend(sorted(y_dir.glob(pattern)))
        if debug_mode:
            all_paths = all_paths[:200]
        return all_paths
    
    # =====================================================================
    # Parallel Parquet Loading
    # =====================================================================
    
    def _parallel_load_parquet(self, paths, max_workers, default_key, data_type):
        """Load parquet files in parallel, adding data_type column."""
        meta_list = []
        
        def load_one_file(path):
            try:
                pf = pq.ParquetFile(path)
                available_columns = [field.name for field in pf.schema_arrow]
                
                # Build columns to read with fallback mapping
                columns_to_read = []
                rename_map = {}
                for col in default_key:
                    if col in available_columns:
                        columns_to_read.append(col)
                    elif col == 'vae_h_shape' and 'latent_shape' in available_columns:
                        columns_to_read.append('latent_shape')
                        rename_map['latent_shape'] = 'vae_h_shape'
                    elif col == 'prompt_embeds_shape' and 'text_embeds_shape' in available_columns:
                        columns_to_read.append('text_embeds_shape')
                        rename_map['text_embeds_shape'] = 'prompt_embeds_shape'
                    elif col == 'source_vae_h_shape' and 'source_latent_shape' in available_columns:
                        columns_to_read.append('source_latent_shape')
                        rename_map['source_latent_shape'] = 'source_vae_h_shape'
                    elif col == 'target_vae_h_shape' and 'target_latent_shape' in available_columns:
                        columns_to_read.append('target_latent_shape')
                        rename_map['target_latent_shape'] = 'target_vae_h_shape'
                    # Skip columns not found (non-critical)
                
                # Also read quality_grade column if available (for quality filtering)
                if 'quality_grade' in available_columns and 'quality_grade' not in columns_to_read:
                    columns_to_read.append('quality_grade')
                
                df = pf.read(columns=columns_to_read).to_pandas()
                
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                df['source_file'] = str(path)
                df['local_index'] = range(len(df))
                df['data_type'] = data_type
                return df
            except Exception as e:
                return f"Error: {path} | {str(e)}"
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(load_one_file, p): p for p in paths}
            for future in tqdm(as_completed(future_to_path), total=len(paths), 
                              desc=f"Scanning {data_type} Parquet"):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    meta_list.append(result)
                elif isinstance(result, str):
                    print(f"⚠️  {result}")
        
        if not meta_list:
            return pd.DataFrame()
        
        return pd.concat(meta_list, ignore_index=True)
    
    # =====================================================================
    # Bucket Filtering
    # =====================================================================
    
    def _filter_small_buckets(self, batch_size, num_replicas):
        """Filter out buckets with too few samples for distributed training.
        Groups by (bucket_h, bucket_w, data_type) to match V3 sampler behavior."""
        group_cols = ['bucket_h', 'bucket_w']
        if 'data_type' in self.meta_df.columns:
            group_cols.append('data_type')
        counts = self.meta_df.groupby(group_cols).indices
        valid_indices = []
        
        for bucket_key, indices in counts.items():
            total_needed = batch_size * num_replicas * 2
            if len(indices) >= total_needed:
                valid_indices.extend(indices)
        
        before = len(self.meta_df)
        self.meta_df = self.meta_df.iloc[valid_indices].reset_index(drop=True)
        print(f"[V4] Bucket filter: {before} → {len(self.meta_df)} samples "
              f"(gen={len(self.meta_df[self.meta_df['data_type'] == 'gen'])}, "
              f"edit={len(self.meta_df[self.meta_df['data_type'] == 'edit'])})")
    
    def _print_stats(self):
        """Print dataset statistics."""
        print(f"\n📊 [V4] Data Statistics:")
        for dtype in ['gen', 'edit']:
            subset = self.meta_df[self.meta_df['data_type'] == dtype]
            if len(subset) == 0:
                continue
            print(f"\n  [{dtype.upper()}] {len(subset)} samples:")
            bucket_counts = subset.groupby(['bucket_h', 'bucket_w']).size().reset_index(name='counts')
            for _, row in bucket_counts.iterrows():
                h, w, count = row['bucket_h'], row['bucket_w'], row['counts']
                print(f"    Resolution {w}x{h}: {count} samples ({count/len(subset)*100:.2f}%)")
    
    # =====================================================================
    # Data Access
    # =====================================================================
    
    def __len__(self):
        return len(self.meta_df)
    
    def get_data_info(self, index):
        index = index % len(self.meta_df)
        return self.meta_df.iloc[index]
    
    def __getitem__(self, index):
        meta_row = self.get_data_info(index)
        data_type = meta_row['data_type']
        
        if data_type == 'gen':
            return self._getitem_gen(meta_row)
        elif data_type == 'edit':
            return self._getitem_edit(meta_row)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    # ==================== Generation __getitem__ ====================
    
    def _getitem_gen(self, meta_row):
        """Load a generation sample (same as V3)."""
        data_base = Path(meta_row['_data_base'])
        cache_path = data_base / meta_row['cache_path']
        
        latents, text_embeds, text_mask = self._read_gen_npz(cache_path, meta_row)
        
        result = {
            "data_type": "gen",
            "latents": latents,          # [C, H, W]
            "text_embeds": text_embeds,   # [SeqLen, Hidden]
            "text_mask": text_mask,       # [SeqLen]
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "aspect_ratio": meta_row['aspect_ratio'],
            "caption": meta_row['caption'],
        }
        return result
    
    def _read_gen_npz(self, npz_path, meta_row):
        """Read generation NPZ: vae_h → sampled latents."""
        try:
            with np.load(npz_path, allow_pickle=True) as npz_data:
                # Load VAE h and sample latents
                if 'vae_h' in npz_data:
                    vae_h = npz_data['vae_h'].astype(np.float32)
                    latents_np = self._sample_latents_from_h(vae_h)
                elif 'latents' in npz_data:
                    latents_np = npz_data['latents'].astype(np.float32)
                else:
                    raise KeyError("Neither 'vae_h' nor 'latents' found in npz")
                
                text_embeds_np = self._read_text_embeds(npz_data)
                text_mask_np = self._read_text_mask(npz_data, text_embeds_np.shape[0])
                
                latents, text_embeds, text_mask = self._to_tensors_and_align(
                    latents_np, text_embeds_np, text_mask_np
                )
                return latents, text_embeds, text_mask
                
        except Exception as e:
            print(f"❌ Error reading gen npz {npz_path}: {e}")
            return self._fallback_gen_tensors(meta_row)
    
    # ==================== Editing __getitem__ ====================
    
    def _getitem_edit(self, meta_row):
        """Load an editing sample with source + target latents."""
        data_base = Path(meta_row['_data_base'])
        cache_path = data_base / meta_row['cache_path']
        
        source_latents, target_latents, text_embeds, text_mask = self._read_edit_npz(
            cache_path, meta_row
        )
        
        result = {
            "data_type": "edit",
            "source_latents": source_latents,   # [C, H, W]
            "target_latents": target_latents,    # [C, H, W]
            "text_embeds": text_embeds,          # [SeqLen, Hidden]
            "text_mask": text_mask,              # [SeqLen]
            "bucket_size": (meta_row['bucket_h'], meta_row['bucket_w']),
            "aspect_ratio": meta_row['aspect_ratio'],
            "caption": meta_row['caption'],
        }
        
        # Compute edit region token mask from ground truth bboxes (if available)
        edit_mask = self._compute_edit_mask_from_bbox(meta_row)
        if edit_mask is not None:
            result["edit_mask"] = edit_mask
        
        return result
    
    def _compute_edit_mask_from_bbox(self, meta_row):
        """
        Compute a binary token-space mask from ground truth edit bounding boxes.
        
        Mapping: pixel coords → token coords via 16× downsampling
          (2× VAE patchify + 8× VAE spatial compression)
        Token ordering: raster scan (row-major) matching _pack_latents_flux2.
        
        Returns:
            torch.Tensor of shape (H_tok * W_tok,) with 1.0 for edit tokens,
            or None if bbox data is not available.
        """
        bbox_str = meta_row.get('edit_bboxes', None)
        if bbox_str is None or (isinstance(bbox_str, float) and math.isnan(bbox_str)):
            return None
        
        try:
            bboxes = json.loads(bbox_str) if isinstance(bbox_str, str) else bbox_str
        except (json.JSONDecodeError, TypeError):
            return None
        
        if not bboxes:
            return None
        
        bucket_h = int(meta_row['bucket_h'])
        bucket_w = int(meta_row['bucket_w'])
        H_tok = bucket_h // 16
        W_tok = bucket_w // 16
        
        spatial_mask = np.zeros((H_tok, W_tok), dtype=np.float32)
        
        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            # Floor for start, ceil for end to ensure full coverage
            tx1 = max(0, x1 // 16)
            ty1 = max(0, y1 // 16)
            tx2 = min(W_tok, -(-x2 // 16))  # ceiling division
            ty2 = min(H_tok, -(-y2 // 16))  # ceiling division
            spatial_mask[ty1:ty2, tx1:tx2] = 1.0
        
        # Flatten to 1D in raster scan order (row-major)
        token_mask = spatial_mask.reshape(-1)
        return torch.from_numpy(token_mask)
    
    def _read_edit_npz(self, npz_path, meta_row):
        """Read editing NPZ: source_vae_h + target_vae_h → sampled latents."""
        try:
            with np.load(npz_path, allow_pickle=True) as npz_data:
                # Source latents
                if 'source_vae_h' in npz_data:
                    src_h = npz_data['source_vae_h'].astype(np.float32)
                    source_np = self._sample_latents_from_h(src_h)
                elif 'source_latents' in npz_data:
                    source_np = npz_data['source_latents'].astype(np.float32)
                else:
                    raise KeyError("Neither 'source_vae_h' nor 'source_latents' found in npz")
                
                # Target latents
                if 'target_vae_h' in npz_data:
                    tgt_h = npz_data['target_vae_h'].astype(np.float32)
                    target_np = self._sample_latents_from_h(tgt_h)
                elif 'target_latents' in npz_data:
                    target_np = npz_data['target_latents'].astype(np.float32)
                else:
                    raise KeyError("Neither 'target_vae_h' nor 'target_latents' found in npz")
                
                text_embeds_np = self._read_text_embeds(npz_data)
                text_mask_np = self._read_text_mask(npz_data, text_embeds_np.shape[0])
                
                source_latents = torch.from_numpy(source_np)
                target_latents = torch.from_numpy(target_np)
                text_embeds = torch.from_numpy(text_embeds_np)
                text_mask = torch.from_numpy(text_mask_np)
                
                # Align text lengths
                L_embed = text_embeds.shape[0]
                L_mask = text_mask.shape[0]
                L = min(L_embed, L_mask)
                if L_embed != L_mask:
                    text_embeds = text_embeds[:L]
                    text_mask = text_mask[:L]
                
                return source_latents, target_latents, text_embeds, text_mask
                
        except Exception as e:
            print(f"❌ Error reading edit npz {npz_path}: {e}")
            return self._fallback_edit_tensors(meta_row)
    
    # =====================================================================
    # Shared Helpers
    # =====================================================================
    
    def _sample_latents_from_h(self, vae_h: np.ndarray) -> np.ndarray:
        """
        Sample latents from VAE encoder hidden states.
        Pure numpy reparameterization trick (no torch overhead in DataLoader workers).
        
        Note: Flux2 VAE does NOT use scaling_factor.
        """
        c = vae_h.shape[0] // 2
        mean = vae_h[:c]
        logvar = vae_h[c:]
        logvar = np.clip(logvar, -30.0, 20.0)
        std = np.exp(0.5 * logvar)
        sample = mean + std * np.random.randn(*mean.shape).astype(np.float32)
        return sample
    
    def _read_text_embeds(self, npz_data) -> np.ndarray:
        """Read text embeddings from NPZ with key fallbacks."""
        if 'prompt_embeds' in npz_data:
            return npz_data['prompt_embeds'].astype(np.float16)
        elif 'text_embeds' in npz_data:
            return npz_data['text_embeds'].astype(np.float16)
        else:
            raise KeyError("Neither 'prompt_embeds' nor 'text_embeds' found in npz")
    
    def _read_text_mask(self, npz_data, seq_len: int) -> np.ndarray:
        """Read attention mask from NPZ with fallback to all-ones."""
        if 'attention_mask' in npz_data:
            return npz_data['attention_mask'].astype(np.int8)
        elif 'text_mask' in npz_data:
            return npz_data['text_mask'].astype(np.int8)
        else:
            return np.ones((seq_len,), dtype=np.int8)
    
    def _to_tensors_and_align(self, latents_np, text_embeds_np, text_mask_np):
        """Convert numpy arrays to tensors and align text sequence lengths."""
        latents = torch.from_numpy(latents_np)
        text_embeds = torch.from_numpy(text_embeds_np)
        text_mask = torch.from_numpy(text_mask_np)
        
        L_embed = text_embeds.shape[0]
        L_mask = text_mask.shape[0]
        L = min(L_embed, L_mask)
        if L_embed != L_mask:
            text_embeds = text_embeds[:L]
            text_mask = text_mask[:L]
        
        return latents, text_embeds, text_mask
    
    def _fallback_gen_tensors(self, meta_row):
        """Return zero tensors when gen sample fails to load."""
        vae_h_shape = meta_row.get('vae_h_shape', [64, 16, 16])
        c = int(vae_h_shape[0]) // 2
        h = int(vae_h_shape[1])
        w = int(vae_h_shape[2])
        latents = torch.zeros((c, h, w), dtype=torch.float32)
        
        text_shape = meta_row.get('prompt_embeds_shape', [512, 4096])
        text_embeds = torch.zeros(tuple(map(int, text_shape)), dtype=torch.float16)
        text_mask = torch.zeros((int(text_shape[0]),), dtype=torch.int8)
        
        return latents, text_embeds, text_mask
    
    def _fallback_edit_tensors(self, meta_row):
        """Return zero tensors when edit sample fails to load."""
        # Try source/target shapes, fall back to vae_h_shape
        src_shape = meta_row.get('source_vae_h_shape', meta_row.get('vae_h_shape', [64, 16, 16]))
        tgt_shape = meta_row.get('target_vae_h_shape', meta_row.get('vae_h_shape', [64, 16, 16]))
        
        c_s = int(src_shape[0]) // 2
        h_s, w_s = int(src_shape[1]), int(src_shape[2])
        source_latents = torch.zeros((c_s, h_s, w_s), dtype=torch.float32)
        
        c_t = int(tgt_shape[0]) // 2
        h_t, w_t = int(tgt_shape[1]), int(tgt_shape[2])
        target_latents = torch.zeros((c_t, h_t, w_t), dtype=torch.float32)
        
        text_shape = meta_row.get('prompt_embeds_shape', [512, 4096])
        text_embeds = torch.zeros(tuple(map(int, text_shape)), dtype=torch.float16)
        text_mask = torch.zeros((int(text_shape[0]),), dtype=torch.int8)
        
        return source_latents, target_latents, text_embeds, text_mask
    
    # =====================================================================
    # Collate Function
    # =====================================================================
    
    def collate_fn(self, batch):
        """
        Collate function handling mixed gen/edit batches.
        
        Since bucket sampler groups by (bucket_h, bucket_w), all samples in a
        batch have the same spatial dimensions. However, a batch may contain
        BOTH gen and edit samples.
        
        Strategy:
        - If ALL samples are gen → return gen batch format (backward compat)
        - If ALL samples are edit → return edit batch format
        - If MIXED → split into gen/edit sub-batches within the same dict
        
        The training iteration function checks 'batch_mode' to decide logic.
        """
        from torch.nn.utils.rnn import pad_sequence
        
        data_types = [x['data_type'] for x in batch]
        unique_types = set(data_types)
        
        # Text embeddings and masks are always present
        embeds_list = [x['text_embeds'] for x in batch]
        masks_list = [x['text_mask'] for x in batch]
        padded_embeds = pad_sequence(embeds_list, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks_list, batch_first=True, padding_value=0)
        
        result = {
            "text_embeds": padded_embeds,
            "text_mask": padded_masks,
            "captions": [x['caption'] for x in batch],
            "bucket_size": batch[0]['bucket_size'],
            "aspect_ratio": batch[0]['aspect_ratio'],
            "data_types": data_types,
        }
        
        if unique_types == {'gen'}:
            # Pure generation batch
            result["batch_mode"] = "gen"
            result["latents"] = torch.stack([x['latents'] for x in batch])
            
        elif unique_types == {'edit'}:
            # Pure editing batch
            result["batch_mode"] = "edit"
            result["source_latents"] = torch.stack([x['source_latents'] for x in batch])
            result["target_latents"] = torch.stack([x['target_latents'] for x in batch])
            # Stack edit region masks if available (from ground truth bboxes)
            if all('edit_mask' in x for x in batch):
                result["edit_mask"] = torch.stack([x['edit_mask'] for x in batch])
            
        else:
            # Mixed batch - provide both, with per-sample indices
            result["batch_mode"] = "mixed"
            gen_indices = [i for i, t in enumerate(data_types) if t == 'gen']
            edit_indices = [i for i, t in enumerate(data_types) if t == 'edit']
            
            result["gen_indices"] = gen_indices
            result["edit_indices"] = edit_indices
            
            # For gen samples, stack latents
            if gen_indices:
                result["latents"] = torch.stack([batch[i]['latents'] for i in gen_indices])
            
            # For edit samples, stack source/target
            if edit_indices:
                result["source_latents"] = torch.stack([batch[i]['source_latents'] for i in edit_indices])
                result["target_latents"] = torch.stack([batch[i]['target_latents'] for i in edit_indices])
        
        return result


# =====================================================================
# DistributedBucketSamplerV3: Type-Aware Grouping
# =====================================================================

@DATASETS.register_module()
class DistributedBucketSamplerV3(Sampler):
    """
    Extends DistributedBucketSamplerV2 with data_type-aware grouping.
    
    Groups by (bucket_h, bucket_w, data_type) instead of just (bucket_h, bucket_w).
    This guarantees every batch contains only one data type (all gen or all edit),
    eliminating the need for complex mixed-batch handling in the iteration function.
    
    Falls back to (bucket_h, bucket_w) grouping if 'data_type' column is missing
    (e.g. when used with V2/V3 gen-only datasets).
    """
    
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None,
                 drop_last=True, shuffle=True, seed=42):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Group by (bucket_h, bucket_w, data_type) if data_type column exists
        if 'data_type' in self.dataset.meta_df.columns:
            self.groups = self.dataset.meta_df.groupby(
                ['bucket_h', 'bucket_w', 'data_type']
            ).indices
        else:
            # Fallback for gen-only datasets without data_type column
            self.groups = self.dataset.meta_df.groupby(
                ['bucket_h', 'bucket_w']
            ).indices
    
    def __iter__(self):
        combined_seed = self.seed + self.epoch
        g = torch.Generator()
        g.manual_seed(combined_seed)
        rng = random.Random(combined_seed)
        
        all_batch_lists = []
        sorted_bucket_keys = sorted(self.groups.keys(), key=str)
        
        for bucket_key in sorted_bucket_keys:
            indices = self.groups[bucket_key].tolist()
            if self.shuffle:
                rng.shuffle(indices)
            
            if self.drop_last:
                total_per_bucket = (
                    len(indices) // (self.num_replicas * self.batch_size)
                ) * (self.num_replicas * self.batch_size)
                indices = indices[:total_per_bucket]
            else:
                total_per_bucket = int(
                    math.ceil(len(indices) / (self.num_replicas * self.batch_size))
                ) * (self.num_replicas * self.batch_size)
                indices += indices[:(total_per_bucket - len(indices))]
            
            bucket_batches = []
            for i in range(0, len(indices), self.batch_size * self.num_replicas):
                chunk = indices[i : i + self.batch_size * self.num_replicas]
                my_batch = chunk[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]
                if len(my_batch) == self.batch_size:
                    bucket_batches.append(my_batch)
            all_batch_lists.extend(bucket_batches)
        
        if self.shuffle:
            rng.shuffle(all_batch_lists)
        return iter(all_batch_lists)
    
    def __len__(self):
        total_batches = 0
        for bucket_key in self.groups:
            indices = self.groups[bucket_key]
            num_samples_per_replica = len(indices) // self.num_replicas
            if self.drop_last:
                total_batches += num_samples_per_replica // self.batch_size
            else:
                total_batches += int(math.ceil(num_samples_per_replica / self.batch_size))
        return total_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch
