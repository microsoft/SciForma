"""
SciForma HuggingFace Hub Dataset — loads directly from microsoft/SciFormaData-700K.

Images are embedded in the HF parquets as PIL.Image objects (HF Image feature).
This class is for users who downloaded data from HuggingFace and want to train
without pre-computing NPZ latent caches.

NOTE: This path is significantly slower than the NPZ-cache path because it
encodes images via VAE on-the-fly during training. For production training,
use ArXiVHFDatasetUnified with pre-computed NPZ caches instead.

Usage:
  from datasets import load_dataset
  # or point data_root to a local HF-format dataset directory

  dataset_cfg = dict(
      type='SciFormaHubDataset',
      hf_repo='microsoft/SciFormaData-700K',   # or local path
      subset='generation_768',                  # 'generation_768'|'generation_1024'|'editing'
      quality_filter='High',                    # None=all, 'High'=high quality only
      hf_token=None,                            # HuggingFace token if private
  )
"""

import io
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from sciforma.registry import DATASETS


@DATASETS.register_module()
class SciFormaHubDataset(Dataset):
    """
    Loads SciFormaData-700K directly from HuggingFace Hub.
    Images are PIL objects (decoded from embedded bytes in parquet).

    For training, images are converted to latents via VAE on-the-fly.
    Text embeddings are computed via the text encoder on-the-fly.
    This is slower than NPZ-cached training but requires no pre-processing.
    """

    def __init__(
        self,
        hf_repo: str = "microsoft/SciFormaData-700K",
        subset: str = "generation_768",
        quality_filter: str = None,
        hf_token: str = None,
        max_samples: int = None,
        debug_mode: bool = False,
        num_workers: int = 4,
    ):
        from datasets import load_dataset

        self.subset = subset
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None

        print(f"Loading {hf_repo} / {subset} from HuggingFace...")
        ds = load_dataset(hf_repo, subset, split="train", token=hf_token)
        print(f"  Loaded {len(ds):,} rows")

        # Filter by quality tier (generation subsets only)
        if quality_filter and "image" in ds.column_names:
            before = len(ds)
            ds = ds.filter(lambda x: x["quality_tier"] == quality_filter)
            print(f"  After quality_filter='{quality_filter}': {len(ds):,} (was {before:,})")

        # Filter empty captions
        ds = ds.filter(lambda x: x["caption"] is not None and len(x["caption"]) >= 10)

        if debug_mode:
            ds = ds.select(range(min(200, len(ds))))
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.ds = ds
        print(f"  Final: {len(self.ds):,} samples (subset={subset})")

    def __len__(self):
        return len(self.ds)

    def _pil_to_tensor(self, img: Image.Image, w: int, h: int) -> np.ndarray:
        """Resize PIL image and convert to float32 tensor for VAE."""
        img = img.resize((w, h), Image.LANCZOS).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        return arr.transpose(2, 0, 1)  # CHW

    def __getitem__(self, index):
        row = self.ds[index]
        bw = int(row.get("bucket_w", 1008))
        bh = int(row.get("bucket_h", 576))

        if self.subset in ("generation_768", "generation_1024"):
            img: Image.Image = row["image"]
            img_arr = self._pil_to_tensor(img, bw, bh)
            return {
                "image_tensor": torch.from_numpy(img_arr),  # CHW float32, [-1,1]
                "caption": row["caption"],
                "bucket_size": (bh, bw),
                "batch_mode": "gen",
                "data_type": "gen",
                "paper_id": row.get("paper_id", ""),
                "quality_tier": row.get("quality_tier", ""),
            }

        elif self.subset == "editing":
            src: Image.Image = row["source_image"]
            tgt: Image.Image = row["target_image"]
            src_arr = self._pil_to_tensor(src, bw, bh)
            tgt_arr = self._pil_to_tensor(tgt, bw, bh)
            return {
                "source_tensor": torch.from_numpy(src_arr),
                "target_tensor": torch.from_numpy(tgt_arr),
                "caption": row["caption"],
                "bucket_size": (bh, bw),
                "batch_mode": "edit",
                "data_type": "edit",
                "paper_id": row.get("paper_id", ""),
                "edit_bboxes": row.get("edit_bboxes", "[]"),
            }

        else:
            raise ValueError(f"Unknown subset: {self.subset}")
