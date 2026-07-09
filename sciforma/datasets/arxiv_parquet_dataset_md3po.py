"""
ArXiV Parquet Dataset - MD3PO (Multi-Dimensional Decoupled DPO)

For each prompt group, build one tuple:
  - winner y+
  - three dimension-anchored losers: y-_component, y-_text, y-_arrow

Design goals:
  - keep training loop API close to DPO
  - prefer hard negatives that are bad on target dim but not catastrophic elsewhere
  - optionally enforce distinct losers for the three dimensions
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from sciforma.registry import DATASETS


@DATASETS.register_module()
class ArXiVParquetDatasetMD3PO(Dataset):
    """Build (winner, loser_c, loser_t, loser_a) tuples from grouped v9r parquet."""

    REQUIRED_COLUMNS = ["group_id", "caption", "cache_path", "bucket_h", "bucket_w"]
    SCORE_COLUMNS = ["reward", "component_score", "text_score", "arrow_score"]
    OPTIONAL_COLUMNS = ["aspect_ratio", "text_cache_path", "shard"]

    _DIM_ALIAS = {
        "component_score": "component",
        "text_score": "text",
        "arrow_score": "arrow",
    }

    def __init__(
        self,
        base_dir: str,
        parquet_base_path: str,
        num_workers: int = 8,
        num_train_examples: int = None,
        debug_mode: bool = False,
        is_main_process: bool = False,
        stat_data: bool = False,
        path_remapping: dict = None,
        parquet_glob: str = "gdro_shard_*.parquet",
        deterministic_latents: bool = True,
        # Winner quality gate.
        min_winner_score: float = 0.0,
        # Compatibility args accepted from existing DPO/ADPO configs.
        min_bucket_samples: int = 0,
        min_score_gap: float = 0.0,
        max_score_gap: float = 0.0,
        max_loser_score: float = 0.0,
        gt_winner_threshold: float = 0.0,
        # MD3PO options.
        target_dims: tuple = ("component_score", "text_score", "arrow_score"),
        winner_key: str = "reward",
        target_min_gap: float = 0.30,
        other_min_gap: float = 0.00,
        other_max_gap: float = 0.40,
        loser_balance_lambda: float = 0.50,
        min_total_gap: float = 0.0,
        strict_all_dims: bool = True,
        require_distinct_losers: bool = True,
        min_group_images: int = 6,
        min_reward: float = None,
        inject_global_worst: bool = False,
        global_worst_min_gap: float = 0.20,
        **kwargs,
    ):
        self.base_path = Path(base_dir)
        self.data_base_path = self.base_path / parquet_base_path
        self.path_remapping = path_remapping or {}
        self.deterministic_latents = deterministic_latents

        self.min_winner_score = float(min_winner_score)
        self.target_dims = tuple(target_dims)
        self.winner_key = str(winner_key)
        self.target_min_gap = float(target_min_gap)
        self.other_min_gap = float(other_min_gap)
        self.other_max_gap = float(other_max_gap)
        self.loser_balance_lambda = float(loser_balance_lambda)
        self.min_total_gap = float(min_total_gap)
        self.strict_all_dims = bool(strict_all_dims)
        self.require_distinct_losers = bool(require_distinct_losers)
        self.min_group_images = int(min_group_images)
        self.inject_global_worst = bool(inject_global_worst)
        self.global_worst_min_gap = float(global_worst_min_gap)

        unknown_keys = list(kwargs.keys())
        if unknown_keys:
            print(f"[MD3PO] Ignoring unsupported extra config keys: {unknown_keys}")

        print(
            f"[MD3PO] Loading group parquets from {self.data_base_path} "
            f"(glob={parquet_glob!r})"
        )
        all_paths = sorted(self.data_base_path.glob(parquet_glob))
        if not all_paths:
            all_paths = sorted(self.data_base_path.glob("*.parquet"))
            print("[MD3PO] parquet_glob matched nothing, fallback to *.parquet")
        if not all_paths:
            raise FileNotFoundError(f"No parquet files found under {self.data_base_path}")
        if debug_mode:
            all_paths = all_paths[:3]

        self.meta_df = self._load_parquets_parallel(all_paths, max_workers=num_workers)
        print(f"[MD3PO] Loaded {len(self.meta_df)} candidate rows")

        missing_dims = [d for d in self.target_dims if d not in self.meta_df.columns]
        if missing_dims:
            raise ValueError(
                f"[MD3PO] Missing target dim columns: {missing_dims}. "
                f"Available: {list(self.meta_df.columns)}"
            )

        if min_reward is not None and "reward" in self.meta_df.columns:
            before = len(self.meta_df)
            self.meta_df = self.meta_df[self.meta_df["reward"] >= float(min_reward)].reset_index(drop=True)
            print(f"[MD3PO] min_reward={min_reward}: {before} -> {len(self.meta_df)} rows")

        self.triple_df = self._build_md3po_triples()

        if num_train_examples is not None and len(self.triple_df) > num_train_examples:
            self.triple_df = self.triple_df.iloc[:num_train_examples].reset_index(drop=True)
            print(f"[MD3PO] Truncated to {len(self.triple_df)} triples")

        if stat_data and is_main_process:
            self._print_stats()

    def _load_parquets_parallel(self, paths, max_workers):
        meta_list = []

        def load_one(path):
            try:
                pf = pq.ParquetFile(path)
                available = {field.name for field in pf.schema_arrow}
                missing = [c for c in self.REQUIRED_COLUMNS if c not in available]
                if missing:
                    return f"Skip (missing {missing}): {path}"

                want = [
                    c for c in (self.REQUIRED_COLUMNS + self.SCORE_COLUMNS + self.OPTIONAL_COLUMNS)
                    if c in available
                ]
                df = pf.read(columns=want).to_pandas()
                df["_source_file"] = str(path)
                return df
            except Exception as e:
                return f"Error: {path} | {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(load_one, p): p for p in paths}
            for future in tqdm(as_completed(futures), total=len(paths), desc="Scanning MD3PO Parquets"):
                result = future.result()
                if isinstance(result, pd.DataFrame):
                    meta_list.append(result)
                else:
                    print(f"[MD3PO] {result}")

        if not meta_list:
            raise RuntimeError("No MD3PO parquet data loaded")
        return pd.concat(meta_list, ignore_index=True)

    def _pick_winner_idx(self, rows: pd.DataFrame) -> int:
        if self.winner_key in rows.columns:
            vals = rows[self.winner_key].astype(float).to_numpy()
            # Tie-break: among tied max values, prefer higher sum of target dims
            max_val = np.nanmax(vals)
            tied = np.where(np.abs(vals - max_val) < 1e-8)[0]
            if len(tied) == 1:
                return int(tied[0])
            tie_scores = np.zeros(len(tied), dtype=np.float64)
            for d in self.target_dims:
                if d in rows.columns:
                    dv = rows[d].astype(float).to_numpy()
                    tie_scores += np.nan_to_num(dv[tied], nan=0.0)
            return int(tied[np.argmax(tie_scores)])

        # Fallback: average of available target dims.
        dims = [d for d in self.target_dims if d in rows.columns]
        if not dims:
            return 0
        score = np.zeros(len(rows), dtype=np.float64)
        for d in dims:
            score += np.nan_to_num(rows[d].astype(float).to_numpy(), nan=0.0)
        return int(np.argmax(score))

    def _select_dim_loser_candidates(self, rows: pd.DataFrame, winner_idx: int, target_dim: str):
        """Return ALL valid loser candidates for `target_dim`, sorted by rank_score descending."""
        vals = {d: rows[d].astype(float).to_numpy() for d in self.target_dims}
        total = rows["reward"].astype(float).to_numpy() if "reward" in rows.columns else None
        others = [d for d in self.target_dims if d != target_dim]

        candidates = []

        for j in range(len(rows)):
            if j == winner_idx:
                continue

            target_gap = float(vals[target_dim][winner_idx] - vals[target_dim][j])
            if target_gap < self.target_min_gap:
                continue

            other_gaps = []
            ok = True
            for od in others:
                g = float(vals[od][winner_idx] - vals[od][j])
                if g < self.other_min_gap:
                    ok = False
                    break
                if g > self.other_max_gap:
                    ok = False
                    break
                other_gaps.append(g)
            if not ok:
                continue

            total_gap = float((total[winner_idx] - total[j]) if total is not None else target_gap)
            if total_gap < self.min_total_gap:
                continue

            mean_other = float(np.mean(other_gaps)) if other_gaps else 0.0
            rank_score = target_gap - self.loser_balance_lambda * mean_other

            candidates.append({
                "loser_idx": j,
                "target_gap": target_gap,
                "total_gap": total_gap,
                "mean_other_gap": mean_other,
                "rank_score": rank_score,
            })

        # Sort descending by rank_score, then by larger target_gap as tie-break
        candidates.sort(key=lambda c: (-c["rank_score"], -c["target_gap"]))
        return candidates

    def _build_md3po_triples(self):
        grouped = self.meta_df.groupby("group_id")
        records = []

        dropped_small = 0
        dropped_nan = 0
        dropped_low_winner = 0
        dropped_missing_dim = 0
        dropped_dup_loser = 0
        dropped_no_global = 0
        resolved_dup = 0
        dim_missing = {d: 0 for d in self.target_dims}

        # Columns that must be finite for a row to participate
        nan_check_cols = [d for d in self.target_dims if d in self.meta_df.columns]
        if self.winner_key in self.meta_df.columns:
            nan_check_cols.append(self.winner_key)
        nan_check_cols = list(dict.fromkeys(nan_check_cols))  # deduplicate

        for group_id, sub_df in grouped:
            if len(sub_df) < self.min_group_images:
                dropped_small += 1
                continue

            # Drop rows with NaN/Inf in any score column
            clean = sub_df.copy()
            for col in nan_check_cols:
                if col in clean.columns:
                    clean = clean[np.isfinite(clean[col].astype(float))]
            if len(clean) < self.min_group_images:
                dropped_nan += 1
                continue

            rows = clean.reset_index(drop=True)
            winner_idx = self._pick_winner_idx(rows)

            # Winner quality gate
            if self.min_winner_score > 0 and self.winner_key in rows.columns:
                winner_reward = float(rows[self.winner_key].iloc[winner_idx])
                if winner_reward < self.min_winner_score:
                    dropped_low_winner += 1
                    continue

            # Collect ranked candidate lists per dimension
            cand_lists = {}
            for d in self.target_dims:
                cands = self._select_dim_loser_candidates(rows, winner_idx, d)
                if not cands:
                    dim_missing[d] += 1
                else:
                    cand_lists[d] = cands

            if self.strict_all_dims and len(cand_lists) < len(self.target_dims):
                dropped_missing_dim += 1
                continue
            if not cand_lists:
                dropped_missing_dim += 1
                continue

            # Greedy distinct assignment: dims with fewer candidates go first
            picks = {}
            if self.require_distinct_losers:
                used_indices = set()
                # Sort dims by number of available candidates (ascending) for greedy
                dim_order = sorted(cand_lists.keys(), key=lambda d: len(cand_lists[d]))
                all_assigned = True
                for d in dim_order:
                    assigned = False
                    for c in cand_lists[d]:
                        if c["loser_idx"] not in used_indices:
                            picks[d] = c
                            used_indices.add(c["loser_idx"])
                            assigned = True
                            break
                    if not assigned:
                        all_assigned = False
                        break
                if not all_assigned:
                    dropped_dup_loser += 1
                    continue
                # Check if greedy resolved a collision (top-1 was not always picked)
                top1_indices = {cand_lists[d][0]["loser_idx"] for d in cand_lists}
                if len(top1_indices) < len(cand_lists):
                    resolved_dup += 1
            else:
                for d in cand_lists:
                    picks[d] = cand_lists[d][0]

            # Global worst: argmin(reward), excluding winner AND dim losers, with gap check
            global_worst_idx = None
            if self.inject_global_worst and "reward" in rows.columns:
                rewards = rows["reward"].astype(float).to_numpy()
                winner_reward = float(rewards[winner_idx])
                # Exclude winner + all already-selected dim losers
                excluded = {winner_idx}
                for p in picks.values():
                    excluded.add(int(p["loser_idx"]))
                candidates = np.array([i for i in range(len(rows)) if i not in excluded])
                if len(candidates) > 0:
                    cand_rewards = rewards[candidates]
                    # Tie-break argmin: among tied worst, prefer lowest sum of dim scores
                    min_val = np.nanmin(cand_rewards)
                    tied = candidates[np.abs(cand_rewards - min_val) < 1e-8]
                    if len(tied) == 1:
                        worst_idx = int(tied[0])
                    else:
                        tie_scores = np.zeros(len(tied), dtype=np.float64)
                        for d in self.target_dims:
                            if d in rows.columns:
                                dv = rows[d].astype(float).to_numpy()
                                tie_scores += np.nan_to_num(dv[tied], nan=0.0)
                        worst_idx = int(tied[np.argmin(tie_scores)])
                    gap = winner_reward - float(rewards[worst_idx])
                    if gap >= self.global_worst_min_gap:
                        global_worst_idx = worst_idx

                if global_worst_idx is None:
                    dropped_no_global += 1
                    # Continue anyway — global worst is auxiliary

            winner_row = rows.iloc[winner_idx]
            text_cache = winner_row.get("text_cache_path", None)
            text_cache_path = (
                self._resolve_cache_path(str(text_cache)) if pd.notna(text_cache)
                else self._resolve_cache_path(str(winner_row["cache_path"]))
            )

            rec = {
                "group_id": str(group_id),
                "caption": str(winner_row.get("caption", "")),
                "winner_cache_path": self._resolve_cache_path(str(winner_row["cache_path"])),
                "text_cache_path": text_cache_path,
                "bucket_h": int(winner_row["bucket_h"]),
                "bucket_w": int(winner_row["bucket_w"]),
                "aspect_ratio": float(winner_row.get("aspect_ratio", 1.0)),
                "winner_score": float(winner_row["reward"]) if "reward" in rows.columns else 0.0,
            }

            for d in self.target_dims:
                alias = self._DIM_ALIAS.get(d, d)
                if d in picks:
                    li = int(picks[d]["loser_idx"])
                    loser_row = rows.iloc[li]
                    rec[f"loser_{alias}_cache_path"] = self._resolve_cache_path(str(loser_row["cache_path"]))
                    rec[f"loser_{alias}_score"] = float(loser_row["reward"]) if "reward" in rows.columns else 0.0
                    rec[f"target_gap_{alias}"] = float(picks[d]["target_gap"])
                    rec[f"score_gap_{alias}"] = float(picks[d]["total_gap"])
                    rec[f"mean_other_gap_{alias}"] = float(picks[d]["mean_other_gap"])
                else:
                    # strict_all_dims=False fallback: point to winner so loader stays valid.
                    rec[f"loser_{alias}_cache_path"] = rec["winner_cache_path"]
                    rec[f"loser_{alias}_score"] = rec["winner_score"]
                    rec[f"target_gap_{alias}"] = 0.0
                    rec[f"score_gap_{alias}"] = 0.0
                    rec[f"mean_other_gap_{alias}"] = 0.0

            # Global worst loser record
            if self.inject_global_worst and global_worst_idx is not None:
                gw_row = rows.iloc[global_worst_idx]
                rec["loser_global_cache_path"] = self._resolve_cache_path(str(gw_row["cache_path"]))
                rec["loser_global_score"] = float(gw_row["reward"]) if "reward" in rows.columns else 0.0
                rec["score_gap_global"] = float(rec["winner_score"] - rec["loser_global_score"])
                rec["has_global_worst"] = True
            elif self.inject_global_worst:
                # No valid global worst found — point to winner (zero gradient)
                rec["loser_global_cache_path"] = rec["winner_cache_path"]
                rec["loser_global_score"] = rec["winner_score"]
                rec["score_gap_global"] = 0.0
                rec["has_global_worst"] = False

            records.append(rec)

        if not records:
            raise RuntimeError("[MD3PO] No valid triples selected. Relax thresholds.")

        df = pd.DataFrame(records).reset_index(drop=True)
        global_str = ""
        if self.inject_global_worst:
            n_with_global = int(df["has_global_worst"].sum()) if "has_global_worst" in df.columns else 0
            global_str = f", no_global={dropped_no_global}, has_global={n_with_global}/{len(df)}"
        print(
            f"[MD3PO] Built {len(df)} triples from {len(grouped)} groups | "
            f"dropped small={dropped_small}, nan={dropped_nan}, low_winner={dropped_low_winner}, "
            f"missing_dim={dropped_missing_dim}, dup_loser={dropped_dup_loser} "
            f"(resolved={resolved_dup}){global_str}"
        )
        for d in self.target_dims:
            print(f"   - missing {d}: {dim_missing[d]}")
        return df

    def _resolve_cache_path(self, p: str) -> str:
        p = self._remap_path(p)
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str(self.data_base_path / path)

    def _remap_path(self, p: str) -> str:
        for old, new in self.path_remapping.items():
            if p.startswith(old):
                return new + p[len(old):]
        return p

    def __len__(self):
        return len(self.triple_df)

    def __getitem__(self, index):
        row = self.triple_df.iloc[index % len(self.triple_df)]
        return self._load_split(row)

    def _load_split(self, meta_row):
        winner_path = Path(str(meta_row["winner_cache_path"]))
        text_path = Path(str(meta_row.get("text_cache_path", winner_path)))

        # Dynamic loser paths based on active dims
        loser_paths = {}
        for d in self.target_dims:
            alias = self._DIM_ALIAS.get(d, d)
            loser_paths[alias] = Path(str(meta_row[f"loser_{alias}_cache_path"]))

        # Global worst loser path
        if self.inject_global_worst and "loser_global_cache_path" in meta_row:
            loser_paths["global"] = Path(str(meta_row["loser_global_cache_path"]))

        try:
            with np.load(str(winner_path), allow_pickle=True) as npz_w:
                winner_latents = self._sample_from_vae_h(self._get_vae_h(npz_w))
            loser_latents = {}
            for alias, lp in loser_paths.items():
                with np.load(str(lp), allow_pickle=True) as npz_l:
                    loser_latents[alias] = self._sample_from_vae_h(self._get_vae_h(npz_l))
            with np.load(str(text_path), allow_pickle=True) as npz_text:
                text_embeds, text_mask = self._read_text(npz_text)
                text_ids = npz_text["text_ids"].astype(np.float32) if "text_ids" in npz_text else None
        except Exception as e:
            print(
                f"[MD3PO] Error loading NPZs "
                f"(w={winner_path}, losers={loser_paths}, t={text_path}): {e}"
            )
            return self._fallback(meta_row)

        return self._build_item(
            winner_latents,
            loser_latents,
            text_embeds,
            text_mask,
            meta_row,
            text_ids=text_ids,
        )

    @staticmethod
    def _get_vae_h(npz):
        if "vae_h" in npz:
            return npz["vae_h"].astype(np.float32)
        if "latents" in npz:
            warnings.warn("MD3PO NPZ has 'latents' but not 'vae_h'; using latents directly.")
            lat = npz["latents"].astype(np.float32)
            return np.concatenate([lat, np.full_like(lat, -100.0)], axis=0)
        raise KeyError("NPZ must contain 'vae_h' or 'latents'")

    def _sample_from_vae_h(self, vae_h: np.ndarray) -> np.ndarray:
        c2 = vae_h.shape[0]
        c = c2 // 2
        mean = vae_h[:c]
        if self.deterministic_latents:
            return mean.astype(np.float32)
        logvar = np.clip(vae_h[c:], -30.0, 20.0)
        std = np.exp(0.5 * logvar)
        return (mean + std * np.random.randn(*mean.shape)).astype(np.float32)

    @staticmethod
    def _read_text(npz):
        if "prompt_embeds" in npz:
            emb = npz["prompt_embeds"].astype(np.float16)
        elif "text_embeds" in npz:
            emb = npz["text_embeds"].astype(np.float16)
        else:
            raise KeyError("NPZ must contain 'prompt_embeds' or 'text_embeds'")

        if "attention_mask" in npz:
            mask = npz["attention_mask"].astype(np.int8)
        elif "text_mask" in npz:
            mask = npz["text_mask"].astype(np.int8)
        else:
            mask = np.ones((emb.shape[0],), dtype=np.int8)

        l = min(emb.shape[0], mask.shape[0])
        return emb[:l], mask[:l]

    def _build_item(
        self,
        winner_np,
        loser_dict,
        text_np,
        mask_np,
        meta_row,
        text_ids=None,
    ):
        item = {
            "winner_latents": torch.from_numpy(winner_np),
            "text_embeds": torch.from_numpy(text_np),
            "text_mask": torch.from_numpy(mask_np),
            "bucket_size": (int(meta_row["bucket_h"]), int(meta_row["bucket_w"])),
            "aspect_ratio": float(meta_row.get("aspect_ratio", 1.0)),
            "caption": str(meta_row.get("caption", "")),
        }
        # Dynamic loser latents
        active_aliases = [self._DIM_ALIAS.get(d, d) for d in self.target_dims]
        # Include global if available
        if self.inject_global_worst and "global" in loser_dict:
            active_aliases = active_aliases + ["global"]
        item["_dim_aliases"] = active_aliases
        for alias in active_aliases:
            item[f"loser_{alias}_latents"] = torch.from_numpy(loser_dict[alias])

        if text_ids is not None:
            item["text_ids"] = torch.from_numpy(text_ids)

        # Score / gap metadata
        for alias in active_aliases:
            for prefix in ("loser_{}_score", "target_gap_{}", "score_gap_{}"):
                key = prefix.format(alias)
                if key in meta_row:
                    item[key] = float(meta_row[key])
        if "winner_score" in meta_row:
            item["winner_score"] = float(meta_row["winner_score"])

        return item

    def _fallback(self, meta_row):
        bh, bw = int(meta_row["bucket_h"]), int(meta_row["bucket_w"])
        lat_h, lat_w = bh // 8, bw // 8
        c = 32
        l, d = 512, 12288
        active_aliases = [self._DIM_ALIAS.get(d_, d_) for d_ in self.target_dims]
        if self.inject_global_worst:
            active_aliases = active_aliases + ["global"]
        item = {
            "winner_latents": torch.zeros(c, lat_h, lat_w),
            "text_embeds": torch.zeros(l, d, dtype=torch.float16),
            "text_mask": torch.zeros(l, dtype=torch.int8),
            "bucket_size": (bh, bw),
            "aspect_ratio": 1.0,
            "caption": str(meta_row.get("caption", "")),
            "_dim_aliases": active_aliases,
        }
        for alias in active_aliases:
            item[f"loser_{alias}_latents"] = torch.zeros(c, lat_h, lat_w)
        return item

    @staticmethod
    def collate_fn(samples):
        d_expected = samples[0]["text_embeds"].shape[1]
        valid = [s for s in samples if s["text_embeds"].shape[1] == d_expected]
        if len(valid) == 0:
            valid = samples[:1]
        samples = valid

        # Discover active dim aliases from first sample
        dim_aliases = samples[0].get("_dim_aliases", ["component", "text", "arrow"])

        batch = {
            "winner_latents": torch.stack([s["winner_latents"] for s in samples]),
            "bucket_size": samples[0]["bucket_size"],
            "aspect_ratio": samples[0]["aspect_ratio"],
            "captions": [s["caption"] for s in samples],
            "_dim_aliases": dim_aliases,
        }
        for alias in dim_aliases:
            key = f"loser_{alias}_latents"
            batch[key] = torch.stack([s[key] for s in samples])

        max_len = max(s["text_embeds"].shape[0] for s in samples)
        d = samples[0]["text_embeds"].shape[1]
        text_embeds = torch.zeros(len(samples), max_len, d, dtype=samples[0]["text_embeds"].dtype)
        text_mask = torch.zeros(len(samples), max_len, dtype=torch.int8)
        for i, s in enumerate(samples):
            ln = s["text_embeds"].shape[0]
            text_embeds[i, :ln] = s["text_embeds"]
            text_mask[i, :ln] = s["text_mask"]
        batch["text_embeds"] = text_embeds
        batch["text_mask"] = text_mask

        if "text_ids" in samples[0]:
            text_ids = torch.zeros(len(samples), max_len, 4, dtype=samples[0]["text_ids"].dtype)
            for i, s in enumerate(samples):
                if "text_ids" in s:
                    ln = s["text_ids"].shape[0]
                    text_ids[i, :ln] = s["text_ids"]
            batch["text_ids"] = text_ids

        # Dynamic metric keys
        for alias in dim_aliases:
            for prefix in ("loser_{}_score", "target_gap_{}", "score_gap_{}"):
                key = prefix.format(alias)
                if key in samples[0]:
                    batch[key] = torch.tensor([s.get(key, 0.0) for s in samples])
        if "winner_score" in samples[0]:
            batch["winner_score"] = torch.tensor([s["winner_score"] for s in samples])

        return batch

    def _print_stats(self):
        n = len(self.triple_df)
        print(f"\n[MD3PO] Statistics ({n} triples, dims={len(self.target_dims)}):")
        for d in self.target_dims:
            suffix = self._DIM_ALIAS.get(d, d)
            tg = f"target_gap_{suffix}"
            sg = f"score_gap_{suffix}"
            if tg in self.triple_df.columns:
                print(f"  {tg}: {self.triple_df[tg].describe()}")
            if sg in self.triple_df.columns:
                desc = self.triple_df[sg].describe()
                neg_n = int((self.triple_df[sg] < 0).sum())
                print(f"  {sg}: {desc}")
                print(f"  {sg} < 0: {neg_n} ({100.0 * neg_n / max(1, n):.3f}%)")
        if self.inject_global_worst and "score_gap_global" in self.triple_df.columns:
            sg_g = self.triple_df["score_gap_global"]
            n_active = int(self.triple_df.get("has_global_worst", pd.Series([False] * n)).sum())
            print(f"  Global worst: active={n_active}/{n}, gap mean={sg_g.mean():.3f}, min={sg_g.min():.3f}")
