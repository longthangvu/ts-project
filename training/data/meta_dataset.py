import math, os, glob, random
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch

class _ShardSeriesDataset:
    """
    Lightweight series provider over *.npy shards produced by the user's script.
    Expected file shape per shard: (k, T) with dtype float32/float64.
    get_a_context() mirrors the old base_dataset API returning (t, v, meta)
      - v: torch tensor shaped [1, T, 1] to satisfy downstream indexing v[0, :, 0]
    """
    def __init__(self, shards_dir: str, mmap: bool = True):
        self.shards_dir = Path(shards_dir)
        if not self.shards_dir.exists():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_dir}")

        self._paths: List[Path] = sorted(self.shards_dir.glob("series_shard_*.npy"))
        if not self._paths:
            raise FileNotFoundError(f"No shard files matched 'series_shard_*.npy' in {self.shards_dir}")

        # Index shard sizes without loading into RAM
        self._sizes: List[int] = []
        self._lengths: List[int] = []
        self._mmap = mmap
        for p in self._paths:
            arr = np.load(p, mmap_mode="r" if mmap else None)
            if arr.ndim != 2:
                raise ValueError(f"Shard {p} must have shape (k, T); got {arr.shape}")
            k, T = arr.shape
            self._sizes.append(k)
            self._lengths.append(T)
        # All shards must share same T for our downstream patching
        if len(set(self._lengths)) != 1:
            raise ValueError(f"All shards must have identical series length T. Found: {set(self._lengths)}")
        self.T = self._lengths[0]
        # Sampling weights proportional to number of series per shard
        total = sum(self._sizes)
        self._weights = [s / total for s in self._sizes]

    def _load_shard(self, idx: int):
        return np.load(self._paths[idx], mmap_mode="r" if self._mmap else None)

    def get_a_context(self) -> Tuple[Optional[torch.Tensor], torch.Tensor, dict]:
        # Pick shard then a row
        shard_idx = random.choices(range(len(self._paths)), weights=self._weights, k=1)[0]
        shard = self._load_shard(shard_idx)
        row = random.randrange(self._sizes[shard_idx])
        y_np = shard[row]  # shape (T,)
        # Safety: ensure finite
        # if not np.isfinite(y_np).all():
        #     # Rare; resample quickly within shard
        #     tries = 0
        #     while tries < 8:
        #         row = random.randrange(self._sizes[shard_idx])
        #         y_np = shard[row]
        #         if np.isfinite(y_np).all():
        #             break
        #         tries += 1
        # Torch tensor shaped [1, T, 1] to match previous code expectations
        # v = torch.from_numpy(np.asarray(y_np, dtype=np.float32)).view(1, -1, 1)
        v = torch.tensor(y_np, dtype=torch.float32).reshape(1, -1, 1)
        meta = {"source_shard": str(self._paths[shard_idx]), "row": row, "T": self.T}
        return v, meta


class VariableMetaDataset:
    """
    Variable-C/Q meta-dataset that draws raw series from pre-generated shards instead of base_dataset synthesis.
      - create_meta_task()
      - sample_context_query_split()
      - _sample_C(), _sample_Q()
    """
    def __init__(
        self,
        shards_dir: str,
        L=50,
        H=20,
        C_range=(4, 256),
        Q_range=(1, 16),
        device="cpu",
        sample_log_uniform=True,
    ):
        self.device = device
        self.L = L
        self.H = H

        # Initialize variable C/Q bookkeeping
        max_C = C_range[1] if isinstance(C_range, tuple) else C_range
        max_Q = Q_range[1] if isinstance(Q_range, tuple) else Q_range
        self.C = max_C
        self.Q = max_Q

        # Preserve flags
        self.C_range = C_range
        self.Q_range = Q_range
        self.sample_log_uniform = sample_log_uniform

        # Replace synthetic base_dataset with shard-backed provider
        self.base_dataset = _ShardSeriesDataset(shards_dir)

        print("Variable SimpleMetaTaskDataset (shards-backed):")
        print(f"  L={L}, H={H}")
        print(f"  C range: {C_range} ({'log-uniform' if sample_log_uniform else 'uniform'})")
        print(f"  Q range: {Q_range}")
        print(f"  Max patches needed: C={max_C}, Q={max_Q}")
        print(f"  Shards dir: {Path(shards_dir).resolve()} | T={self.base_dataset.T}")

    def _sample_C(self):
        if isinstance(self.C_range, tuple):
            min_C, max_C = self.C_range
            if self.sample_log_uniform:
                log_min, log_max = math.log(min_C), math.log(max_C)
                log_C = np.random.uniform(log_min, log_max)
                return min(max_C, max(min_C, int(math.exp(log_C))))
            else:
                return np.random.randint(min_C, max_C + 1)
        else:
            return self.C_range

    def _sample_Q(self):
        if isinstance(self.Q_range, tuple):
            min_Q, max_Q = self.Q_range
            return np.random.randint(min_Q, max_Q + 1)
        else:
            return self.Q_range

    def sample_context_query_split(self, endpoints: np.ndarray, series_len: int, C_actual=None, Q_actual=None):
        split = int(0.8 * series_len)
        C_use = C_actual if C_actual is not None else self.C
        Q_use = Q_actual if Q_actual is not None else self.Q

        ctx_cand = np.nonzero((endpoints + self.H) <= split)[0]
        qry_cand = np.nonzero((endpoints - self.L + 1) >= split)[0]

        rep_c = ctx_cand.size < C_use
        rep_q = qry_cand.size < Q_use

        if ctx_cand.size == 0:
            ctx_cand = np.arange(len(endpoints))
            rep_c = True
        if qry_cand.size == 0:
            qry_cand = np.arange(len(endpoints))
            rep_q = True

        ctx_idx = np.random.choice(ctx_cand, size=C_use, replace=rep_c)
        qry_idx = np.random.choice(qry_cand, size=Q_use, replace=rep_q)

        if rep_c or rep_q:
            ctx_set = set(ctx_idx.tolist())
            q_fixed = []
            for q in qry_idx:
                if q in ctx_set:
                    pool = [x for x in qry_cand if x not in ctx_set]
                    q = np.random.choice(pool) if len(pool) else q
                q_fixed.append(q)
            qry_idx = np.array(q_fixed, dtype=int)

        return ctx_idx, qry_idx

    def create_patches(self, series: torch.Tensor):
        T = len(series)
        endpoints = np.arange(self.L - 1, T - self.H, dtype=int)
        patch_X = torch.stack([series[t - self.L + 1 : t + 1] for t in endpoints], axis=0)
        patch_Y = torch.stack([series[t + 1 : t + 1 + self.H] for t in endpoints], axis=0)
        return patch_X, patch_Y, endpoints

    def compute_scaler(self, patches_x: torch.Tensor, ctx_idx=None):
        N = patches_x.shape[0]
        if ctx_idx is None:
            split = int(0.8 * N)
            x_ctx = patches_x[:split]
        else:
            x_ctx = patches_x[ctx_idx]
        mu = x_ctx.mean()
        med = x_ctx.median()
        mad = (x_ctx - med).abs().median()
        sigma = (1.4826 * mad).clamp_min(0.10)
        return mu, sigma

    def create_meta_task(self):
        C_actual = self._sample_C()
        Q_actual = self._sample_Q()

        v, meta = self.base_dataset.get_a_context()
        y_series = v[0, :, 0].to(self.device)

        patches_x, patches_z, endpoints = self.create_patches(y_series)
        endpoints_np = np.asarray(endpoints, dtype=int)

        mu, sigma = self.compute_scaler(patches_x)
        x_norm = torch.clamp((patches_x - mu) / sigma, -10.0, 10.0)
        z_norm = torch.clamp((patches_z - mu) / sigma, -10.0, 10.0)

        ctx_idx, qry_idx = self.sample_context_query_split(endpoints_np, len(y_series), C_actual, Q_actual)

        meta_task = {
            "ctx_x": x_norm[ctx_idx],
            "ctx_z": z_norm[ctx_idx],
            "qry_x": x_norm[qry_idx],
            "qry_z": z_norm[qry_idx],
            "stats": {"mu": mu, "sigma": sigma},
            "raw_series": y_series,
            "endpoints": {
                "ctx": [endpoints[i] for i in ctx_idx],
                "qry": [endpoints[i] for i in qry_idx],
            },
            "source": meta,
        }
        return meta_task
