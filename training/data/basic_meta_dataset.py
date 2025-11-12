import torch
import numpy as np
from data.priors.LaTPFN_dataset import LaTPFNDataset

class SimpleMetaDataset:
    """
    Simple meta-task dataset.
    
    For a synthetic univariate series y_{1:T+H}, creates patches:
    - x^(t) = (y_{t-L+1}, ..., y_t) ∈ R^L (history)
    - z^(t) = (y_{t+1}, ..., y_{t+H}) ∈ R^H (forecast)
    
    Splits into context/query sets and normalizes using context series statistics.
    """
    def __init__(
        self,
        shape_config,
        hyperprior_params,
        L=20,           # Patch history length
        H=10,           # Forecast horizon
        C=8,            # Number of context patches
        Q=4,            # Number of query patches
        device="cpu"
    ):
        self.shape_config = shape_config
        self.hyperprior_params = hyperprior_params
        self.L = L  # History length
        self.H = H  # Forecast horizon
        self.C = C  # Context patches
        self.Q = Q  # Query patches
        self.device = device
        
        # Create base dataset for generating synthetic series
        self.base_dataset = LaTPFNDataset(
            shape=shape_config,
            hyperprior_params=hyperprior_params,
            batch_size=1,
            length=shape_config.n_sequence,
            is_train=False,
            device=device,
            return_components=False,
            scale_noise=True,
            separate_noise=False  # Single values only
        )
    
    def create_patches(self, series):
        """
        From a 1D series of length T, build all possible patches.

        Returns:
          patch_X : [N, L]  histories
          patch_Y : [N, H]  targets
          endpoints : [N]   index t = last time of history (so history covers [t-L+1..t])
        """
        T = len(series)
        endpoints = np.arange(self.L - 1, T - self.H, dtype=int)
        patch_X = torch.stack([series[t - self.L + 1 : t + 1] for t in endpoints], axis=0)
        patch_Y = torch.stack([series[t + 1 : t + 1 + self.H] for t in endpoints], axis=0)
        return patch_X, patch_Y, endpoints
    
    def sample_context_query_split(
        self,
        endpoints: np.ndarray,  # shape [N], history ends at t; targets are [t+1 .. t+H]
        series_length: int,     # len(y)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        context_patch_indices: np.ndarray of shape [self.C]
        query_patch_indices:   np.ndarray of shape [self.Q]
        Policy:
        - Context patches are fully in the first 80% by time (t + H <= split).
        - Query patches are in the last 20%. history start >= split (t - L + 1 >= split).
        """
        split_index = int(0.8 * series_length)

        # candidates by time
        ctx_candidate_indices = np.nonzero((endpoints + self.H) <= split_index)[0]
        qry_candidata_indices = np.nonzero((endpoints - self.L + 1) >= split_index)[0]

        # sample (with replace fallback if tail/head too short)
        replace_ctx = ctx_candidate_indices.size < self.C
        replace_qry = qry_candidata_indices.size < self.Q

        ctx_indices = np.random.choice(
            ctx_candidate_indices, size=self.C, replace=replace_ctx
        )
        qry_indices = np.random.choice(
            qry_candidata_indices, size=self.Q, replace=replace_qry
        )
        return ctx_indices, qry_indices

    # def compute_scaler(
    #     self,
    #     patch_X: np.ndarray,
    #     ctx_indices: np.ndarray,
    # ) -> tuple[float, float]:
    #     """Fit (mu, sigma) on context histories only."""
    #     mu = float(patch_X[ctx_indices].mean())
    #     sigma = float(patch_X[ctx_indices].std() + 1e-6)
    #     return mu, sigma
    
    def compute_scaler(self, patches_x, ctx_idx=None):
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
        """
        Create a meta-task following the specification:
        D_ctx = {(x_norm_i, z_norm_i)} for i in I_ctx
        D_qry = {x_norm_star_j} for j in I_qry
        """
        # Generate synthetic univariate series
        t, v, _ = self.base_dataset.get_a_context()
        series = v[0, :, 0]  # Extract univariate series

        T = len(series)
        # Create patches
        patches_x, patches_z, endpoints = self.create_patches(series)

        ctx_idx, qry_idx = self.sample_context_query_split(endpoints=endpoints, series_length=T)
        mu, sigma = self.compute_scaler(patches_x, ctx_idx)
        x_norm = torch.clamp((patches_x - mu) / sigma, -10.0, 10.0) 
        z_norm = torch.clamp((patches_z - mu) / sigma, -10.0, 10.0)
        
        # Create meta-task
        meta_task = {
            'ctx_x': x_norm[ctx_idx],  # Context inputs [C, L]
            'ctx_z': z_norm[ctx_idx],  # Context targets [C, H]
            'qry_x': x_norm[qry_idx],  # Query inputs [Q, L]
            'qry_z': z_norm[qry_idx],  # Query targets [Q, H]
            'stats': {'mu': mu, 'sigma': sigma},
            'raw_series': series,
            'endpoints': {'ctx': [endpoints[i] for i in ctx_idx],
                         'qry': [endpoints[i] for i in qry_idx]}
        }
        
        return meta_task