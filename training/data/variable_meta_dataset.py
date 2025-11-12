import math
import torch
import numpy as np
from .basic_meta_dataset import SimpleMetaDataset

class VariableMetaDataset(SimpleMetaDataset):
    """
    Extension of SimpleMetaDataset with variable C,Q.
    Preserves all original functionality while adding support for variable context/query sizes.
    """
    
    def __init__(
        self,
        shape_config,
        hyperprior_params,
        L=50,           # History length
        H=20,           # Forecast horizon  
        C_range=(4, 256),  # Variable context range
        Q_range=(1, 16),   # Variable query range
        device="cpu",
        sample_log_uniform=True
    ):
        # Initialize with maximum C,Q for internal consistency
        max_C = C_range[1] if isinstance(C_range, tuple) else C_range
        max_Q = Q_range[1] if isinstance(Q_range, tuple) else Q_range
        
        # Call parent with max sizes
        super().__init__(
            shape_config=shape_config,
            hyperprior_params=hyperprior_params,
            L=L, H=H, C=max_C, Q=max_Q,
            device=device
        )
        
        # Store variable sampling configuration
        self.C_range = C_range
        self.Q_range = Q_range
        self.sample_log_uniform = sample_log_uniform
        
        print(f"Variable SimpleMetaTaskDataset:")
        print(f"  L={L}, H={H}")
        print(f"  C range: {C_range} ({'log-uniform' if sample_log_uniform else 'uniform'})")
        print(f"  Q range: {Q_range}")
        print(f"  Max patches needed: C={max_C}, Q={max_Q}")
    
    def _sample_C(self):
        """Sample context size C using log-uniform or uniform distribution"""
        if isinstance(self.C_range, tuple):
            min_C, max_C = self.C_range
            if self.sample_log_uniform:
                # Log-uniform sampling (TabPFN style)
                log_min, log_max = math.log(min_C), math.log(max_C)
                log_C = np.random.uniform(log_min, log_max)
                return min(max_C, max(min_C, int(math.exp(log_C))))
            else:
                return np.random.randint(min_C, max_C + 1)
        else:
            return self.C_range
    
    def _sample_Q(self):
        """Sample query size Q using uniform distribution"""
        if isinstance(self.Q_range, tuple):
            min_Q, max_Q = self.Q_range
            return np.random.randint(min_Q, max_Q + 1)
        else:
            return self.Q_range
    
    def sample_context_query_split(self, endpoints: np.ndarray, series_len: int, C_actual=None, Q_actual=None):
        """
        Override parent method to support variable C,Q sizes.
        Sample disjoint index sets with actual sizes.
        """
        split = int(0.8 * series_len)
        C_use = C_actual if C_actual is not None else self.C
        Q_use = Q_actual if Q_actual is not None else self.Q

        ctx_cand = np.nonzero((endpoints + self.H) <= split)[0]
        qry_cand = np.nonzero((endpoints - self.L + 1) >= split)[0]

        rep_c = ctx_cand.size < C_use
        rep_q = qry_cand.size < Q_use

        # If *no* candidates exist, relax minimally by borrowing across split.
        if ctx_cand.size == 0:
            ctx_cand = np.arange(len(endpoints))
            rep_c = True
        if qry_cand.size == 0:
            qry_cand = np.arange(len(endpoints))
            rep_q = True

        ctx_idx = np.random.choice(ctx_cand, size=C_use, replace=rep_c)
        qry_idx = np.random.choice(qry_cand, size=Q_use, replace=rep_q)

        # Ensure disjoint sets; if collision happens under replacement, fix greedily.
        if rep_c or rep_q:
            ctx_set = set(ctx_idx.tolist())
            q_fixed = []
            for q in qry_idx:
                if q in ctx_set:
                    # pick a new one from qry_cand not in ctx_set if possible
                    pool = [x for x in qry_cand if x not in ctx_set]
                    q = np.random.choice(pool) if len(pool) else q
                q_fixed.append(q)
            qry_idx = np.array(q_fixed, dtype=int)

        return ctx_idx, qry_idx
    
    def create_meta_task(self):
        """
        Create meta-task with variable C,Q sizes.
        Preserves all original functionality from your implementation.
        """
        # Sample actual sizes for this task
        C_actual = self._sample_C()
        Q_actual = self._sample_Q()
        
        t, v, _ = self.base_dataset.get_a_context()
        y_series = v[0, :, 0]  # Extract univariate series
        
        patches_x, patches_z, endpoints = self.create_patches(y_series)
        endpoints_np = np.asarray(endpoints, dtype=int)
        
        # x_norm, z_norm, mu, sigma = self.normalize_patches(patches_x, patches_z, y_series)
        
        # Sample context/query split with actual sizes
        ctx_idx, qry_idx = self.sample_context_query_split(endpoints_np, len(y_series), C_actual, Q_actual)
        # ctx_indices, qry_indices = self.sample_context_query_split(
        #     len(patches_x), C_actual, Q_actual
        # )
        mu, sigma = self.compute_scaler(patches_x)
        
        x_norm = torch.clamp((patches_x - mu) / sigma, -10.0, 10.0)
        z_norm = torch.clamp((patches_z - mu) / sigma, -10.0, 10.0)
        
        # Create meta-task (your original structure)
        meta_task = {
            'ctx_x': x_norm[ctx_idx],  # Context inputs [C_actual, L]
            'ctx_z': z_norm[ctx_idx],  # Context targets [C_actual, H]
            'qry_x': x_norm[qry_idx],  # Query inputs [Q_actual, L]
            'qry_z': z_norm[qry_idx],  # Query targets [Q_actual, H]
            'stats': {'mu': mu, 'sigma': sigma},
            'raw_series': y_series,
            'endpoints': {'ctx': [endpoints[i] for i in ctx_idx],
                         'qry': [endpoints[i] for i in qry_idx]}
        }
        
        return meta_task