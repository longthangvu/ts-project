import torch
import torch.nn as nn

# ---- Recency bias module (per-head learnable α) ----
class RecencyBias(nn.Module):
    def __init__(self, n_heads: int, init_alpha: float = 1e-2, learnable: bool = True):
        super().__init__()
        alpha = torch.full((n_heads,), float(init_alpha))
        if learnable:
            self.alpha = nn.Parameter(alpha)  # [H]
        else:
            self.register_buffer("alpha", alpha)

    def forward(self, t_q: torch.LongTensor, t_k: torch.LongTensor):
        """
        t_q: [B, Nq]  t_k: [B, Nk]
        returns bias: [B, H, Nq, Nk] where bias <= 0
        """
        dqk = (t_q.unsqueeze(-1) - t_k.unsqueeze(-2)).clamp_min(0).to(torch.float32)  # [B,Nq,Nk]
        return -self.alpha.view(1, -1, 1, 1) * dqk.unsqueeze(1)  # [B,H,Nq,Nk]

# ---- MHA with additive per-head, per-batch bias ----
class MHAWithRecency(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias_mod: RecencyBias, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.bias_mod = bias_mod
        self.n_heads = n_heads

    def forward(self, X, t_idx, attn_bool_mask=None):
        """
        X: [B, N, d]   t_idx: [B, N] (int cut-off per token)
        attn_bool_mask: [B, N, N] bool, True = disallow
        """
        B, N, _ = X.shape
        # Build recency bias for self-attention
        bias = self.bias_mod(t_idx, t_idx)  # [B,H,N,N]
        # Base boolean mask -> additive float
        if attn_bool_mask is None:
            base = torch.zeros(B, 1, N, N, dtype=torch.bool, device=X.device)
        else:
            base = attn_bool_mask.unsqueeze(1)  # [B,1,N,N]
        add = bias.clone()
        m = base.expand_as(add)
        add[m] = float("-inf")  # block structurally
        # Flatten heads into batch for PyTorch MHA attn_mask
        attn_mask = add.reshape(B * self.n_heads, N, N)  # float, -inf where blocked
        out, _ = self.mha(X, X, X, attn_mask=attn_mask, need_weights=False)
        return out

# ---- PFN encoder block with recency ----
class PFNEncoderLayerRecency(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, bias_mod: RecencyBias):
        super().__init__()
        self.attn = MHAWithRecency(d_model, n_heads, bias_mod, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, t_idx, base_bool_mask):
        # Norm-first style
        x = x + self.attn(self.ln1(x), t_idx, attn_bool_mask=base_bool_mask)
        x = x + self.ff(self.ln2(x))
        return x