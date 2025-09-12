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

# ---- SimpleLinearPFN with time-indexed recency only ----
class LinearPFN(nn.Module):
    def __init__(self, L=20, H=10, d=256, L_blk=6, n_heads=8, d_ff=1024, dropout=0.1, recency_init=1e-2):
        super().__init__()
        self.L, self.H, self.d = L, H, d
        self.L_blk = L_blk

        # Encode [history, target] for contexts; [history, zeros(H)] for queries
        self.phi_ctx = nn.Linear(L + H, d)
        self.phi_qry = nn.Linear(L + H, d)

        self.recency = RecencyBias(n_heads, init_alpha=recency_init, learnable=True)
        self.blocks = nn.ModuleList([
            PFNEncoderLayerRecency(d, n_heads, d_ff, dropout, self.recency)
        for _ in range(L_blk)])

        self.output_head = nn.Linear(d, H)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    @staticmethod
    def _struct_mask(C: int, Q: int, device):
        """PFN structure. True = disallowed. [S,S], S=C+Q."""
        S = C + Q
        m = torch.zeros(S, S, dtype=torch.bool, device=device)
        # contexts cannot attend queries; queries cannot attend queries
        m[:C, C:] = True
        m[C:, C:] = True
        return m

    @staticmethod
    def _nofuture_mask(t_all: torch.LongTensor):
        """
        Block attending to strictly future tokens by time index.
        True = disallowed. Shape [B,S,S].
        """
        # future if key time > query time  => t_q < t_k
        return (t_all.unsqueeze(2) < t_all.unsqueeze(1))

    def forward(self, ctx_x, ctx_z, qry_x, t_ctx: torch.LongTensor, t_qry: torch.LongTensor):
        """
        ctx_x: [B, C, L]   ctx_z: [B, C, H]   qry_x: [B, Q, L]
        t_ctx: [B, C] long  (cut-off index per context patch)
        t_qry: [B, Q] long  (cut-off index per query patch)
        """
        B, C, _ = ctx_x.shape
        Q = qry_x.shape[1]
        device = ctx_x.device

        # Inputs
        ctx_in = torch.cat([ctx_x, ctx_z], dim=-1)                       # [B,C,L+H]
        qry_in = torch.cat([qry_x, torch.zeros(B, Q, self.H, device=device)], dim=-1)

        ctx_e = self.phi_ctx(ctx_in)                                     # [B,C,d]
        qry_e = self.phi_qry(qry_in)                                     # [B,Q,d]
        Z = torch.cat([ctx_e, qry_e], dim=1)                              # [B,S,d] S=C+Q

        # Indices in the same order
        t_all = torch.cat([t_ctx, t_qry], dim=1)                          # [B,S]

        # Base masks
        struct = self._struct_mask(C, Q, device).unsqueeze(0).expand(B, -1, -1)  # [B,S,S]
        nofuture = self._nofuture_mask(t_all)                                        # [B,S,S]
        base_mask = struct | nofuture  # True = disallow

        # Transformer with recency bias
        for blk in self.blocks:
            Z = blk(Z, t_all, base_mask)

        U = Z[:, C:, :]                                                  # [B,Q,d]
        out = self.output_head(U)                                        # [B,Q,H]
        return out
