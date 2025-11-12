import torch
import torch.nn as nn

from layers.pfn_encoder import RecencyBias, PFNEncoderLayerRecency
from layers.pfn_backbone import PFNBackbone
from layers.input_encoder import InputEncoder
from layers.mask_builder import MaskBuilder

# ---- SimpleLinearPFN with time-indexed recency only ----
class LinearPFN_old(nn.Module):
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

class LinearPFN(nn.Module):
    def __init__(self, L=20, H=10, d=256, L_blk=6, n_heads=8, d_ff=1024, dropout=0.1, recency_init=1e-2):
        super().__init__()
        self.L, self.H, self.d = L, H, d
        self.enc = InputEncoder(L, H, d)
        self.recency = RecencyBias(n_heads, init_alpha=recency_init, learnable=True)
        self.backbone = PFNBackbone(d, n_heads, d_ff, dropout, self.recency, L_blk)
        self.masker = MaskBuilder()
        self.head = nn.Linear(d, H)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, ctx_x, ctx_z, qry_x, t_ctx, t_qry):
        ctx_e, qry_e = self.enc(ctx_x, ctx_z, qry_x)              # [B,C,d],[B,Q,d]
        Z = torch.cat([ctx_e, qry_e], dim=1)                      # [B,S,d]
        t_all, base_mask = self.masker(t_ctx, t_qry)              # [B,S],[B,S,S]
        Z = self.backbone(Z, t_all, base_mask)                    # [B,S,d]
        C = ctx_x.shape[1]
        U = Z[:, C:, :]                                           # [B,Q,d]
        return self.head(U)                                       # [B,Q,H]