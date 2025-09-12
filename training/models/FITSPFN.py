import torch
import torch.nn as nn
from .layers.fits_encoder_1d import PhiCtxFITS1D, PhiQryFITS1D

class FitsPFN(nn.Module):
    def __init__(self, L=20, H=10, d=256, L_blk=6, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.L, self.H, self.d = L, H, d
        self.L_blk = L_blk
        F_cut = max(4, (L // 2 + 1) // 3)

        # Encode [history, target] for contexts; [history, zeros(H)] for queries
        self.phi_ctx = PhiCtxFITS1D(L=self.L, H=self.H, F_cut=F_cut, d_model=self.d,
                            alpha=0.0, freeze_fits=True)
        self.phi_qry = PhiQryFITS1D(L=self.L, H=self.H, F_cut=F_cut, d_model=self.d,
                            freeze_fits=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=L_blk)
        # self.output_head = nn.Linear(d, 2 * H)
        self.output_head = nn.Linear(d, H)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_attn_mask(self, C: int, Q: int, device):
        """Boolean mask: True = disallowed. Shape [S, S] with S=C+Q."""
        S = C + Q
        mask = torch.zeros(S, S, dtype=torch.bool, device=device)
        # rows 0..C-1 (contexts): forbid attending to queries (cols C..S-1)
        mask[:C, C:] = True
        # rows C..S-1 (queries): forbid attending to queries (cols C..S-1)
        mask[C:, C:] = True   # blocks query->query (incl. self)
        return mask

    def forward(self, ctx_x, ctx_z, qry_x):
        """
        ctx_x: [B, C, L]   (normalized histories)
        ctx_z: [B, C, H]   (normalized targets)
        qry_x: [B, Q, L]   (normalized histories)
        """
        B, C, L = ctx_x.shape
        Q = qry_x.shape[1]
        device = ctx_x.device

        # Flatten patches to match φ’s (B*, L) / (B*, H) interface
        ctx_x_flat = ctx_x.reshape(B * C, L)          # [B*C, L]
        ctx_z_flat = ctx_z.reshape(B * C, self.H)     # [B*C, H]
        qry_x_flat = qry_x.reshape(B * Q, L)          # [B*Q, L]

        # φ_ctx expects (x_ctx, z_ctx); φ_qry expects (x_qry)
        ctx_e = self.phi_ctx(ctx_x_flat, ctx_z_flat)  # [B*C, d]
        qry_e = self.phi_qry(qry_x_flat)              # [B*Q, d]

        # Restore [B, C/Q, d]
        ctx_e = ctx_e.view(B, C, -1)                  # [B, C, d]
        qry_e = qry_e.view(B, Q, -1)                  # [B, Q, d]

        Z0 = torch.cat([ctx_e, qry_e], dim=1)                             # [B, C+Q, d]

        # Attention mask to enforce PFN semantics
        attn_mask = self._build_attn_mask(C, Q, device)                  # [C+Q, C+Q]

        Zf = self.transformer(Z0, mask=attn_mask)                        # [B, C+Q, d]
        U = Zf[:, C:, :]                                                 # [B, Q, d]

        out = self.output_head(U)                                        # [B, Q, 2H]
        # mu, log_sigma2 = out[..., :self.H], out[..., self.H:]
        return out