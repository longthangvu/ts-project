# fits_encoder_1d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Complex linear (F_in complex bins -> F_out complex bins) ----
class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # store real/imag separately as real-valued parameters
        self.Wr = nn.Parameter(torch.empty(out_features, in_features))
        self.Wi = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.br = nn.Parameter(torch.empty(out_features))
            self.bi = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming-uniform on real/imag; small scale is fine
        nn.init.kaiming_uniform_(self.Wr, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wi, a=math.sqrt(5))
        if self.br is not None:
            fan_in = self.Wr.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.br, -bound, bound)
            nn.init.uniform_(self.bi, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F_in) complex (torch.cfloat)
        xr = x.real   # (B, F_in)
        xi = x.imag
        # y = (Wr + j Wi) @ (xr + j xi)
        yr = F.linear(xr, self.Wr) - F.linear(xi, self.Wi)
        yi = F.linear(xr, self.Wi) + F.linear(xi, self.Wr)
        if self.br is not None:
            yr = yr + self.br
            yi = yi + self.bi
        return torch.complex(yr, yi)  # (B, F_out)
        

# ---- Core FITS block (RIN -> rFFT -> LPF -> complex interp -> pad -> irFFT -> iRIN) ----
class FITSBlock1D(nn.Module):
    def __init__(self, L: int, H: int, F_cut: int, scale_comp: bool = True):
        """
        L: look-back length
        H: horizon
        F_cut: number of low-frequency bins to keep after LPF (1..L//2+1)
        """
        super().__init__()
        assert 1 <= F_cut <= (L // 2 + 1)
        self.L, self.H, self.F_cut = L, H, F_cut
        eta = (L + H) / L  # frequency interpolation ratio
        F_out = max(1, int(math.floor(F_cut * eta)))
        self.cinterp = ComplexLinear(F_cut, F_out, bias=True)
        self.scale_comp = scale_comp  # energy compensation after irFFT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) real
        returns y_hat: (B, L+H) real (reconstructed + forecasted)
        """
        eps = 1e-5

        # RIN
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
        xn = (x - mu) / std

        # rFFT
        X = torch.fft.rfft(xn, dim=1)  # (B, L//2+1) complex

        # LPF keep first F_cut bins (including DC)
        X_lpf = X[:, : self.F_cut]  # (B, F_cut) complex

        # Complex-valued linear interpolation to ~eta * F_cut bins
        X_up = self.cinterp(X_lpf)  # (B, F_out) complex

        # Zero-pad spectrum to target length (L+H)//2 + 1
        spec_len = (self.L + self.H) // 2 + 1
        B = x.size(0)
        Z = torch.zeros(B, spec_len, dtype=torch.cfloat, device=x.device)
        keep = min(spec_len, X_up.size(1))
        Z[:, :keep] = X_up[:, :keep]

        # irFFT back to time domain with length L+H
        y_hat = torch.fft.irfft(Z, n=self.L + self.H, dim=1)  # (B, L+H) real

        # Energy compensation (optional but stabilizes magnitude after length change)
        if self.scale_comp:
            y_hat = y_hat * ((self.L + self.H) / self.L)

        # iRIN
        y_hat = y_hat * std + mu
        return y_hat  # (B, L+H)


# ---- Encoders to replace φ layers ----
class PhiQryFITS1D(nn.Module):
    def __init__(self, L: int, H: int, F_cut: int, d_model: int, freeze_fits: bool = True):
        super().__init__()
        self.fits = FITSBlock1D(L, H, F_cut)
        if freeze_fits:
            for p in self.fits.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(H, d_model)  # project forecast slice -> token

    def forward(self, x_qry: torch.Tensor) -> torch.Tensor:
        """
        x_qry: (B, L)
        returns: (B, d_model)
        """
        y_hat = self.fits(x_qry)            # (B, L+H)
        z_hat = y_hat[:, -self.fits.H :]    # forecast slice (B, H)
        return self.proj(z_hat)             # (B, d_model)


class PhiCtxFITS1D(nn.Module):
    def __init__(self, L: int, H: int, F_cut: int, d_model: int,
                 alpha: float = 0.0, freeze_fits: bool = True):
        """
        alpha: mix factor between FITS forecast and ground-truth z for context tokens.
               0.0 -> use ground-truth only (pure teacher forcing).
               (0,1] -> blend in model's forecast.
        """
        super().__init__()
        self.fits = FITSBlock1D(L, H, F_cut)
        if freeze_fits:
            for p in self.fits.parameters():
                p.requires_grad = False
        self.alpha = float(alpha)
        self.H = H
        self.proj = nn.Linear(H, d_model)

    def forward(self, x_ctx: torch.Tensor, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        x_ctx: (B, L)
        z_ctx: (B, H) ground-truth horizon for the context samples
        returns: (B, d_model)
        """
        if self.alpha == 0.0:
            feat = z_ctx
        else:
            y_hat = self.fits(x_ctx)           # (B, L+H)
            z_hat = y_hat[:, -self.H :]        # (B, H)
            feat = self.alpha * z_hat + (1.0 - self.alpha) * z_ctx
        return self.proj(feat)                 # (B, d_model)
