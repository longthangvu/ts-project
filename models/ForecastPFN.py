import torch
import torch.nn as nn
import math
from einops import rearrange
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
NUM_TASKS = 10
YEAR = 0
MONTH = 1
DAY = 2
DOW = 3

class RobustScaler(nn.Module):
    """
    RobustScaler normalizes input time series while ignoring outliers and missing values.
    It masks out zeros, clips extreme values above mean + 2*std, rescales using
    mean + std of clipped data, and clips final output to [0, 3].
    This improves robustness in the presence of noise or missing data.
    """
    def forward(self, x, epsilon):
        # x: [B, T, 1]
        B, T, _ = x.shape
        x = x.squeeze(-1)  # → [B, T]
        scale = torch.zeros((B, 1, 1), device=x.device)
        scaled = torch.zeros((B, T, 1), device=x.device)

        for b in range(B):
            series = x[b]  # shape: [T]

            # First mask and stats
            non_zero = series[series != 0]
            if non_zero.numel() == 0:
                mean = std = torch.tensor(0.0, device=x.device)
            else:
                mean = non_zero.mean()
                std = non_zero.std(unbiased=False)

            upper = mean + 2 * std
            clipped = torch.clamp(series, min=0.0, max=upper)

            # Second pass stats
            non_zero_clipped = clipped[clipped != 0]
            if non_zero_clipped.numel() == 0:
                mean_clip = std_clip = torch.tensor(0.0, device=x.device)
            else:
                mean_clip = non_zero_clipped.mean()
                std_clip = non_zero_clipped.std(unbiased=False)

            s = mean_clip + std_clip + epsilon
            scale[b, 0, 0] = s
            scaled[b, :, 0] = torch.clamp(series / s, 0.0, 3.0)

        return scale, scaled


class MaxScaler(nn.Module):
    """
    MaxScaler normalizes input time series by dividing by the global maximum value.
    It assumes all values are valid (no masking), and provides a fast,
    simple normalization suitable for dense, bounded inputs.
    """
    def forward(self, x, epsilon):
        # x: [B, T, 1]
        scale = x.max(dim=1, keepdim=True).values + epsilon
        scaled = x / scale
        return scale, scaled

class PositionExpansion(nn.Module):
    def __init__(self, periods: int, freqs: int):
        super().__init__()
        self.periods = periods
        self.channels = freqs * 2

        i = torch.arange(periods + 1).unsqueeze(1)  # shape: [periods+1, 1]
        j = torch.arange(freqs).unsqueeze(0)        # shape: [1, freqs]
        angles = math.pi / periods * (2 ** j) * (i - 1)  # i-1 matches TF

        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [P+1, 2F]
        self.register_buffer("embedding", pe)

    def forward(self, tc):
        return self.embedding[tc]  # expects tc ∈ [0, periods]


class CustomScaling(nn.Module):
    """
    Used to normalize the historical input series before encoding.
    It ensures that time series values are on a comparable scale across samples
    """
    def __init__(self, method='robust'):
        super().__init__()
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'max':
            self.scaler = MaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def forward(self, history_channels, epsilon):
        return self.scaler(history_channels, epsilon)

class CustomSelfAttention(nn.Module):
    def __init__(self, embed_dim=72, num_heads=4, value_dim=72):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, num_heads * value_dim)
        self.k_proj = nn.Linear(embed_dim, num_heads * value_dim)
        self.v_proj = nn.Linear(embed_dim, num_heads * value_dim)
        self.out_proj = nn.Linear(num_heads * value_dim, embed_dim)
        self.num_heads = num_heads
        self.value_dim = value_dim

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H, V = self.num_heads, self.value_dim

        q = self.q_proj(x).view(B, T, H, V).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, V).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, V).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(V)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None, :, :], float(-1e4))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).reshape(B, T, H * V)

        return self.out_proj(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=72, heads=4, value_dim=72):
        super().__init__()
        self.attn = CustomSelfAttention(d_model, heads, value_dim)
        self.ff1 = nn.Linear(d_model, 4 * heads * value_dim)  # 4×288=1152
        self.ff2 = nn.Linear(4 * heads * value_dim, heads * value_dim)  # → 288
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask)
        x = self.activation(self.ff1(x))
        x = self.activation(self.ff2(x))
        return x

class PAttnBlock(nn.Module):
    """
    Patch‐based attention from https://arxiv.org/abs/2406.16964
    Inputs:
      - x: [B, seq_len, d_model]
    Outputs:
      - [B, pred_len, d_model]
    """
    def __init__(
        self,
        d_model:int,
        seq_len:int=100+1,
        pred_len:int=1,
        n_heads:int=4,
        factor:float=5,
        dropout:float=0.1,
        activation:str='gelu',
        patch_size:int=16,
        stride:int=8
    ):
        super().__init__()
        self.d_ff       = 4 * n_heads * d_model
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.d_model    = d_model
        self.patch_size = patch_size
        self.stride     = stride
        # number of overlapping patches you’ll get
        self.patch_num  = (seq_len - patch_size) // stride + 2

        self.pad   = nn.ReplicationPad1d((0, stride))
        self.inp   = nn.Linear(patch_size, d_model)
        self.enc   = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads
                ),
                d_model, self.d_ff, dropout=dropout, activation=activation
            )],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.outp  = nn.Linear(d_model * self.patch_num, pred_len)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        B, T, D = x.shape

        # 1) normalize in‐place
        m = x.mean(1, keepdim=True).detach()
        x = x - m
        s = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        x = x / s

        # 2) patch‐unfold
        x = x.permute(0, 2, 1)                     # → [B, D, T]
        x = self.pad(x)                            # → [B, D, T+stride]
        x = x.unfold(-1, self.patch_size, self.stride)   # → [B, D, patch_num, patch_size]
        x = rearrange(x, 'b d m p -> b m d p')     # → [B, patch_num, D, patch_size]
        x = self.inp(x)                            # → [B, patch_num, D, d_model]
        x = rearrange(x, 'b m d d_model -> (b d) m d_model')

        # 3) attend + ff
        x, _ = self.enc(x)                         # → [(B·D), patch_num, d_model]
        x = rearrange(x, '(b d) m d_model -> b d (m d_model)', b=B, d=D)

        # 4) project back to pred_len
        x = self.outp(x).permute(0, 2, 1)                           # → [B, pred_len, D]
        return x

class ForecastPFN_PAttn(nn.Module):
    def __init__(self,
                 epsilon:float=1e-4,
                 scaler:str='robust',
                 patch_size:int=16,
                 stride:int=8):
        """
        config: dict with keys
          - seq_len, pred_len, n_heads, d_ff, factor, dropout, activation, d_model
        """
        super().__init__()
        self.epsilon   = epsilon
        # --- time‐feature embedders (unchanged) ---
        self.pos_year  = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day   = PositionExpansion(31, 6)
        self.pos_dow   = PositionExpansion(7,  4)
        self.scaler    = CustomScaling(scaler)

        # --- history embedding size = 2×embed_size ---
        embed_size = sum(e.channels for e in (
            self.pos_year, self.pos_month, self.pos_day, self.pos_dow
        ))
        self.d_model = embed_size * 2

        self.no_pos = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.w_pos  = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.task_m = nn.Embedding(NUM_TASKS, embed_size)

        # --- two PAttn blocks, first “narrow,” then “wide” ---
        self.p0 = PAttnBlock(d_model=self.d_model,patch_size=patch_size, stride=stride)

        # final scalar head
        self.head  = nn.Sequential(nn.Linear(self.d_model, 1), nn.ReLU())

    def forward(self, X):
        # 1) unpack & pos‐embed history
        ts  = X["ts"]               # [B, T, 5]
        yr  = ts[:,:,0]
        dy  = (yr[:,-1:] - yr).clamp(0, self.pos_year.periods)
        pos = torch.cat([
            self.pos_year(dy),
            self.pos_month(ts[:,:,1]),
            self.pos_day(ts[:,:,2]),
            self.pos_dow(ts[:,:,3]),
        ], dim=-1)                  # → [B, T, embed_size]

        # 2) scale & embed values
        h = X["history"].unsqueeze(-1)           # [B, T,1]
        scale, h = self.scaler(h, self.epsilon)   # [B,1,1], [B,T,1]
        e0 = self.no_pos(h)                       # [B, T, embed_size]
        e1 = self.w_pos(h) + pos                  # [B, T, embed_size]
        hist = torch.cat([e0, e1], dim=-1)        # [B, T, d_model]

        # 3) embed query
        tt = X["target_ts"].view(-1,1,5)          # ensure [B,1,5]
        yrq= (yr[:,-1:] - tt[:,:,0]).clamp(0, self.pos_year.periods)
        qp = torch.cat([
            self.pos_year(yrq),
            self.pos_month(tt[:,:,1]),
            self.pos_day(tt[:,:,2]),
            self.pos_dow(tt[:,:,3])
        ], dim=-1).squeeze(1)                     # [B, embed_size]

        t  = self.task_m(X["task"])               # [B, embed_size]
        tgt= torch.cat([t, t + qp], dim=-1)       # [B, d_model]

        # 4) one‐stage PAttn
        hist = torch.cat([hist, tgt.unsqueeze(1)], dim=1)  # [B, T+1, d_model]
        out0 = self.p0(hist)                # [B, pred_len, d_model]

        # 5) head + de‐scale
        y   = self.head(out0).squeeze(-1) * scale[:,:,0]
        y   = y.squeeze(1)
        return {"result": y, "scale": scale}

class ResBlock(nn.Module):
    def __init__(self, d_model: int, seq_len: int = 100+1, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1,2)).transpose(1,2)
        x = x + self.channel(x)
        return x


class ForecastPFN_TSMixer(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-4,
        scaler: str = 'robust',
        patch_size: int = 16,      # kept only for signature-compatibility
        stride:     int = 8        # ditto
    ):
        super().__init__()
        # --- exactly the same signature as ForecastPFN_PAttn ---
        self.epsilon = epsilon
        self.scaler  = CustomScaling(scaler)

        # time-feature embedders (unchanged)
        self.pos_year  = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day   = PositionExpansion(31, 6)
        self.pos_dow   = PositionExpansion(7,  4)

        # compute d_model the same way
        embed_size   = sum(e.channels for e in (
            self.pos_year, self.pos_month, self.pos_day, self.pos_dow
        ))
        self.d_model = embed_size * 2

        # history value embedders
        self.no_pos  = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.w_pos   = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.task_m  = nn.Embedding(NUM_TASKS, embed_size)

        # two TSMixer blocks (seq_len defaults to 100+1 inside ResBlock)
        self.ts_blocks = nn.ModuleList([
            ResBlock(d_model=self.d_model) for _ in range(2)
        ])

        # project along time from length 101→1
        self.proj_time = nn.Linear(100 + 1, 1)
        # scalar head on each embedding
        self.head      = nn.Sequential(nn.Linear(self.d_model, 1), nn.ReLU())


    def forward(self, X: dict) -> dict:
        # 1) time-pos + scaling
        ts   = X['ts']                # [B, T, 5]
        yr   = ts[:,:,0]
        dy   = (yr[:,-1:] - yr).clamp(0, self.pos_year.periods)
        pos  = torch.cat([
            self.pos_year(dy),
            self.pos_month(ts[:,:,1]),
            self.pos_day(  ts[:,:,2]),
            self.pos_dow(  ts[:,:,3])
        ], dim=-1)                   # [B, T, embed_size]

        h     = X['history'].unsqueeze(-1)         # [B, T, 1]
        scale, h = self.scaler(h, self.epsilon)    # [B,1,1], [B,T,1]
        e0    = self.no_pos(h)                     # [B, T, embed_size]
        e1    = self.w_pos(h) + pos                # [B, T, embed_size]
        hist  = torch.cat([e0, e1], dim=-1)        # [B, T, d_model]

        # 2) build tgt exactly as in PAttn
        tt   = X['target_ts'].view(-1,1,5)         # [B,1,5]
        yrq  = (yr[:,-1:] - tt[:,:,0]).clamp(0, self.pos_year.periods)
        qp   = torch.cat([
                    self.pos_year(yrq),
                    self.pos_month(tt[:,:,1]),
                    self.pos_day(  tt[:,:,2]),
                    self.pos_dow(  tt[:,:,3])
               ], dim=-1).squeeze(1)               # [B, embed_size]
        t    = self.task_m(X['task'])             # [B, embed_size]
        tgt  = torch.cat([t, t + qp], dim=-1)     # [B, d_model]

        # prepend it → [B, T+1, d_model]
        hist = torch.cat([hist, tgt.unsqueeze(1)], dim=1)

        # 3) mix via TSMixer
        x = hist
        for block in self.ts_blocks:
            x = block(x)                          # [B, T+1, d_model]

        # 4) project along time → [B, pred_len=1, d_model]
        x = x.permute(0,2,1)                      # → [B, d_model, T+1]
        x = self.proj_time(x)                     # → [B, d_model, 1]
        x = x.permute(0,2,1)                      # → [B, 1, d_model]

        # 5) head + de-scale → keep the same output logic
        y = self.head(x).squeeze(-1)              # → [B, 1]
        y = y * scale[:,:,0]                      # de-scale
        y = y + X['history'].mean(1, keepdim=True)  # add back mean

        # exactly the same final squeeze for pred_len=1
        y = y.squeeze(1)                          # → [B]

        return {'result': y, 'scale': scale}

class ForecastPFN_Linear(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-4,
        scaler: str = 'robust',
        patch_size: int = 16,   # kept for signature compatibility
        stride:     int = 8     # kept for signature compatibility
    ):
        super().__init__()
        self.epsilon = epsilon
        self.scaler  = CustomScaling(scaler)

        # time‐feature embedders (unchanged)
        self.pos_year  = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day   = PositionExpansion(31, 6)
        self.pos_dow   = PositionExpansion(7,  4)

        # compute d_model
        embed_size   = sum(e.channels for e in (
            self.pos_year, self.pos_month, self.pos_day, self.pos_dow
        ))
        self.d_model = embed_size * 2

        # history‐value embedders
        self.no_pos = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.w_pos  = nn.Sequential(nn.Linear(1, embed_size), nn.ReLU())
        self.task_m = nn.Embedding(NUM_TASKS, embed_size)

        # simple linear mixer along time: length=100+1 → 1
        self.mix = nn.Linear(100 + 1, 1)

        # scalar head on each embedding
        self.head = nn.Sequential(nn.Linear(self.d_model, 1), nn.ReLU())

    def forward(self, X: dict) -> dict:
        # 1) time‐pos + scaling
        ts  = X["ts"]               # [B, T=100, 5]
        yr  = ts[:,:,0]
        dy  = (yr[:,-1:] - yr).clamp(0, self.pos_year.periods)
        pos = torch.cat([
            self.pos_year(dy),
            self.pos_month(ts[:,:,1]),
            self.pos_day(  ts[:,:,2]),
            self.pos_dow(  ts[:,:,3])
        ], dim=-1)                  # [B, 100, embed_size]

        h      = X["history"].unsqueeze(-1)    # [B, 100, 1]
        scale, h = self.scaler(h, self.epsilon)# scale:[B,1,1], h:[B,100,1]
        e0     = self.no_pos(h)                # [B, 100, embed_size]
        e1     = self.w_pos(h) + pos           # [B, 100, embed_size]
        hist   = torch.cat([e0, e1], dim=-1)   # [B, 100, d_model]

        # 2) build tgt embedding
        tt   = X["target_ts"].view(-1,1,5)     # [B,1,5]
        yrq  = (yr[:,-1:] - tt[:,:,0]).clamp(0, self.pos_year.periods)
        qp   = torch.cat([
                    self.pos_year(yrq),
                    self.pos_month(tt[:,:,1]),
                    self.pos_day(  tt[:,:,2]),
                    self.pos_dow(  tt[:,:,3])
               ], dim=-1).squeeze(1)           # [B, embed_size]
        t    = self.task_m(X["task"])          # [B, embed_size]
        tgt  = torch.cat([t, t + qp], dim=-1)  # [B, d_model]

        # prepend tgt → [B, 101, d_model]
        hist = torch.cat([hist, tgt.unsqueeze(1)], dim=1)

        # 3) simple linear mix along time
        x = hist.permute(0,2,1)    # [B, d_model, 101]
        x = self.mix(x)            # [B, d_model, 1]
        x = x.permute(0,2,1)       # [B, 1, d_model]

        # 4) head + de‐scale + add back mean
        y = self.head(x).squeeze(-1)               # [B, 1]
        y = y * scale[:,:,0]                       # de‐scale
        y = y + X["history"].mean(1, keepdim=True) # add mean
        y = y.squeeze(1)                           # → [B]

        return {"result": y, "scale": scale}

class ForecastPFN(nn.Module):
    def __init__(self, epsilon=1e-4, scaler='robust'):
        super().__init__()
        self.epsilon = epsilon
        self.pos_year = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day = PositionExpansion(31, 6)
        self.pos_dow = PositionExpansion(7, 4)
        self.scaler = CustomScaling(scaler)
        self.embed_size = sum(emb.channels for emb in (self.pos_year, self.pos_month, self.pos_day, self.pos_dow))
        self.expand_target_nopos = nn.Sequential(nn.Linear(1, 36),
                                                 nn.ReLU())
        self.expand_target_forpos = nn.Sequential(nn.Linear(1, 36),
                                                  nn.ReLU())
        self.target_marker = nn.Embedding(NUM_TASKS, self.embed_size)
        # Transformer Blocks
        self.d_model = self.embed_size * 2
        self.encoder0 = TransformerBlock(d_model=self.d_model)
        self.encoder1 = TransformerBlock(d_model=self.d_model * 4)
        self.final_output = nn.Sequential(
            nn.Linear(self.d_model * 4, 1),
            nn.ReLU()
        )
        
        
    @staticmethod
    def tc(ts, time_index):
        return ts[:, :, time_index]
    def forward(self, x):
        ts, history, target_ts, task = x['ts'], x['history'], x['target_ts'], x['task']
        
        # Build position encodings
        year = self.tc(ts, YEAR)
        delta_year = (year[:, -1:] - year).clamp(min=0, max=self.pos_year.periods)
        pos_embedding = torch.cat([
            self.pos_year(delta_year),
            self.pos_month(self.tc(ts, MONTH)),
            self.pos_day(self.tc(ts, DAY)),
            self.pos_dow(self.tc(ts, DOW)),
            ], dim=-1)
        
        # Embed history
        history_channels = history.unsqueeze(-1)
        scale, scaled = self.scaler(history_channels, self.epsilon)
        embed_nopos = self.expand_target_nopos(scaled)
        embed_pos = self.expand_target_forpos(scaled) + pos_embedding
        embedded = torch.cat([embed_nopos, embed_pos], dim=-1)
        
        # Embed target
        target_year = (year[:, -1:] - self.tc(target_ts, YEAR)).clamp(min=0, max=self.pos_year.periods)
        target_pos_embed = torch.cat([
            self.pos_year(target_year),
            self.pos_month(self.tc(target_ts, MONTH)),
            self.pos_day(self.tc(target_ts, DAY)),
            self.pos_dow(self.tc(target_ts, DOW))
            ], dim=-1)
        target_pos_embed = target_pos_embed.squeeze(1)
        task_embed = self.target_marker(task)
        target = torch.cat([task_embed, task_embed + target_pos_embed], dim=-1)
        
        # Mask
        seq_mask = (year > 0)  # → [B, T], bool
        seq_mask = torch.cat([seq_mask, torch.ones_like(seq_mask[:, :1], dtype=torch.bool)], dim=1)  # [B, T+1]

        # Broadcast to [B, T+1, T+1]
        mask = seq_mask.unsqueeze(1) & seq_mask.unsqueeze(2)
        
        x = torch.cat([embedded, target.unsqueeze(1)], dim=1)

        x = self.encoder0(x, mask=mask)
        x = self.encoder1(x, mask=mask)
        scale = scale[:, -1, 0:1]
        result = self.final_output(x[:, -1, :]) * scale
        return {'result': result, 'scale': scale}
    
if __name__ == "__main__":
    # Model config
    embed_size = 36
    num_tasks = 10
    seq_len = 36
    batch_size = 2

    model = ForecastPFN()

    # Generate time feature inputs within valid ranges
    ts = torch.cat([
        torch.randint(0, 10, (batch_size, seq_len, 1)),  # year delta (0–9)
        torch.randint(0, 12, (batch_size, seq_len, 1)),  # month (0–11)
        torch.randint(0, 31, (batch_size, seq_len, 1)),  # day (0–30)
        torch.randint(0, 7,  (batch_size, seq_len, 1))   # day of week (0–6)
    ], dim=-1)

    target_ts = torch.cat([
        torch.randint(0, 10, (batch_size, 1, 1)),  # year delta
        torch.randint(0, 12, (batch_size, 1, 1)),  # month
        torch.randint(0, 31, (batch_size, 1, 1)),  # day
        torch.randint(0, 7,  (batch_size, 1, 1))   # day of week
    ], dim=-1)

    batch = {
        "ts": ts,                                   # [B, T, 4]
        "history": torch.rand(batch_size, seq_len), # [B, T]
        "target_ts": target_ts,                     # [B, 1, 4]
        "task": torch.randint(0, num_tasks, (batch_size,))  # [B]
    }

    print(model)
    output = model(batch)
    print("Output:", output)
    print("Output shape:", output['result'].shape)
