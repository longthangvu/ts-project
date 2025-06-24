import torch
import torch.nn as nn
import math

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
