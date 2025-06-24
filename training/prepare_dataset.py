import numpy as np
import pandas as pd
import torch
from constants import PADDING, HISTORY_LEN, TARGET_LEN, TRIM_LEN, TARGET_INDEX, SINGLE_POINT, MEAN_TO_DATE, STDEV_TO_DATE
from config_variables import Config


def compute_time_features(ts: np.ndarray) -> np.ndarray:
    """
    Compute time features analogous to the TF version.
    """
    ts = pd.to_datetime(ts)
    if Config.is_sub_day:
        return np.stack([ts.minute, ts.hour, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)
    return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)


def position_encoding(periods: int, freqs: int) -> np.ndarray:
    """
    Same as TF version: build sin/cos embedding table.
    """
    return np.hstack([
        np.fromfunction(lambda i, j: np.sin(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs)),
        np.fromfunction(lambda i, j: np.cos(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs))
    ])


def build_frames(rec: dict):
    """
    Port of `build_frames` from TF: returns 6 tensors:
      date_info_frames, history_frames, noise_frames,
      target_dates, target_values, target_noise
    Each has shape [batch_size, ...]
    """
    # to numpy
    ts = rec['ts'].cpu().numpy()
    y = rec['y'].cpu().numpy()
    noise = rec['noise'].cpu().numpy()

    raw_date_info = compute_time_features(ts)
    # 1) pad & frame date_info
    di_pad = np.pad(raw_date_info, ((PADDING, 0), (0, 0)))
    date_info = np.stack([di_pad[i:i+HISTORY_LEN] for i in range(di_pad.shape[0] - HISTORY_LEN + 1)], axis=0)
    # 2) pad & frame y / noise
    y_pad = np.pad(y, (PADDING, 0))
    history = np.stack([y_pad[i:i+HISTORY_LEN] for i in range(y_pad.shape[0] - HISTORY_LEN + 1)], axis=0)[..., None]
    n_pad = np.pad(noise, (PADDING, 0))
    noise_f = np.stack([n_pad[i:i+HISTORY_LEN] for i in range(n_pad.shape[0] - HISTORY_LEN + 1)], axis=0)[..., None]
    # 3) frame targets
    td = np.stack([raw_date_info[i:i+TARGET_LEN] for i in range(raw_date_info.shape[0] - TARGET_LEN + 1)], axis=0)
    tv = np.stack([y[i:i+TARGET_LEN] for i in range(y.shape[0] - TARGET_LEN + 1)], axis=0)[..., None]
    tn = np.stack([noise[i:i+TARGET_LEN] for i in range(noise.shape[0] - TARGET_LEN + 1)], axis=0)[..., None]
    # 4) slice according to TRIM_LEN and TARGET_INDEX
    start = tv.shape[0] - TRIM_LEN
    date_info = date_info[-start:-TARGET_LEN]
    history = history[-start:-TARGET_LEN]
    noise_f = noise_f[-start:-TARGET_LEN]
    td = td[TARGET_INDEX:]
    tv = tv[TARGET_INDEX:]
    tn = tn[TARGET_INDEX:]

    # to torch
    return (
        torch.from_numpy(date_info).long(),
        torch.from_numpy(history).float(),
        torch.from_numpy(noise_f).float(),
        torch.from_numpy(td).long(),
        torch.from_numpy(tv).float(),
        torch.from_numpy(tn).float()
    )


def gen_random_single_point(date_info, history, noise, target_dates, target_values, target_noise):
    """
    Single-point task: pick random index in [0, TARGET_LEN).
    """
    batch = target_dates.shape[0]
    device = history.device
    idx = torch.randint(0, TARGET_LEN, (batch,), device=device)
    sel = torch.arange(batch, device=device)
    target_ts = target_dates[sel, idx]
    target_val = target_values[sel, idx, 0]
    return {
        'ts': date_info,
        'history': history * noise,
        'noise': noise,
        'target_ts': target_ts,
        'task': torch.full((batch,), SINGLE_POINT, dtype=torch.long, device=device),
        'target_noise': target_noise
    }, target_val


def gen_mean_to_random_date(date_info, history, noise, target_dates, target_values, target_noise):
    """
    Mean-to-random-date task.
    """
    batch = target_dates.shape[0]
    device = history.device
    idxs = torch.randint(0, TARGET_LEN, (batch,), device=device)
    sel = torch.arange(batch, device=device)
    target_ts = target_dates[sel, idxs]
    means = torch.stack([target_values[i, :k+1, 0].mean() for i, k in enumerate(idxs)], dim=0)
    return {
        'ts': date_info,
        'history': history * noise * 0.75,
        'noise': noise,
        'target_ts': target_ts,
        'task': torch.full((batch,), MEAN_TO_DATE, dtype=torch.long, device=device),
        'target_noise': target_noise
    }, means


def gen_std_to_random_date(date_info, history, noise, target_dates, target_values, target_noise):
    """
    Std-to-random-date task.
    """
    batch = target_dates.shape[0]
    device = history.device
    low = TARGET_LEN // 2
    idxs = torch.randint(low, TARGET_LEN, (batch,), device=device)
    sel = torch.arange(batch, device=device)
    target_ts = target_dates[sel, idxs]
    stds = torch.stack([target_values[i, :k+1, 0].std(unbiased=False) for i, k in enumerate(idxs)], dim=0)
    noise_stds = torch.stack([target_noise[i, :k+1, 0].std(unbiased=False) for i, k in enumerate(idxs)], dim=0)
    combined = torch.sqrt(stds**2 + noise_stds**2)
    return {
        'ts': date_info,
        'history': history * noise,
        'noise': noise,
        'target_ts': target_ts,
        'task': torch.full((batch,), STDEV_TO_DATE, dtype=torch.long, device=device),
        'target_noise': target_noise
    }, combined

def gen_random_single_point_no_noise(
    date_info, history, noise, target_dates, target_values, target_noise
):
    """
    TF: return only ts, history, target_ts, task (no noise fields) and the clean target.
    """
    batch = target_dates.shape[0]
    device = history.device
    idxs = torch.randint(0, TARGET_LEN, (batch,), device=device)
    sel  = torch.arange(batch, device=device)
    target_ts  = target_dates[sel, idxs]
    target_val = target_values[sel, idxs, 0]
    return {
        'ts':       date_info,
        'history':  history,            # NO multiplication by noise
        'task':     torch.full((batch,), SINGLE_POINT, device=device, dtype=torch.long),
        'target_ts': target_ts,
    }, target_val


def gen_mean_to_random_date_no_noise(
    date_info, history, noise, target_dates, target_values, target_noise
):
    batch = target_dates.shape[0]
    device = history.device
    idxs = torch.randint(0, TARGET_LEN, (batch,), device=device)
    sel  = torch.arange(batch, device=device)
    target_ts = target_dates[sel, idxs]
    # mean over values[0:idx+1]
    means = torch.stack(
        [target_values[i, :k+1, 0].mean() for i, k in enumerate(idxs)], 
        dim=0)
    return {
        'ts':      date_info,
        'history': history,             # NO noise scaling
        'task':    torch.full((batch,), MEAN_TO_DATE, device=device, dtype=torch.long),
        'target_ts': target_ts,
    }, means


def gen_std_to_random_date_no_noise(
    date_info, history, noise, target_dates, target_values, target_noise
):
    batch = target_dates.shape[0]
    device = history.device
    low = TARGET_LEN // 2
    idxs = torch.randint(low, TARGET_LEN, (batch,), device=device)
    sel  = torch.arange(batch, device=device)
    target_ts = target_dates[sel, idxs]
    # std over values and noise separately
    stds = torch.stack(
        [target_values[i, :k+1, 0].std(unbiased=False) for i, k in enumerate(idxs)],
        dim=0)
    noise_stds = torch.stack(
        [target_noise[i, :k+1, 0].std(unbiased=False) for i, k in enumerate(idxs)],
        dim=0)
    combined = torch.sqrt(stds**2 + noise_stds**2)
    return {
        'ts':      date_info,
        'history': history,             # NO noise scaling
        'task':    torch.full((batch,), STDEV_TO_DATE, device=device, dtype=torch.long),
        'target_ts': target_ts,
    }, combined


def remove_noise(X: dict, y: torch.Tensor):
    """
    Drop the now‐unused 'noise' and 'target_noise' fields, leaving only ts, history, target_ts, task.
    """
    return (
        {
            'ts':        X['ts'],
            'history':   X['history'],
            'target_ts': X['target_ts'],
            'task':      X['task'],
        },
        y
    )


def filter_unusable_points(X: dict, y: torch.Tensor) -> bool:
    """
    Drop examples where max(history) <= 0.1 or y is non-finite.
    """
    return (X['history'].max() > 0.1) and torch.isfinite(y).all()
