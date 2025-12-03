from util.prior import shift_axis

import numpy as np
import torch
import typing as T


def generate_trend_component(
    trend_linear_scaler: torch.Tensor,
    trend_exp_scaler: torch.Tensor,
    offset_linear: torch.Tensor,
    offset_exp: torch.Tensor,
    x: torch.Tensor,
    min_exp_scaler: float = 0.00001,
):
    """
    Method to generate trend component of the time series
    Args:
    trend_linear_scaler: Linear scaler for the trend
    trend_exp_scaler: Exponential scaler for the trend
    offset_linear: Offset for the linear trend
    offset_exp: Offset for the exponential trend
    x: Input tensor
    min_exp_scaler: Minimum value for the exponential scaler
    return: Tuple of values, linear trend and exponential trend
    """
    values = torch.ones_like(x)
    origin = x[:, 0].unsqueeze(-1).expand(-1, x.shape[-1])
    distance_to_origin = torch.sub(x, origin)
    # span = (x[:, -1] - x[:, 0]).unsqueeze(-1).expand_as(x).clamp_min(1e-8)
    # linear_trend = torch.zeros_like(x)
    
    if trend_linear_scaler is not None:
        trend_linear_scaler = trend_linear_scaler.unsqueeze(-1).expand(-1, x.shape[-1])
        linear_trend = torch.mul(
            # shift_axis(distance_to_origin, offset_linear) / span, trend_linear_scaler
            shift_axis(distance_to_origin, offset_linear), trend_linear_scaler
        )
        values = torch.add(values, linear_trend)

    if trend_exp_scaler is not None:
        trend_exp_scaler = (
            trend_exp_scaler.clip(min=min_exp_scaler)
            .unsqueeze(-1)
            .expand(-1, x.shape[-1])
        )
        # exp_trend = torch.pow(
        #     trend_exp_scaler, shift_axis(distance_to_origin, offset_exp)
        # )
        # scale time to [0,1] per series to de-couple growth from n_units/sequence length
        span = (x[:, -1] - x[:, 0]).unsqueeze(-1).expand_as(x).clamp_min(1e-8)
        exp_trend = torch.pow(
            trend_exp_scaler, shift_axis(distance_to_origin, offset_exp) / span
        )

        values = torch.mul(values, exp_trend)

        return values, linear_trend, exp_trend

    return values, linear_trend, None


def get_freq_component(
    frequency_feature: torch.Tensor,
    n_harmonics: torch.Tensor,
    cycle: T.Union[int, float],
    device: str = "cpu",
):
    """
    Method to get systematic movement of values across time
    """

    harmonics = (
        torch.arange(1, n_harmonics.item() + 1)
        .unsqueeze(0)
        .expand(frequency_feature.shape[0], -1)
        .to(device)
    )
    sin_coef = torch.normal(mean=0, std=1 / harmonics)
    cos_coef = torch.normal(mean=0, std=1 / harmonics)

    # normalize the coefficients such that their sum of squares is 1
    coef_sq_sum = torch.sqrt(torch.sum(sin_coef**2) + torch.sum(cos_coef**2))
    sin_coef /= coef_sq_sum
    cos_coef /= coef_sq_sum

    # construct the result for systematic movement which
    # comprises of patterns of varying frequency
    freq_pattern = torch.div(frequency_feature, cycle)
    sin = (
        sin_coef.unsqueeze(-1)
        * torch.sin(2 * torch.pi * harmonics.unsqueeze(-1) * freq_pattern.unsqueeze(1))
    ).sum(1)
    cos = (
        cos_coef.unsqueeze(-1)
        * torch.cos(2 * torch.pi * harmonics.unsqueeze(-1) * freq_pattern.unsqueeze(1))
    ).sum(1)
    comp = torch.add(sin, cos)
    comp = comp / (2.0 * n_harmonics.item()) ** 0.5
    return comp


def binning_function(
    x: torch.Tensor, bins: T.Union[int, float], cycle: float, n_units: torch.Tensor
):
    out = (x / cycle).floor() + 1
    mask = out > bins
    out[mask] = (out[mask] % bins) + 1
    return out

def _bounded_part(phi, amp, cap=0.95):
    # phi: [B,T] raw periodic basis; amp: [B]
    phi_b = torch.tanh(phi)                      # bound basis to [-1,1]
    a = torch.tanh(amp) * cap                    # bound amplitude to [-cap,cap]
    return 1.0 + a.unsqueeze(-1) * phi_b         # strictly > 0 for cap<1

def generate_seasonal_component(
    annual_param: torch.Tensor,
    monthly_param: torch.Tensor,
    weekly_param: torch.Tensor,
    x: torch.Tensor,
    n_units: torch.Tensor,
    n_harmonics: torch.Tensor,
    device: str = "cpu",
):
    # write docstring for this function
    """
    Method to generate seasonal component of the time series
    Args:
    annual_param: Annual parameter
    monthly_param: Monthly parameter
    weekly_param: Weekly parameter
    x: Input tensor
    n_units: Number of units
    n_harmonics: Number of harmonics
    device: Device to run the code
    return: Seasonal component
    """

    seasonal = torch.ones(x.shape[0], x.shape[1], 4).to(device)

    if annual_param is not None:
        phi_ann = get_freq_component(binning_function(x, 12, 30.417, n_units), 
                                     n_harmonics[0], 12, device)
        annual_component = _bounded_part(phi_ann, annual_param.to(device), 0.95)
        seasonal[:, :, 1] = annual_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], annual_component)

    if monthly_param is not None:
        phi_mth = get_freq_component(binning_function(x, 30.417, 1, n_units), 
                                     n_harmonics[1], 30.417, device)
        monthly_component = _bounded_part(phi_mth, monthly_param.to(device), 0.95)
        seasonal[:, :, 2] = monthly_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], monthly_component)

    if weekly_param is not None:
        phi_wkl = get_freq_component(binning_function(x, 7, 1, n_units), 
                                     n_harmonics[2], 7, device)
        weekly_component = _bounded_part(phi_wkl, weekly_param.to(device), 0.95)
        seasonal[:, :, 3] = weekly_component
        seasonal[:, :, 0] = torch.mul(seasonal[:, :, 0], weekly_component)

    # seasonal dimensions = total_seasonality, annual, monthly, weekly
    return seasonal


def generate_noise_component(
    k: torch.Tensor,
    noise_mean: torch.Tensor,
    shape: T.Tuple[int, int],
    device: str = "cpu",
):
    """
    Method to generate noise component of the time series
    Args:
    k: Shape parameter for the weibull distribution
    noise_mean: Mean of the noise
    shape: Shape of the noise
    device: Device to run the code
    return: Noise component
    """
    lambda_ = noise_mean / (np.log(2) ** (1 / k))
    return torch.from_numpy(np.random.weibull(k.unsqueeze(-1).cpu(), size=shape)).to(
        device
    ) * lambda_.unsqueeze(-1)
