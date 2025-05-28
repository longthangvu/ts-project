import torch
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def dhr_arima_forecast(x, pred_len, season_len):
    preds = []
    for s in x:
        arr = s.numpy()
        t_idx = np.arange(len(arr))
        t_future = np.arange(len(arr), len(arr) + pred_len)

        # Dynamically set Fourier order to satisfy constraint
        order = max(1, min(3, season_len // 2))

        try:
            fourier = Fourier(period=season_len, order=order)
            X = fourier.in_sample(index=t_idx)
            X_fore = fourier.in_sample(index=t_future)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                model = SARIMAX(
                    arr,
                    exog=X,
                    order=(1, 0, 0),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

            if not model.mle_retvals.get('converged', True):
                raise RuntimeError("DHR-ARIMA failed to converge")

            forecast = model.forecast(steps=pred_len, exog=X_fore)
            preds.append(torch.tensor(forecast, dtype=torch.float32))

        except Exception:
            preds.append(torch.full((pred_len,), arr[-1], dtype=torch.float32))  # fallback
    return torch.stack(preds)

def naive2_forecast(x, pred_len, season_len):
    preds = []
    for s in x:
        arr = s.numpy()
        if season_len > 1 and len(arr) >= season_len:
            # Seasonal naive
            repeated = arr[-season_len:].repeat((pred_len + season_len - 1) // season_len)[:pred_len]
            preds.append(torch.tensor(repeated, dtype=torch.float32))
        else:
            # ARIMA fallback
            try:
                model = auto_arima(
                    arr,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    maxiter=10
                )
                preds.append(torch.tensor(model.predict(n_periods=pred_len), dtype=torch.float32))
            except:
                # Fallback to naive last if ARIMA fails
                preds.append(torch.full((pred_len,), arr[-1], dtype=torch.float32))
    return torch.stack(preds)

def naive_last(x, pred_len):
    return torch.stack([
        torch.full((pred_len,), s[-1].item()) for s in x
    ])

def naive_mean(x, pred_len):
    return torch.stack([
        torch.full((pred_len,), s.mean().item()) for s in x
    ])

def naive_seasonal(x, pred_len, season_len):
    preds = []
    for s in x:
        if len(s) < season_len:
            raise ValueError("Time series too short for given season length")
        # Repeat the last season values to fill pred_len
        last_season = s[-season_len:]
        num_repeats = (pred_len + season_len - 1) // season_len  # ceil division
        repeated = last_season.repeat(num_repeats)[:pred_len]
        preds.append(repeated)
    return torch.stack(preds)


def ets_forecast(x, pred_len, season_len):
    preds = []
    for s in x:
        arr = s.numpy()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                model = ExponentialSmoothing(
                    arr,
                    trend='add',
                    seasonal='add' if season_len > 1 else None,
                    seasonal_periods=season_len if season_len > 1 else None
                ).fit()

            if not model.mle_retvals.get('converged', True):
                raise RuntimeError("ETS failed to converge.")
            forecast = model.forecast(pred_len)
            if np.isnan(forecast).any():
                raise ValueError("ETS forecast returned NaNs.")
            preds.append(torch.tensor(forecast, dtype=torch.float32))
        except Exception as e:
            print(f"Warning: {str(e)} — fallback to naive_last")
            fallback = torch.full((pred_len,), arr[-1], dtype=torch.float32)
            preds.append(fallback)

    return torch.stack(preds)


