from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average over the previous n monthly closes."""
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """EMA_t = (P_t - EMA_{t-1}) * 2/(n+1) + EMA_{t-1}; EMA_{n-1} = SMA of first n closes."""
    out = pd.Series(float("nan"), index=close.index, dtype=float)
    if len(close) < n:
        return out

    seeded = close.copy()
    seeded.iloc[: n - 1] = np.nan
    seeded.iloc[n - 1] = close.iloc[:n].mean()
    return seeded.ewm(alpha=2 / (n + 1), adjust=False).mean()


def bollinger(close: pd.Series, n: int, k: float) -> pd.DataFrame:
    """Compute Bollinger bands using rolling mean and sample std dev."""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame({f"bb_mid_{n}": mid, f"bb_upper_{n}": upper, f"bb_lower_{n}": lower})


def donchian(high: pd.Series, low: pd.Series, n: int) -> pd.DataFrame:
    """Compute Donchian channel upper/lower/mid using rolling n periods."""
    upper = high.rolling(n).max()
    lower_band = low.rolling(n).min()
    mid = (upper + lower_band) / 2
    return pd.DataFrame(
        {
            f"donchian_upper_{n}": upper,
            f"donchian_lower_{n}": lower_band,
            f"donchian_mid_{n}": mid,
        }
    )


def zscore(close: pd.Series, n: int) -> pd.Series:
    """Rolling z-score using the same window as Bollinger bands."""
    rolling_mean = close.rolling(n).mean()
    rolling_std = close.rolling(n).std(ddof=1)
    rolling_std = rolling_std.replace(0, np.nan)
    return (close - rolling_mean) / rolling_std
