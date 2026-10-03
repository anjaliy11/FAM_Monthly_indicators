from __future__ import annotations

import pandas as pd


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily OHLCV data to monthly OHLCV by ticker and month."""
    sorted_daily = daily.sort_values(["ticker", "date"], kind="stable").copy()
    sorted_daily["month"] = sorted_daily["date"].dt.to_period("M")
    monthly = (
        sorted_daily.groupby(["ticker", "month"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    monthly["month"] = monthly["month"].astype(str)
    return monthly[["ticker", "month", "open", "high", "low", "close", "volume"]]
