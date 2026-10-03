from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def daily_df() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2020-01-02",
            "2020-01-08",
            "2020-02-03",
            "2020-02-10",
            "2020-02-20",
            "2020-03-05",
        ]
    )
    data = {
        "ticker": ["AAPL"] * 6,
        "date": dates,
        "volume": [100, 200, 300, 400, 500, 600],
        "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "close": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def monthly_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AMD", "AMD"],
            "month": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "open": [10.0, 12.0, 50.0, 52.0],
            "high": [12.0, 14.0, 55.0, 57.0],
            "low": [9.0, 11.0, 48.0, 50.0],
            "close": [11.0, 13.0, 51.0, 53.0],
            "volume": [300, 1200, 900, 1500],
        }
    )
