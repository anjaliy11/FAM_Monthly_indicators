from __future__ import annotations

import pandas as pd

from fam_pipeline.pipeline import build_ticker_frame


def test_build_ticker_frame_all_columns_and_order(monthly_df: pd.DataFrame) -> None:
    frame = build_ticker_frame(monthly_df)

    expected_columns = (
        "ticker",
        "month",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_10",
        "sma_20",
        "ema_10",
        "ema_20",
        "donchian_upper_20",
        "donchian_lower_20",
        "donchian_mid_20",
        "bb_mid_20",
        "bb_upper_20",
        "bb_lower_20",
        "zscore_20",
    )

    assert list(frame.columns) == list(expected_columns)
    assert frame["month"].tolist() == ["2020-01", "2020-02"]
