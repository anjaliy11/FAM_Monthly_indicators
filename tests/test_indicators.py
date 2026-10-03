from __future__ import annotations

import numpy as np
import pandas as pd

from fam_pipeline.indicators import bollinger, donchian, ema, sma, zscore


def test_ema_and_sma_match_hand_calculation() -> None:
    close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=float)

    expected_sma = pd.Series([np.nan, np.nan, 12.0, 14.0, 16.0, 18.0])
    pd.testing.assert_series_equal(sma(close, 3), expected_sma, check_names=False)

    expected_ema = pd.Series([np.nan, np.nan, 12.0, 14.0, 16.0, 18.0])
    result_ema = ema(close, 3)
    pd.testing.assert_series_equal(result_ema, expected_ema, check_names=False)


def test_bollinger_and_zscore_properties() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=float)
    bands = bollinger(close, 5, 2.0)
    z = zscore(close, 5)

    assert list(bands.columns) == ["bb_mid_5", "bb_upper_5", "bb_lower_5"]
    assert z.notna().sum() >= 1
    assert np.isclose(
        (bands["bb_upper_5"] - bands["bb_mid_5"]).iloc[-1], 2 * (close.rolling(5).std().iloc[-1])
    )

    constant = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], dtype=float)
    z_constant = zscore(constant, 5)
    assert z_constant.isna().all()


def test_donchian_channel() -> None:
    high = pd.Series([10.0, 12.0, 11.0, 15.0, 13.0, 17.0])
    low = pd.Series([8.0, 9.0, 7.0, 10.0, 12.0, 15.0])
    ch = donchian(high, low, 3)

    assert list(ch.columns) == ["donchian_upper_3", "donchian_lower_3", "donchian_mid_3"]
    assert ch["donchian_upper_3"].iloc[-1] == 17.0
    assert ch["donchian_lower_3"].iloc[-1] == 10.0
