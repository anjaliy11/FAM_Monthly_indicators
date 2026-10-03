from __future__ import annotations

import pandas as pd

from fam_pipeline.resample import to_monthly


def test_to_monthly_shuffled_input() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(
                ["2020-01-08", "2020-01-02", "2020-02-05", "2020-02-11", "2020-02-20", "2020-03-01"]
            ),
            "volume": [100, 50, 200, 300, 250, 400],
            "open": [11.0, 10.0, 12.0, 13.0, 14.0, 15.0],
            "high": [12.0, 11.0, 15.0, 16.0, 17.0, 18.0],
            "low": [10.0, 9.0, 11.0, 12.0, 13.0, 14.0],
            "close": [11.5, 10.5, 14.0, 15.0, 16.0, 17.0],
        }
    )

    out = to_monthly(df)

    expected = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "month": ["2020-01", "2020-02", "2020-03"],
            "open": [10.0, 12.0, 15.0],
            "high": [12.0, 17.0, 18.0],
            "low": [9.0, 11.0, 14.0],
            "close": [11.5, 16.0, 17.0],
            "volume": [150, 750, 400],
        }
    )

    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)
