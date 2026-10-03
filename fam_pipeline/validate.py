from __future__ import annotations

import pandas as pd

from fam_pipeline.config import (
    EXPECTED_MONTH_END,
    EXPECTED_MONTH_START,
    EXPECTED_MONTHS,
    EXPECTED_TICKERS,
    INDICATOR_WINDOWS,
    OUTPUT_COLUMNS,
)


def validate_output(frames: dict[str, pd.DataFrame]) -> None:
    """Check all output frames meet the required structure and row contract."""
    if set(frames) != set(EXPECTED_TICKERS):
        missing = sorted(set(EXPECTED_TICKERS) - set(frames))
        extra = sorted(set(frames) - set(EXPECTED_TICKERS))
        raise ValueError(f"Output frame keys mismatch. Missing={missing}, extra={extra}.")

    expected_months = (
        pd.period_range(EXPECTED_MONTH_START, EXPECTED_MONTH_END, freq="M").astype(str).tolist()
    )

    for ticker in EXPECTED_TICKERS:
        frame = frames[ticker]
        if len(frame) != EXPECTED_MONTHS:
            raise ValueError(
                f"Ticker {ticker} output has {len(frame)} rows; expected {EXPECTED_MONTHS}."
            )
        if list(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Ticker {ticker} output columns mismatch. Expected {OUTPUT_COLUMNS}, "
                f"got {list(frame.columns)}."
            )

        actual_tickers = frame["ticker"].drop_duplicates().tolist()
        if actual_tickers != [ticker] or frame["ticker"].isna().any():
            raise ValueError(
                f"Ticker {ticker} column must contain only {ticker}; got {actual_tickers}."
            )

        actual_months = frame["month"].astype(str).tolist()
        if actual_months != expected_months:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(actual_months, expected_months, strict=True))
                    if pair[0] != pair[1]
                ),
                min(len(actual_months), len(expected_months)),
            )
            actual = actual_months[mismatch] if mismatch < len(actual_months) else "<missing>"
            expected = (
                expected_months[mismatch] if mismatch < len(expected_months) else "<no month>"
            )
            raise ValueError(
                f"Ticker {ticker} month column mismatch at row {mismatch + 1}: "
                f"expected {expected}, got {actual}; months must be ascending, contiguous, "
                "and unique."
            )

        for column in OUTPUT_COLUMNS:
            numeric_values = pd.to_numeric(frame[column], errors="coerce")
            infinite = numeric_values.isin([float("inf"), float("-inf")])
            if infinite.any():
                row = frame.index[infinite][0]
                raise ValueError(
                    f"Ticker {ticker} month {frame.at[row, 'month']} column {column} is infinite."
                )

        for column in OUTPUT_COLUMNS:
            if column in INDICATOR_WINDOWS:
                window = INDICATOR_WINDOWS[column]
                for row in range(EXPECTED_MONTHS):
                    should_be_nan = row < window - 1
                    is_nan = pd.isna(frame.iloc[row][column])
                    if is_nan != should_be_nan:
                        expected = "NaN" if should_be_nan else "finite"
                        actual = "NaN" if is_nan else "finite"
                        raise ValueError(
                            f"Ticker {ticker} month {frame.iloc[row]['month']} column {column} "
                            f"must be {expected}, got {actual}."
                        )
            elif frame[column].isna().any():
                row = frame.index[frame[column].isna()][0]
                raise ValueError(
                    f"Ticker {ticker} month {frame.at[row, 'month']} column {column} "
                    "must not be NaN."
                )

        invalid_high_low = frame["high"] < frame["low"]
        if invalid_high_low.any():
            row = frame.index[invalid_high_low][0]
            raise ValueError(
                f"Ticker {ticker} month {frame.at[row, 'month']} column high is below column low."
            )
        invalid_high = frame["high"] < frame[["open", "close"]].max(axis=1)
        if invalid_high.any():
            row = frame.index[invalid_high][0]
            raise ValueError(
                f"Ticker {ticker} month {frame.at[row, 'month']} column high is below open "
                "or close."
            )
        invalid_low = frame["low"] > frame[["open", "close"]].min(axis=1)
        if invalid_low.any():
            row = frame.index[invalid_low][0]
            raise ValueError(
                f"Ticker {ticker} month {frame.at[row, 'month']} column low is above open or close."
            )

        for upper, middle, lower in (
            ("bb_upper_20", "bb_mid_20", "bb_lower_20"),
            ("donchian_upper_20", "donchian_mid_20", "donchian_lower_20"),
        ):
            complete = frame[[upper, middle, lower]].notna().all(axis=1)
            invalid_order = complete & (
                (frame[upper] < frame[middle]) | (frame[middle] < frame[lower])
            )
            if invalid_order.any():
                row = frame.index[invalid_order][0]
                raise ValueError(
                    f"Ticker {ticker} month {frame.at[row, 'month']} columns "
                    f"{upper}, {middle}, {lower} are not in descending order."
                )
