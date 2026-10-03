from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fam_pipeline.config import EXPECTED_MONTHS, EXPECTED_TICKERS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def load_daily(path: Path) -> pd.DataFrame:
    """Load daily OHLCV data and enforce schema and value validation."""
    try:
        frame = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Missing required columns in {path}: {REQUIRED_COLUMNS}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date value in column 'date': {exc}") from exc

    numeric_columns = ["volume", "open", "high", "low", "close", "adjclose"]
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value in column '{column}': {exc}") from exc

    frame["volume"] = frame["volume"].astype("int64")
    for column in ["open", "high", "low", "close", "adjclose"]:
        frame[column] = frame[column].astype("float64")

    if frame.isnull().any().any():
        null_columns = frame.columns[frame.isnull().any()].tolist()
        raise ValueError(f"Null values found in required columns: {null_columns}")

    if (frame["volume"] < 0).any():
        raise ValueError("Negative volume values are not allowed.")
    for price_name in ["open", "high", "low", "close", "adjclose"]:
        if (frame[price_name] <= 0).any():
            raise ValueError(f"Non-positive price values found in column '{price_name}'.")

    if frame.duplicated(subset=["ticker", "date"], keep=False).any():
        dupes = frame[frame.duplicated(subset=["ticker", "date"], keep=False)][["ticker", "date"]]
        raise ValueError(
            f"Duplicate (ticker, date) rows found: {dupes.head().to_dict(orient='records')}"
        )

    if set(frame["ticker"].unique()) != set(EXPECTED_TICKERS):
        actual = sorted(frame["ticker"].unique().tolist())
        raise ValueError(
            f"Unexpected ticker set. Expected {sorted(EXPECTED_TICKERS)}, got {actual}."
        )

    months_per_ticker = (
        frame.assign(month=frame["date"].dt.to_period("M")).groupby("ticker")["month"].nunique()
    )
    bad_months = months_per_ticker[months_per_ticker != EXPECTED_MONTHS]
    if not bad_months.empty:
        raise ValueError(
            "Months per ticker mismatch. Expected 24 months for each ticker, got "
            f"{bad_months.to_dict()}."
        )

    if (frame["high"] < frame["low"]).any():
        bad = frame.loc[frame["high"] < frame["low"], ["ticker", "date", "high", "low"]]
        raise ValueError(f"High below low observed: {bad.head().to_dict(orient='records')}")

    max_ohlc = frame[["open", "close"]].max(axis=1)
    min_ohlc = frame[["open", "close"]].min(axis=1)
    if (frame["high"] < max_ohlc).any():
        bad = frame.loc[frame["high"] < max_ohlc, ["ticker", "date", "open", "close", "high"]]
        raise ValueError(
            f"High below max(open, close) observed: {bad.head().to_dict(orient='records')}"
        )
    if (frame["low"] > min_ohlc).any():
        bad = frame.loc[frame["low"] > min_ohlc, ["ticker", "date", "open", "close", "low"]]
        raise ValueError(
            f"Low above min(open, close) observed: {bad.head().to_dict(orient='records')}"
        )

    sorted_frame = frame.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    logger.info(
        "Loaded %s daily rows for %s tickers.", len(sorted_frame), sorted_frame["ticker"].nunique()
    )
    return sorted_frame
