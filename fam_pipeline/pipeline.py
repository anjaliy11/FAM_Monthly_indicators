from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fam_pipeline.config import EXPECTED_TICKERS, OUTPUT_COLUMNS
from fam_pipeline.indicators import bollinger, donchian, ema, sma, zscore
from fam_pipeline.loader import load_daily
from fam_pipeline.resample import to_monthly
from fam_pipeline.validate import validate_output
from fam_pipeline.writer import write_results

logger = logging.getLogger(__name__)


def build_ticker_frame(monthly_one_ticker: pd.DataFrame) -> pd.DataFrame:
    """Build the full indicator frame for one ticker."""
    frame = monthly_one_ticker.copy()
    if "ticker" in frame.columns and frame["ticker"].nunique() > 1:
        first_ticker = frame["ticker"].dropna().iloc[0]
        frame = frame[frame["ticker"] == first_ticker].copy()

    frame = frame.sort_values("month", kind="stable").reset_index(drop=True).copy()
    close = frame["close"]

    base = frame[["ticker", "month", "open", "high", "low", "close", "volume"]].copy()
    metrics = pd.concat(
        [
            sma(close, 10).rename("sma_10"),
            sma(close, 20).rename("sma_20"),
            ema(close, 10).rename("ema_10"),
            ema(close, 20).rename("ema_20"),
            donchian(frame["high"], frame["low"], 20)["donchian_upper_20"].rename(
                "donchian_upper_20"
            ),
            donchian(frame["high"], frame["low"], 20)["donchian_lower_20"].rename(
                "donchian_lower_20"
            ),
            donchian(frame["high"], frame["low"], 20)["donchian_mid_20"].rename("donchian_mid_20"),
            bollinger(close, 20, 2.0)["bb_mid_20"],
            bollinger(close, 20, 2.0)["bb_upper_20"],
            bollinger(close, 20, 2.0)["bb_lower_20"],
            zscore(close, 20).rename("zscore_20"),
        ],
        axis=1,
    )

    result = pd.concat([base, metrics], axis=1)
    result = result[OUTPUT_COLUMNS]
    logger.info(
        "Built %s rows for %s.",
        len(result),
        result["ticker"].iloc[0] if not result.empty else "ticker",
    )
    return result


def build_all(monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build indicator outputs for each ticker in the configured order."""
    frames: dict[str, pd.DataFrame] = {}
    for ticker in EXPECTED_TICKERS:
        subset = (
            monthly[monthly["ticker"] == ticker]
            .sort_values("month", kind="stable")
            .reset_index(drop=True)
        )
        if subset.empty:
            raise ValueError(f"No monthly rows available for ticker {ticker}.")
        frames[ticker] = build_ticker_frame(subset)
    return frames


def run_pipeline(input_path: Path | str, out_dir: Path | str) -> list[Path]:
    """Run the full pipeline from raw file to output CSVs."""
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    logger.info("Starting pipeline for %s.", input_path)

    daily = load_daily(input_path)
    logger.info("Loaded %s rows across %s tickers.", len(daily), daily["ticker"].nunique())

    monthly = to_monthly(daily)
    logger.info("Resampled to %s monthly rows.", len(monthly))

    frames = build_all(monthly)
    validate_output(frames)
    written = write_results(frames, out_dir)
    logger.info("Pipeline completed successfully; wrote %s files.", len(written))
    return written
