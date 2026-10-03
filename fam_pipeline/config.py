from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "input.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

EXPECTED_TICKERS = (
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "CSCO",
    "MSFT",
    "NFLX",
    "PEP",
    "TMUS",
    "TSLA",
)
EXPECTED_MONTHS = 24
EXPECTED_MONTH_START = "2018-01"
EXPECTED_MONTH_END = "2019-12"
SMA_WINDOWS = (10, 20)
EMA_WINDOWS = (10, 20)
BB_WINDOW = 20
BB_K = 2.0
DONCHIAN_WINDOW = 20
ZSCORE_WINDOW = 20
INDICATOR_WINDOWS = {
    "sma_10": 10,
    "ema_10": 10,
    "sma_20": 20,
    "ema_20": 20,
    "donchian_upper_20": 20,
    "donchian_lower_20": 20,
    "donchian_mid_20": 20,
    "bb_mid_20": 20,
    "bb_upper_20": 20,
    "bb_lower_20": 20,
    "zscore_20": 20,
}
FLOAT_FORMAT = "%.4f"
REQUIRED_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adjclose", "ticker"]
OUTPUT_COLUMNS = [
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
]
