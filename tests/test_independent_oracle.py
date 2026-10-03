from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "input.csv"
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


def _monthly_oracle(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = daily.copy()
    ordered["ticker"] = ordered["ticker"].astype(str).str.strip().str.upper()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="raise")
    ordered = ordered.sort_values(["ticker", "date"], kind="stable")
    ordered["month"] = ordered["date"].dt.to_period("M").astype(str)

    groups: dict[str, list[dict[str, object]]] = {}
    for (ticker, month), daily_month in ordered.groupby(["ticker", "month"], sort=True):
        daily_month = daily_month.sort_values("date", kind="stable")
        groups.setdefault(ticker, []).append(
            {
                "ticker": ticker,
                "month": month,
                "open": float(daily_month["open"].iloc[0]),
                "high": float(daily_month["high"].max()),
                "low": float(daily_month["low"].min()),
                "close": float(daily_month["close"].iloc[-1]),
                "volume": int(daily_month["volume"].sum()),
            }
        )

    results: dict[str, pd.DataFrame] = {}
    for ticker, rows in groups.items():
        closes = [float(row["close"]) for row in rows]
        highs = [float(row["high"]) for row in rows]
        lows = [float(row["low"]) for row in rows]
        length = len(rows)

        for window in (10, 20):
            sma_values = [math.nan] * length
            ema_values = [math.nan] * length
            if length >= window:
                for index in range(window - 1, length):
                    sma_values[index] = sum(closes[index - window + 1 : index + 1]) / window
                ema_values[window - 1] = sum(closes[:window]) / window
                multiplier = 2 / (window + 1)
                for index in range(window, length):
                    prior = ema_values[index - 1]
                    ema_values[index] = (closes[index] - prior) * multiplier + prior
            for index in range(length):
                rows[index][f"sma_{window}"] = sma_values[index]
                rows[index][f"ema_{window}"] = ema_values[index]

        for index in range(length):
            if index < 19:
                for column in (
                    "donchian_upper_20",
                    "donchian_lower_20",
                    "donchian_mid_20",
                    "bb_mid_20",
                    "bb_upper_20",
                    "bb_lower_20",
                    "zscore_20",
                ):
                    rows[index][column] = math.nan
                continue

            start = index - 19
            close_window = closes[start : index + 1]
            high_window = highs[start : index + 1]
            low_window = lows[start : index + 1]
            mean = sum(close_window) / 20
            variance = sum((value - mean) ** 2 for value in close_window) / 19
            standard_deviation = math.sqrt(variance)
            upper = max(high_window)
            lower = min(low_window)
            rows[index]["donchian_upper_20"] = upper
            rows[index]["donchian_lower_20"] = lower
            rows[index]["donchian_mid_20"] = (upper + lower) / 2
            rows[index]["bb_mid_20"] = mean
            rows[index]["bb_upper_20"] = mean + 2 * standard_deviation
            rows[index]["bb_lower_20"] = mean - 2 * standard_deviation
            rows[index]["zscore_20"] = (
                (closes[index] - mean) / standard_deviation if standard_deviation else math.nan
            )

        results[ticker] = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return results


def test_all_outputs_match_independent_loop_oracle(tmp_path: Path) -> None:
    output_dir = tmp_path / "oracle-output"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--input",
        str(INPUT_PATH),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    expected_by_ticker = _monthly_oracle(pd.read_csv(INPUT_PATH))
    max_differences = {column: 0.0 for column in OUTPUT_COLUMNS}
    assert len(list(output_dir.glob("result_*.csv"))) == 10

    for ticker, expected in expected_by_ticker.items():
        output_path = output_dir / f"result_{ticker}.csv"
        actual = pd.read_csv(output_path)
        assert list(actual.columns) == OUTPUT_COLUMNS
        assert len(actual) == 24
        assert actual["ticker"].tolist() == [ticker] * 24
        assert actual["month"].tolist() == expected["month"].tolist()

        for column in OUTPUT_COLUMNS:
            if column in ("ticker", "month"):
                continue
            actual_values = actual[column].to_numpy(dtype=float)
            expected_values = expected[column].to_numpy(dtype=float)
            actual_nan = np.isnan(actual_values)
            expected_nan = np.isnan(expected_values)
            np.testing.assert_array_equal(
                actual_nan, expected_nan, err_msg=f"{ticker}.{column} NaN positions"
            )
            differences = np.abs(actual_values[~actual_nan] - expected_values[~expected_nan])
            if differences.size:
                max_differences[column] = max(max_differences[column], float(differences.max()))
            assert np.allclose(actual_values, expected_values, atol=1e-4, rtol=0, equal_nan=True), (
                f"{ticker}.{column} differs from independent oracle"
            )

    print("MAX_ABS_DIFF_BY_COLUMN")
    for column, difference in max_differences.items():
        print(f"{column}={difference:.10g}")
