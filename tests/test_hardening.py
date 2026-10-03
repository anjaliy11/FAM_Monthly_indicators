from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fam_pipeline import writer as writer_module
from fam_pipeline.config import EXPECTED_TICKERS, OUTPUT_COLUMNS
from fam_pipeline.indicators import bollinger, ema, sma, zscore
from fam_pipeline.loader import load_daily
from fam_pipeline.pipeline import build_all, run_pipeline
from fam_pipeline.resample import to_monthly
from fam_pipeline.validate import validate_output
from fam_pipeline.writer import write_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "input.csv"


def _valid_output_frames() -> dict[str, pd.DataFrame]:
    months = pd.period_range("2018-01", "2019-12", freq="M").astype(str).tolist()
    frames: dict[str, pd.DataFrame] = {}
    for ticker in EXPECTED_TICKERS:
        rows: list[dict[str, object]] = []
        for index, month in enumerate(months):
            close = 100.0 + index
            row: dict[str, object] = {
                "ticker": ticker,
                "month": month,
                "open": close,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1000,
            }
            for column in OUTPUT_COLUMNS[7:]:
                window = 10 if column in {"sma_10", "ema_10"} else 20
                row[column] = np.nan if index < window - 1 else close
            if index >= 19:
                row.update(
                    {
                        "donchian_upper_20": close + 2,
                        "donchian_lower_20": close - 2,
                        "donchian_mid_20": close,
                        "bb_mid_20": close,
                        "bb_upper_20": close + 2,
                        "bb_lower_20": close - 2,
                        "zscore_20": 0.0,
                    }
                )
            rows.append(row)
        frames[ticker] = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return frames


def test_shuffled_input_matches_sorted_monthly_and_final_output(tmp_path: Path) -> None:
    daily = pd.read_csv(INPUT_PATH)
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d")
    ordered = daily.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    shuffled = daily.sample(frac=1, random_state=20261003).reset_index(drop=True)
    expected_monthly = (
        ordered.set_index("date")
        .groupby("ticker")
        .resample("MS")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    expected_monthly["month"] = expected_monthly["date"].dt.to_period("M").astype(str)
    expected_monthly = expected_monthly[
        ["ticker", "month", "open", "high", "low", "close", "volume"]
    ]
    pd.testing.assert_frame_equal(to_monthly(shuffled), expected_monthly, check_dtype=False)
    pd.testing.assert_frame_equal(to_monthly(ordered), expected_monthly, check_dtype=False)

    sorted_path = tmp_path / "sorted.csv"
    shuffled_path = tmp_path / "shuffled.csv"
    ordered.to_csv(sorted_path, index=False)
    shuffled.to_csv(shuffled_path, index=False)
    sorted_output = tmp_path / "sorted-output"
    shuffled_output = tmp_path / "shuffled-output"
    run_pipeline(sorted_path, sorted_output)
    run_pipeline(shuffled_path, shuffled_output)
    for ticker in EXPECTED_TICKERS:
        assert (sorted_output / f"result_{ticker}.csv").read_bytes() == (
            shuffled_output / f"result_{ticker}.csv"
        ).read_bytes()


def test_ohlc_hand_built_two_months() -> None:
    daily = pd.DataFrame(
        {
            "ticker": ["ONE"] * 4,
            "date": pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"]),
            "open": [10.0, 12.0, 14.0, 13.0],
            "high": [11.0, 15.0, 16.0, 17.0],
            "low": [9.0, 11.0, 12.0, 12.0],
            "close": [12.0, 14.0, 13.0, 16.0],
            "volume": [100, 200, 300, 400],
        }
    )
    actual = to_monthly(daily)
    expected = pd.DataFrame(
        {
            "ticker": ["ONE", "ONE"],
            "month": ["2020-01", "2020-02"],
            "open": [10.0, 14.0],
            "high": [15.0, 17.0],
            "low": [9.0, 12.0],
            "close": [14.0, 16.0],
            "volume": [300, 700],
        }
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_sma_ema_hand_computation_and_seed() -> None:
    close = pd.Series([10.0, 20.0, 15.0, 30.0, 28.0])
    sma_values = sma(close, 3)
    ema_values = ema(close, 3)
    assert sma_values.iloc[:2].isna().all()
    assert sma_values.iloc[2] == pytest.approx(15.0)
    assert ema_values.iloc[:2].isna().all()
    assert ema_values.iloc[2] == pytest.approx(15.0)
    assert ema_values.iloc[3] == pytest.approx((30.0 - 15.0) * 0.5 + 15.0)
    assert ema_values.iloc[4] == pytest.approx((28.0 - 22.5) * 0.5 + 22.5)


def test_bollinger_zscore_identities_and_constant_prices() -> None:
    close = pd.Series([10.0, 12.0, 9.0, 15.0, 14.0, 18.0])
    bands = bollinger(close, 4, 2.0)
    z_values = zscore(close, 4)
    rolling_std = close.rolling(4).std(ddof=1)
    valid = bands["bb_mid_4"].notna()
    assert np.allclose(z_values[valid], ((close - bands["bb_mid_4"]) / rolling_std)[valid])
    assert np.allclose((bands["bb_upper_4"] - bands["bb_mid_4"])[valid], (2.0 * rolling_std)[valid])
    constant_z = zscore(pd.Series([5.0] * 6), 4)
    assert constant_z.isna().all()
    assert not np.isinf(constant_z.to_numpy()).any()


def test_stacked_tickers_have_independent_rolling_windows() -> None:
    months = pd.period_range("2018-01", periods=20, freq="M").astype(str)
    monthly = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "month": month,
                "open": base + index,
                "high": base + index + 2,
                "low": base + index - 2,
                "close": base + index,
                "volume": 100,
            }
            for ticker_index, ticker in enumerate(EXPECTED_TICKERS)
            for index, month in enumerate(months)
            for base in [100.0 + 1000 * ticker_index]
        ]
    )
    outputs = build_all(monthly)
    for ticker_index, ticker in enumerate(EXPECTED_TICKERS):
        frame = outputs[ticker]
        assert frame["sma_10"].first_valid_index() == 9
        assert frame["ema_10"].first_valid_index() == 9
        assert frame["sma_20"].first_valid_index() == 19
        assert frame.loc[9, "ema_10"] == pytest.approx(104.5 + 1000 * ticker_index)


def test_output_contract_and_no_extra_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    paths = run_pipeline(INPUT_PATH, output_dir)
    expected_names = {f"result_{ticker}.csv" for ticker in EXPECTED_TICKERS}
    assert {path.name for path in paths} == expected_names
    assert {path.name for path in output_dir.iterdir()} == expected_names
    for path in paths:
        frame = pd.read_csv(path)
        ticker = path.stem.removeprefix("result_")
        assert len(frame) == 24
        assert list(frame.columns) == OUTPUT_COLUMNS
        assert frame["ticker"].tolist() == [ticker] * 24


def test_pandas_resample_cross_check_and_month_edges() -> None:
    daily = pd.read_csv(INPUT_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"], kind="stable")
    expected = (
        daily.set_index("date")
        .groupby("ticker")
        .resample("MS")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    expected["month"] = expected["date"].dt.to_period("M").astype(str)
    expected = expected[["ticker", "month", "open", "high", "low", "close", "volume"]]
    actual = to_monthly(daily)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)

    raw = daily.assign(month=daily["date"].dt.to_period("M").astype(str))
    for (ticker, month), group in raw.groupby(["ticker", "month"], sort=True):
        result = actual[(actual["ticker"] == ticker) & (actual["month"] == month)].iloc[0]
        first = group.sort_values("date", kind="stable").iloc[0]
        last = group.sort_values("date", kind="stable").iloc[-1]
        assert result["open"] == first["open"]
        assert result["close"] == last["close"]


def test_atomic_writer_failure_leaves_no_results_or_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "failed-output"
    frames = {ticker: _valid_output_frames()[ticker] for ticker in EXPECTED_TICKERS}
    original_to_csv = pd.DataFrame.to_csv
    calls = 0

    def fail_on_second_write(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_on_second_write)
    with pytest.raises(OSError, match="simulated write failure"):
        write_results(frames, output_dir)
    assert list(output_dir.glob("result_*.csv")) == []
    assert list(output_dir.glob("*.tmp")) == []


def test_atomic_writer_commit_failure_rolls_back_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "commit-failure-output"
    output_dir.mkdir()
    prior_output = output_dir / "result_AAPL.csv"
    prior_output.write_text("previous complete output\n", encoding="utf-8")
    frames = _valid_output_frames()
    original_replace = os.replace
    calls = 0

    def fail_on_second_replace(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(writer_module.os, "replace", fail_on_second_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_results(frames, output_dir)
    assert list(output_dir.glob("result_*.csv")) == [prior_output]
    assert prior_output.read_text(encoding="utf-8") == "previous complete output\n"
    assert list(output_dir.glob("*.tmp")) == []
    assert list(output_dir.glob("*.bak")) == []


@pytest.mark.parametrize("invalid_case", ["duplicate", "missing-column"])
def test_main_returns_one_and_logs_invalid_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    invalid_case: str,
) -> None:
    daily = pd.read_csv(INPUT_PATH)
    if invalid_case == "duplicate":
        daily = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    else:
        daily = daily.drop(columns=["volume"])
    input_path = tmp_path / f"{invalid_case}.csv"
    daily.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--input", str(input_path), "--output-dir", str(tmp_path / "out")],
    )
    from main import main

    with caplog.at_level(logging.ERROR):
        assert main() == 1
    assert "Pipeline failed:" in caplog.text


def test_main_returns_zero_for_valid_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--input", str(INPUT_PATH), "--output-dir", str(tmp_path / "good-output")],
    )
    from main import main

    assert main() == 0
    assert len(list((tmp_path / "good-output").glob("result_*.csv"))) == 10


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("non-numeric-price", "close"),
        ("unparsable-date", "date"),
        ("non-positive-price", "close"),
        ("wrong-ticker-set", "ticker"),
    ],
)
def test_loader_rejects_invalid_input(tmp_path: Path, mutation: str, message: str) -> None:
    daily = pd.read_csv(INPUT_PATH)
    if mutation == "non-numeric-price":
        daily["close"] = daily["close"].astype(object)
        daily.loc[0, "close"] = "not-a-price"
    elif mutation == "unparsable-date":
        daily.loc[0, "date"] = "not-a-date"
    elif mutation == "non-positive-price":
        daily.loc[0, "close"] = 0
    else:
        daily.loc[daily.index[0], "ticker"] = "UNKNOWN"
    input_path = tmp_path / f"{mutation}.csv"
    daily.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match=message):
        load_daily(input_path)


def test_validate_accepts_valid_frames() -> None:
    validate_output(_valid_output_frames())


@pytest.mark.parametrize(
    ("violation", "column", "row", "value"),
    [
        ("month-range-gap", "month", 0, "2018-02"),
        ("duplicate-month", "month", 1, "2018-01"),
        ("descending-month", "month", 0, "2019-12"),
        ("ticker-mismatch", "ticker", 0, "OTHER"),
        ("infinity", "close", 0, np.inf),
        ("unexpected-base-nan", "open", 0, np.nan),
        ("unexpected-indicator-nan", "sma_10", 9, np.nan),
        ("high-below-open-close", "high", 0, 99.0),
        ("low-above-open-close", "low", 0, 101.0),
        ("high-below-low", "high", 0, -1.0),
        ("bollinger-order", "bb_upper_20", 19, 99.0),
        ("donchian-order", "donchian_mid_20", 19, 1000.0),
    ],
)
def test_validate_rejects_semantic_violations(
    violation: str, column: str, row: int, value: object
) -> None:
    frames = _valid_output_frames()
    frames[EXPECTED_TICKERS[0]].loc[row, column] = value
    with pytest.raises(ValueError, match="2018-01|2019-12|column|ticker"):
        validate_output(frames)
