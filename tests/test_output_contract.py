from __future__ import annotations

import hashlib
from pathlib import Path

from fam_pipeline.pipeline import run_pipeline


def test_run_pipeline_creates_deterministic_outputs(tmp_path: Path) -> None:
    out_1 = tmp_path / "out_1"
    out_2 = tmp_path / "out_2"

    paths_1 = run_pipeline(Path("data/input.csv"), out_1)
    paths_2 = run_pipeline(Path("data/input.csv"), out_2)

    assert len(paths_1) == 10
    assert sorted(p.name for p in paths_1) == sorted(f"result_{ticker}.csv" for ticker in [
        "AAPL", "AMD", "AMZN", "AVGO", "CSCO", "MSFT", "NFLX", "PEP", "TMUS", "TSLA"
    ])

    hashes_1 = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths_1]
    hashes_2 = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths_2]
    assert hashes_1 == hashes_2
