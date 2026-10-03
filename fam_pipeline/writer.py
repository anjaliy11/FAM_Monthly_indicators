from __future__ import annotations

import logging
import os
from pathlib import Path

from fam_pipeline.config import EXPECTED_TICKERS, FLOAT_FORMAT

logger = logging.getLogger(__name__)


def write_results(frames: dict[str, object], out_dir: Path | str) -> list[Path]:
    """Write each ticker result to disk atomically and clean up temporary files on failure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for temp_file in out_dir.glob("*.tmp"):
        temp_file.unlink(missing_ok=True)
    for backup_file in out_dir.glob("result_*.csv.bak"):
        backup_file.unlink(missing_ok=True)

    staged_paths: list[tuple[Path, Path, int]] = []
    committed_paths: list[tuple[Path, Path | None]] = []
    try:
        for ticker in EXPECTED_TICKERS:
            frame = frames[ticker].copy()
            target = out_dir / f"result_{ticker}.csv"
            temp_path = out_dir / f"result_{ticker}.csv.tmp"
            frame.to_csv(temp_path, index=False, float_format=FLOAT_FORMAT, na_rep="")
            staged_paths.append((temp_path, target, len(frame)))

        written_paths: list[Path] = []
        for temp_path, target, row_count in staged_paths:
            backup_path = target.with_name(f"{target.name}.bak") if target.exists() else None
            if backup_path is not None:
                os.replace(target, backup_path)
            committed_paths.append((target, backup_path))
            os.replace(temp_path, target)
            written_paths.append(target)
            logger.info("Wrote %s with %s rows.", target, row_count)
    except Exception:
        for target, backup_path in reversed(committed_paths):
            if backup_path is not None and backup_path.exists():
                target.unlink(missing_ok=True)
                os.replace(backup_path, target)
            else:
                target.unlink(missing_ok=True)
        for temp_file in out_dir.glob("*.tmp"):
            temp_file.unlink(missing_ok=True)
        raise

    for _, backup_path in committed_paths:
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)

    return written_paths
