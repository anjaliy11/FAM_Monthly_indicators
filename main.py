from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from fam_pipeline.config import INPUT_PATH, OUTPUT_DIR
from fam_pipeline.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly OHLC and indicator outputs for each ticker."
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for result CSV output."
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    start = time.perf_counter()
    try:
        args = parse_args()
        run_pipeline(args.input, args.output_dir)
        logger.info("Pipeline finished successfully.")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    finally:
        logger.info("Total runtime: %.3f seconds.", time.perf_counter() - start)


if __name__ == "__main__":
    raise SystemExit(main())
