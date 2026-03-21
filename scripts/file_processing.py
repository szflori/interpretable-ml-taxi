from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def list_parquet_files(input_dir: str | Path, pattern: str = "*.parquet") -> list[Path]:
    """Return sorted parquet files from a directory."""
    folder = Path(input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder.resolve()}")
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder.resolve()}")
    LOGGER.info("Found %d parquet files in %s", len(files), folder)
    return files

