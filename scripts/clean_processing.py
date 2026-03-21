from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

from .config import DEFAULT_CLEAN_COLUMNS, DEFAULT_COLUMNS_TO_KEEP, DEFAULT_RENAME_MAP
from .data_cleaning import clean_missing_and_duplicates
from .data_processing import keep_and_rename_columns, normalize_dtypes, validate_data_file
from .file_processing import list_parquet_files

LOGGER = logging.getLogger(__name__)


def _intersect_subset(df: pd.DataFrame, subset: list[str] | None) -> list[str] | None:
    if subset is None:
        return None
    valid = [column for column in subset if column in df.columns]
    return valid if valid else None


def clean_single_parquet_file(
    input_path: str | Path,
    output_path: str | Path,
    columns_to_keep: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    numeric_columns: list[str] | None = None,
    na_subset: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    batch_size: int = 250_000,
    compression: str = "snappy",
    use_threads: bool = True,
    log_every_n_batches: int = 10,
) -> dict[str, Any]:
    """Stream-clean one parquet file and write the cleaned output parquet."""
    input_file = validate_data_file(input_path)
    output_file = Path(output_path)

    selected_columns = columns_to_keep or DEFAULT_COLUMNS_TO_KEEP
    selected_rename_map = rename_map or DEFAULT_RENAME_MAP
    selected_numeric_columns = numeric_columns
    selected_na_subset = na_subset or DEFAULT_CLEAN_COLUMNS
    selected_duplicate_subset = duplicate_subset or DEFAULT_CLEAN_COLUMNS

    output_file.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(input_file)

    LOGGER.info("Start cleaning file: %s", input_file.name)
    LOGGER.info(
        "Settings | batch_size=%d, compression=%s, use_threads=%s",
        batch_size,
        compression,
        use_threads,
    )

    writer = None
    batch_index = 0
    initial_rows = 0
    after_nan_drop_rows = 0
    final_rows = 0

    try:
        for record_batch in parquet_file.iter_batches(
            columns=selected_columns,
            batch_size=batch_size,
            use_threads=use_threads,
        ):
            batch_index += 1
            chunk_df = record_batch.to_pandas(split_blocks=True, self_destruct=True)

            chunk_df = keep_and_rename_columns(
                chunk_df,
                columns_to_keep=selected_columns,
                rename_map=selected_rename_map,
            )
            chunk_df = normalize_dtypes(
                chunk_df,
                numeric_columns=selected_numeric_columns,
            )

            current_na_subset = _intersect_subset(chunk_df, selected_na_subset)
            current_duplicate_subset = _intersect_subset(chunk_df, selected_duplicate_subset)

            cleaned_chunk_df, chunk_stats = clean_missing_and_duplicates(
                chunk_df,
                na_subset=current_na_subset,
                duplicate_subset=current_duplicate_subset,
            )

            initial_rows += int(chunk_stats["initial_rows"])
            after_nan_drop_rows += int(chunk_stats["after_nan_drop_rows"])
            final_rows += int(chunk_stats["final_rows"])

            if not cleaned_chunk_df.empty:
                table = pa.Table.from_pandas(cleaned_chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        where=str(output_file),
                        schema=table.schema,
                        compression=compression,
                    )
                writer_obj = writer
                if writer_obj is None:
                    raise RuntimeError("Parquet writer initialization failed.")
                writer_obj.write_table(table)

            if batch_index == 1 or batch_index % log_every_n_batches == 0:
                LOGGER.info(
                    "File %s | batch=%d | rows_in=%s | rows_out=%s",
                    input_file.name,
                    batch_index,
                    f"{initial_rows:,}",
                    f"{final_rows:,}",
                )

            del chunk_df, cleaned_chunk_df
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pd.DataFrame(columns=DEFAULT_CLEAN_COLUMNS).to_parquet(output_file, index=False)

    result = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "initial_rows": initial_rows,
        "after_nan_drop_rows": after_nan_drop_rows,
        "final_rows": final_rows,
        "nan_removed_rows": initial_rows - after_nan_drop_rows,
        "duplicate_removed_rows": after_nan_drop_rows - final_rows,
        "total_removed_rows": initial_rows - final_rows,
        "na_subset": selected_na_subset,
        "duplicate_subset": selected_duplicate_subset,
        "batch_size": batch_size,
    }

    LOGGER.info(
        "Finished %s | initial=%s | final=%s | removed=%s",
        input_file.name,
        f"{initial_rows:,}",
        f"{final_rows:,}",
        f"{result['total_removed_rows']:,}",
    )
    return result


def clean_list_parquet_files(
    input_paths: Sequence[str | Path],
    clean_dir: str | Path,
    columns_to_keep: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    numeric_columns: list[str] | None = None,
    na_subset: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    batch_size: int = 250_000,
    compression: str = "snappy",
    use_threads: bool = True,
    log_every_n_batches: int = 10,
) -> pd.DataFrame:
    """Clean a list of parquet files and return file-level summary DataFrame."""
    if not input_paths:
        LOGGER.warning("No input parquet files provided.")
        return pd.DataFrame()

    clean_folder = Path(clean_dir)
    clean_folder.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting list cleaning for %d files", len(input_paths))

    records: list[dict[str, Any]] = []
    for file_index, input_item in enumerate(input_paths, start=1):
        input_file = Path(input_item)
        output_file = clean_folder / input_file.name

        LOGGER.info("[%d/%d] Processing %s", file_index, len(input_paths), input_file.name)
        stats = clean_single_parquet_file(
            input_path=input_file,
            output_path=output_file,
            columns_to_keep=columns_to_keep,
            rename_map=rename_map,
            numeric_columns=numeric_columns,
            na_subset=na_subset,
            duplicate_subset=duplicate_subset,
            batch_size=batch_size,
            compression=compression,
            use_threads=use_threads,
            log_every_n_batches=log_every_n_batches,
        )
        records.append(stats)

    summary_df = pd.DataFrame(records)
    summary_df["removed_pct"] = (
        summary_df["total_removed_rows"] / summary_df["initial_rows"]
    ).fillna(0.0)

    LOGGER.info(
        "Finished list cleaning | files=%d | input_rows=%s | output_rows=%s",
        len(summary_df),
        f"{int(summary_df['initial_rows'].sum()):,}",
        f"{int(summary_df['final_rows'].sum()):,}",
    )
    return summary_df


def clean_parquet_folder(
    raw_dir: str | Path,
    clean_dir: str | Path,
    pattern: str = "*.parquet",
    columns_to_keep: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
    numeric_columns: list[str] | None = None,
    na_subset: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    batch_size: int = 250_000,
    compression: str = "snappy",
    use_threads: bool = True,
    log_every_n_batches: int = 10,
) -> pd.DataFrame:
    """Clean every parquet file from a folder and return file-level summary."""
    input_files = list_parquet_files(raw_dir, pattern=pattern)
    return clean_list_parquet_files(
        input_paths=input_files,
        clean_dir=clean_dir,
        columns_to_keep=columns_to_keep,
        rename_map=rename_map,
        numeric_columns=numeric_columns,
        na_subset=na_subset,
        duplicate_subset=duplicate_subset,
        batch_size=batch_size,
        compression=compression,
        use_threads=use_threads,
        log_every_n_batches=log_every_n_batches,
    )
