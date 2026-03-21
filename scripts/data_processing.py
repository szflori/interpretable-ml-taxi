from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_COLUMNS_TO_KEEP, DEFAULT_NUMERIC_COLUMNS, DEFAULT_RENAME_MAP

LOGGER = logging.getLogger(__name__)


def validate_data_file(path: str | Path) -> Path:
    """Validate parquet path and return normalized Path object."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path.resolve()}")
    if file_path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a parquet file, got: {file_path.name}")
    return file_path


def load_single_month(
    parquet_path: str | Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single parquet file into a pandas DataFrame."""
    file_path = validate_data_file(parquet_path)
    LOGGER.info("Loading parquet file: %s", file_path)
    df = pd.read_parquet(file_path, columns=columns)
    LOGGER.info("Loaded %s rows, %s columns", f"{len(df):,}", df.shape[1])
    return df


def keep_and_rename_columns(
    df: pd.DataFrame,
    columns_to_keep: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Select requested columns (if present) and rename them."""
    selected_columns = columns_to_keep or DEFAULT_COLUMNS_TO_KEEP
    selected_rename_map = rename_map or DEFAULT_RENAME_MAP

    available_columns = [column for column in selected_columns if column in df.columns]
    if not available_columns:
        raise ValueError("None of the requested columns exist in DataFrame.")

    LOGGER.info(
        "Selecting %d/%d requested columns",
        len(available_columns),
        len(selected_columns),
    )
    out = df[available_columns].copy()
    out = out.rename(columns=selected_rename_map)
    return out


def normalize_dtypes(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
    datetime_columns: tuple[str, str] = ("pickup_at", "dropoff_at"),
) -> pd.DataFrame:
    """Normalize datetime and numeric column types in a safe way."""
    out = df.copy()
    numeric_targets = numeric_columns or DEFAULT_NUMERIC_COLUMNS

    for column in datetime_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")

    for column in numeric_targets:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    return out


def basic_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Generate basic quality metrics for a DataFrame."""
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    duplicate_rows = int(df.duplicated().sum())
    report: dict[str, Any] = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": duplicate_rows,
        "missing_pct": missing_pct,
    }
    if {"pickup_at", "dropoff_at"}.issubset(df.columns):
        report["pickup_min"] = df["pickup_at"].min()
        report["pickup_max"] = df["pickup_at"].max()
    return report


def infer_feature_roles(
    df: pd.DataFrame,
    max_unique_for_discrete: int = 200,
    int_like_sample_size: int = 50_000,
) -> dict[str, list[str]]:
    """Classify columns into datetime, categorical, discrete, continuous."""
    roles: dict[str, list[str]] = {
        "datetime": [],
        "categorical": [],
        "discrete": [],
        "continuous": [],
    }

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            roles["datetime"].append(column)
            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            roles["categorical"].append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            unique_count = non_null.nunique()
            is_int_like = pd.api.types.is_integer_dtype(series)

            if (not is_int_like) and len(non_null) > 0:
                sample = non_null.iloc[:int_like_sample_size].to_numpy()
                is_int_like = bool(np.all(np.isclose(sample, np.round(sample))))

            if is_int_like and unique_count <= max_unique_for_discrete:
                roles["discrete"].append(column)
            else:
                roles["continuous"].append(column)
            continue

        roles["categorical"].append(column)

    return roles


def preprocess_trips(
    df: pd.DataFrame,
    passenger_range: tuple[int, int] = (1, 8),
    distance_range: tuple[float, float] = (0.0, 200.0),
    duration_range_min: tuple[float, float] = (1.0, 300.0),
) -> pd.DataFrame:
    """Apply simple domain filters and engineer time-based features."""
    out = df.copy()

    if {"pickup_at", "dropoff_at"}.issubset(out.columns):
        out = out[
            out["pickup_at"].notna()
            & out["dropoff_at"].notna()
            & (out["dropoff_at"] >= out["pickup_at"])
        ]

    if "passenger_count" in out.columns:
        out = out[out["passenger_count"].between(*passenger_range)]

    if "trip_distance" in out.columns:
        out = out[out["trip_distance"].between(*distance_range)]

    if {"pickup_at", "dropoff_at"}.issubset(out.columns):
        out["trip_duration_min"] = (
            (out["dropoff_at"] - out["pickup_at"]).dt.total_seconds() / 60.0
        )
        out = out[out["trip_duration_min"].between(*duration_range_min)]

    if "pickup_at" in out.columns:
        out["pickup_hour"] = out["pickup_at"].dt.hour
        out["pickup_weekday"] = out["pickup_at"].dt.weekday
        out["pickup_month"] = out["pickup_at"].dt.month

    return out


def cast_model_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast cleaned data to memory-friendly dtypes for model-ready use."""
    out = df.copy()

    discrete_columns = [
        column
        for column in ["passenger_count", "pickup_hour", "pickup_weekday", "pickup_month"]
        if column in out.columns
    ]
    for column in discrete_columns:
        out[column] = out[column].round().astype("Int32")

    continuous_columns = [
        column
        for column in [
            "trip_distance",
            "pickup_lon",
            "pickup_lat",
            "dropoff_lon",
            "dropoff_lat",
            "trip_duration_min",
        ]
        if column in out.columns
    ]
    for column in continuous_columns:
        out[column] = out[column].astype("float32")

    return out


def prepare_single_month_dataset(
    parquet_path: str | Path,
    columns_to_keep: list[str] | None = None,
    rename_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    """Run loading, normalization, report, cleaning, and final casting for one file."""
    raw_df = load_single_month(parquet_path, columns=columns_to_keep)
    trips_df = keep_and_rename_columns(
        raw_df,
        columns_to_keep=columns_to_keep,
        rename_map=rename_map,
    )
    trips_df = normalize_dtypes(trips_df)

    roles = infer_feature_roles(trips_df)
    report = basic_quality_report(trips_df)

    clean_df = preprocess_trips(trips_df)
    final_df = cast_model_dtypes(clean_df)
    return trips_df, final_df, roles, report
