from __future__ import annotations

import logging
from typing import Any, Literal, cast

import pandas as pd

LOGGER = logging.getLogger(__name__)


def drop_nan_null_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    how: Literal["any", "all"] = "any",
) -> pd.DataFrame:
    """Drop rows containing NaN/null values."""
    return df.dropna(axis=0, subset=subset, how=cast(Any, how))


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: Literal["first", "last", False] = "first",
) -> pd.DataFrame:
    """Drop duplicated rows."""
    return df.drop_duplicates(subset=subset, keep=cast(Any, keep))


def clean_missing_and_duplicates(
    df: pd.DataFrame,
    na_subset: list[str] | None = None,
    duplicate_subset: list[str] | None = None,
    na_how: Literal["any", "all"] = "any",
    duplicate_keep: Literal["first", "last", False] = "first",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove NaN/null and duplicated rows in two explicit steps.

    Returns:
        cleaned_df, stats
    """
    initial_rows = len(df)

    no_nan_df = drop_nan_null_rows(df, subset=na_subset, how=na_how)
    after_nan_rows = len(no_nan_df)

    cleaned_df = drop_duplicate_rows(
        no_nan_df,
        subset=duplicate_subset,
        keep=duplicate_keep,
    )
    final_rows = len(cleaned_df)

    stats: dict[str, Any] = {
        "initial_rows": initial_rows,
        "after_nan_drop_rows": after_nan_rows,
        "final_rows": final_rows,
        "nan_removed_rows": initial_rows - after_nan_rows,
        "duplicate_removed_rows": after_nan_rows - final_rows,
        "total_removed_rows": initial_rows - final_rows,
        "na_subset": na_subset,
        "duplicate_subset": duplicate_subset,
    }

    LOGGER.debug(
        "Clean step | input=%s | after_nan=%s | final=%s",
        f"{initial_rows:,}",
        f"{after_nan_rows:,}",
        f"{final_rows:,}",
    )

    return cleaned_df, stats
