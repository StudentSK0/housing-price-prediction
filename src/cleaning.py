"""Data quality checks and conservative cleaning utilities for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .config import PROCESSED_DATA_FILE, REQUIRED_COLUMNS
    from .data import load_raw_dataset, save_processed_dataset, validate_expected_columns
except ImportError:  # pragma: no cover - supports running as a script
    from config import PROCESSED_DATA_FILE, REQUIRED_COLUMNS  # type: ignore
    from data import load_raw_dataset, save_processed_dataset, validate_expected_columns  # type: ignore


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped column names for stable downstream usage."""
    return df.rename(columns=lambda column: column.strip() if isinstance(column, str) else column)


def build_data_quality_report(
    df: pd.DataFrame,
    near_constant_threshold: float = 0.99,
) -> dict[str, Any]:
    """Build a compact report on core data quality indicators."""
    missing_by_column = df.isna().sum().sort_values(ascending=False)
    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]

    near_constant_columns: dict[str, float] = {}
    for column in df.columns:
        value_share = df[column].value_counts(dropna=False, normalize=True)
        if not value_share.empty and float(value_share.iloc[0]) >= near_constant_threshold:
            near_constant_columns[column] = float(value_share.iloc[0])

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(missing_by_column.sum()),
        "missing_by_column": missing_by_column.to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
    }


def _coerce_required_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns are numeric, coercing invalid values to NaN."""
    coerced = df.copy()
    for column in REQUIRED_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def clean_housing_data(
    df: pd.DataFrame,
    fill_missing: bool = True,
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply conservative cleaning for the housing dataset."""
    cleaned = normalize_column_names(df)
    validate_expected_columns(cleaned)

    rows_before = int(len(cleaned))
    cleaned = _coerce_required_columns_to_numeric(cleaned)

    duplicates_removed = 0
    if remove_duplicates:
        duplicates_removed = int(cleaned.duplicated().sum())
        if duplicates_removed:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    missing_before = cleaned.isna().sum()
    imputation_values: dict[str, float] = {}

    if fill_missing:
        numeric_columns = cleaned.select_dtypes(include="number").columns
        columns_with_missing = [column for column in numeric_columns if cleaned[column].isna().any()]
        if columns_with_missing:
            medians = cleaned[columns_with_missing].median()
            cleaned[columns_with_missing] = cleaned[columns_with_missing].fillna(medians)
            imputation_values = {column: float(medians[column]) for column in columns_with_missing}

    missing_after = cleaned.isna().sum()

    summary = {
        "rows_before": rows_before,
        "rows_after": int(len(cleaned)),
        "duplicates_removed": duplicates_removed,
        "missing_before_total": int(missing_before.sum()),
        "missing_after_total": int(missing_after.sum()),
        "imputed_numeric_columns": list(imputation_values.keys()),
        "imputation_values": imputation_values,
    }
    return cleaned, summary


def save_cleaned_housing_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save cleaned housing data to disk."""
    output_path = path or PROCESSED_DATA_FILE
    return save_processed_dataset(df, output_path)


def run_stage2_cleaning(
    raw_path: Path | None = None,
    processed_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run Stage 2 cleaning end-to-end using local CSV files."""
    raw_df = load_raw_dataset(raw_path)
    cleaned_df, summary = clean_housing_data(raw_df)
    output_path = save_cleaned_housing_data(cleaned_df, processed_path)
    summary["output_path"] = str(output_path)
    return output_path, summary


if __name__ == "__main__":
    final_path, cleaning_summary = run_stage2_cleaning()
    print(f"Cleaned dataset saved to: {final_path}")
    print(f"Rows before/after: {cleaning_summary['rows_before']} -> {cleaning_summary['rows_after']}")
    print(f"Duplicates removed: {cleaning_summary['duplicates_removed']}")
    print(
        "Missing values before/after: "
        f"{cleaning_summary['missing_before_total']} -> {cleaning_summary['missing_after_total']}"
    )
