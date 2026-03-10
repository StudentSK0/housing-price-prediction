"""Reusable data loading utilities for the California Housing project."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

try:
    from .config import (
        BASELINE_RESULTS_FILE,
        DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_FILE,
        PROCESSED_DATA_DIR,
        RAW_DATA_FILE,
        RAW_DATA_DIR,
        REQUIRED_COLUMNS,
        REPORTS_DIR,
        SKLEARN_DATA_HOME,
        SPLIT_METADATA_FILE,
        STAGE4_ERROR_ANALYSIS_FILE,
        TARGET_COLUMN,
        TUNED_RESULTS_FILE,
        TUNING_SUMMARY_FILE,
    )
except ImportError:  # pragma: no cover - supports running as a script
    from config import (  # type: ignore
        BASELINE_RESULTS_FILE,
        DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_FILE,
        PROCESSED_DATA_DIR,
        RAW_DATA_FILE,
        RAW_DATA_DIR,
        REQUIRED_COLUMNS,
        REPORTS_DIR,
        SKLEARN_DATA_HOME,
        SPLIT_METADATA_FILE,
        STAGE4_ERROR_ANALYSIS_FILE,
        TARGET_COLUMN,
        TUNED_RESULTS_FILE,
        TUNING_SUMMARY_FILE,
    )


def ensure_project_directories() -> None:
    """Create required project directories if they do not exist."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SKLEARN_DATA_HOME,
        REPORTS_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_california_housing_dataframe() -> pd.DataFrame:
    """Load the California Housing dataset as a pandas DataFrame."""
    dataset = fetch_california_housing(
        as_frame=True,
        data_home=str(SKLEARN_DATA_HOME),
    )
    return dataset.frame.copy()


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix and target series."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' is not present in the DataFrame.")

    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()
    return features, target


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw housing dataset from local CSV storage."""
    input_path = path or RAW_DATA_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {input_path}")
    return pd.read_csv(input_path)


def load_processed_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned housing dataset from local CSV storage."""
    input_path = path or PROCESSED_DATA_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {input_path}")
    return pd.read_csv(input_path)


def save_processed_dataset(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save processed dataset to CSV and return the output path."""
    ensure_project_directories()
    output_path = path or PROCESSED_DATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def validate_expected_columns(df: pd.DataFrame) -> None:
    """Validate that all expected columns are present in the dataset."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def save_report_dataframe(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save a reporting DataFrame to CSV."""
    ensure_project_directories()
    output_path = path or BASELINE_RESULTS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_json_report(payload: dict, path: Path) -> Path:
    """Save a dictionary payload to a JSON report file."""
    ensure_project_directories()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def get_tuned_results_path() -> Path:
    """Return the default Stage 4 tuned results path."""
    return TUNED_RESULTS_FILE


def get_tuning_summary_path() -> Path:
    """Return the default Stage 4 tuning summary path."""
    return TUNING_SUMMARY_FILE


def get_stage4_error_analysis_path() -> Path:
    """Return the default Stage 4 error-analysis output path."""
    return STAGE4_ERROR_ANALYSIS_FILE


def get_split_metadata_path() -> Path:
    """Return the default Stage 3 split metadata path."""
    return SPLIT_METADATA_FILE


def save_raw_dataset(path: Path | None = None) -> Path:
    """Save the raw California Housing dataset to CSV and return the file path."""
    ensure_project_directories()

    output_path = path or RAW_DATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_california_housing_dataframe()
    df.to_csv(output_path, index=False)
    return output_path


def load_features_and_target() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return features and target separately."""
    df = load_california_housing_dataframe()
    return split_features_and_target(df)


if __name__ == "__main__":
    saved_path = save_raw_dataset()
    print(f"Raw dataset saved to: {saved_path}")
