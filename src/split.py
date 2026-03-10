"""Train/validation/test splitting utilities for reproducible experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import (
        RANDOM_STATE,
        SPLIT_METADATA_FILE,
        TARGET_COLUMN,
        TEST_SIZE,
        VALIDATION_SIZE,
    )
    from .data import validate_expected_columns
except ImportError:  # pragma: no cover - supports running as a script
    from config import (  # type: ignore
        RANDOM_STATE,
        SPLIT_METADATA_FILE,
        TARGET_COLUMN,
        TEST_SIZE,
        VALIDATION_SIZE,
    )
    from data import validate_expected_columns  # type: ignore


@dataclass
class TrainValTestSplit:
    """Container for reproducible train/validation/test subsets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from a cleaned dataset."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' is missing from the dataset.")

    features = df.drop(columns=[target_column]).copy()
    target = df[target_column].copy()
    return features, target


def create_train_val_test_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainValTestSplit:
    """Create reproducible train/validation/test splits with fixed random_state."""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")
    if val_size <= 0 or val_size >= 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    validate_expected_columns(df)
    X, y = split_features_target(df, target_column=target_column)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    val_fraction_in_train_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_in_train_val,
        random_state=random_state,
        shuffle=True,
    )

    return TrainValTestSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_split_metadata(
    split: TrainValTestSplit,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
) -> dict[str, Any]:
    """Build compact split metadata for reproducibility records."""
    total_rows = len(split.X_train) + len(split.X_val) + len(split.X_test)
    return {
        "random_state": random_state,
        "test_size": test_size,
        "validation_size": val_size,
        "train_rows": int(len(split.X_train)),
        "validation_rows": int(len(split.X_val)),
        "test_rows": int(len(split.X_test)),
        "total_rows": int(total_rows),
        "feature_count": int(split.X_train.shape[1]),
    }


def save_split_metadata(
    metadata: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Save split metadata as JSON."""
    output_path = path or SPLIT_METADATA_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def load_split_metadata(path: Path | None = None) -> dict[str, Any]:
    """Load split metadata from JSON."""
    input_path = path or SPLIT_METADATA_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))
