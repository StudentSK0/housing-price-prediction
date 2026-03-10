"""Minimal single-sample inference utility for the frozen final model."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from .config import FINAL_MODEL_ARTIFACT_FILE, REQUIRED_COLUMNS, TARGET_COLUMN
except ImportError:  # pragma: no cover - supports running as a script
    from config import FINAL_MODEL_ARTIFACT_FILE, REQUIRED_COLUMNS, TARGET_COLUMN  # type: ignore


REQUIRED_FEATURE_COLUMNS = tuple(column for column in REQUIRED_COLUMNS if column != TARGET_COLUMN)


def load_final_model(path: Path | None = None) -> Pipeline:
    """Load the saved final sklearn pipeline artifact."""
    model_path = path or FINAL_MODEL_ARTIFACT_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Final model artifact not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError("Loaded final artifact is not a sklearn Pipeline.")
    return model


def prepare_single_sample(sample: Mapping[str, float] | pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize one input sample into a 1-row DataFrame."""
    if isinstance(sample, pd.DataFrame):
        if len(sample) != 1:
            raise ValueError("Input DataFrame must contain exactly one row for single-sample inference.")
        frame = sample.copy()
    else:
        frame = pd.DataFrame([dict(sample)])

    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input sample is missing required feature columns: {missing}")

    return frame.loc[:, list(REQUIRED_FEATURE_COLUMNS)].copy()


def predict_single_sample(
    sample: Mapping[str, float] | pd.DataFrame,
    model_path: Path | None = None,
) -> float:
    """Run inference for one California Housing sample and return predicted value."""
    model = load_final_model(path=model_path)
    sample_frame = prepare_single_sample(sample)
    prediction = model.predict(sample_frame)
    return float(prediction[0])
