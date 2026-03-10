"""Stage 5 finalization helpers for frozen model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

try:
    from .config import (
        FINAL_ADD_STAGE4_FEATURES,
        FINAL_MODEL_ARTIFACT_FILE,
        FINAL_MODEL_FAMILY,
        FINAL_MODEL_LABEL,
        FINAL_MODEL_PARAMS,
        RANDOM_STATE,
        TUNED_RESULTS_FILE,
    )
    from .modeling import build_model_pipeline
except ImportError:  # pragma: no cover - supports running as a script
    from config import (  # type: ignore
        FINAL_ADD_STAGE4_FEATURES,
        FINAL_MODEL_ARTIFACT_FILE,
        FINAL_MODEL_FAMILY,
        FINAL_MODEL_LABEL,
        FINAL_MODEL_PARAMS,
        RANDOM_STATE,
        TUNED_RESULTS_FILE,
    )
    from modeling import build_model_pipeline  # type: ignore


@dataclass(frozen=True)
class FrozenFinalModelConfig:
    """Frozen final model configuration selected in Stage 4."""

    model_label: str
    model_family: str
    model_params: dict[str, Any]
    add_stage4_features: bool


def get_frozen_final_model_config() -> FrozenFinalModelConfig:
    """Return the frozen final model configuration."""
    return FrozenFinalModelConfig(
        model_label=FINAL_MODEL_LABEL,
        model_family=FINAL_MODEL_FAMILY,
        model_params=dict(FINAL_MODEL_PARAMS),
        add_stage4_features=FINAL_ADD_STAGE4_FEATURES,
    )


def load_stage4_validation_reference(
    path: Path | None = None,
    model_label: str = FINAL_MODEL_LABEL,
) -> dict[str, float]:
    """Load Stage 4 validation metrics for the frozen model label."""
    input_path = path or TUNED_RESULTS_FILE
    if not input_path.exists():
        raise FileNotFoundError(f"Stage 4 tuned results file not found: {input_path}")

    results = pd.read_csv(input_path)
    row = results.loc[results["Model"] == model_label]
    if row.empty:
        raise ValueError(f"Frozen model label '{model_label}' not found in {input_path}.")

    record = row.iloc[0]
    return {
        "RMSE_val": float(record["RMSE_val"]),
        "MAE_val": float(record["MAE_val"]),
        "R2_val": float(record["R2_val"]),
    }


def build_frozen_final_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the frozen final pipeline without any new model decisions."""
    estimator = GradientBoostingRegressor(
        random_state=random_state,
        **FINAL_MODEL_PARAMS,
    )
    return build_model_pipeline(
        estimator=estimator,
        scale_numeric=False,
        include_feature_engineering=True,
        feature_engineering_params={"add_stage4_features": FINAL_ADD_STAGE4_FEATURES},
    )


def make_final_dev_set(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine train and validation subsets into a final development set."""
    X_dev = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_dev = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    return X_dev, y_dev


def save_final_model_artifact(
    pipeline: Pipeline,
    path: Path | None = None,
) -> Path:
    """Persist the final fitted pipeline using joblib."""
    output_path = path or FINAL_MODEL_ARTIFACT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def extract_feature_importance(
    fitted_pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract normalized feature importances from fitted GradientBoosting model."""
    preprocess = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    frame["importance_norm"] = frame["importance"] / frame["importance"].sum()
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
