"""Evaluation and error-analysis utilities for Stage 3 baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


@dataclass
class EvaluationOutput:
    """Container for baseline model evaluation artifacts."""

    results: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    validation_predictions: dict[str, np.ndarray]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE_val": rmse, "MAE_val": mae, "R2_val": r2}


def evaluate_baseline_models(
    model_pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> EvaluationOutput:
    """Fit baseline models and evaluate on validation data."""
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    validation_predictions: dict[str, np.ndarray] = {}

    for model_name, pipeline in model_pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        fitted_models[model_name] = pipeline
        validation_predictions[model_name] = y_pred

        metrics = regression_metrics(y_val, y_pred)
        rows.append({"Model": model_name, **metrics})

    results = pd.DataFrame(rows).sort_values("RMSE_val", ascending=True).reset_index(drop=True)
    return EvaluationOutput(
        results=results,
        fitted_models=fitted_models,
        validation_predictions=validation_predictions,
    )


def evaluate_single_model_on_validation(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, float], Pipeline, np.ndarray]:
    """Fit a single model and evaluate on validation data."""
    fitted = pipeline.fit(X_train, y_train)
    y_pred = fitted.predict(X_val)
    metrics = regression_metrics(y_val, y_pred)
    metrics["Model"] = model_name
    return metrics, fitted, y_pred


def cross_validation_summary(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Compute CV summary metrics on training data only."""
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    rmse_values = -cv_results["test_rmse"]
    mae_values = -cv_results["test_mae"]
    r2_values = cv_results["test_r2"]

    return {
        "CV_RMSE_mean": float(np.mean(rmse_values)),
        "CV_RMSE_std": float(np.std(rmse_values)),
        "CV_MAE_mean": float(np.mean(mae_values)),
        "CV_MAE_std": float(np.std(mae_values)),
        "CV_R2_mean": float(np.mean(r2_values)),
        "CV_R2_std": float(np.std(r2_values)),
    }


def build_residual_frame(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a residual analysis DataFrame."""
    residual_df = pd.DataFrame(
        {
            "actual": y_true.to_numpy(),
            "predicted": y_pred,
        }
    )
    residual_df["residual"] = residual_df["actual"] - residual_df["predicted"]
    residual_df["abs_error"] = residual_df["residual"].abs()
    residual_df["squared_error"] = residual_df["residual"] ** 2
    return residual_df


def residual_summary(residual_df: pd.DataFrame) -> pd.Series:
    """Summarize residual behavior for quick diagnostics."""
    summary = {
        "residual_mean": float(residual_df["residual"].mean()),
        "residual_std": float(residual_df["residual"].std()),
        "residual_median": float(residual_df["residual"].median()),
        "mae": float(residual_df["abs_error"].mean()),
        "rmse": float(np.sqrt(residual_df["squared_error"].mean())),
        "p90_abs_error": float(residual_df["abs_error"].quantile(0.90)),
        "p95_abs_error": float(residual_df["abs_error"].quantile(0.95)),
    }
    return pd.Series(summary, name="value")


def absolute_error_by_target_quantile(
    residual_df: pd.DataFrame,
    bins: int = 5,
) -> pd.DataFrame:
    """Assess whether absolute error grows with larger target values."""
    frame = residual_df.copy()
    frame["target_bin"] = pd.qcut(frame["actual"], q=bins, duplicates="drop")
    grouped = (
        frame.groupby("target_bin", observed=True)["abs_error"]
        .agg(["mean", "median", "max", "count"])
        .reset_index()
    )
    grouped.columns = ["target_bin", "abs_error_mean", "abs_error_median", "abs_error_max", "count"]
    return grouped
