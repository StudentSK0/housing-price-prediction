"""Model and preprocessing builders for Stage 3 baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import RANDOM_STATE
    from .features import BaselineFeatureEngineer
except ImportError:  # pragma: no cover - supports running as a script
    from config import RANDOM_STATE  # type: ignore
    from features import BaselineFeatureEngineer  # type: ignore


@dataclass(frozen=True)
class ModelSpec:
    """Baseline model configuration."""

    estimator: object
    scale_numeric: bool


def get_baseline_model_specs(random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """Return the baseline model set for Stage 3."""
    return {
        "DummyRegressor": ModelSpec(
            estimator=DummyRegressor(strategy="mean"),
            scale_numeric=False,
        ),
        "LinearRegression": ModelSpec(
            estimator=LinearRegression(),
            scale_numeric=True,
        ),
        "Ridge": ModelSpec(
            estimator=Ridge(alpha=1.0, random_state=random_state),
            scale_numeric=True,
        ),
        "RandomForestRegressor": ModelSpec(
            estimator=RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "GradientBoostingRegressor": ModelSpec(
            estimator=GradientBoostingRegressor(random_state=random_state),
            scale_numeric=False,
        ),
    }


def get_selected_model_specs(
    model_names: list[str],
    random_state: int = RANDOM_STATE,
) -> dict[str, ModelSpec]:
    """Return selected model specs by name."""
    all_specs = get_baseline_model_specs(random_state=random_state)
    missing = [name for name in model_names if name not in all_specs]
    if missing:
        raise KeyError(f"Unknown model names requested: {missing}")
    return {name: all_specs[name] for name in model_names}


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build numeric preprocessing with optional scaling."""
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_model_pipeline(
    estimator: object,
    scale_numeric: bool,
    include_feature_engineering: bool = True,
    feature_engineering_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a model pipeline that avoids leakage by fitting steps on train data only."""
    steps: list[tuple[str, object]] = []
    if include_feature_engineering:
        params = feature_engineering_params or {}
        steps.append(("feature_engineering", BaselineFeatureEngineer(**params)))
    steps.append(("preprocess", build_preprocessor(scale_numeric=scale_numeric)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def build_baseline_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Build all baseline pipelines with consistent preprocessing design."""
    specs = get_baseline_model_specs(random_state=random_state)
    return {
        model_name: build_model_pipeline(
            estimator=spec.estimator,
            scale_numeric=spec.scale_numeric,
            include_feature_engineering=True,
        )
        for model_name, spec in specs.items()
    }
