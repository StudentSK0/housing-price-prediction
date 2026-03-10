"""Hyperparameter tuning utilities for Stage 4 model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from .config import RANDOM_STATE, TUNING_CV_SPLITS
    from .modeling import build_model_pipeline, get_selected_model_specs
except ImportError:  # pragma: no cover - supports running as a script
    from config import RANDOM_STATE, TUNING_CV_SPLITS  # type: ignore
    from modeling import build_model_pipeline, get_selected_model_specs  # type: ignore


@dataclass
class TunedModelResult:
    """Container for one tuned model output."""

    model_name: str
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_cv_rmse: float
    best_cv_rmse_std: float
    search_type: str


def get_stage4_candidate_model_names(include_ridge: bool = True) -> list[str]:
    """Return model names selected for Stage 4 tuning."""
    names = ["RandomForestRegressor", "GradientBoostingRegressor"]
    if include_ridge:
        names.append("Ridge")
    return names


def build_stage4_baseline_pipelines(
    random_state: int = RANDOM_STATE,
    include_ridge: bool = True,
) -> dict[str, Pipeline]:
    """Build baseline candidate pipelines for Stage 4 comparison."""
    model_names = get_stage4_candidate_model_names(include_ridge=include_ridge)
    specs = get_selected_model_specs(model_names=model_names, random_state=random_state)
    return {
        model_name: build_model_pipeline(
            estimator=spec.estimator,
            scale_numeric=spec.scale_numeric,
            include_feature_engineering=True,
            feature_engineering_params={"add_stage4_features": False},
        )
        for model_name, spec in specs.items()
    }


def get_stage4_search_spaces(include_ridge: bool = True) -> dict[str, dict[str, list[Any]]]:
    """Return compact high-value hyperparameter spaces for Stage 4 tuning."""
    spaces: dict[str, dict[str, list[Any]]] = {
        "RandomForestRegressor": {
            "feature_engineering__add_stage4_features": [False, True],
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt", 0.7],
        },
        "GradientBoostingRegressor": {
            "feature_engineering__add_stage4_features": [False, True],
            "model__n_estimators": [150, 250, 350],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.7, 0.85, 1.0],
        },
    }
    if include_ridge:
        spaces["Ridge"] = {
            "feature_engineering__add_stage4_features": [False, True],
            "model__alpha": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        }
    return spaces


def _build_cv(
    cv_splits: int = TUNING_CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> KFold:
    """Build deterministic KFold object for tuning."""
    return KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)


def _fit_search(
    model_name: str,
    pipeline: Pipeline,
    param_space: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    random_state: int,
) -> TunedModelResult:
    """Fit model-specific hyperparameter search and return best result."""
    scoring = "neg_root_mean_squared_error"

    if model_name == "Ridge":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search_type = "GridSearchCV"
    else:
        n_iter = 12
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
        )
        search_type = "RandomizedSearchCV"

    search.fit(X_train, y_train)
    best_index = int(search.best_index_)

    return TunedModelResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_cv_rmse=float(-search.best_score_),
        best_cv_rmse_std=float(search.cv_results_["std_test_score"][best_index]),
        search_type=search_type,
    )


def run_stage4_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    include_ridge: bool = True,
    cv_splits: int = TUNING_CV_SPLITS,
) -> dict[str, TunedModelResult]:
    """Run Stage 4 tuning for selected candidate models."""
    model_names = get_stage4_candidate_model_names(include_ridge=include_ridge)
    specs = get_selected_model_specs(model_names=model_names, random_state=random_state)
    spaces = get_stage4_search_spaces(include_ridge=include_ridge)
    cv = _build_cv(cv_splits=cv_splits, random_state=random_state)

    results: dict[str, TunedModelResult] = {}
    for model_name in model_names:
        spec = specs[model_name]
        pipeline = build_model_pipeline(
            estimator=spec.estimator,
            scale_numeric=spec.scale_numeric,
            include_feature_engineering=True,
            feature_engineering_params={"add_stage4_features": False},
        )
        tuned = _fit_search(
            model_name=model_name,
            pipeline=pipeline,
            param_space=spaces[model_name],
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            random_state=random_state,
        )
        results[model_name] = tuned

    return results


def tuning_results_frame(results: dict[str, TunedModelResult]) -> pd.DataFrame:
    """Convert tuning outputs to a compact DataFrame."""
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        rows.append(
            {
                "Model": model_name,
                "search_type": result.search_type,
                "best_cv_rmse": result.best_cv_rmse,
                "best_cv_rmse_std": result.best_cv_rmse_std,
                "best_params": str(result.best_params),
            }
        )
    return pd.DataFrame(rows).sort_values("best_cv_rmse", ascending=True).reset_index(drop=True)


def tuning_summary_payload(
    results: dict[str, TunedModelResult],
    random_state: int = RANDOM_STATE,
    cv_splits: int = TUNING_CV_SPLITS,
) -> dict[str, Any]:
    """Build JSON-serializable summary for Stage 4 tuning."""
    payload: dict[str, Any] = {
        "random_state": random_state,
        "tuning_cv_splits": cv_splits,
        "models": {},
    }
    for model_name, result in results.items():
        payload["models"][model_name] = {
            "search_type": result.search_type,
            "best_cv_rmse": result.best_cv_rmse,
            "best_cv_rmse_std": result.best_cv_rmse_std,
            "best_params": result.best_params,
        }
    return payload
