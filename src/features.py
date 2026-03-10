"""Baseline feature engineering for Stage 3 experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaselineFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add a small set of interpretable ratio features.

    Notes:
    - `AveRooms` and `AveOccup` already represent household-level averages.
    - Added features focus on compact, interview-friendly ratios.
    - Stage 4 can enable two extra occupancy/income ratios for stronger models.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        add_stage4_features: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.add_stage4_features = add_stage4_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaselineFeatureEngineer":
        """No-op fit for sklearn compatibility."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a transformed DataFrame with additional ratio-based features."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BaselineFeatureEngineer expects a pandas DataFrame input.")

        required = {"AveRooms", "AveBedrms", "AveOccup", "MedInc"}
        missing = sorted(required - set(X.columns))
        if missing:
            raise KeyError(f"Missing required columns for feature engineering: {missing}")

        transformed = X.copy()
        safe_rooms = transformed["AveRooms"].astype(float).replace(0.0, np.nan)
        safe_occupancy = transformed["AveOccup"].astype(float).replace(0.0, np.nan)

        transformed["BedroomsPerRoom"] = transformed["AveBedrms"] / (safe_rooms + self.epsilon)
        transformed["IncomePerRoom"] = transformed["MedInc"] / (safe_rooms + self.epsilon)
        transformed["RoomsPerPerson"] = transformed["AveRooms"] / (safe_occupancy + self.epsilon)

        if self.add_stage4_features:
            transformed["IncomePerPerson"] = transformed["MedInc"] / (safe_occupancy + self.epsilon)
            transformed["OccupancyPerRoom"] = transformed["AveOccup"] / (safe_rooms + self.epsilon)

        transformed = transformed.replace([np.inf, -np.inf], np.nan)
        return transformed
