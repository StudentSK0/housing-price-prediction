"""Lightweight reusable EDA helpers for Stage 2."""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .config import TARGET_COLUMN
except ImportError:  # pragma: no cover - supports running as a script
    from config import TARGET_COLUMN  # type: ignore


def summarize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all numeric columns."""
    return df.select_dtypes(include="number").describe().T


def find_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that contain one unique value only."""
    return [column for column in df.columns if df[column].nunique(dropna=False) <= 1]


def find_near_constant_columns(
    df: pd.DataFrame,
    threshold: float = 0.99,
) -> pd.Series:
    """Return columns where one value dominates at or above the threshold share."""
    top_value_shares: dict[str, float] = {}
    for column in df.columns:
        value_share = df[column].value_counts(dropna=False, normalize=True)
        if not value_share.empty and float(value_share.iloc[0]) >= threshold:
            top_value_shares[column] = float(value_share.iloc[0])

    if not top_value_shares:
        return pd.Series(dtype=float, name="top_value_share")
    return pd.Series(top_value_shares, name="top_value_share").sort_values(ascending=False)


def compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IQR-based outlier counts for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    rows: list[dict[str, float | int | str]] = []

    for column in numeric_df.columns:
        q1 = float(numeric_df[column].quantile(0.25))
        q3 = float(numeric_df[column].quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (numeric_df[column] < lower_bound) | (numeric_df[column] > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_share = outlier_count / len(numeric_df) if len(numeric_df) else 0.0

        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_share": outlier_share,
            }
        )

    return pd.DataFrame(rows).set_index("column").sort_values("outlier_share", ascending=False)


def correlation_with_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.Series:
    """Return correlations of numeric features with the target (sorted by absolute value)."""
    numeric_df = df.select_dtypes(include="number")
    if target_column not in numeric_df.columns:
        raise KeyError(f"Target column '{target_column}' is not numeric or not found.")

    target_corr = numeric_df.corr(numeric_only=True)[target_column].drop(target_column)
    return target_corr.sort_values(key=lambda series: series.abs(), ascending=False)


def plot_numeric_histograms(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    bins: int = 30,
    figsize: tuple[int, int] = (14, 10),
) -> None:
    """Plot histograms for selected numeric columns."""
    selected = df[list(columns)] if columns else df.select_dtypes(include="number")
    selected.hist(bins=bins, figsize=figsize)
    plt.suptitle("Numeric Feature Distributions", y=1.02)
    plt.tight_layout()


def plot_numeric_boxplots(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Plot boxplots for selected numeric columns."""
    selected = df[list(columns)] if columns else df.select_dtypes(include="number")
    fig, axis = plt.subplots(figsize=figsize)
    selected.plot(kind="box", ax=axis, rot=45)
    axis.set_title("Numeric Feature Boxplots")
    axis.set_ylabel("Value")
    plt.tight_layout()


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Plot a numeric correlation matrix using matplotlib."""
    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(corr.index)), corr.index)
    axis.set_title("Correlation Matrix")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_target_vs_features(
    df: pd.DataFrame,
    features: Sequence[str],
    target_column: str = TARGET_COLUMN,
    figsize: tuple[int, int] = (16, 10),
    alpha: float = 0.2,
) -> None:
    """Plot target vs selected features as scatter charts."""
    valid_features = [feature for feature in features if feature in df.columns]
    if not valid_features:
        raise ValueError("No valid features provided for target relationship plots.")

    columns_count = min(3, len(valid_features))
    rows_count = math.ceil(len(valid_features) / columns_count)

    fig, axes = plt.subplots(rows_count, columns_count, figsize=figsize, squeeze=False)
    for index, feature in enumerate(valid_features):
        axis = axes[index // columns_count][index % columns_count]
        axis.scatter(df[feature], df[target_column], alpha=alpha, s=12)
        axis.set_xlabel(feature)
        axis.set_ylabel(target_column)
        axis.set_title(f"{target_column} vs {feature}")

    for index in range(len(valid_features), rows_count * columns_count):
        axes[index // columns_count][index % columns_count].axis("off")

    plt.tight_layout()
