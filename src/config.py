"""Project-wide paths and constants for the housing Stage 1 scaffold."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SKLEARN_DATA_HOME = DATA_DIR / "sklearn_cache"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_FILE = RAW_DATA_DIR / "california_housing.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "california_housing_clean.csv"
TARGET_COLUMN = "MedHouseVal"
REQUIRED_COLUMNS = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    TARGET_COLUMN,
)

# Stage 3 experiment constants
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

SPLIT_METADATA_FILE = REPORTS_DIR / "split_metadata.json"
BASELINE_RESULTS_FILE = REPORTS_DIR / "baseline_model_results.csv"

# Stage 4 tuning outputs
TUNED_RESULTS_FILE = REPORTS_DIR / "tuned_model_results.csv"
TUNING_SUMMARY_FILE = REPORTS_DIR / "tuning_summary.json"
STAGE4_ERROR_ANALYSIS_FILE = REPORTS_DIR / "error_analysis_stage4.csv"

# Stage 4 validation settings
TUNING_CV_SPLITS = 3
ROBUSTNESS_CV_SPLITS = 5

# Stage 5 finalization outputs
FINAL_MODEL_ARTIFACT_FILE = MODELS_DIR / "final_gradient_boosting_pipeline.joblib"
FINAL_TEST_METRICS_JSON_FILE = REPORTS_DIR / "final_test_metrics.json"
FINAL_TEST_METRICS_CSV_FILE = REPORTS_DIR / "final_test_metrics.csv"
FINAL_ERROR_ANALYSIS_FILE = REPORTS_DIR / "final_error_analysis.csv"
FINAL_FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "final_feature_importance.csv"
FINAL_MODEL_SUMMARY_FILE = REPORTS_DIR / "final_model_summary.md"

# Stage 5 frozen final model choice (from Stage 4)
FINAL_MODEL_LABEL = "GradientBoostingRegressor_tuned"
FINAL_MODEL_FAMILY = "GradientBoostingRegressor"
FINAL_MODEL_PARAMS = {
    "subsample": 0.85,
    "n_estimators": 350,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_depth": 4,
    "learning_rate": 0.05,
}
FINAL_ADD_STAGE4_FEATURES = False
