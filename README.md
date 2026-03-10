# Housing Price Prediction

## Current Status (Short)
- End-to-end project is complete through final hold-out evaluation.
- Final frozen model: `GradientBoostingRegressor_tuned`.
- Final test metrics:
  - `RMSE_test = 0.4845`
  - `MAE_test = 0.3280`
  - `R2_test = 0.8209`
- Validation-to-test gap is small, indicating stable generalization.

## Core Artifacts
- Final model: `models/final_gradient_boosting_pipeline.joblib`
- Final metrics: `reports/final_test_metrics.json`, `reports/final_test_metrics.csv`
- Final error analysis: `reports/final_error_analysis.csv`
- Final feature importance: `reports/final_feature_importance.csv`

## Inference Demo
- Utility: `src/predict.py`
- Notebook: `notebooks/06_single_sample_inference_demo.ipynb`
- Purpose: predict one manual California Housing sample with the frozen final pipeline.
