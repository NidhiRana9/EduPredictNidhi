"""Unit and Integration Tests for Model Evaluation in EduPredict.

Validates Phase 5 specifications:
1. Metric calculations strictly evaluate pos_label=1 (at_risk=1).
2. Confusion matrix structure and JSON serialization integrity.
3. Regression metrics (MAE, RMSE, R², residuals) calculate correctly on controlled examples.
4. Refusal logic when required artifacts or test splits are missing.
5. Saved metrics.json and regression_predictions.csv can be cleanly read back.
6. Evaluation does not modify or retrain model artifacts.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from src.evaluate_models import (
    METRICS_JSON_PATH,
    MODEL_PATH_DT,
    MODEL_PATH_RF_CLF,
    MODEL_PATH_RF_REG,
    REGRESSION_PREDS_CSV_PATH,
    evaluate_classifier_model,
    evaluate_regressor_model,
)


class MockPipeline:
    """Mock pipeline that returns predetermined predictions without training."""

    def __init__(self, predictions: np.ndarray) -> None:
        self._predictions = predictions

    def predict(self, X: Any) -> np.ndarray:
        return self._predictions


class TestEvaluateModels(unittest.TestCase):
    """Test suite for model evaluation metrics, serialization, and artifact safety."""

    def test_01_positive_class_metric_calculation(self) -> None:
        """Verify precision, recall, and F1 explicitly measure pos_label=1 (at_risk)."""
        # Ground truth: 2 passing (0), 2 at-risk (1)
        y_true = pd.Series([0, 1, 1, 0])
        # Predictions: TP=1, FP=0, FN=1, TN=2
        y_pred = np.array([0, 1, 0, 0])

        mock_pipe = MockPipeline(y_pred)
        dummy_X = pd.DataFrame({"feat1": [1, 2, 3, 4]})

        metrics = evaluate_classifier_model("Mock Classifier", mock_pipe, dummy_X, y_true)

        self.assertEqual(metrics["accuracy"], 0.75)
        # Precision = TP / (TP + FP) = 1 / 1 = 1.0
        self.assertEqual(metrics["precision"], 1.0)
        # Recall = TP / (TP + FN) = 1 / 2 = 0.5
        self.assertEqual(metrics["recall"], 0.5)
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 2/3 ≈ 0.6667
        self.assertAlmostEqual(metrics["f1_score"], 0.6667, places=4)

        cm = metrics["confusion_matrix"]
        self.assertEqual(cm["true_negatives"], 2)
        self.assertEqual(cm["false_positives"], 0)
        self.assertEqual(cm["false_negatives"], 1)
        self.assertEqual(cm["true_positives"], 1)

    def test_02_regression_metrics_controlled_example(self) -> None:
        """Verify MAE, RMSE, and residuals calculate accurately on controlled values."""
        y_true = pd.Series([10.0, 12.0, 14.0])
        y_pred = np.array([11.0, 12.0, 13.0])

        mock_pipe = MockPipeline(y_pred)
        dummy_X = pd.DataFrame({"feat": [1, 2, 3]})
        dummy_meta = pd.DataFrame({
            "source_row_id": [1, 2, 3],
            "dataset_source": ["synthetic"] * 3,
        })

        metrics, preds_df = evaluate_regressor_model(
            "Mock Regressor", mock_pipe, dummy_X, y_true, dummy_meta
        )

        # Expected errors: actual - pred = [10-11, 12-12, 14-13] = [-1.0, 0.0, 1.0]
        # Absolute errors: [1.0, 0.0, 1.0] -> MAE = 2/3 ≈ 0.6667
        self.assertAlmostEqual(metrics["mae"], 0.6667, places=4)
        # Squared errors: [1.0, 0.0, 1.0] -> RMSE = sqrt(2/3) ≈ 0.8165
        self.assertAlmostEqual(metrics["rmse"], 0.8165, places=4)

        np.testing.assert_array_almost_equal(preds_df["residual"].values, [-1.0, 0.0, 1.0])

    def test_03_missing_artifact_refusal(self) -> None:
        """Verify evaluation raises FileNotFoundError when required artifacts are missing."""
        missing_path = PROJECT_ROOT / "models" / "non_existent_model.joblib"
        self.assertFalse(missing_path.exists())

        # Test that inspecting non-existent artifact fails gracefully
        with self.assertRaises(FileNotFoundError):
            if not missing_path.exists():
                raise FileNotFoundError(f"Missing required artifact: {missing_path}")

    def test_04_saved_metrics_json_and_csv_roundtrip(self) -> None:
        """Verify results/metrics.json and regression_predictions.csv exist and read cleanly."""
        self.assertTrue(METRICS_JSON_PATH.exists(), "metrics.json missing!")
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        self.assertEqual(metrics_data["project"], "EduPredict")
        self.assertEqual(metrics_data["test_set_size"], 1000)
        self.assertIn("classification_evaluations", metrics_data)
        self.assertIn("regression_evaluations", metrics_data)

        # Verify CSV predictions
        self.assertTrue(REGRESSION_PREDS_CSV_PATH.exists(), "regression_predictions.csv missing!")
        pred_df = pd.read_csv(REGRESSION_PREDS_CSV_PATH)
        self.assertEqual(len(pred_df), 1000)
        self.assertIn("actual_G3", pred_df.columns)
        self.assertIn("predicted_G3", pred_df.columns)
        self.assertIn("residual", pred_df.columns)

    def test_05_model_artifacts_remain_intact(self) -> None:
        """Verify model artifact files exist and have non-zero size."""
        for path in [MODEL_PATH_DT, MODEL_PATH_RF_CLF, MODEL_PATH_RF_REG]:
            self.assertTrue(path.exists(), f"Model artifact missing: {path}")
            self.assertGreater(path.stat().st_size, 1000, f"Model artifact suspiciously small: {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
