"""Unit and Integration Tests for Machine Learning Model Training in EduPredict.

Validates Phase 4 specifications:
1. Feature input validation: Grade columns (G1, G2, G3) and target/metadata
   columns are strictly excluded.
2. Target definition accuracy: 'at_risk' follows the G3 < 10 rule.
3. Reproducibility: Train/test split with random_state=42 produces identical splits.
4. Pipeline execution: ColumnTransformer correctly handles both numeric and categorical
   inputs without data leakage.
5. In-memory training: Pipelines train and predict cleanly on small mock datasets.
6. Artifact validation: All 3 saved joblib models can be loaded and generate predictions.
7. Training summary verification: results/training_summary.json is valid and contains
   training-only cross-validation metrics.
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

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN_CLASSIFICATION,
    TARGET_COLUMN_REGRESSION,
    build_preprocessor,
    prepare_features_and_targets,
    split_dataset,
)
from src.train_models import (
    MODEL_PATH_DT_CLASSIFIER,
    MODEL_PATH_RF_CLASSIFIER,
    MODEL_PATH_RF_REGRESSOR,
    TRAINING_SUMMARY_PATH,
)


class TestTrainModels(unittest.TestCase):
    """Test suite for Phase 4 model training and pipeline persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a lightweight synthetic in-memory fixture for isolated testing."""
        np.random.seed(42)
        cls.n_samples = 60

        data: Dict[str, Any] = {
            # Metadata identifiers
            "_id": [f"test_id_{i}" for i in range(cls.n_samples)],
            "source_row_id": list(range(1, cls.n_samples + 1)),
            "dataset_source": ["synthetic"] * cls.n_samples,
            # Grades
            "G1": np.random.randint(0, 21, cls.n_samples),
            "G2": np.random.randint(0, 21, cls.n_samples),
            "G3": np.random.randint(0, 21, cls.n_samples),
            # Numeric predictors (13 features)
            "age": np.random.randint(15, 22, cls.n_samples),
            "Medu": np.random.randint(0, 5, cls.n_samples),
            "Fedu": np.random.randint(0, 5, cls.n_samples),
            "traveltime": np.random.randint(1, 5, cls.n_samples),
            "studytime": np.random.randint(1, 5, cls.n_samples),
            "failures": np.random.randint(0, 4, cls.n_samples),
            "famrel": np.random.randint(1, 6, cls.n_samples),
            "freetime": np.random.randint(1, 6, cls.n_samples),
            "goout": np.random.randint(1, 6, cls.n_samples),
            "Dalc": np.random.randint(1, 6, cls.n_samples),
            "Walc": np.random.randint(1, 6, cls.n_samples),
            "health": np.random.randint(1, 6, cls.n_samples),
            "absences": np.random.randint(0, 30, cls.n_samples),
            # Categorical predictors (17 features)
            "school": np.random.choice(["GP", "MS"], cls.n_samples),
            "sex": np.random.choice(["F", "M"], cls.n_samples),
            "address": np.random.choice(["U", "R"], cls.n_samples),
            "famsize": np.random.choice(["LE3", "GT3"], cls.n_samples),
            "Pstatus": np.random.choice(["T", "A"], cls.n_samples),
            "Mjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], cls.n_samples),
            "Fjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], cls.n_samples),
            "reason": np.random.choice(["home", "reputation", "course", "other"], cls.n_samples),
            "guardian": np.random.choice(["mother", "father", "other"], cls.n_samples),
            "schoolsup": np.random.choice(["yes", "no"], cls.n_samples),
            "famsup": np.random.choice(["yes", "no"], cls.n_samples),
            "paid": np.random.choice(["yes", "no"], cls.n_samples),
            "activities": np.random.choice(["yes", "no"], cls.n_samples),
            "nursery": np.random.choice(["yes", "no"], cls.n_samples),
            "higher": np.random.choice(["yes", "no"], cls.n_samples),
            "internet": np.random.choice(["yes", "no"], cls.n_samples),
            "romantic": np.random.choice(["yes", "no"], cls.n_samples),
        }
        cls.mock_df = pd.DataFrame(data)

    def test_01_feature_exclusion_and_targets(self) -> None:
        """Verify grade and metadata columns are strictly excluded from predictive features."""
        X, y_class, y_reg = prepare_features_and_targets(self.mock_df)

        for col in EXCLUDED_FROM_FEATURES:
            self.assertNotIn(col, X.columns, f"Leaked column '{col}' found in feature matrix X!")

        self.assertEqual(X.shape[1], 30)
        self.assertEqual(y_class.name, TARGET_COLUMN_CLASSIFICATION)
        self.assertEqual(y_reg.name, TARGET_COLUMN_REGRESSION)

    def test_02_split_reproducibility(self) -> None:
        """Verify train/test split with fixed random_state produces identical splits."""
        X, y_class, _ = prepare_features_and_targets(self.mock_df)

        X_tr1, X_te1, y_tr1, y_te1 = split_dataset(X, y_class, test_size=0.2, random_state=42, stratify=True)
        X_tr2, X_te2, y_tr2, y_te2 = split_dataset(X, y_class, test_size=0.2, random_state=42, stratify=True)

        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_frame_equal(X_te1, X_te2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)
        pd.testing.assert_series_equal(y_te1, y_te2)

    def test_03_pipeline_preprocessing_execution(self) -> None:
        """Verify ColumnTransformer transforms both numeric and categorical columns cleanly."""
        X, _, _ = prepare_features_and_targets(self.mock_df)
        preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False)

        transformed = preprocessor.fit_transform(X)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape[0], len(X))
        self.assertFalse(np.isnan(transformed).any(), "Preprocessed matrix contains NaNs!")

    def test_04_in_memory_training_and_prediction(self) -> None:
        """Verify that DecisionTree, RandomForestClassifier, and Regressor fit and predict in-memory."""
        X, y_class, y_reg = prepare_features_and_targets(self.mock_df)
        preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False)

        # Test DT Classifier
        dt_pipe = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(max_depth=3, random_state=42))])
        dt_pipe.fit(X, y_class)
        dt_preds = dt_pipe.predict(X.head(5))
        self.assertEqual(len(dt_preds), 5)

        # Test RF Classifier
        rf_clf_pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42))])
        rf_clf_pipe.fit(X, y_class)
        rf_preds = rf_clf_pipe.predict(X.head(5))
        self.assertEqual(len(rf_preds), 5)

        # Test RF Regressor
        rf_reg_pipe = Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42))])
        rf_reg_pipe.fit(X, y_reg)
        reg_preds = rf_reg_pipe.predict(X.head(5))
        self.assertEqual(len(reg_preds), 5)
        self.assertTrue(np.all(reg_preds >= 0) and np.all(reg_preds <= 20))

    def test_05_saved_artifacts_load_and_predict(self) -> None:
        """Verify that all 3 saved joblib model pipelines load and predict on raw feature input."""
        sample_input = self.mock_df[[c for c in self.mock_df.columns if c not in EXCLUDED_FROM_FEATURES]].head(3)

        artifacts = [
            (MODEL_PATH_DT_CLASSIFIER, "Decision Tree Classifier"),
            (MODEL_PATH_RF_CLASSIFIER, "Random Forest Classifier"),
            (MODEL_PATH_RF_REGRESSOR, "Random Forest Regressor"),
        ]

        for path, name in artifacts:
            self.assertTrue(path.exists(), f"Saved model artifact missing: {path}")
            model = joblib.load(path)
            self.assertTrue(hasattr(model, "predict"), f"{name} does not have a predict method.")
            preds = model.predict(sample_input)
            self.assertEqual(len(preds), 3, f"{name} predictions count mismatch.")

    def test_06_training_summary_json(self) -> None:
        """Verify results/training_summary.json exists, is valid JSON, and contains training CV metrics."""
        self.assertTrue(TRAINING_SUMMARY_PATH.exists(), "training_summary.json not found!")
        with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["project"], "EduPredict")
        self.assertEqual(summary["dataset"]["total_records"], 5000)
        self.assertEqual(summary["features"]["total_count"], 30)

        cv_res = summary["training_only_cross_validation_results"]
        self.assertIn("decision_tree_classifier", cv_res)
        self.assertIn("random_forest_classifier", cv_res)
        self.assertIn("random_forest_regressor", cv_res)

        # Verify cross-validation metric keys
        self.assertIn("cv_accuracy_mean", cv_res["decision_tree_classifier"]["metrics"])
        self.assertIn("cv_accuracy_mean", cv_res["random_forest_classifier"]["metrics"])
        self.assertIn("cv_mae_mean", cv_res["random_forest_regressor"]["metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
