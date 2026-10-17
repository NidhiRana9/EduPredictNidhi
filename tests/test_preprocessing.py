"""Unit Tests for Feature Engineering and Preprocessing in EduPredict.

Validates Phase 3 specifications:
1. Target transformation: 'at_risk' strictly follows G3 < 10 threshold.
2. Feature exclusion: 'G1', 'G2', 'G3', 'at_risk', '_id', 'source_row_id',
   and 'dataset_source' are strictly excluded from predictive features.
3. Feature categorization: Numeric (13) and categorical (17) columns are
   correctly identified.
4. Pipeline construction: Scikit-learn ColumnTransformer is built un-fitted.
5. Train/test splitting: Feature and target dimensions match across splits,
   preserving stratification without data leakage.
"""

import sys
import unittest
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import (
    AT_RISK_THRESHOLD,
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN_CLASSIFICATION,
    TARGET_COLUMN_REGRESSION,
    build_preprocessor,
    identify_feature_types,
    prepare_features_and_targets,
    split_dataset,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing logic, feature partitioning, and pipeline construction."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a lightweight synthetic in-memory DataFrame for fast testing."""
        np.random.seed(42)
        cls.sample_size = 100

        data = {
            # Metadata identifiers
            "_id": [f"mongo_id_{i}" for i in range(cls.sample_size)],
            "source_row_id": list(range(1, cls.sample_size + 1)),
            "dataset_source": ["synthetic"] * cls.sample_size,
            # Grades
            "G1": np.random.randint(0, 21, cls.sample_size),
            "G2": np.random.randint(0, 21, cls.sample_size),
            "G3": np.random.randint(0, 21, cls.sample_size),
            # Numeric predictors (13 features)
            "age": np.random.randint(15, 22, cls.sample_size),
            "Medu": np.random.randint(0, 5, cls.sample_size),
            "Fedu": np.random.randint(0, 5, cls.sample_size),
            "traveltime": np.random.randint(1, 5, cls.sample_size),
            "studytime": np.random.randint(1, 5, cls.sample_size),
            "failures": np.random.randint(0, 4, cls.sample_size),
            "famrel": np.random.randint(1, 6, cls.sample_size),
            "freetime": np.random.randint(1, 6, cls.sample_size),
            "goout": np.random.randint(1, 6, cls.sample_size),
            "Dalc": np.random.randint(1, 6, cls.sample_size),
            "Walc": np.random.randint(1, 6, cls.sample_size),
            "health": np.random.randint(1, 6, cls.sample_size),
            "absences": np.random.randint(0, 30, cls.sample_size),
            # Categorical predictors (17 features)
            "school": np.random.choice(["GP", "MS"], cls.sample_size),
            "sex": np.random.choice(["F", "M"], cls.sample_size),
            "address": np.random.choice(["U", "R"], cls.sample_size),
            "famsize": np.random.choice(["LE3", "GT3"], cls.sample_size),
            "Pstatus": np.random.choice(["T", "A"], cls.sample_size),
            "Mjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], cls.sample_size),
            "Fjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], cls.sample_size),
            "reason": np.random.choice(["home", "reputation", "course", "other"], cls.sample_size),
            "guardian": np.random.choice(["mother", "father", "other"], cls.sample_size),
            "schoolsup": np.random.choice(["yes", "no"], cls.sample_size),
            "famsup": np.random.choice(["yes", "no"], cls.sample_size),
            "paid": np.random.choice(["yes", "no"], cls.sample_size),
            "activities": np.random.choice(["yes", "no"], cls.sample_size),
            "nursery": np.random.choice(["yes", "no"], cls.sample_size),
            "higher": np.random.choice(["yes", "no"], cls.sample_size),
            "internet": np.random.choice(["yes", "no"], cls.sample_size),
            "romantic": np.random.choice(["yes", "no"], cls.sample_size),
        }
        cls.mock_df = pd.DataFrame(data)

    def test_01_at_risk_threshold_rule(self) -> None:
        """Verify that at_risk target is strictly 1 when G3 < 10, and 0 when G3 >= 10."""
        test_grades = pd.DataFrame({"G3": [0, 5, 9, 10, 11, 20]})
        X, y_class, y_reg = prepare_features_and_targets(test_grades)

        expected_risk = [1, 1, 1, 0, 0, 0]
        self.assertEqual(y_class.tolist(), expected_risk)
        self.assertEqual(y_reg.tolist(), [0, 5, 9, 10, 11, 20])
        self.assertEqual(y_class.name, TARGET_COLUMN_CLASSIFICATION)
        self.assertEqual(y_reg.name, TARGET_COLUMN_REGRESSION)

    def test_02_feature_exclusion(self) -> None:
        """Verify that G1, G2, G3, at_risk, _id, source_row_id, and dataset_source are excluded."""
        X, y_class, y_reg = prepare_features_and_targets(self.mock_df)

        for excluded_col in EXCLUDED_FROM_FEATURES:
            self.assertNotIn(
                excluded_col,
                X.columns,
                f"Leaked column '{excluded_col}' found in feature DataFrame X!",
            )

        # Expected 30 predictor features (13 numeric + 17 categorical)
        self.assertEqual(
            X.shape[1],
            30,
            f"Expected 30 features, got {X.shape[1]}: {list(X.columns)}",
        )

    def test_03_feature_type_identification(self) -> None:
        """Verify identify_feature_types correctly categorizes numeric and categorical columns."""
        X, _, _ = prepare_features_and_targets(self.mock_df)
        num_cols, cat_cols = identify_feature_types(X)

        self.assertEqual(set(num_cols), set(NUMERIC_FEATURES))
        self.assertEqual(set(cat_cols), set(CATEGORICAL_FEATURES))
        self.assertEqual(len(num_cols), 13)
        self.assertEqual(len(cat_cols), 17)

    def test_04_pipeline_construction_unfitted(self) -> None:
        """Verify build_preprocessor creates an un-fitted ColumnTransformer with expected steps."""
        preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        self.assertIsInstance(preprocessor, ColumnTransformer)

        # Confirm pipeline is un-fitted
        self.assertFalse(
            hasattr(preprocessor, "transformers_"),
            "Preprocessor should remain UN-FITTED until model training in Phase 4!",
        )

        # Confirm transformers configuration
        transformer_names = [t[0] for t in preprocessor.transformers]
        self.assertIn("num", transformer_names)
        self.assertIn("cat", transformer_names)

    def test_05_train_test_split_shapes_and_stratification(self) -> None:
        """Verify train/test splitting maintains dimension integrity and handles stratification."""
        X, y_class, _ = prepare_features_and_targets(self.mock_df)

        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y_class,
            test_size=0.2,
            random_state=42,
            stratify=True,
        )

        # 80/20 partition of 100 samples
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

        # Feature column alignment
        self.assertEqual(list(X_train.columns), list(X_test.columns))
        self.assertEqual(X_train.shape[1], 30)

        # Check stratification preserves positive proportion closely
        orig_ratio = y_class.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        self.assertAlmostEqual(orig_ratio, train_ratio, delta=0.05)
        self.assertAlmostEqual(orig_ratio, test_ratio, delta=0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
