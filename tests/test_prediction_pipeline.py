"""Unit and Integration Tests for MongoDB + ML Prediction Pipeline in EduPredict.

Validates Phase 6 specifications:
1. PredictionService loads fitted pipelines and validates feature schemas.
2. Feature validation strictly rejects missing or non-numeric features.
3. Feature leakage prevention: Strips out _id, G1, G2, G3, and at_risk.
4. Single prediction structure: Correct types, probability bounds, and UTC timestamp.
5. Batch prediction resilience: Handles individual bad records without crashing.
6. Idempotent MongoDB upserts into 'student_predictions'.
"""

import datetime
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import COLLECTION_STUDENT_PREDICTIONS, get_database, get_mongo_client
from src.predict_students import fetch_students_to_predict, save_predictions_to_mongodb
from src.prediction_service import ALL_PREDICTOR_FEATURES, PredictionService


class TestPredictionPipeline(unittest.TestCase):
    """Test suite for Phase 6 Prediction Service and MongoDB prediction workflow."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the PredictionService and a valid mock student dictionary."""
        cls.service = PredictionService(model_version="v1_test")

        # Create valid mock student with all 30 features + leaked/metadata columns
        cls.valid_student: Dict[str, Any] = {
            "_id": "mock_obj_id_12345",
            "source_row_id": 999,
            "dataset_source": "synthetic",
            # Leaked columns that MUST be stripped
            "G1": 15,
            "G2": 14,
            "G3": 16,
            "at_risk": 0,
            # 13 Numeric features
            "age": 16,
            "Medu": 3,
            "Fedu": 2,
            "traveltime": 1,
            "studytime": 2,
            "failures": 0,
            "famrel": 4,
            "freetime": 3,
            "goout": 2,
            "Dalc": 1,
            "Walc": 1,
            "health": 5,
            "absences": 2,
            # 17 Categorical features
            "school": "GP",
            "sex": "F",
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "T",
            "Mjob": "other",
            "Fjob": "services",
            "reason": "course",
            "guardian": "mother",
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
        }

    def test_01_service_initialization(self) -> None:
        """Verify that PredictionService loads all 3 model pipelines properly."""
        self.assertIsNotNone(self.service.dt_pipeline)
        self.assertIsNotNone(self.service.rf_clf_pipeline)
        self.assertIsNotNone(self.service.rf_reg_pipeline)
        self.assertEqual(len(self.service.artifact_names), 3)

    def test_02_feature_validation_and_leakage_exclusion(self) -> None:
        """Verify that validation strips G1, G2, G3, at_risk, and retains exactly 30 features."""
        cleaned = self.service.validate_and_clean_features(self.valid_student)

        self.assertEqual(len(cleaned), 30)
        self.assertEqual(set(cleaned.keys()), set(ALL_PREDICTOR_FEATURES))

        for leaked in ["G1", "G2", "G3", "at_risk", "_id", "source_row_id", "dataset_source"]:
            self.assertNotIn(leaked, cleaned)

    def test_03_missing_feature_rejection(self) -> None:
        """Verify that missing required features raises ValueError."""
        incomplete = self.valid_student.copy()
        del incomplete["studytime"]
        del incomplete["school"]

        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_clean_features(incomplete)
        self.assertIn("missing", str(ctx.exception).lower())

    def test_04_corrupt_numeric_feature_rejection(self) -> None:
        """Verify that non-numeric data in numeric features raises ValueError."""
        corrupt = self.valid_student.copy()
        corrupt["age"] = "not_a_number"

        with self.assertRaises(ValueError) as ctx:
            self.service.validate_and_clean_features(corrupt)
        self.assertIn("invalid numeric value", str(ctx.exception).lower())

    def test_05_single_prediction_structure_and_types(self) -> None:
        """Verify prediction output document structure, types, and probability ranges."""
        pred_doc = self.service.predict_single(self.valid_student, student_id="test_student_001")

        self.assertEqual(pred_doc["student_id"], "test_student_001")
        self.assertEqual(pred_doc["model_version"], "v1_test")
        self.assertIsInstance(pred_doc["created_at"], datetime.datetime)

        p = pred_doc["predictions"]
        self.assertIn(p["decision_tree_at_risk"], [0, 1])
        self.assertIn(p["random_forest_at_risk"], [0, 1])
        self.assertIsInstance(p["random_forest_risk_probability"], float)
        self.assertGreaterEqual(p["random_forest_risk_probability"], 0.0)
        self.assertLessEqual(p["random_forest_risk_probability"], 1.0)
        self.assertIsInstance(p["random_forest_estimated_g3"], float)
        self.assertGreaterEqual(p["random_forest_estimated_g3"], 0.0)
        self.assertLessEqual(p["random_forest_estimated_g3"], 20.0)

    def test_06_batch_prediction_resilience(self) -> None:
        """Verify batch prediction processes valid records while isolating corrupt ones."""
        corrupt_student = self.valid_student.copy()
        corrupt_student["_id"] = "bad_student_666"
        corrupt_student["failures"] = "corrupt_data"

        batch = [self.valid_student, corrupt_student]
        successful, errors = self.service.predict_batch(batch)

        self.assertEqual(len(successful), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["student_id"], "bad_student_666")

    @classmethod
    def tearDownClass(cls) -> None:
        """Safely close any opened MongoDB clients."""
        if hasattr(cls, "client") and cls.client is not None:
            cls.client.close()

    def test_07_idempotent_mongodb_storage(self) -> None:
        """Verify that saving predictions to MongoDB uses upsert and does not duplicate."""
        client = get_mongo_client()
        try:
            db = get_database(client)
            pred_doc = self.service.predict_single(self.valid_student, student_id="test_upsert_student")

            # First save
            stats_1 = save_predictions_to_mongodb(db, [pred_doc])
            self.assertEqual(stats_1["total"], 1)

            # Second save (idempotency check)
            stats_2 = save_predictions_to_mongodb(db, [pred_doc])
            self.assertEqual(stats_2["upserted"], 0)
            self.assertEqual(stats_2["matched"], 1)

            # Clean up test artifact document from collection
            db[COLLECTION_STUDENT_PREDICTIONS].delete_one({"student_id": "test_upsert_student"})
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
