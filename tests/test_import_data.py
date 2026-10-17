"""Unit and Integration Tests for Dataset Validation and MongoDB Import in EduPredict.

Validates Phase 2 functionality:
1. CSV validation rules (33 columns, delimiter, required columns, grade ranges).
2. Data preparation (primitive types, source_row_id, dataset_source tagging).
3. Unique compound index creation on ('dataset_source', 'source_row_id').
4. Idempotent import behavior (ensuring rerun does not duplicate records).
5. Document count and sample data integrity in MongoDB.
"""

import sys
import unittest
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.database import (
    COLLECTION_STUDENTS,
    DATABASE_NAME,
    get_database,
    get_mongo_client,
    ping_database,
)
from src.import_data import (
    DATASET_SOURCE_NAME,
    DEFAULT_DATASET_PATH,
    EXPECTED_COLUMNS_COUNT,
    REQUIRED_COLUMNS,
    ensure_database_indexes,
    import_students_data,
    prepare_documents,
    validate_csv_dataset,
    verify_import,
)


class TestImportData(unittest.TestCase):
    """Test suite for dataset validation and idempotent MongoDB import."""

    client: Optional[MongoClient] = None
    db: Optional[Database] = None
    collection: Optional[Collection] = None
    df: Optional[pd.DataFrame] = None

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize MongoDB connection and load the dataset once."""
        cls.client = get_mongo_client()
        ping_database(cls.client)
        cls.db = get_database(cls.client)
        cls.collection = cls.db[COLLECTION_STUDENTS]
        cls.df = validate_csv_dataset(DEFAULT_DATASET_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        """Safely close client connections after tests finish."""
        if cls.client is not None:
            cls.client.close()

    def test_01_csv_validation(self) -> None:
        """Verify that the CSV has 5000 rows, 33 columns, and valid required fields."""
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 5000, f"Expected 5000 rows, found {len(self.df)}.")
        self.assertEqual(
            len(self.df.columns),
            EXPECTED_COLUMNS_COUNT,
            f"Expected {EXPECTED_COLUMNS_COUNT} columns, found {len(self.df.columns)}.",
        )
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, self.df.columns, f"Missing required column: {col}")

        # Check grade bounds
        for g in ["G1", "G2", "G3"]:
            self.assertGreaterEqual(self.df[g].min(), 0, f"{g} min < 0")
            self.assertLessEqual(self.df[g].max(), 20, f"{g} max > 20")

    def test_02_document_preparation(self) -> None:
        """Verify that DataFrame rows convert to sanitized, tagged Python dictionaries."""
        docs = prepare_documents(self.df.head(10), dataset_source=DATASET_SOURCE_NAME)
        self.assertEqual(len(docs), 10)

        first_doc = docs[0]
        self.assertEqual(first_doc["source_row_id"], 1)
        self.assertEqual(first_doc["dataset_source"], DATASET_SOURCE_NAME)

        # Check that types are native Python primitives
        self.assertIsInstance(first_doc["age"], int)
        self.assertIsInstance(first_doc["G1"], int)
        self.assertIsInstance(first_doc["school"], str)

    def test_03_index_creation(self) -> None:
        """Verify that ensure_database_indexes creates the expected unique compound index."""
        ensure_database_indexes(self.collection)
        indexes = self.collection.index_information()
        self.assertIn("idx_dataset_source_row_id", indexes)
        self.assertTrue(indexes["idx_dataset_source_row_id"].get("unique", False))

    def test_04_idempotent_import_and_rerun(self) -> None:
        """Verify import writes records and rerunning does NOT increase document count."""
        docs = prepare_documents(self.df, dataset_source=DATASET_SOURCE_NAME)

        # First run
        stats_run_1 = import_students_data(self.collection, docs, dataset_source=DATASET_SOURCE_NAME)
        self.assertEqual(stats_run_1["count_after"], 5000)

        # Second run (Idempotency test)
        stats_run_2 = import_students_data(self.collection, docs, dataset_source=DATASET_SOURCE_NAME)
        self.assertEqual(
            stats_run_2["count_after"],
            5000,
            f"Document count increased after second run! Count: {stats_run_2['count_after']}",
        )
        self.assertEqual(
            stats_run_2["upserted_count"],
            0,
            f"Expected 0 new upserts on rerun, got {stats_run_2['upserted_count']}",
        )
        self.assertEqual(
            stats_run_2["matched_count"],
            5000,
            f"Expected 5000 matched documents on rerun, got {stats_run_2['matched_count']}",
        )

    def test_05_verification_helper(self) -> None:
        """Verify that verify_import retrieves and asserts data integrity correctly."""
        summary = verify_import(self.collection, expected_count=5000, dataset_source=DATASET_SOURCE_NAME)
        self.assertEqual(summary["verified_count"], 5000)
        self.assertIn("sample_doc", summary)
        self.assertNotIn("_id", summary["sample_doc"])
        self.assertEqual(summary["sample_doc"]["dataset_source"], DATASET_SOURCE_NAME)


if __name__ == "__main__":
    unittest.main(verbosity=2)
