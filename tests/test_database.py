"""Unit and Integration Tests for MongoDB Connectivity in EduPredict.

This module validates that:
1. The MongoDB client can be instantiated with appropriate timeout settings.
2. An admin ping command successfully reaches the local MongoDB daemon.
3. The selected database handle is correctly identified as 'edupredict_db'.
4. Required collection handles ('students', 'predictions', 'model_evaluations')
   are instantiated properly without modifying or writing documents.

Can be run directly via:
    python tests/test_database.py
or via Python's standard unittest runner:
    python -m unittest tests/test_database.py
"""

import sys
import unittest
from pathlib import Path
from typing import Optional

# Ensure the project root directory is in Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymongo import MongoClient
from pymongo.database import Database
from src.database import (
    DATABASE_NAME,
    COLLECTION_STUDENTS,
    COLLECTION_PREDICTIONS,
    COLLECTION_MODEL_EVALUATIONS,
    get_mongo_client,
    ping_database,
    get_database,
    get_collections,
)


class TestDatabaseConnection(unittest.TestCase):
    """Test suite for verifying MongoDB configuration and connectivity."""

    client: Optional[MongoClient] = None

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize the client once for the test suite."""
        cls.client = get_mongo_client()

    @classmethod
    def tearDownClass(cls) -> None:
        """Safely close client connections after tests finish."""
        if cls.client is not None:
            cls.client.close()

    def test_01_client_initialization(self) -> None:
        """Verify that MongoClient initializes properly with configured options."""
        self.assertIsNotNone(self.client, "MongoClient should not be None.")
        self.assertIsInstance(self.client, MongoClient)

    def test_02_database_ping(self) -> None:
        """Verify that an admin ping command reaches MongoDB successfully."""
        try:
            response = ping_database(self.client)
            self.assertIsInstance(response, dict, "Ping response should be a dictionary.")
            self.assertEqual(
                response.get("ok"),
                1.0,
                f"Ping response ok status was not 1.0 (received: {response})",
            )
            print("\n[PASS] MongoDB ping succeeded! Server is active and reachable.")
        except ConnectionError as err:
            self.fail(f"MongoDB connection failed: {err}")

    def test_03_database_selection(self) -> None:
        """Verify that the database handle corresponds to 'edupredict_db'."""
        db: Database = get_database(self.client)
        self.assertEqual(
            db.name,
            DATABASE_NAME,
            f"Expected database name '{DATABASE_NAME}', but got '{db.name}'.",
        )
        print(f"[PASS] Successfully selected target database: '{db.name}'.")

    def test_04_collection_handles(self) -> None:
        """Verify collection handles are defined without writing documents to disk."""
        db: Database = get_database(self.client)
        collections = get_collections(db)

        expected_collections = {
            COLLECTION_STUDENTS,
            COLLECTION_PREDICTIONS,
            COLLECTION_MODEL_EVALUATIONS,
        }

        self.assertEqual(
            set(collections.keys()),
            expected_collections,
            f"Collection handles mismatch. Expected: {expected_collections}, Got: {set(collections.keys())}",
        )

        for col_name, col_handle in collections.items():
            self.assertEqual(
                col_handle.name,
                col_name,
                f"Collection handle name mismatch for {col_name}",
            )

        print(
            f"[PASS] Defined collection handles: {list(collections.keys())} "
            "(no documents written or modified)."
        )


def main() -> None:
    """Standalone runner providing beginner-friendly status and actionable messages."""
    print("=" * 60)
    print(" EduPredict: Phase 1 - MongoDB Connectivity Test Runner")
    print("=" * 60)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabaseConnection)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("-" * 60)
    if result.wasSuccessful():
        print("[SUCCESS] All database tests passed! MongoDB is properly configured.")
        print("Project setup and Phase 1 database verification completed successfully.")
        sys.exit(0)
    else:
        print("[FAILURE] MongoDB connectivity tests encountered errors or failures.")
        print("\nActionable Troubleshooting Steps:")
        print("  1. Check if MongoDB Service is running in PowerShell:")
        print("       Get-Service -Name MongoDB")
        print("     To start the service (Run as Administrator):")
        print("       Start-Service MongoDB")
        print("  2. In Command Prompt (Run as Administrator):")
        print("       net start MongoDB")
        print("  3. Check if MongoDB is listening on port 27017:")
        print("       Test-NetConnection -ComputerName localhost -Port 27017")
        print("  4. If MongoDB is installed in a non-default location or port,")
        print("     verify the connection URI in src/database.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
