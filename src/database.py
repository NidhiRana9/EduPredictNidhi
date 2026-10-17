"""Database Connection Module for EduPredict.

This module provides a reusable, safe connection manager for MongoDB
using PyMongo. It handles client initialization, server ping verification,
and retrieval of database and collection handles for the EduPredict project.

Default configuration:
- URI: mongodb://localhost:27017/
- Database: edupredict_db
- Collections:
    - students: stores raw and processed student academic/demographic records.
    - predictions: stores model prediction outputs and scores.
    - model_evaluations: stores metrics and performance logs of ML models.
"""

from typing import Any, Dict, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

# ==============================================================================
# Configuration Constants
# ==============================================================================
# Standard local MongoDB connection string (unauthenticated by default)
DEFAULT_MONGO_URI: str = "mongodb://localhost:27017/"

# Target database name for the EduPredict application
DATABASE_NAME: str = "edupredict_db"

# Server selection timeout in milliseconds (5 seconds)
SERVER_SELECTION_TIMEOUT_MS: int = 5000

# Collection names defined for current and future phases
COLLECTION_STUDENTS: str = "students"
COLLECTION_PREDICTIONS: str = "predictions"
COLLECTION_STUDENT_PREDICTIONS: str = "student_predictions"
COLLECTION_MODEL_EVALUATIONS: str = "model_evaluations"


def get_mongo_client(
    uri: str = DEFAULT_MONGO_URI,
    timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
) -> MongoClient:
    """Create and return a PyMongo MongoClient instance.

    Note on MongoDB lazy connections:
    Creating a MongoClient does NOT immediately communicate over the network.
    Network requests occur when a command (like ping) or database query is executed.

    Args:
        uri: MongoDB connection string URI.
        timeout_ms: Maximum time in milliseconds to wait for server response.

    Returns:
        An initialized MongoClient instance.
    """
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def ping_database(client: MongoClient) -> Dict[str, Any]:
    """Verify that the MongoDB server is reachable by sending an admin ping command.

    This function actively confirms network connectivity and server responsiveness
    without reading, writing, modifying, or deleting any user data.

    Args:
        client: An active MongoClient instance.

    Returns:
        A dictionary containing the server ping response, typically {'ok': 1.0}.

    Raises:
        ConnectionError: If the server cannot be reached within the timeout period
                         or if connection parameters are invalid.
        RuntimeError: If an unexpected error occurs during ping execution.
    """
    try:
        # Run the lightweight 'ping' command against the admin database
        response = client.admin.command("ping")
        return response
    except (ServerSelectionTimeoutError, ConnectionFailure) as conn_err:
        error_msg = (
            "MongoDB Connection Failed: Unable to reach MongoDB server.\n"
            f"Details: {conn_err}\n"
            "Actionable steps to resolve:\n"
            "  1. Ensure the MongoDB service is installed and running.\n"
            "     - Windows PowerShell (Admin): Start-Service MongoDB\n"
            "     - Windows Command Prompt: net start MongoDB\n"
            "  2. Verify that MongoDB is listening on the expected port (default: 27017).\n"
            "  3. Check MongoDB log files for startup errors (typically in C:\\Program Files\\MongoDB\\Server\\<version>\\log)."
        )
        raise ConnectionError(error_msg) from conn_err
    except PyMongoError as mongo_err:
        error_msg = f"MongoDB PyMongo Error during ping: {mongo_err}"
        raise ConnectionError(error_msg) from mongo_err
    except Exception as exc:
        error_msg = f"Unexpected error while verifying MongoDB connection: {exc}"
        raise RuntimeError(error_msg) from exc


def get_database(
    client: Optional[MongoClient] = None,
    db_name: str = DATABASE_NAME,
) -> Database:
    """Retrieve a handle to the EduPredict MongoDB database.

    In PyMongo, obtaining a Database handle is a local in-memory operation
    and does not create a database on the server until documents are written.

    Args:
        client: Optional existing MongoClient. If None, a new client will be created.
        db_name: Target database name (defaults to 'edupredict_db').

    Returns:
        A pymongo.database.Database instance.
    """
    if client is None:
        client = get_mongo_client()
    return client[db_name]


def get_collections(
    db: Optional[Database] = None,
) -> Dict[str, Collection]:
    """Retrieve handles for all required EduPredict collections.

    In MongoDB, collection handles are references. Creating these handles
    does not create collections on disk until data is inserted, keeping
    the test read-only and non-destructive.

    Args:
        db: Optional Database handle. If None, the default database is resolved.

    Returns:
        A dictionary mapping collection identifiers to PyMongo Collection objects:
        - 'students': collection for student data
        - 'predictions': collection for model predictions
        - 'model_evaluations': collection for evaluation metrics
    """
    if db is None:
        db = get_database()

    return {
        COLLECTION_STUDENTS: db[COLLECTION_STUDENTS],
        COLLECTION_PREDICTIONS: db[COLLECTION_PREDICTIONS],
        COLLECTION_MODEL_EVALUATIONS: db[COLLECTION_MODEL_EVALUATIONS],
    }


def verify_connection(
    uri: str = DEFAULT_MONGO_URI,
    timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
) -> bool:
    """Convenience helper to test connectivity and print diagnostic information.

    Returns:
        True if connection succeeded, False otherwise.
    """
    client: Optional[MongoClient] = None
    try:
        client = get_mongo_client(uri=uri, timeout_ms=timeout_ms)
        response = ping_database(client)
        db = get_database(client)
        collections = get_collections(db)

        print("[SUCCESS] MongoDB connection verified successfully.")
        print(f"  - Server Ping Response : {response}")
        print(f"  - Active Database      : {db.name}")
        print(f"  - Collection Handles   : {list(collections.keys())}")
        return True
    except ConnectionError as err:
        print(f"[ERROR] {err}")
        return False
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
        return False
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    import sys

    print("--- Running EduPredict Database Connectivity Check ---")
    success = verify_connection()
    sys.exit(0 if success else 1)
