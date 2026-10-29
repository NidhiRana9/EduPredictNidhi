
"""Database Connection Module for EduPredict.

Supports MongoDB Atlas through Streamlit Secrets or the MONGO_URI
environment variable, with a local MongoDB fallback for development.
"""

import os
from typing import Any, Dict, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    ConnectionFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

# ==============================================================================
# Configuration Constants
# ==============================================================================

DATABASE_NAME: str = "edupredict_db"
SERVER_SELECTION_TIMEOUT_MS: int = 10000

COLLECTION_STUDENTS: str = "students"
COLLECTION_PREDICTIONS: str = "predictions"
COLLECTION_STUDENT_PREDICTIONS: str = "student_predictions"
COLLECTION_MODEL_EVALUATIONS: str = "model_evaluations"


def get_configured_mongo_uri() -> str:
    """Read the MongoDB URI from Streamlit Secrets or environment variables."""

    # Streamlit Cloud configuration
    try:
        import streamlit as st

        uri = st.secrets.get("MONGO_URI")
        if uri:
            return str(uri).strip()
    except Exception:
        pass

    # Local development configuration
    return os.getenv(
        "MONGO_URI",
        "mongodb://localhost:27017/",
    ).strip()


DEFAULT_MONGO_URI: str = get_configured_mongo_uri()


def get_mongo_client(
    uri: Optional[str] = None,
    timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
) -> MongoClient:
    """Create a MongoDB client using the configured connection URI."""

    configured_uri = uri or get_configured_mongo_uri()

    if not configured_uri:
        raise ValueError(
            "MongoDB URI is empty. Configure MONGO_URI in Streamlit Secrets "
            "or your environment variables."
        )

    return MongoClient(
        configured_uri,
        serverSelectionTimeoutMS=timeout_ms,
    )


def ping_database(client: MongoClient) -> Dict[str, Any]:
    """Check MongoDB connectivity without modifying database contents."""

    try:
        return client.admin.command("ping")

    except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
        raise ConnectionError(
            "MongoDB connection failed. Check your connection URI, "
            "Atlas Network Access settings, database credentials, "
            "and network connectivity. "
            f"Details: {exc}"
        ) from exc

    except PyMongoError as exc:
        raise ConnectionError(
            f"MongoDB error during connection verification: {exc}"
        ) from exc


def get_database(
    client: Optional[MongoClient] = None,
    db_name: str = DATABASE_NAME,
) -> Database:
    """Return a handle to the EduPredict database."""

    if client is None:
        client = get_mongo_client()

    return client[db_name]


def get_collections(
    db: Optional[Database] = None,
) -> Dict[str, Collection]:
    """Return handles for the required MongoDB collections."""

    if db is None:
        db = get_database()

    return {
        COLLECTION_STUDENTS: db[COLLECTION_STUDENTS],
        COLLECTION_PREDICTIONS: db[COLLECTION_PREDICTIONS],
        COLLECTION_MODEL_EVALUATIONS: db[COLLECTION_MODEL_EVALUATIONS],
    }


def verify_connection(
    uri: Optional[str] = None,
    timeout_ms: int = SERVER_SELECTION_TIMEOUT_MS,
) -> bool:
    """Test MongoDB connectivity and print diagnostic information."""

    client: Optional[MongoClient] = None

    try:
        client = get_mongo_client(
            uri=uri,
            timeout_ms=timeout_ms,
        )

        response = ping_database(client)
        db = client[DATABASE_NAME]
        collections = get_collections(db)

        print("[SUCCESS] MongoDB connection verified successfully.")
        print(f"  - Server Ping Response : {response}")
        print(f"  - Active Database      : {db.name}")
        print(f"  - Collection Handles   : {list(collections.keys())}")

        return True

    except (ConnectionError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return False

    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
        return False

    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    import sys

    print("--- EduPredict Database Connectivity Check ---")
    success = verify_connection()
    sys.exit(0 if success else 1)
