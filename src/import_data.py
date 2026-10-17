"""Dataset Validation and MongoDB Import Module for EduPredict.

Phase 2 Implementation:
- Reads and validates the synthetic student dataset from CSV.
- Sanitizes and converts pandas records into MongoDB-compliant Python primitives.
- Ensures unique compound database indexing on ('dataset_source', 'source_row_id').
- Performs idempotent bulk upserts to prevent duplicate records upon rerunning.
- Verifies MongoDB document counts and field integrity without modifying or clearing
  unrelated collections.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, PyMongoError

# Ensure project root is in sys.path for direct module execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    COLLECTION_STUDENTS,
    DATABASE_NAME,
    get_database,
    get_mongo_client,
    ping_database,
)

# ==============================================================================
# Constants & Dataset Specifications
# ==============================================================================
DEFAULT_DATASET_PATH: Path = PROJECT_ROOT / "data" / "student-mat-synthetic.csv"
DATASET_SOURCE_NAME: str = "synthetic"
EXPECTED_COLUMNS_COUNT: int = 33
CSV_DELIMITER: str = ";"
BATCH_SIZE: int = 1000

REQUIRED_COLUMNS: List[str] = [
    "school",
    "sex",
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2",
    "G3",
]

GRADE_COLUMNS: List[str] = ["G1", "G2", "G3"]
GRADE_MIN: int = 0
GRADE_MAX: int = 20


def validate_csv_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the synthetic student dataset CSV.

    Validates:
      1. File existence at the specified path.
      2. Correct parsing with semicolon (';') delimiter.
      3. Presence of all 33 expected columns and required key fields.
      4. Unique and clean column headers.
      5. Absence of corrupt, non-numeric values in numeric and grade fields.
      6. Grade ranges lie strictly within 0-20.

    Args:
        csv_path: Absolute or relative Path to the CSV dataset.

    Returns:
        pd.DataFrame containing the validated dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If validation checks fail.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {csv_path.resolve()}\n"
            "Please ensure 'student-mat-synthetic.csv' is present in the 'data/' directory."
        )

    print(f"[INFO] Reading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=CSV_DELIMITER)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file with delimiter '{CSV_DELIMITER}': {exc}") from exc

    # Check 1: Multi-column parsing (not parsed into a single column)
    num_cols = len(df.columns)
    if num_cols < 2:
        raise ValueError(
            f"CSV parsed into only {num_cols} column(s). "
            f"Check if the delimiter is indeed '{CSV_DELIMITER}'."
        )

    # Check 2: Total column count
    if num_cols != EXPECTED_COLUMNS_COUNT:
        raise ValueError(
            f"Unexpected column count: got {num_cols}, expected {EXPECTED_COLUMNS_COUNT}."
        )

    # Check 3: Unique, non-empty column names
    col_names = [str(c).strip() for c in df.columns]
    if len(col_names) != len(set(col_names)):
        raise ValueError("Duplicate column names detected in CSV.")
    df.columns = col_names

    # Check 4: Required key columns
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    # Check 5: Grade ranges (0-20)
    for grade_col in GRADE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[grade_col]):
            raise ValueError(f"Grade column '{grade_col}' contains non-numeric data.")

        min_val = df[grade_col].min()
        max_val = df[grade_col].max()
        if min_val < GRADE_MIN or max_val > GRADE_MAX:
            raise ValueError(
                f"Grade values in '{grade_col}' exceed expected range [0, 20]: min={min_val}, max={max_val}"
            )

    # Diagnostic reporting
    num_rows = len(df)
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    duplicate_rows = int(df.duplicated().sum())

    print("[SUCCESS] CSV dataset validation passed:")
    print(f"  - Total rows: {num_rows}")
    print(f"  - Total columns: {num_cols}")
    print(f"  - Missing values: {total_missing}")
    print(f"  - Full duplicate rows: {duplicate_rows}")
    print(f"  - Grade ranges: G1 [{df['G1'].min()}, {df['G1'].max()}], "
          f"G2 [{df['G2'].min()}, {df['G2'].max()}], "
          f"G3 [{df['G3'].min()}, {df['G3'].max()}]")

    return df


def prepare_documents(
    df: pd.DataFrame,
    dataset_source: str = DATASET_SOURCE_NAME,
) -> List[Dict[str, Any]]:
    """Convert pandas DataFrame rows to MongoDB-compatible Python dictionaries.

    Ensures:
      - numpy scalar types (int64, float64) are converted to native Python primitives.
      - NaN / NaT values are safely converted to None.
      - A deterministic 'source_row_id' (1-indexed row position) is attached.
      - 'dataset_source' is set (default: 'synthetic').

    Args:
        df: Validated pandas DataFrame.
        dataset_source: Identifier tag for this dataset.

    Returns:
        List of dictionaries ready for MongoDB insertion/upsert.
    """
    documents: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        doc: Dict[str, Any] = {
            "source_row_id": int(idx + 1),  # Deterministic 1-based row index
            "dataset_source": dataset_source,
        }

        for col, val in row.items():
            if pd.isna(val):
                doc[col] = None
            elif isinstance(val, (np.integer, int)):
                doc[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                doc[col] = float(val)
            elif isinstance(val, (np.bool_, bool)):
                doc[col] = bool(val)
            else:
                doc[col] = str(val).strip()

        documents.append(doc)

    return documents


def ensure_database_indexes(collection: Collection) -> None:
    """Ensure appropriate unique indexes exist for idempotent upserts.

    Creates a compound unique index on ('dataset_source', 'source_row_id').
    This ensures that:
      1. Rerunning this script updates records rather than duplicating them.
      2. Synthetic data can coexist safely with official or future datasets.
    """
    index_name = "idx_dataset_source_row_id"
    existing_indexes = collection.index_information()

    if index_name not in existing_indexes:
        print(f"[INFO] Creating unique compound index '{index_name}' on ('dataset_source', 'source_row_id')...")
        collection.create_index(
            [("dataset_source", 1), ("source_row_id", 1)],
            unique=True,
            name=index_name,
        )
        print("[SUCCESS] Unique compound index created.")
    else:
        print(f"[INFO] Unique compound index '{index_name}' already exists.")


def import_students_data(
    collection: Collection,
    documents: List[Dict[str, Any]],
    dataset_source: str = DATASET_SOURCE_NAME,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, int]:
    """Execute idempotent bulk upsert of student documents into MongoDB.

    Uses PyMongo UpdateOne with upsert=True matched against the unique
    compound key ('dataset_source', 'source_row_id').
    Does NOT use delete_many() or drop collection, preserving any existing records.

    Args:
        collection: PyMongo Collection instance for 'students'.
        documents: List of sanitized document dictionaries.
        dataset_source: Dataset identifier tag.
        batch_size: Number of operations executed per bulk write.

    Returns:
        Dictionary containing counts of operations and collection statistics.
    """
    count_before = collection.count_documents({"dataset_source": dataset_source})
    print(f"[INFO] Existing documents with dataset_source='{dataset_source}': {count_before}")

    operations: List[UpdateOne] = [
        UpdateOne(
            filter={
                "dataset_source": doc["dataset_source"],
                "source_row_id": doc["source_row_id"],
            },
            update={"$set": doc},
            upsert=True,
        )
        for doc in documents
    ]

    total_ops = len(operations)
    total_upserted = 0
    total_matched = 0
    total_modified = 0

    print(f"[INFO] Performing bulk upsert for {total_ops} documents in batches of {batch_size}...")

    for i in range(0, total_ops, batch_size):
        batch = operations[i : i + batch_size]
        try:
            res = collection.bulk_write(batch, ordered=False)
            total_upserted += len(res.upserted_ids)
            total_matched += res.matched_count
            total_modified += res.modified_count
        except BulkWriteError as bwe:
            raise RuntimeError(f"Bulk write error encountered during import: {bwe.details}") from bwe
        except PyMongoError as pme:
            raise RuntimeError(f"PyMongo error during bulk upsert: {pme}") from pme

    count_after = collection.count_documents({"dataset_source": dataset_source})

    return {
        "count_before": count_before,
        "count_after": count_after,
        "total_documents": total_ops,
        "upserted_count": total_upserted,
        "matched_count": total_matched,
        "modified_count": total_modified,
    }


def verify_import(
    collection: Collection,
    expected_count: int,
    dataset_source: str = DATASET_SOURCE_NAME,
) -> Dict[str, Any]:
    """Verify that the imported collection accurately mirrors the dataset.

    Args:
        collection: PyMongo Collection handle.
        expected_count: Number of expected rows.
        dataset_source: Dataset identifier filter.

    Returns:
        Summary dict containing verified counts and a sample document.
    """
    actual_count = collection.count_documents({"dataset_source": dataset_source})
    if actual_count != expected_count:
        raise ValueError(
            f"Import verification failed! Expected {expected_count} documents, "
            f"but found {actual_count} in MongoDB."
        )

    # Fetch a sample document (omitting internal _id for clean display)
    sample_doc = collection.find_one(
        {"dataset_source": dataset_source},
        {"_id": 0},
    )

    if not sample_doc:
        raise ValueError("Verification failed: Could not retrieve sample document from collection.")

    # Validate required fields exist in sample
    for req in REQUIRED_COLUMNS:
        if req not in sample_doc:
            raise ValueError(f"Verification failed: Sample document missing required field '{req}'.")

    # Aggregate min/max grades to confirm database integrity
    pipeline = [
        {"$match": {"dataset_source": dataset_source}},
        {
            "$group": {
                "_id": None,
                "min_G1": {"$min": "$G1"},
                "max_G1": {"$max": "$G1"},
                "min_G2": {"$min": "$G2"},
                "max_G2": {"$max": "$G2"},
                "min_G3": {"$min": "$G3"},
                "max_G3": {"$max": "$G3"},
            }
        },
    ]
    grade_stats = list(collection.aggregate(pipeline))

    return {
        "verified_count": actual_count,
        "sample_doc": sample_doc,
        "grade_stats": grade_stats[0] if grade_stats else {},
    }


def main(csv_path: Optional[Path] = None) -> None:
    """Main execution function for Phase 2 dataset validation and import."""
    target_csv = csv_path or DEFAULT_DATASET_PATH

    print("=" * 70)
    print(" EduPredict: Phase 2 - Dataset Validation and MongoDB Import")
    print("=" * 70)

    # Step 1: Connect and ping MongoDB
    print("\n[Step 1/5] Checking MongoDB connection...")
    client = get_mongo_client()
    try:
        ping_res = ping_database(client)
        print(f"[SUCCESS] Connected to MongoDB on localhost:27017 (ping: {ping_res})")
    except Exception as exc:
        print(f"[ERROR] Could not connect to MongoDB: {exc}")
        sys.exit(1)

    db: Database = get_database(client)
    students_col: Collection = db[COLLECTION_STUDENTS]
    print(f"[INFO] Using database: '{db.name}', collection: '{students_col.name}'")

    # Step 2: Validate CSV dataset
    print("\n[Step 2/5] Validating CSV dataset...")
    try:
        df = validate_csv_dataset(target_csv)
    except Exception as exc:
        print(f"[ERROR] CSV Validation failed: {exc}")
        sys.exit(1)

    # Step 3: Ensure unique database indexes
    print("\n[Step 3/5] Configuring database indexes...")
    try:
        ensure_database_indexes(students_col)
    except Exception as exc:
        print(f"[ERROR] Failed to create database indexes: {exc}")
        sys.exit(1)

    # Step 4: Prepare documents and execute idempotent upsert
    print("\n[Step 4/5] Preparing documents and importing to MongoDB...")
    try:
        documents = prepare_documents(df, dataset_source=DATASET_SOURCE_NAME)
        stats = import_students_data(students_col, documents, dataset_source=DATASET_SOURCE_NAME)

        print("\n[IMPORT SUMMARY]")
        print(f"  - Count before import   : {stats['count_before']}")
        print(f"  - Documents processed   : {stats['total_documents']}")
        print(f"  - New documents upserted: {stats['upserted_count']}")
        print(f"  - Documents matched     : {stats['matched_count']}")
        print(f"  - Documents modified    : {stats['modified_count']}")
        print(f"  - Count after import    : {stats['count_after']}")
    except Exception as exc:
        print(f"[ERROR] Data import failed: {exc}")
        sys.exit(1)

    # Step 5: Verify import and display sample
    print("\n[Step 5/5] Verifying MongoDB collection integrity...")
    try:
        verification = verify_import(students_col, expected_count=len(df), dataset_source=DATASET_SOURCE_NAME)
        print(f"[SUCCESS] All {verification['verified_count']} records verified in MongoDB.")
        print("\n[SAMPLE IMPORTED DOCUMENT (First record)]:")
        for k, v in verification["sample_doc"].items():
            print(f"  {k}: {v}")

        if "grade_stats" in verification:
            gs = verification["grade_stats"]
            print(f"\n[VERIFIED GRADE STATS IN DB]:")
            print(f"  G1 range: [{gs.get('min_G1')}, {gs.get('max_G1')}]")
            print(f"  G2 range: [{gs.get('min_G2')}, {gs.get('max_G2')}]")
            print(f"  G3 range: [{gs.get('min_G3')}, {gs.get('max_G3')}]")

        print("\n" + "=" * 70)
        print(" [SUCCESS] Phase 2: Dataset Validation & MongoDB Import Completed!")
        print("=" * 70)
    except Exception as exc:
        print(f"[ERROR] Verification failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
