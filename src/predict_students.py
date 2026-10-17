"""CLI Workflow for Generating and Storing Student Predictions in MongoDB.

Phase 6 Implementation:
- Queries eligible student documents from MongoDB 'students'.
- Provides flexible CLI arguments:
    --limit N: Process a safe batch of N students (default: 10).
    --all: Explicitly process all eligible students.
    --student-id ID: Score a single student by MongoDB _id or source_row_id.
- Uses PredictionService to validate features and generate multi-model inferences.
- Idempotently upserts predictions into MongoDB collection 'student_predictions'.
- Preserves original student documents in 'students' collection without modification.
- Generates a markdown report summarizing the execution run (reports/prediction_run_report.md).
"""

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bson.errors import InvalidId
from bson.objectid import ObjectId
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    COLLECTION_STUDENT_PREDICTIONS,
    COLLECTION_STUDENTS,
    DATABASE_NAME,
    get_database,
    get_mongo_client,
    ping_database,
)
from src.prediction_service import PredictionService

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
PREDICTION_REPORT_PATH: Path = REPORTS_DIR / "prediction_run_report.md"
DEFAULT_BATCH_LIMIT: int = 10
DATASET_SOURCE_NAME: str = "synthetic"


def fetch_students_to_predict(
    db: Database,
    student_id_query: Optional[str] = None,
    limit: Optional[int] = DEFAULT_BATCH_LIMIT,
    fetch_all: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve target student documents from MongoDB 'students' collection.

    Supports:
      - Query by ObjectId string or integer source_row_id.
      - Safe default limit (10 records) to prevent accidental full-collection scans.
      - Explicit --all flag.

    Returns:
        List of student document dictionaries.
    """
    students_col = db[COLLECTION_STUDENTS]

    # Case A: Single student lookup
    if student_id_query:
        query: Dict[str, Any] = {}
        # Try as ObjectId
        try:
            query = {"_id": ObjectId(student_id_query)}
        except (InvalidId, ValueError):
            # Try as integer source_row_id
            try:
                query = {"source_row_id": int(student_id_query)}
            except ValueError:
                # String query
                query = {"student_id": student_id_query}

        record = students_col.find_one(query)
        if not record:
            # Fallback search by string representation of _id
            record = students_col.find_one({"source_row_id": student_id_query})

        if not record:
            raise ValueError(
                f"No student record found matching identifier: '{student_id_query}' "
                f"in collection '{COLLECTION_STUDENTS}'."
            )
        return [record]

    # Case B: Batch query
    query = {"dataset_source": DATASET_SOURCE_NAME}

    cursor = students_col.find(query).sort("source_row_id", 1)
    if not fetch_all and limit is not None and limit > 0:
        cursor = cursor.limit(limit)

    records = list(cursor)
    return records


def save_predictions_to_mongodb(
    db: Database,
    predictions: List[Dict[str, Any]],
    collection_name: str = COLLECTION_STUDENT_PREDICTIONS,
) -> Dict[str, int]:
    """Idempotently save prediction documents into MongoDB collection 'student_predictions'.

    Uses UpdateOne with upsert=True on ('student_id', 'model_version') so
    repeated executions update existing predictions in-place.
    """
    if not predictions:
        return {"total": 0, "upserted": 0, "matched": 0, "modified": 0}

    col: Collection = db[collection_name]
    operations: List[UpdateOne] = []

    for pred in predictions:
        filter_doc = {
            "student_id": pred["student_id"],
            "model_version": pred["model_version"],
        }
        operations.append(UpdateOne(filter_doc, {"$set": pred}, upsert=True))

    res = col.bulk_write(operations, ordered=False)

    return {
        "total": len(predictions),
        "upserted": len(res.upserted_ids),
        "matched": res.matched_count,
        "modified": res.modified_count,
    }


def generate_prediction_run_report(
    run_timestamp: datetime.datetime,
    model_version: str,
    artifact_names: List[str],
    total_attempted: int,
    successful_predictions: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    db_stats: Dict[str, int],
    output_path: Path = PREDICTION_REPORT_PATH,
) -> None:
    """Generate a clean, readable prediction execution summary report in Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample table rows
    table_rows = []
    for pred in successful_predictions[:10]:
        sid = pred["student_id"]
        row_id = pred.get("source_row_id", "N/A")
        p = pred["predictions"]
        dt_risk = "At-Risk" if p["decision_tree_at_risk"] == 1 else "Passing"
        rf_risk = "At-Risk" if p["random_forest_at_risk"] == 1 else "Passing"
        prob = f"{p['random_forest_risk_probability'] * 100:.1f}%"
        g3 = f"{p['random_forest_estimated_g3']:.2f}"

        table_rows.append(
            f"| `{sid}` | {row_id} | {dt_risk} | {rf_risk} | {prob} | {g3} |"
        )

    table_md = "\n".join(table_rows) if table_rows else "| None | - | - | - | - | - |"

    error_summary = "None (All records scored successfully)."
    if errors:
        error_lines = [f"- Student `{err['student_id']}`: {err['error']}" for err in errors[:5]]
        if len(errors) > 5:
            error_lines.append(f"- ... and {len(errors) - 5} more error(s).")
        error_summary = "\n".join(error_lines)

    md_content = f"""# EduPredict — Prediction Execution Run Report

> **Dataset Notice:** All predictions were generated for **synthetic student records** (`data/student-mat-synthetic.csv` stored in MongoDB `edupredict_db.students`).  
> **Predictive Disclaimer:** Predictions represent **statistical estimates** based solely on non-grade demographic, lifestyle, and behavioral factors; they do **not** constitute real-world academic evaluations or guarantees of future academic standing.

---

## 1. Run Execution Metadata

| Attribute | Details |
|---|---|
| **Execution Date & Time** | `{run_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}` |
| **Model Version** | `{model_version}` |
| **Model Artifacts Used** | `{', '.join(artifact_names)}` |
| **Database & Collection** | `{DATABASE_NAME}.{COLLECTION_STUDENT_PREDICTIONS}` |
| **Input Features Evaluated** | **30 features** (Grade columns G1, G2, G3 strictly excluded) |

---

## 2. Processing Summary

| Processing Metric | Count |
|---|---|
| **Total Records Attempted** | **{total_attempted:,}** |
| **Successful Predictions** | **{len(successful_predictions):,}** ({len(successful_predictions)/max(total_attempted, 1)*100:.1f}%) |
| **Failed Records** | **{len(errors):,}** |
| **MongoDB Upserted Records** | **{db_stats['upserted']:,}** |
| **MongoDB Matched / Updated Records** | **{db_stats['matched']:,}** |

---

## 3. Sample Scored Predictions (First 10 Records)

| Student ID | Row ID | Decision Tree Risk | Random Forest Risk | RF Risk Probability | Estimated G3 (0–20) |
|---|---|---|---|---|---|
{table_md}

---

## 4. Error & Validation Diagnostics

{error_summary}

---

## 5. Architectural Safeguards

- **Zero Target Leakage:** Features `G1`, `G2`, `G3`, and `at_risk` were stripped prior to prediction; inference was conducted strictly on non-grade attributes.
- **Idempotency Confirmed:** MongoDB predictions collection uses a compound upsert filter on `(student_id, model_version)`, ensuring duplicate executions update records cleanly in-place without data duplication.
- **Source Integrity Preserved:** Original documents in `edupredict_db.students` were accessed read-only with zero modifications.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Exported prediction run report to: {output_path}")


def main() -> None:
    """CLI Entrypoint for student prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="EduPredict Phase 6: Student Performance & Risk Prediction Pipeline"
    )
    parser.add_argument(
        "-s",
        "--student-id",
        type=str,
        default=None,
        help="Score a single student by MongoDB _id (hex string) or source_row_id (integer).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=DEFAULT_BATCH_LIMIT,
        help=f"Number of student documents to process (default: {DEFAULT_BATCH_LIMIT}). Use --all for full batch.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all eligible student records in the collection.",
    )
    parser.add_argument(
        "-v",
        "--model-version",
        type=str,
        default="v1",
        help="Model version tag for prediction tracking (default: 'v1').",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" EduPredict: Phase 6 - MongoDB + ML Prediction Pipeline")
    print("=" * 70)

    # 1. Initialize MongoDB connection
    print("\n[Step 1/5] Connecting to MongoDB...")
    client = get_mongo_client()
    try:
        ping_database(client)
        print(f"[SUCCESS] Connected to MongoDB on localhost:27017")
    except Exception as exc:
        print(f"[ERROR] Could not connect to MongoDB: {exc}")
        sys.exit(1)

    db = get_database(client)

    # 2. Initialize Prediction Service (loads models)
    print("\n[Step 2/5] Initializing Prediction Service and loading models...")
    try:
        service = PredictionService(model_version=args.model_version)
        print(f"[SUCCESS] Loaded models: {', '.join(service.artifact_names)}")
    except Exception as exc:
        print(f"[ERROR] Failed to initialize Prediction Service: {exc}")
        sys.exit(1)

    # 3. Retrieve eligible student documents
    print("\n[Step 3/5] Querying target student records from MongoDB...")
    try:
        students = fetch_students_to_predict(
            db=db,
            student_id_query=args.student_id,
            limit=args.limit,
            fetch_all=args.all,
        )
        print(f"[INFO] Retrieved {len(students)} eligible student record(s) to score.")
    except Exception as exc:
        print(f"[ERROR] Failed to query student records: {exc}")
        sys.exit(1)

    # 4. Generate predictions
    print("\n[Step 4/5] Executing prediction service on student records...")
    successful_preds, errors = service.predict_batch(students)

    print(f"[INFO] Predictions generated: {len(successful_preds)} succeeded, {len(errors)} failed.")

    # Print sample predictions to console
    if successful_preds:
        print("\n[SAMPLE PREDICTIONS (First 3 Records)]:")
        for pred in successful_preds[:3]:
            p = pred["predictions"]
            sid = pred["student_id"]
            row = pred.get("source_row_id", "N/A")
            dt = "At-Risk" if p["decision_tree_at_risk"] == 1 else "Passing"
            rf = "At-Risk" if p["random_forest_at_risk"] == 1 else "Passing"
            prob = p["random_forest_risk_probability"] * 100
            g3 = p["random_forest_estimated_g3"]
            print(f"  • Student [{sid}] (Row {row}):")
            print(f"      Decision Tree : {dt}")
            print(f"      Random Forest : {rf} (Risk Probability: {prob:.1f}%)")
            print(f"      Estimated G3  : {g3:.2f} / 20.00")

    if errors:
        print("\n[WARNING] Errors encountered:")
        for err in errors[:5]:
            print(f"  • Student [{err['student_id']}]: {err['error']}")

    # 5. Idempotent upsert into student_predictions
    print("\n[Step 5/5] Storing predictions in MongoDB 'student_predictions'...")
    try:
        db_stats = save_predictions_to_mongodb(db, successful_preds)
        print(f"[SUCCESS] Stored in MongoDB collection '{COLLECTION_STUDENT_PREDICTIONS}':")
        print(f"  - Total records : {db_stats['total']}")
        print(f"  - Upserted (new): {db_stats['upserted']}")
        print(f"  - Matched (old) : {db_stats['matched']}")
        print(f"  - Modified      : {db_stats['modified']}")
    except Exception as exc:
        print(f"[ERROR] Failed to save predictions to MongoDB: {exc}")
        sys.exit(1)

    # Generate Markdown Report
    run_timestamp = datetime.datetime.now(datetime.timezone.utc)
    generate_prediction_run_report(
        run_timestamp=run_timestamp,
        model_version=args.model_version,
        artifact_names=service.artifact_names,
        total_attempted=len(students),
        successful_predictions=successful_preds,
        errors=errors,
        db_stats=db_stats,
    )

    print("\n" + "=" * 70)
    print(" [SUCCESS] Phase 6: Prediction Workflow Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
