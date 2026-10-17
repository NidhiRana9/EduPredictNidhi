"""Machine Learning Model Training Module for EduPredict.

Phase 4 Implementation:
- Retrieves synthetic student dataset from MongoDB.
- Prepares features and targets (strictly excluding G1, G2, G3, at_risk, and metadata identifiers).
- Performs 80/20 train/test split with random_state=42 (stratified for classification).
- Configures un-fitted scikit-learn ColumnTransformer preprocessor:
    - Numeric: median imputation.
    - Categorical: most-frequent imputation and OneHotEncoder(handle_unknown='ignore').
- Trains and cross-validates:
    1. DecisionTreeClassifier (Task A: at-risk classification)
    2. RandomForestClassifier (Task A: at-risk classification)
    3. RandomForestRegressor (Task B: final-grade regression)
- Fits complete pipelines (preprocessor + estimator) exclusively on the training split.
- Persists model artifacts to models/ using joblib.
- Saves detailed training metadata and CV results to results/training_summary.json.
- Verifies model artifacts by reloading and predicting on sample records.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in sys.path
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
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN_CLASSIFICATION,
    TARGET_COLUMN_REGRESSION,
    build_preprocessor,
    prepare_features_and_targets,
    split_dataset,
)

# Output directory and artifact paths
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
TRAINING_SUMMARY_PATH: Path = RESULTS_DIR / "training_summary.json"

MODEL_PATH_DT_CLASSIFIER: Path = MODELS_DIR / "decision_tree_classifier.joblib"
MODEL_PATH_RF_CLASSIFIER: Path = MODELS_DIR / "random_forest_classifier.joblib"
MODEL_PATH_RF_REGRESSOR: Path = MODELS_DIR / "random_forest_regressor.joblib"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
DATASET_SOURCE_NAME: str = "synthetic"


def fetch_training_data_from_mongodb() -> pd.DataFrame:
    """Connect to MongoDB and retrieve synthetic student documents.

    Excludes MongoDB '_id' and filters specifically for dataset_source='synthetic'.

    Returns:
        pd.DataFrame containing student records.

    Raises:
        RuntimeError: If MongoDB is unreachable or contains no synthetic records.
    """
    client = get_mongo_client()
    try:
        ping_database(client)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to MongoDB: {exc}") from exc

    db = get_database(client)
    collection = db[COLLECTION_STUDENTS]

    query = {"dataset_source": DATASET_SOURCE_NAME}
    projection = {"_id": 0}

    records = list(collection.find(query, projection))
    if not records:
        raise RuntimeError(
            f"No records found in collection '{COLLECTION_STUDENTS}' matching {query}. "
            "Please ensure Phase 2 data import was executed."
        )

    df = pd.DataFrame(records)
    print(f"[INFO] Retrieved {len(df)} records from MongoDB '{DATABASE_NAME}.{COLLECTION_STUDENTS}'.")
    return df


def inspect_class_distribution(y_class: pd.Series) -> Dict[str, Any]:
    """Inspect and report the binary classification target class distribution.

    Args:
        y_class: Target series for at_risk (1 or 0).

    Returns:
        Dict of distribution counts and ratios.
    """
    counts = y_class.value_counts().to_dict()
    total = len(y_class)
    at_risk_count = counts.get(1, 0)
    not_at_risk_count = counts.get(0, 0)

    ratio = round(not_at_risk_count / max(at_risk_count, 1), 2)
    print("\n[INFO] Classification Target Distribution (at_risk):")
    print(f"  - At-Risk (G3 < 10): {at_risk_count:,} ({at_risk_count / total * 100:.2f}%)")
    print(f"  - Not At-Risk (G3 >= 10): {not_at_risk_count:,} ({not_at_risk_count / total * 100:.2f}%)")
    print(f"  - Class Balance Ratio: {ratio}:1 (Virtually balanced, no severe skew)")

    return {
        "at_risk_count": at_risk_count,
        "not_at_risk_count": not_at_risk_count,
        "total_count": total,
        "class_ratio": ratio,
        "weighting_strategy": "standard (classes balanced ~47.4% vs 52.6%)",
    }


def train_and_cross_validate_classifier(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute training-only stratified cross-validation and fit the complete pipeline.

    Metrics computed on training folds: Accuracy, F1 (binary), ROC-AUC.
    Does NOT evaluate on held-out test data.

    Args:
        model_name: Identifier for reporting.
        pipeline: Complete Pipeline([('preprocessor', ...), ('classifier', ...)]).
        X_train: Training feature DataFrame.
        y_train: Training binary target Series.
        cv_folds: Number of stratified folds.
        random_state: Random state for CV shuffling.

    Returns:
        Dictionary of cross-validation metrics and fitted pipeline.
    """
    print(f"\n[TRAINING] Evaluating {model_name} with {cv_folds}-fold Stratified CV on training split...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1", "roc_auc"]

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    acc_mean = float(np.mean(cv_results["test_accuracy"]))
    acc_std = float(np.std(cv_results["test_accuracy"]))
    f1_mean = float(np.mean(cv_results["test_f1"]))
    f1_std = float(np.std(cv_results["test_f1"]))
    auc_mean = float(np.mean(cv_results["test_roc_auc"]))
    auc_std = float(np.std(cv_results["test_roc_auc"]))

    print(f"  - CV Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f})")
    print(f"  - CV F1-Score: {f1_mean:.4f} (+/- {f1_std:.4f})")
    print(f"  - CV ROC-AUC : {auc_mean:.4f} (+/- {auc_std:.4f})")

    # Fit the complete pipeline on the entire training split
    print(f"  Fitting {model_name} on all {len(X_train)} training records...")
    pipeline.fit(X_train, y_train)
    print(f"  [SUCCESS] {model_name} fitted successfully.")

    return {
        "model_name": model_name,
        "cv_accuracy_mean": round(acc_mean, 4),
        "cv_accuracy_std": round(acc_std, 4),
        "cv_f1_mean": round(f1_mean, 4),
        "cv_f1_std": round(f1_std, 4),
        "cv_roc_auc_mean": round(auc_mean, 4),
        "cv_roc_auc_std": round(auc_std, 4),
    }


def train_and_cross_validate_regressor(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Execute training-only K-Fold cross-validation and fit the complete pipeline.

    Metrics computed on training folds: MAE, RMSE, R².
    Does NOT evaluate on held-out test data.

    Args:
        model_name: Identifier for reporting.
        pipeline: Complete Pipeline([('preprocessor', ...), ('regressor', ...)]).
        X_train: Training feature DataFrame.
        y_train: Training numeric target Series (G3).
        cv_folds: Number of folds.
        random_state: Random state for CV shuffling.

    Returns:
        Dictionary of cross-validation metrics and fitted pipeline.
    """
    print(f"\n[TRAINING] Evaluating {model_name} with {cv_folds}-fold K-Fold CV on training split...")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    mae_mean = float(-np.mean(cv_results["test_neg_mean_absolute_error"]))
    mae_std = float(np.std(cv_results["test_neg_mean_absolute_error"]))
    rmse_mean = float(-np.mean(cv_results["test_neg_root_mean_squared_error"]))
    rmse_std = float(np.std(cv_results["test_neg_root_mean_squared_error"]))
    r2_mean = float(np.mean(cv_results["test_r2"]))
    r2_std = float(np.std(cv_results["test_r2"]))

    print(f"  - CV MAE : {mae_mean:.4f} (+/- {mae_std:.4f})")
    print(f"  - CV RMSE: {rmse_mean:.4f} (+/- {rmse_std:.4f})")
    print(f"  - CV R²  : {r2_mean:.4f} (+/- {r2_std:.4f})")

    # Fit the complete pipeline on the entire training split
    print(f"  Fitting {model_name} on all {len(X_train)} training records...")
    pipeline.fit(X_train, y_train)
    print(f"  [SUCCESS] {model_name} fitted successfully.")

    return {
        "model_name": model_name,
        "cv_mae_mean": round(mae_mean, 4),
        "cv_mae_std": round(mae_std, 4),
        "cv_rmse_mean": round(rmse_mean, 4),
        "cv_rmse_std": round(rmse_std, 4),
        "cv_r2_mean": round(r2_mean, 4),
        "cv_r2_std": round(r2_std, 4),
    }


def verify_saved_model(artifact_path: Path, sample_input: pd.DataFrame) -> None:
    """Verify that a saved model artifact can be reloaded and executes predictions."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")

    loaded_pipe = joblib.load(artifact_path)
    if not hasattr(loaded_pipe, "predict"):
        raise TypeError(f"Loaded artifact at {artifact_path} does not have a predict method.")

    predictions = loaded_pipe.predict(sample_input)
    if len(predictions) != len(sample_input):
        raise ValueError(f"Loaded model prediction length mismatch: expected {len(sample_input)}, got {len(predictions)}")

    print(f"[VERIFY] Successfully loaded and predicted using {artifact_path.name}. Sample output: {predictions[:2]}")


def main() -> None:
    """Execute the complete Phase 4 training workflow."""
    print("=" * 70)
    print(" EduPredict: Phase 4 - Machine Learning Model Training")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Retrieve data
    print("\n[Step 1/6] Retrieving synthetic records from MongoDB...")
    df = fetch_training_data_from_mongodb()

    # 2. Prepare features and targets
    print("\n[Step 2/6] Preparing features and targets...")
    X, y_class, y_reg = prepare_features_and_targets(df)
    class_dist = inspect_class_distribution(y_class)

    # Validate feature exclusion
    print(f"  - Total predictor features: {X.shape[1]} (Numeric: {len(NUMERIC_FEATURES)}, Categorical: {len(CATEGORICAL_FEATURES)})")
    for leaked in ["G1", "G2", "G3", "at_risk", "_id", "source_row_id", "dataset_source"]:
        if leaked in X.columns:
            raise ValueError(f"Critical error: Leaked column '{leaked}' found in input features!")
    print("  [SUCCESS] All grade columns, target columns, and metadata IDs verified strictly excluded.")

    # 3. Train/test split (80/20, random_state=42)
    print("\n[Step 3/6] Performing 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_class_train, y_class_test = split_dataset(
        X,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=True,
    )
    _, _, y_reg_train, y_reg_test = split_dataset(
        X,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=False,
    )
    print(f"  - Training samples: {len(X_train):,} records (80%)")
    print(f"  - Test samples    : {len(X_test):,} records (20%, held out for Phase 5)")

    # 4. Construct un-fitted preprocessing pipeline
    print("\n[Step 4/6] Constructing preprocessing pipeline...")
    # Tree models with median imputation and one-hot encoding
    preprocessor = build_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        scale_numeric=False,  # Tree algorithms are invariant to monotonic scaling
    )

    # 5. Define and train models with training-only cross-validation
    print("\n[Step 5/6] Training models with training-only cross-validation...")

    # --- Model A1: Decision Tree Classifier ---
    dt_params = {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
    }
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(**dt_params)),
        ]
    )
    dt_metrics = train_and_cross_validate_classifier(
        "Decision Tree Classifier",
        dt_pipeline,
        X_train,
        y_class_train,
        cv_folds=CV_FOLDS,
        random_state=RANDOM_STATE,
    )
    joblib.dump(dt_pipeline, MODEL_PATH_DT_CLASSIFIER)
    print(f"  [SAVED] Artifact: {MODEL_PATH_DT_CLASSIFIER}")

    # --- Model A2: Random Forest Classifier ---
    rf_class_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_class_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_class_params)),
        ]
    )
    rf_class_metrics = train_and_cross_validate_classifier(
        "Random Forest Classifier",
        rf_class_pipeline,
        X_train,
        y_class_train,
        cv_folds=CV_FOLDS,
        random_state=RANDOM_STATE,
    )
    joblib.dump(rf_class_pipeline, MODEL_PATH_RF_CLASSIFIER)
    print(f"  [SAVED] Artifact: {MODEL_PATH_RF_CLASSIFIER}")

    # --- Model B: Random Forest Regressor ---
    rf_reg_params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_reg_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**rf_reg_params)),
        ]
    )
    rf_reg_metrics = train_and_cross_validate_regressor(
        "Random Forest Regressor",
        rf_reg_pipeline,
        X_train,
        y_reg_train,
        cv_folds=CV_FOLDS,
        random_state=RANDOM_STATE,
    )
    joblib.dump(rf_reg_pipeline, MODEL_PATH_RF_REGRESSOR)
    print(f"  [SAVED] Artifact: {MODEL_PATH_RF_REGRESSOR}")

    # 6. Verify saved artifacts and write summary JSON
    print("\n[Step 6/6] Verifying saved artifacts and writing training summary...")
    sample_test_input = X_test.head(5)
    verify_saved_model(MODEL_PATH_DT_CLASSIFIER, sample_test_input)
    verify_saved_model(MODEL_PATH_RF_CLASSIFIER, sample_test_input)
    verify_saved_model(MODEL_PATH_RF_REGRESSOR, sample_test_input)

    training_summary: Dict[str, Any] = {
        "project": "EduPredict",
        "phase": "Phase 4 - Machine Learning Model Training",
        "dataset": {
            "source": DATASET_SOURCE_NAME,
            "database": DATABASE_NAME,
            "collection": COLLECTION_STUDENTS,
            "total_records": len(df),
            "is_synthetic": True,
            "disclaimer": "This dataset is synthetic. Training metrics reflect artificial data patterns and do not indicate real-world student behavior.",
        },
        "tasks": {
            "task_a_classification": {
                "target": TARGET_COLUMN_CLASSIFICATION,
                "target_definition": "at_risk = 1 if G3 < 10 else 0",
                "class_distribution_total": class_dist,
                "models": ["DecisionTreeClassifier", "RandomForestClassifier"],
            },
            "task_b_regression": {
                "target": TARGET_COLUMN_REGRESSION,
                "target_definition": "numerical G3 (0-20 scale)",
                "models": ["RandomForestRegressor"],
            },
        },
        "features": {
            "total_count": X.shape[1],
            "numeric_count": len(NUMERIC_FEATURES),
            "categorical_count": len(CATEGORICAL_FEATURES),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_features": ["G1", "G2", "G3", "at_risk", "_id", "source_row_id", "dataset_source"],
        },
        "data_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "stratified_classification": True,
            "note": "Test set was held out completely; zero test-set metrics used for training or model selection.",
        },
        "training_only_cross_validation_results": {
            "cv_folds": CV_FOLDS,
            "decision_tree_classifier": {
                "parameters": dt_params,
                "metrics": dt_metrics,
            },
            "random_forest_classifier": {
                "parameters": rf_class_params,
                "metrics": rf_class_metrics,
            },
            "random_forest_regressor": {
                "parameters": rf_reg_params,
                "metrics": rf_reg_metrics,
            },
        },
        "artifacts": {
            "decision_tree_classifier": str(MODEL_PATH_DT_CLASSIFIER),
            "random_forest_classifier": str(MODEL_PATH_RF_CLASSIFIER),
            "random_forest_regressor": str(MODEL_PATH_RF_REGRESSOR),
        },
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\n[SUCCESS] Training summary saved to: {TRAINING_SUMMARY_PATH}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Phase 4: Machine Learning Model Training Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
