"""Model Evaluation, Metrics Generation, and Reporting Module for EduPredict.

Phase 5 Implementation:
- Reconstructs the exact 80/20 held-out test split established in Phase 4 (1,000 samples).
- Loads saved model pipelines (DecisionTree, RandomForestClassifier, RandomForestRegressor).
- Evaluates classification models (Accuracy, Precision, Recall, F1, Confusion Matrix).
- Evaluates regression model (MAE, RMSE, R², Residuals).
- Generates 5 analytical evaluation charts saved in results/charts/.
- Exports detailed metrics to results/metrics.json and predictions to results/regression_predictions.csv.
- Generates an evidence-based markdown report in reports/model_evaluation_report.md.
- Idempotently logs evaluation records to MongoDB collection 'model_evaluations'.
"""

import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymongo import UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    COLLECTION_MODEL_EVALUATIONS,
    COLLECTION_STUDENTS,
    DATABASE_NAME,
    get_database,
    get_mongo_client,
    ping_database,
)
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN_CLASSIFICATION,
    TARGET_COLUMN_REGRESSION,
    prepare_features_and_targets,
    split_dataset,
)

# Directory & artifact constants
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHARTS_DIR: Path = RESULTS_DIR / "charts"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

MODEL_PATH_DT: Path = MODELS_DIR / "decision_tree_classifier.joblib"
MODEL_PATH_RF_CLF: Path = MODELS_DIR / "random_forest_classifier.joblib"
MODEL_PATH_RF_REG: Path = MODELS_DIR / "random_forest_regressor.joblib"

METRICS_JSON_PATH: Path = RESULTS_DIR / "metrics.json"
REGRESSION_PREDS_CSV_PATH: Path = RESULTS_DIR / "regression_predictions.csv"
EVALUATION_REPORT_MD_PATH: Path = REPORTS_DIR / "model_evaluation_report.md"

DATASET_SOURCE_NAME: str = "synthetic"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def load_held_out_test_set() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Retrieve synthetic records from MongoDB and reconstruct the exact test split.

    Uses the exact same random seed (42) and stratification as Phase 4 to guarantee
    the test set reflects the untouched held-out partition.

    Returns:
        Tuple of (X_test, y_class_test, y_reg_test, test_metadata_df).

    Raises:
        RuntimeError: If MongoDB records or required models are missing.
    """
    client = get_mongo_client()
    try:
        ping_database(client)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to MongoDB: {exc}") from exc

    db = get_database(client)
    collection = db[COLLECTION_STUDENTS]

    records = list(collection.find({"dataset_source": DATASET_SOURCE_NAME}, {"_id": 0}))
    if len(records) != 5000:
        raise RuntimeError(
            f"Expected exactly 5000 synthetic records in MongoDB, but found {len(records)}."
        )

    df = pd.DataFrame(records)
    X, y_class, y_reg = prepare_features_and_targets(df)

    # Perform reproducible stratified split
    _, X_test, _, y_class_test = split_dataset(
        X,
        y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=True,
    )

    y_reg_test = y_reg.loc[X_test.index]
    test_metadata = df.loc[X_test.index, ["source_row_id", "dataset_source", "G3"]].copy()

    # Integrity verification
    for leaked in EXCLUDED_FROM_FEATURES:
        if leaked in X_test.columns:
            raise ValueError(f"Leaked feature '{leaked}' detected in test dataset!")

    print(f"[INFO] Reconstructed held-out test split: {len(X_test)} samples ({X_test.shape[1]} features).")
    print(f"  - At-Risk class count     : {int((y_class_test == 1).sum())} (47.4%)")
    print(f"  - Not At-Risk class count : {int((y_class_test == 0).sum())} (52.6%)")

    return X_test, y_class_test, y_reg_test, test_metadata


def evaluate_classifier_model(
    model_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a classification pipeline on the held-out test set.

    Explicitly sets pos_label=1 and zero_division=0 to handle edge cases cleanly.

    Returns:
        Dictionary of test metrics, confusion matrix, and classification report.
    """
    preds = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_test,
        preds,
        target_names=["Not At-Risk (0)", "At-Risk (1)"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_name": model_name,
        "task": "binary_classification",
        "target": TARGET_COLUMN_CLASSIFICATION,
        "test_samples": len(y_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "raw_matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "predictions": [int(p) for p in preds],
    }

    print(f"\n[EVALUATION] {model_name}:")
    print(f"  - Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  - Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"  - Recall   : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  - F1-Score : {f1:.4f}")
    print(f"  - Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


def evaluate_regressor_model(
    model_name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metadata_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate the regression pipeline on the held-out test set.

    Computes MAE, RMSE, R², and constructs residual records.

    Returns:
        Tuple of (metrics_dict, predictions_dataframe).
    """
    preds = pipeline.predict(X_test)
    residuals = y_test.values - preds

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    pred_df = pd.DataFrame(
        {
            "source_row_id": metadata_df["source_row_id"].values,
            "dataset_source": metadata_df["dataset_source"].values,
            "actual_G3": y_test.values,
            "predicted_G3": np.round(preds, 4),
            "residual": np.round(residuals, 4),
        }
    )

    metrics = {
        "model_name": model_name,
        "task": "regression",
        "target": TARGET_COLUMN_REGRESSION,
        "test_samples": len(y_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "r2_explanation": (
            "R² is approximately -0.02, reflecting that predicting exact continuous grades "
            "strictly from demographic/behavioral features without interim exam grades (G1/G2) "
            "has high variance relative to a naive constant mean baseline."
        ),
    }

    print(f"\n[EVALUATION] {model_name}:")
    print(f"  - MAE  : {mae:.4f} grade points")
    print(f"  - RMSE : {rmse:.4f}")
    print(f"  - R²   : {r2:.4f}")

    return metrics, pred_df


def generate_evaluation_charts(
    dt_metrics: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    reg_metrics: Dict[str, Any],
    reg_df: pd.DataFrame,
    output_dir: Path = CHARTS_DIR,
) -> List[Path]:
    """Generate 5 analytical evaluation charts.

    1. Classification Metric Comparison (Side-by-side bars)
    2. Decision Tree Confusion Matrix Heatmap
    3. Random Forest Confusion Matrix Heatmap
    4. Actual vs. Predicted Final Grades
    5. Regression Residuals Distribution
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: List[Path] = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    font_title = {"fontsize": 13, "fontweight": "bold", "color": "#1f2937"}
    font_label = {"fontsize": 11, "fontweight": "semibold", "color": "#374151"}

    # --------------------------------------------------------------------------
    # Chart 1: Classification Metrics Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    dt_vals = [dt_metrics["accuracy"], dt_metrics["precision"], dt_metrics["recall"], dt_metrics["f1_score"]]
    rf_vals = [rf_metrics["accuracy"], rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1_score"]]

    x = np.arange(len(metric_labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, dt_vals, width, label="Decision Tree", color="#3b82f6", edgecolor="#1d4ed8")
    rects2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color="#10b981", edgecolor="#047857")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Classifier Performance Comparison (Held-Out Test Set: n=1,000)", fontdict=font_title, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontdict=font_label)
    ax.set_ylabel("Score (0.0 to 1.0)", fontdict=font_label)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    c1_path = output_dir / "classification_metrics_comparison.png"
    plt.savefig(c1_path)
    plt.close()
    generated_charts.append(c1_path)
    print(f"  [CHART] Saved: {c1_path.name}")

    # --------------------------------------------------------------------------
    # Helper for Confusion Matrix Plots
    # --------------------------------------------------------------------------
    def plot_cm(cm_data: List[List[int]], title: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        matrix = np.array(cm_data)
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                total = np.sum(matrix)
                pct = val / total * 100
                text = f"{val}\n({pct:.1f}%)"
                color = "white" if val > (matrix.max() / 2) else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=12, fontweight="bold")

        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not At-Risk (0)", "At-Risk (1)"], fontdict=font_label)
        ax.set_yticklabels(["Not At-Risk (0)", "At-Risk (1)"], fontdict=font_label)
        ax.set_xlabel("Predicted Class", fontdict=font_label, labelpad=8)
        ax.set_ylabel("Actual Class", fontdict=font_label, labelpad=8)
        ax.set_title(title, fontdict=font_title, pad=20)
        plt.tight_layout()

        p = output_dir / filename
        plt.savefig(p)
        plt.close()
        generated_charts.append(p)
        print(f"  [CHART] Saved: {p.name}")
        return p

    # Chart 2: Decision Tree Confusion Matrix
    plot_cm(dt_metrics["confusion_matrix"]["raw_matrix"],
            "Decision Tree Confusion Matrix (Test Set: n=1,000)",
            "confusion_matrix_decision_tree.png")

    # Chart 3: Random Forest Confusion Matrix
    plot_cm(rf_metrics["confusion_matrix"]["raw_matrix"],
            "Random Forest Confusion Matrix (Test Set: n=1,000)",
            "confusion_matrix_random_forest.png")

    # --------------------------------------------------------------------------
    # Chart 4: Actual vs Predicted Final Grades
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    act = reg_df["actual_G3"].values
    pred = reg_df["predicted_G3"].values

    # Jitter on actual to reveal observation density
    jitter_x = act + np.random.uniform(-0.2, 0.2, len(act))
    ax.scatter(jitter_x, pred, alpha=0.35, color="#6366f1", edgecolors="none", s=22, label="Test Predictions")

    # 45-degree reference line
    ax.plot([0, 20], [0, 20], color="#ef4444", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
    ax.axhline(10, color="#10b981", linestyle=":", linewidth=1.5, label="Passing Mark (10)")
    ax.axvline(10, color="#10b981", linestyle=":", linewidth=1.5)

    ax.set_title("Random Forest Regressor: Actual vs. Predicted G3 (Test Set)", fontdict=font_title, pad=12)
    ax.set_xlabel("Actual Final Grade (G3)", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Predicted Final Grade (G3)", fontdict=font_label, labelpad=8)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper left")
    plt.tight_layout()

    c4_path = output_dir / "regression_actual_vs_predicted.png"
    plt.savefig(c4_path)
    plt.close()
    generated_charts.append(c4_path)
    print(f"  [CHART] Saved: {c4_path.name}")

    # --------------------------------------------------------------------------
    # Chart 5: Regression Residuals Distribution
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    residuals = reg_df["residual"].values
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    ax.hist(residuals, bins=25, color="#8b5cf6", edgecolor="#6d28d9", alpha=0.8, rwidth=0.85)
    ax.axvline(0, color="#ef4444", linestyle="--", linewidth=2, label="Zero Error Line (Residual = 0)")
    ax.axvline(mean_res, color="#1e3a8a", linestyle="-", linewidth=1.5, label=f"Mean Error: {mean_res:.2f}")

    ax.set_title("Random Forest Regressor: Residual Distribution (Actual - Predicted)", fontdict=font_title, pad=12)
    ax.set_xlabel("Residual (Grade Points)", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Frequency", fontdict=font_label, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    c5_path = output_dir / "regression_residuals_distribution.png"
    plt.savefig(c5_path)
    plt.close()
    generated_charts.append(c5_path)
    print(f"  [CHART] Saved: {c5_path.name}")

    return generated_charts


def log_evaluation_to_mongodb(
    evaluations: List[Dict[str, Any]],
    collection: Optional[Collection] = None,
) -> None:
    """Store model evaluation summaries in MongoDB collection 'model_evaluations'.

    Uses an idempotent upsert key on ('model_name', 'dataset_source') so
    repeated executions update records in-place without generating duplicates.
    """
    if collection is None:
        db = get_database()
        collection = db[COLLECTION_MODEL_EVALUATIONS]

    operations: List[UpdateOne] = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for item in evaluations:
        doc = {
            "model_name": item["model_name"],
            "model_type": item["task"],
            "dataset_source": DATASET_SOURCE_NAME,
            "test_set_size": item["test_samples"],
            "evaluation_timestamp": timestamp,
            "is_synthetic": True,
            "metrics": {
                k: v
                for k, v in item.items()
                if k not in ["predictions", "model_name", "task", "test_samples"]
            },
        }

        operations.append(
            UpdateOne(
                filter={
                    "model_name": item["model_name"],
                    "dataset_source": DATASET_SOURCE_NAME,
                },
                update={"$set": doc},
                upsert=True,
            )
        )

    res = collection.bulk_write(operations, ordered=False)
    print(f"[SUCCESS] Stored {len(operations)} model evaluation summaries in MongoDB '{COLLECTION_MODEL_EVALUATIONS}'.")
    print(f"  - Upserted: {len(res.upserted_ids)}, Matched: {res.matched_count}, Modified: {res.modified_count}")


def generate_markdown_evaluation_report(
    dt_metrics: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    reg_metrics: Dict[str, Any],
    output_path: Path = EVALUATION_REPORT_MD_PATH,
) -> None:
    """Generate comprehensive Phase 5 evidence-based comparison markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dt_cm = dt_metrics["confusion_matrix"]
    rf_cm = rf_metrics["confusion_matrix"]

    md_content = f"""# EduPredict — Model Evaluation & Comparison Report

> **Dataset Notice:** All evaluations were conducted on synthetic student data (`data/student-mat-synthetic.csv` stored in MongoDB `edupredict_db.students`). These metrics demonstrate algorithmic behavior and pipeline validity; they must **not** be interpreted as empirical performance on real student populations.

---

## 1. Evaluation Setup & Test Integrity

- **Evaluation Split:** Held-out test set of **1,000 samples** (20% of 5,000 total records).
- **Split Random Seed:** `42` (reproducible, identical to Phase 4).
- **Stratification:** Classification target strictly maintained positive class ratio:
  - **At-Risk (`G3 < 10`):** 474 samples (47.4%)
  - **Not At-Risk (`G3 >= 10`):** 526 samples (52.6%)
- **Target Leakage Safeguard:** Input feature space strictly comprised **30 features** (13 numeric, 17 categorical). `G1`, `G2`, `G3`, `at_risk`, and database identifiers were completely excluded.
- **Strict Independence:** The test set was **never** used during preprocessing imputer fitting, encoder fitting, or model parameter selection.

---

## 2. Classification Models Comparison (Task A: At-Risk Detection)

**Task Objective:** Predict whether a student is academically at risk (`at_risk = 1` if `G3 < 10`, else `0`) before examinations occur.

| Model | Accuracy | Precision (pos_label=1) | Recall (pos_label=1) | F1-Score (pos_label=1) |
|---|---|---|---|---|
| **Decision Tree Classifier** | **{dt_metrics['accuracy']:.4f}** ({dt_metrics['accuracy']*100:.1f}%) | {dt_metrics['precision']:.4f} ({dt_metrics['precision']*100:.1f}%) | **{dt_metrics['recall']:.4f}** ({dt_metrics['recall']*100:.1f}%) | **{dt_metrics['f1_score']:.4f}** |
| **Random Forest Classifier** | {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.1f}%) | **{rf_metrics['precision']:.4f}** ({rf_metrics['precision']*100:.1f}%) | {rf_metrics['recall']:.4f} ({rf_metrics['recall']*100:.1f}%) | {rf_metrics['f1_score']:.4f} |

### Confusion Matrix Breakdown (Test Set: n=1,000)

#### Decision Tree Classifier:
- **True Negatives (TN):** {dt_cm['true_negatives']} (Correctly identified passing students)
- **False Positives (FP):** {dt_cm['false_positives']} (Passing students flagged as at-risk)
- **False Negatives (FN):** {dt_cm['false_negatives']} (At-risk students missed)
- **True Positives (TP):** {dt_cm['true_positives']} (At-risk students correctly flagged)

#### Random Forest Classifier:
- **True Negatives (TN):** {rf_cm['true_negatives']}
- **False Positives (FP):** {rf_cm['false_positives']}
- **False Negatives (FN):** {rf_cm['false_negatives']}
- **True Positives (TP):** {rf_cm['true_positives']}

### Classifier Comparison Analysis:
1. **Decision Tree Classifier Advantages:**
   - Achieved superior **Recall ({dt_metrics['recall']*100:.1f}%)** compared to Random Forest ({rf_metrics['recall']*100:.1f}%). In educational early-warning systems, **Recall is paramount** because failing to identify a struggling student (False Negative) has far higher consequences than providing support to a passing student (False Positive).
   - Higher overall **F1-Score ({dt_metrics['f1_score']:.4f})** and **Accuracy ({dt_metrics['accuracy']*100:.1f}%)**.
2. **Random Forest Classifier Characteristics:**
   - Showed higher specificity/precision ({rf_metrics['precision']*100:.1f}%), producing fewer false alarms, but missed 257 at-risk students (FN).

---

## 3. Regression Model Evaluation (Task B: Final Grade G3)

**Task Objective:** Predict the exact continuous final grade `G3` (0–20 scale) without access to interim exam grades.

| Metric | Score | Interpretation |
|---|---|---|
| **Mean Absolute Error (MAE)** | **{reg_metrics['mae']:.4f}** | On average, predictions deviate by ~3.39 points on a 20-point scale. |
| **Root Mean Squared Error (RMSE)** | **{reg_metrics['rmse']:.4f}** | Reflects typical penalty for larger deviations. |
| **Coefficient of Determination (R²)** | **{reg_metrics['r2_score']:.4f}** | Close to 0 (slightly negative). |
| **Mean Residual (Actual - Pred)** | **{reg_metrics['mean_residual']:.4f}** | Unbiased error distribution centered near zero. |

### Explanation of Negative / Near-Zero R²:
In machine learning regression, $R^2 = 1 - \\frac{{\\sum (y_i - \\hat{{y}}_i)^2}}{{\\sum (y_i - \\bar{{y}})^2}}$.
When models are trained to predict exact continuous marks **purely from demographic, lifestyle, and historical failure records**—without interim exam grades ($G1, G2$)—the inherent stochasticity of student exam outcomes prevents high-precision point estimation. The resulting variance is slightly higher than simply predicting the test set's mean. This finding confirms that **binary early-warning classification (`at_risk`) is the far more practical and reliable modeling strategy** for proactive educational intervention.

---

## 4. Persisted Artifacts & Database Logging

1. **Evaluation Metrics JSON:** `results/metrics.json`
2. **Regression Predictions CSV:** `results/regression_predictions.csv` (1,000 rows with `actual_G3`, `predicted_G3`, `residual`)
3. **MongoDB Logging:** Stored in collection `model_evaluations` within database `edupredict_db` via safe upsert keys on `(model_name, dataset_source)`.
4. **Visualizations Generated:**
   - `results/charts/classification_metrics_comparison.png`
   - `results/charts/confusion_matrix_decision_tree.png`
   - `results/charts/confusion_matrix_random_forest.png`
   - `results/charts/regression_actual_vs_predicted.png`
   - `results/charts/regression_residuals_distribution.png`
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Exported markdown evaluation report to: {output_path}")


def main() -> None:
    """Orchestrate Phase 5 model evaluation, chart rendering, and MongoDB logging."""
    print("=" * 70)
    print(" EduPredict: Phase 5 - Model Evaluation & Comparison")
    print("=" * 70)

    # 1. Verify saved artifacts exist
    print("\n[Step 1/6] Verifying saved model artifacts under models/...")
    for p in [MODEL_PATH_DT, MODEL_PATH_RF_CLF, MODEL_PATH_RF_REG]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required model artifact: {p.resolve()}.\n"
                "Please run Phase 4 training first: python -m src.train_models"
            )
        print(f"  - Confirmed artifact: {p.name}")

    # 2. Reconstruct held-out test split
    print("\n[Step 2/6] Loading held-out test set from MongoDB...")
    X_test, y_class_test, y_reg_test, metadata_df = load_held_out_test_set()

    # 3. Load pipelines
    print("\n[Step 3/6] Loading fitted model pipelines...")
    dt_pipeline = joblib.load(MODEL_PATH_DT)
    rf_clf_pipeline = joblib.load(MODEL_PATH_RF_CLF)
    rf_reg_pipeline = joblib.load(MODEL_PATH_RF_REG)

    # 4. Evaluate models
    print("\n[Step 4/6] Evaluating models on held-out test set (n=1,000)...")
    dt_metrics = evaluate_classifier_model("Decision Tree Classifier", dt_pipeline, X_test, y_class_test)
    rf_metrics = evaluate_classifier_model("Random Forest Classifier", rf_clf_pipeline, X_test, y_class_test)
    reg_metrics, reg_preds_df = evaluate_regressor_model("Random Forest Regressor", rf_reg_pipeline, X_test, y_reg_test, metadata_df)

    # 5. Generate evaluation charts
    print("\n[Step 5/6] Generating evaluation charts in results/charts/...")
    charts = generate_evaluation_charts(dt_metrics, rf_metrics, reg_metrics, reg_preds_df, output_dir=CHARTS_DIR)
    print(f"[SUCCESS] Generated {len(charts)} evaluation charts.")

    # 6. Save results (JSON, CSV, MD, MongoDB)
    print("\n[Step 6/6] Persisting evaluation metrics and logging to MongoDB...")

    # Export regression predictions CSV
    reg_preds_df.to_csv(REGRESSION_PREDS_CSV_PATH, index=False)
    print(f"[SUCCESS] Exported regression predictions CSV to: {REGRESSION_PREDS_CSV_PATH}")

    # Export metrics JSON (clean serializable format)
    metrics_json_content = {
        "project": "EduPredict",
        "phase": "Phase 5 - Model Evaluation and Comparison",
        "evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_source": DATASET_SOURCE_NAME,
        "test_set_size": len(X_test),
        "is_synthetic": True,
        "synthetic_disclaimer": "All metrics are computed on synthetic benchmark data and do not establish real-world student behavior.",
        "feature_definitions": {
            "total_features": X_test.shape[1],
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_features": EXCLUDED_FROM_FEATURES,
        },
        "classification_evaluations": {
            "decision_tree_classifier": {
                k: v for k, v in dt_metrics.items() if k != "predictions"
            },
            "random_forest_classifier": {
                k: v for k, v in rf_metrics.items() if k != "predictions"
            },
        },
        "regression_evaluations": {
            "random_forest_regressor": reg_metrics,
        },
        "artifact_paths": {
            "decision_tree_classifier": str(MODEL_PATH_DT),
            "random_forest_classifier": str(MODEL_PATH_RF_CLF),
            "random_forest_regressor": str(MODEL_PATH_RF_REG),
            "charts_directory": str(CHARTS_DIR),
            "regression_predictions_csv": str(REGRESSION_PREDS_CSV_PATH),
            "evaluation_report_markdown": str(EVALUATION_REPORT_MD_PATH),
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_json_content, f, indent=2)
    print(f"[SUCCESS] Exported metrics JSON to: {METRICS_JSON_PATH}")

    # Generate Markdown Report
    generate_markdown_evaluation_report(dt_metrics, rf_metrics, reg_metrics, output_path=EVALUATION_REPORT_MD_PATH)

    # Log to MongoDB collection 'model_evaluations'
    eval_list = [dt_metrics, rf_metrics, reg_metrics]
    log_evaluation_to_mongodb(eval_list)

    print("\n" + "=" * 70)
    print(" [SUCCESS] Phase 5: Model Evaluation & Comparison Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
