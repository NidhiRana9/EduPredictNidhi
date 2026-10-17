# EduPredict — Student Performance Analysis and Prediction

EduPredict is an end-to-end Machine Learning and Data Analysis system designed to analyze student demographic and academic factors, predict student performance, and provide actionable insights for educators.

> **Current Status:**  
> - **Phase 1: Project Setup and MongoDB Connectivity** (Completed & Verified)  
> - **Phase 2: Dataset Validation and MongoDB Import** (Completed & Verified)  
> - **Phase 3: Exploratory Data Analysis & Feature Preprocessing** (Completed & Verified)  
> - **Phase 4: Machine Learning Model Training & Pipeline Persistence** (Completed & Verified)  
> - **Phase 5: Model Evaluation & Evidence-Based Comparison** (Completed & Verified)  
> - **Phase 6: MongoDB + ML Prediction Pipeline** (Completed & Verified)

---

## 1. Technology Stack

- **Language:** Python (3.10+; verified with Python 3.14)
- **Database:** MongoDB Community Server (local unauthenticated instance on port `27017`)
- **Database Driver:** PyMongo (`>=4.6.0`)
- **Data Manipulation:** pandas (`>=2.2.0`), NumPy (`>=2.3.0`)
- **Machine Learning & Preprocessing:** scikit-learn (`>=1.4.0`)
- **Model Persistence:** joblib (`>=1.3.0`)
- **Visualization:** matplotlib (`>=3.8.0`)
- **Testing:** Python standard library `unittest` (32 automated tests passed)

---

## 2. Project Directory Structure

```text
EduPredict/
│
├── data/
│   ├── .gitkeep
│   └── student-mat-synthetic.csv # Synthetic student dataset (5,000 rows, 33 columns)
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── database.py               # Reusable MongoDB client & collections manager
│   ├── import_data.py            # CSV validator and idempotent MongoDB importer
│   ├── analyze_data.py           # MongoDB data retriever, EDA charts & report generator
│   ├── preprocessing.py          # Feature partitioning, pipeline builder & train/test splitter
│   ├── train_models.py           # Model training, CV evaluation & joblib artifact persistence
│   ├── evaluate_models.py        # Held-out test evaluation, chart generation & MongoDB logging
│   ├── prediction_service.py     # Reusable inference service with leakage prevention
│   └── predict_students.py       # Batch/single student prediction CLI & MongoDB upsert
│
├── models/                       # Persisted, fitted scikit-learn pipelines (.joblib)
│   ├── .gitkeep
│   ├── decision_tree_classifier.joblib
│   ├── random_forest_classifier.joblib
│   └── random_forest_regressor.joblib
│
├── results/                      # Metric outputs, predictions, and reports
│   ├── charts/                   # 11 Generated visualizations (.png)
│   │   ├── absences_vs_g3.png
│   │   ├── at_risk_distribution.png
│   │   ├── classification_metrics_comparison.png
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── failures_vs_g3.png
│   │   ├── g3_distribution.png
│   │   ├── numeric_correlation_heatmap.png
│   │   ├── regression_actual_vs_predicted.png
│   │   ├── regression_residuals_distribution.png
│   │   └── studytime_distribution.png
│   ├── cleaned_students.csv      # Exported clean analysis dataset (5,000 rows)
│   ├── data_quality_report.json  # Machine-readable data quality report
│   ├── training_summary.json     # Training-only CV metrics & model specifications
│   ├── metrics.json              # Phase 5 held-out test evaluation metrics
│   └── regression_predictions.csv# Held-out test predictions (1,000 rows with residuals)
│
├── reports/                      # Documentation and analysis reports
│   ├── data_analysis_report.md   # Comprehensive Phase 3 EDA report
│   ├── model_evaluation_report.md# Comprehensive Phase 5 Model Evaluation & Comparison report
│   └── prediction_run_report.md  # Comprehensive Phase 6 Prediction Run & Validation report
│
├── tests/
│   ├── test_database.py          # Phase 1: MongoDB connectivity and handle tests
│   ├── test_import_data.py       # Phase 2: CSV validation and idempotent import tests
│   ├── test_preprocessing.py     # Phase 3: Feature separation, pipeline & split tests
│   ├── test_train_models.py      # Phase 4: Model training, artifact loading & split tests
│   ├── test_evaluate_models.py   # Phase 5: Test evaluation, metric calculation & roundtrip tests
│   └── test_prediction_pipeline.py# Phase 6: Prediction service, validation & MongoDB upsert tests
│
├── .gitignore                    # Git ignore rules for caches, venv, and large artifacts
├── requirements.txt              # Project dependencies and version bounds
└── README.md                     # Project overview and setup documentation
```

---

## 3. Dataset Overview & Synthetic Data Disclaimer

### Important Dataset Notice
> **Notice:** The dataset used in this project (`data/student-mat-synthetic.csv` and MongoDB collection `students`) is **entirely synthetic**. It was generated to model the feature distribution and schema of the benchmark UCI Student Performance dataset (P. Cortez and A. Silva, 2008) for academic modeling purposes. It contains **no real, personally identifiable student records**.
>
> **Implication for Model Evaluation:** The evaluation results demonstrate the correctness of the analytical pipeline, data preprocessing, and algorithm behavior. They **must not** be interpreted as empirical findings about real-world student populations or educational systems.

### Quality Summary
- **Source:** MongoDB `edupredict_db.students` (`dataset_source: "synthetic"`)
- **Total Records:** 5,000 documents
- **Missing / Null Values:** 0 across all fields
- **Duplicate Rows:** 0 full duplicates
- **Passing Threshold:** `G3 >= 10`
- **At-Risk Students (`G3 < 10`):** 2,369 students (47.38%)
- **Not At-Risk Students (`G3 >= 10`):** 2,631 students (52.62%)

---

## 4. Virtual Environment & Setup

Run from the root of the `EduPredict` project:

### A. Create Environment
```powershell
python -m venv venv
```

### B. Install Dependencies
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

> **Note on PowerShell Script Execution:**  
> If PowerShell blocks `.ps1` script activation, you can call executables directly (`.\venv\Scripts\python.exe`) or bypass script execution for the active terminal session:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

---

## 5. Execution Workflow (Phases 1 through 6)

### Step 1: Verify MongoDB Connection (Phase 1)
```powershell
.\venv\Scripts\python.exe tests/test_database.py
```

### Step 2: Validate & Import Dataset into MongoDB (Phase 2)
```powershell
.\venv\Scripts\python.exe -m src.import_data
```
- Performs idempotent bulk upsert on `(dataset_source, source_row_id)`.

### Step 3: Run Exploratory Data Analysis (Phase 3)
```powershell
.\venv\Scripts\python.exe -m src.analyze_data
```
- Generates 6 high-resolution EDA charts in `results/charts/`.
- Exports `results/data_quality_report.json`, `reports/data_analysis_report.md`, and `results/cleaned_students.csv`.

### Step 4: Train & Persist Machine Learning Models (Phase 4)
```powershell
.\venv\Scripts\python.exe -m src.train_models
```
- Evaluates 5-fold cross-validation on the 4,000 training records.
- Persists fitted pipelines to `models/*.joblib`.
- Exports `results/training_summary.json`.

### Step 5: Evaluate Models on Held-Out Test Set (Phase 5)
```powershell
.\venv\Scripts\python.exe -m src.evaluate_models
```
- Evaluates models on the untouched 1,000 test samples.
- Generates 5 evaluation visualizations in `results/charts/`.
- Exports `results/metrics.json` and `results/regression_predictions.csv`.
- Generates `reports/model_evaluation_report.md`.
- Idempotently logs evaluation summaries to MongoDB collection `model_evaluations`.

### Step 6: Generate Predictions on Student Records (Phase 6)
```powershell
# Safe default run (processes first 10 students):
.\venv\Scripts\python.exe -m src.predict_students

# Predict single student by source_row_id or ObjectId:
.\venv\Scripts\python.exe -m src.predict_students --student-id 1

# Predict custom batch (e.g., 25 students):
.\venv\Scripts\python.exe -m src.predict_students --limit 25

# Process all eligible students:
.\venv\Scripts\python.exe -m src.predict_students --all
```
- Performs inference using all 3 trained model pipelines without fitting or retraining.
- Strips any grade or target fields (`G1`, `G2`, `G3`, `at_risk`, `_id`) to prevent grade leakage.
- Idempotently upserts predictions into MongoDB `student_predictions` on `(student_id, model_version)`.
- Generates execution summary in `reports/prediction_run_report.md`.

---

## 6. Model Evaluation & Comparison (Phase 5)

### A. Evaluation Metric Definitions

1. **Accuracy:** Overall proportion of correct classifications:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (pos_label=1):** Exactness of at-risk flags (proportion of predicted at-risk students who are actually at risk):
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall / Sensitivity (pos_label=1):** Completeness of at-risk detection (proportion of actual at-risk students successfully flagged):
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score:** Harmonic mean balancing precision and recall:
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Mean Absolute Error (MAE):** Average magnitude of prediction error on a continuous grade scale:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
6. **Coefficient of Determination (R²):** Proportion of variance in final grades explained by the model relative to a naive constant mean prediction baseline:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

### B. Held-Out Test Set Results (n = 1,000 Records)

#### Task A: At-Risk Classification

| Metric | Decision Tree Classifier | Random Forest Classifier | Best Model & Rationale |
|---|---|---|---|
| **Accuracy** | **60.40%** | 58.60% | **Decision Tree** (+1.8%) |
| **Precision (pos_label=1)** | 57.56% | **58.02%** | Random Forest (+0.46%) |
| **Recall (pos_label=1)** | **62.66%** | 45.78% | **Decision Tree (+16.88%)** |
| **F1-Score (pos_label=1)** | **0.6000** | 0.5118 | **Decision Tree (+0.0882)** |
| **True Negatives (TN)** | 307 | 369 | Random Forest produces fewer false alarms |
| **False Positives (FP)** | 219 | 157 | Random Forest is more conservative |
| **False Negatives (FN)** | **177** | 257 | **Decision Tree misses 80 fewer struggling students** |
| **True Positives (TP)** | **297** | 217 | **Decision Tree identifies 80 more at-risk students** |

#### Task B: Final Grade Regression

| Metric | Random Forest Regressor | Baseline Interpretation |
|---|---|---|
| **Mean Absolute Error (MAE)** | **3.3870** grade points | Average deviation of ~3.39 marks on a 0–20 scale. |
| **Root Mean Squared Error (RMSE)**| **4.2779** grade points | Penalizes larger deviations. |
| **Coefficient of Determination (R²)**| **-0.0234** | Near zero / slightly negative. |

---

### C. Interpretation of Results & Best Model Selection

1. **Why Decision Tree is the Superior Early-Warning Classifier:**
   - In educational intervention, **Recall is the critical operational metric**. A False Negative means a struggling student receives no support and risks failing, whereas a False Positive merely results in providing check-ins or tutoring to a passing student.
   - The Decision Tree achieved **62.7% recall** (identifying 297 out of 474 at-risk students) vs. only **45.8% recall** for Random Forest (which missed more than half of the at-risk cohort).
   - Combined with higher overall Accuracy (60.4%) and F1-score (0.6000), Decision Tree is the recommended production model for proactive early alerts.

2. **Understanding Negative R² in Continuous Regression:**
   - When models attempt to predict exact continuous marks ($G3$) *strictly* from pre-semester demographic and lifestyle factors without interim exam scores ($G1, G2$), the variance of student test performance is high.
   - The resulting $R^2 \approx -0.02$ indicates that continuous point estimation has high variance relative to a naive mean predictor.
   - **Key Finding:** Binary classification (`at_risk`) is far more dependable, actionable, and stable than continuous grade regression for early academic intervention.

---

## 7. Prediction Service & Pipeline Architecture (Phase 6)

### A. Inference Architecture & Anti-Leakage Safeguards
The prediction subsystem consists of two modular layers:
1. **`PredictionService` (`src/prediction_service.py`):**
   - **Pre-trained Pipelines:** Automatically loads `decision_tree_classifier.joblib`, `random_forest_classifier.joblib`, and `random_forest_regressor.joblib`.
   - **No In-Flight Retraining:** Executes purely in inference mode (`predict()` and `predict_proba()`), never calling `fit()` or `partial_fit()`.
   - **Feature Cleansing & Validation:** Automatically scrubs out metadata (`_id`, `dataset_source`, `source_row_id`) and ground-truth grades/targets (`G1`, `G2`, `G3`, `at_risk`). Verifies that all 30 pre-semester predictors are present and conform to numeric/categorical type bounds.
   - **Native Python Type Coercion:** Ensures all outputs (`int`, `float`) are converted from NumPy primitives to native Python types prior to serialization.

2. **`predict_students` CLI (`src/predict_students.py`):**
   - Reads eligible documents from MongoDB `edupredict_db.students`.
   - Supports targeted single-student prediction (`--student-id`) or batched execution (`--limit N`, default 10; `--all`).
   - Implements per-record error isolation: bad or missing features on a single document are recorded as failures without terminating batch processing.
   - Idempotently stores outputs to `edupredict_db.student_predictions` using MongoDB bulk `UpdateOne(..., upsert=True)`.

### B. MongoDB Prediction Document Schema
```json
{
  "student_id": "1",
  "model_version": "v1",
  "predictions": {
    "decision_tree_at_risk": 0,
    "random_forest_at_risk": 0,
    "random_forest_risk_probability": 0.3542,
    "random_forest_estimated_g3": 11.83
  },
  "created_at": "2026-10-17T04:12:15.000Z",
  "metadata": {
    "source_collection": "students",
    "mongo_id": "673891...",
    "features_used_count": 30,
    "leakage_guard_verified": true,
    "models": {
      "classification_dt": "models/decision_tree_classifier.joblib",
      "classification_rf": "models/random_forest_classifier.joblib",
      "regression_rf": "models/random_forest_regressor.joblib"
    }
  }
}
```

### C. Educational & Operational Guidelines
- **Binary vs. Continuous Guidance:** Decision Tree is the recommended early-warning classification model due to its superior **62.7% recall** (detecting significantly more at-risk students than Random Forest's 45.8%).
- **Probabilistic Risk Scoring:** Random Forest risk probability (`random_forest_risk_probability`) provides a calibrated continuum of risk (e.g., 0.75 indicates strong risk) for nuanced tier-based advisor intervention.
- **Continuous Grade Estimates:** Regression outputs (`random_forest_estimated_g3`) serve as broad indicator benchmarks (MAE ≈ 3.39 marks), not exact final grade determinations.
- **Advisory Status:** All model outputs are advisory estimates designed for early support, not deterministic evaluations of student capabilities.

---

## 8. Automated Testing (32 Tests Passed)

Run the full project test suite covering all 6 phases:

**PowerShell:**
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Command Prompt:**
```cmd
venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test Suite Breakdown:
- **`TestDatabaseConnection` (4 tests):** Client instantiation, MongoDB admin ping, database selection (`edupredict_db`), and collection handles.
- **`TestImportData` (5 tests):** CSV parsing (33 columns, delimiter), data sanitization, compound unique indexing, idempotency check, and verification helper.
- **`TestPreprocessing` (5 tests):** `at_risk` threshold logic (`G3 < 10`), feature leakage exclusion (`G1`, `G2`, `G3` excluded), feature typing, ColumnTransformer construction, and stratified splitting.
- **`TestTrainModels` (6 tests):** Target definition accuracy, train/test split reproducibility, ColumnTransformer execution, in-memory model fitting & predicting, joblib artifact reloading verification, and `training_summary.json` schema validation.
- **`TestEvaluateModels` (5 tests):** Explicit `pos_label=1` metric calculation, regression metric validation on controlled values, missing artifact refusal handling, `metrics.json` and `regression_predictions.csv` roundtrip parsing, and model artifact preservation.
- **`TestPredictionPipeline` (7 tests):** Pipeline artifact loading, feature leakage scrubbing & validation, missing feature rejection, corrupted numeric feature rejection, single prediction document schema & types, batch resilience with per-record fault tolerance, and idempotent MongoDB upsert storage.

