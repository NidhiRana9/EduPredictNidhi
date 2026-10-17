# EduPredict — Student Performance Analysis and Prediction

EduPredict is an end-to-end Machine Learning and Data Analysis system designed to analyze student demographic and academic factors, predict student performance, and provide actionable insights for educators.

> **Current Status:**  
> - **Phase 1: Project Setup and MongoDB Connectivity** (Completed & Verified)  
> - **Phase 2: Dataset Validation and MongoDB Import** (Completed & Verified)  
> - **Phase 3: Exploratory Data Analysis & Feature Preprocessing** (Completed & Verified)  
> - **Phase 4: Machine Learning Model Training & Pipeline Persistence** (Completed & Verified)

---

## 1. Technology Stack

- **Language:** Python (3.10+; verified with Python 3.14)
- **Database:** MongoDB Community Server (local unauthenticated instance on port `27017`)
- **Database Driver:** PyMongo (`>=4.6.0`)
- **Data Manipulation:** pandas (`>=2.2.0`), NumPy (`>=2.3.0`)
- **Machine Learning & Preprocessing:** scikit-learn (`>=1.4.0`)
- **Model Persistence:** joblib (`>=1.3.0`)
- **Visualization:** matplotlib (`>=3.8.0`)
- **Testing:** Python standard library `unittest` (20 automated tests passed)

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
│   └── train_models.py           # Model training, CV evaluation & joblib artifact persistence
│
├── models/                       # Persisted, fitted scikit-learn pipelines (.joblib)
│   ├── .gitkeep
│   ├── decision_tree_classifier.joblib
│   ├── random_forest_classifier.joblib
│   └── random_forest_regressor.joblib
│
├── results/                      # Metric outputs, predictions, and reports
│   ├── charts/                   # 6 Generated EDA visualizations (.png)
│   │   ├── absences_vs_g3.png
│   │   ├── at_risk_distribution.png
│   │   ├── failures_vs_g3.png
│   │   ├── g3_distribution.png
│   │   ├── numeric_correlation_heatmap.png
│   │   └── studytime_distribution.png
│   ├── cleaned_students.csv      # Exported clean analysis dataset (5,000 rows)
│   ├── data_quality_report.json  # Machine-readable data quality report
│   └── training_summary.json     # Training-only CV metrics & model specifications
│
├── reports/                      # Documentation and analysis reports
│   └── data_analysis_report.md   # Comprehensive Phase 3 EDA report
│
├── tests/
│   ├── test_database.py          # Phase 1: MongoDB connectivity and handle tests
│   ├── test_import_data.py       # Phase 2: CSV validation and idempotent import tests
│   ├── test_preprocessing.py     # Phase 3: Feature separation, pipeline & split tests
│   └── test_train_models.py      # Phase 4: Model training, artifact loading & split tests
│
├── .gitignore                    # Git ignore rules for caches, venv, and large artifacts
├── requirements.txt              # Project dependencies and version bounds
└── README.md                     # Project overview and setup documentation
```

---

## 3. Dataset Overview

### Important Dataset Notice
> **Notice:** The dataset used in this project (`data/student-mat-synthetic.csv` and MongoDB collection `students`) is **entirely synthetic**. It was generated to model the feature distribution and schema of the benchmark UCI Student Performance dataset (P. Cortez and A. Silva, 2008) for academic modeling purposes. It contains **no real, personally identifiable student records**. Results on synthetic data demonstrate pipeline correctness and must not be treated as empirical evidence of real-world student behavior.

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

## 5. Execution Workflow (Phases 1 through 4)

### Step 1: Verify MongoDB Connection (Phase 1)
```powershell
.\venv\Scripts\python.exe tests/test_database.py
```

### Step 2: Validate & Import Dataset into MongoDB (Phase 2)
```powershell
.\venv\Scripts\python.exe -m src.import_data
```
- Performs idempotent bulk upsert on `(dataset_source, source_row_id)`.
- Rerunning the script updates in-place without duplicating documents.

### Step 3: Run Exploratory Data Analysis (Phase 3)
```powershell
.\venv\Scripts\python.exe -m src.analyze_data
```
- Generates 6 high-resolution charts in `results/charts/`.
- Exports `results/data_quality_report.json`, `reports/data_analysis_report.md`, and `results/cleaned_students.csv`.

### Step 4: Train & Persist Machine Learning Models (Phase 4)
```powershell
.\venv\Scripts\python.exe -m src.train_models
```

---

## 6. Machine Learning Architecture (Phase 4)

### A. The Two Prediction Tasks

1. **Task A: At-Risk Classification (Early Warning)**
   - **Target:** `at_risk = 1` if `G3 < 10` else `0`.
   - **Models:**
     - `DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)`
     - `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)`
   - **Class Balance:** 47.4% at-risk vs. 52.6% not-at-risk. Because the classes are virtually balanced (~1.1:1 ratio), standard unweighted training is employed without requiring synthetic oversampling (SMOTE).

2. **Task B: Final Grade Regression**
   - **Target:** Numerical final score `G3` (0–20 continuous scale).
   - **Model:**
     - `RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)`

### B. Target Leakage Prevention & Feature Exclusion

To build an authentic early-warning system that can flag students *before* exams occur:
- **Strictly Excluded Columns:** `G1` (Period 1 exam), `G2` (Period 2 exam), `G3` (Final exam), `at_risk` (Target), and database identifiers (`_id`, `source_row_id`, `dataset_source`).
- **Remaining 30 Predictor Features:**
  - **13 Numeric Features:** `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
  - **17 Categorical Features:** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.

### C. Data Splitting & Held-Out Test Set Integrity

- **Split Ratio:** 80% Train (4,000 records) / 20% Test (1,000 records) using `random_state=42`.
- **Stratification:** Classification split uses stratified sampling to preserve the 47.4% / 52.6% risk ratio.
- **Strict Held-Out Rule:** The test set of 1,000 records was **completely held out** during Phase 4. Zero test samples were used in feature selection, imputer fitting, encoder fitting, or model selection. Comprehensive test-set evaluation is reserved for Phase 5.

### D. Training-Only Cross-Validation Results (5-Fold CV on 4,000 Training Samples)

| Model | Task | Primary CV Metric | Secondary CV Metric | Secondary CV Metric |
|---|---|---|---|---|
| **Decision Tree Classifier** | At-Risk Classification | Accuracy: **59.40%** (±2.4%) | F1-Score: **0.5517** (±0.043) | ROC-AUC: **0.6332** (±0.017) |
| **Random Forest Classifier** | At-Risk Classification | Accuracy: **60.82%** (±0.6%) | F1-Score: **0.5356** (±0.016) | ROC-AUC: **0.6568** (±0.008) |
| **Random Forest Regressor** | Final Grade (G3) | MAE: **3.3194** (±0.058) | RMSE: **4.1453** (±0.065) | R²: **-0.0100** (±0.004) |

### E. Persisted Model Pipelines (`models/*.joblib`)

The entire fitted scikit-learn `Pipeline` (including `ColumnTransformer` preprocessing + estimator) is persisted using `joblib`:
- `models/decision_tree_classifier.joblib`
- `models/random_forest_classifier.joblib`
- `models/random_forest_regressor.joblib`

Because the preprocessor is packaged inside the pipeline, future prediction modules can pass raw pandas DataFrames or dictionaries directly into `.predict()` without manual encoding or scaling.

---

## 7. Automated Testing (20 Tests Passed)

Run the full project test suite:

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
- **`TestImportData` (5 tests):** CSV parsing (33 columns, delimiter), data sanitization, compound unique indexing, idempotency check (rerun count unchanged), and verification helper.
- **`TestPreprocessing` (5 tests):** `at_risk` threshold logic (`G3 < 10`), feature leakage exclusion (`G1`, `G2`, `G3` excluded), feature typing, un-fitted ColumnTransformer construction, and stratified train/test splitting.
- **`TestTrainModels` (6 tests):** Target definition accuracy, train/test split reproducibility, ColumnTransformer execution, in-memory model fitting & predicting, joblib artifact reloading verification, and `training_summary.json` schema validation.

---

## 8. Next Steps (Phase 5 Preview)

Once Phase 4 is approved:
- **Phase 5: Model Evaluation, Prediction Pipeline & Final Reporting**
  - Evaluate the 3 persisted pipelines on the held-out test set (1,000 samples).
  - Compute test Confusion Matrix, Classification Report (Precision, Recall, F1), ROC Curves, MAE, RMSE, and R².
  - Store model evaluation metrics in MongoDB `model_evaluations` collection.
  - Generate model prediction outputs and log them to MongoDB `predictions` collection.
  - Compile the final assignment and technical project report in `reports/`.
