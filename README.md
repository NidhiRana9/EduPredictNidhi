# EduPredict — Student Performance Analysis and Prediction

EduPredict is an end-to-end Machine Learning and Data Analysis system designed to analyze student demographic and academic factors, predict student performance, and provide actionable insights for educators.

> **Current Status:**  
> - **Phase 1: Project Setup and MongoDB Connectivity** (Completed & Verified)  
> - **Phase 2: Dataset Validation and MongoDB Import** (Completed & Verified)  
> - **Phase 3: Exploratory Data Analysis & Feature Preprocessing** (Completed & Verified)

---

## 1. Technology Stack

- **Language:** Python (3.10+; verified with Python 3.14)
- **Database:** MongoDB Community Server (local unauthenticated instance on port `27017`)
- **Database Driver:** PyMongo (`>=4.6.0`)
- **Data Manipulation:** pandas (`>=2.2.0`), NumPy (`>=2.3.0`)
- **Machine Learning & Preprocessing:** scikit-learn (`>=1.4.0`)
- **Model Persistence:** joblib (`>=1.3.0`)
- **Visualization:** matplotlib (`>=3.8.0`)
- **Testing:** Python standard library `unittest` (14 automated tests passed)

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
│   └── preprocessing.py          # Feature partitioning, pipeline builder & train/test splitter
│
├── models/                       # Serialized trained machine learning models (Phase 4)
│   └── .gitkeep
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
│   └── data_quality_report.json  # Machine-readable data quality report
│
├── reports/                      # Documentation and analysis reports
│   └── data_analysis_report.md   # Comprehensive Phase 3 EDA report
│
├── tests/
│   ├── test_database.py          # Phase 1: MongoDB connectivity and handle tests
│   ├── test_import_data.py       # Phase 2: CSV validation and idempotent import tests
│   └── test_preprocessing.py     # Phase 3: Feature separation, pipeline & split tests
│
├── .gitignore                    # Git ignore rules for caches, venv, and large artifacts
├── requirements.txt              # Project dependencies and version bounds
└── README.md                     # Project overview and setup documentation
```

---

## 3. Dataset Overview

### Important Dataset Notice
> **Notice:** The dataset used in this project (`data/student-mat-synthetic.csv` and MongoDB collection `students`) is **entirely synthetic**. It was generated to model the feature distribution and schema of the benchmark UCI Student Performance dataset (P. Cortez and A. Silva, 2008) for academic modeling purposes. It contains **no real, personally identifiable student records**.

### Specifications & Quality Summary
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

## 5. Running the Pipeline (Phases 1, 2, and 3)

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

### Step 3: Run Exploratory Data Analysis & Preprocessing (Phase 3)
```powershell
.\venv\Scripts\python.exe -m src.analyze_data
```
This script:
1. Retrieves only synthetic documents from MongoDB (excluding internal `_id`).
2. Validates types, uniqueness, and grade bounds (`0 <= G1, G2, G3 <= 20`).
3. Generates 6 high-resolution charts in `results/charts/`:
   - `g3_distribution.png`: Histogram of final grades with mean (9.76) and median (10.0).
   - `studytime_distribution.png`: Student counts across study time tiers.
   - `absences_vs_g3.png`: Jittered scatter plot and trend line of school absences vs. final grade.
   - `failures_vs_g3.png`: Boxplot showing performance decline across past failure counts.
   - `at_risk_distribution.png`: Bar chart of at-risk (47.4%) vs. passing (52.6%) student counts.
   - `numeric_correlation_heatmap.png`: Full correlation matrix across all 16 numeric variables.
4. Exports machine-readable report: `results/data_quality_report.json`.
5. Exports technical markdown report: `reports/data_analysis_report.md`.
6. Exports cleaned modeling dataset: `results/cleaned_students.csv` (preserves all 5,000 rows with `at_risk` target).

---

## 6. Machine Learning Preprocessing Design (`src/preprocessing.py`)

Phase 3 establishes clean, reusable feature pipelines for later training in Phase 4:

### A. Target Variable Separation
- **Binary Classification Target:** `at_risk = 1` if `G3 < 10` else `0`.
- **Regression Target:** Numerical `G3` (0–20 scale).

### B. Target Leakage Prevention
To ensure authentic early-warning predictions before exams occur:
- **Strictly Excluded Columns:** `G1`, `G2`, `G3`, `at_risk`, `_id`, `source_row_id`, `dataset_source`.
- **Remaining Predictors (30 Features):**
  - **13 Numeric Features:** `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
  - **17 Categorical Features:** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.

### C. Pipeline Architecture (Scikit-Learn ColumnTransformer)
- **Numeric Pipeline:** `SimpleImputer(strategy="median")` + `StandardScaler()`.
- **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- **Strict Rule:** Preprocessing transformers remain **un-fitted** in Phase 3. They will only be fitted on training splits in Phase 4 to prevent data leakage.

### D. Reproducible Data Splitting
- `split_dataset(X, y, test_size=0.2, random_state=42, stratify=True)`
- Partitions 5,000 samples into 4,000 train (80%) and 1,000 test (20%) while preserving the exact 47.4% / 52.6% risk class ratio.

---

## 7. Automated Testing (14 Tests Passed)

Run the complete test suite covering Phase 1, Phase 2, and Phase 3:

**PowerShell:**
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Command Prompt:**
```cmd
venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Test Suite Breakdown:
- **`TestDatabaseConnection` (4 tests):** Client instantiation, MongoDB admin ping, database selection (`edupredict_db`), and non-destructive collection handles.
- **`TestImportData` (5 tests):** CSV parsing (33 columns, delimiter), data sanitization, compound unique indexing, idempotency check (rerun count unchanged), and verification helper.
- **`TestPreprocessing` (5 tests):** `at_risk` threshold logic (`G3 < 10`), feature leakage exclusion (`G1`, `G2`, `G3` excluded), feature type identification, un-fitted ColumnTransformer construction, and stratified train/test splitting.

---

## 8. Next Steps (Phase 4 Preview)

Once Phase 3 is approved:
- **Phase 4: Model Training and Evaluation**
  - Fit preprocessing pipelines exclusively on `X_train`.
  - Train classification models (Logistic Regression, Random Forest, Gradient Boosting) to predict `at_risk`.
  - Train regression models (Ridge, Random Forest Regressor) to predict final grade `G3`.
  - Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC (classification) and MAE, RMSE, R² (regression).
  - Persist trained models to `models/` using `joblib`.
  - Log evaluation metrics into MongoDB `model_evaluations` collection.
