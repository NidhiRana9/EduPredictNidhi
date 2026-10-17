# EduPredict — Model Evaluation & Comparison Report

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
| **Decision Tree Classifier** | **0.6040** (60.4%) | 0.5756 (57.6%) | **0.6266** (62.7%) | **0.6000** |
| **Random Forest Classifier** | 0.5860 (58.6%) | **0.5802** (58.0%) | 0.4578 (45.8%) | 0.5118 |

### Confusion Matrix Breakdown (Test Set: n=1,000)

#### Decision Tree Classifier:
- **True Negatives (TN):** 307 (Correctly identified passing students)
- **False Positives (FP):** 219 (Passing students flagged as at-risk)
- **False Negatives (FN):** 177 (At-risk students missed)
- **True Positives (TP):** 297 (At-risk students correctly flagged)

#### Random Forest Classifier:
- **True Negatives (TN):** 369
- **False Positives (FP):** 157
- **False Negatives (FN):** 257
- **True Positives (TP):** 217

### Classifier Comparison Analysis:
1. **Decision Tree Classifier Advantages:**
   - Achieved superior **Recall (62.7%)** compared to Random Forest (45.8%). In educational early-warning systems, **Recall is paramount** because failing to identify a struggling student (False Negative) has far higher consequences than providing support to a passing student (False Positive).
   - Higher overall **F1-Score (0.6000)** and **Accuracy (60.4%)**.
2. **Random Forest Classifier Characteristics:**
   - Showed higher specificity/precision (58.0%), producing fewer false alarms, but missed 257 at-risk students (FN).

---

## 3. Regression Model Evaluation (Task B: Final Grade G3)

**Task Objective:** Predict the exact continuous final grade `G3` (0–20 scale) without access to interim exam grades.

| Metric | Score | Interpretation |
|---|---|---|
| **Mean Absolute Error (MAE)** | **3.3870** | On average, predictions deviate by ~3.39 points on a 20-point scale. |
| **Root Mean Squared Error (RMSE)** | **4.2779** | Reflects typical penalty for larger deviations. |
| **Coefficient of Determination (R²)** | **-0.0234** | Close to 0 (slightly negative). |
| **Mean Residual (Actual - Pred)** | **0.0574** | Unbiased error distribution centered near zero. |

### Explanation of Negative / Near-Zero R²:
In machine learning regression, $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
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
