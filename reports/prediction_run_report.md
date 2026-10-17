# EduPredict — Prediction Execution Run Report

> **Dataset Notice:** All predictions were generated for **synthetic student records** (`data/student-mat-synthetic.csv` stored in MongoDB `edupredict_db.students`).  
> **Predictive Disclaimer:** Predictions represent **statistical estimates** based solely on non-grade demographic, lifestyle, and behavioral factors; they do **not** constitute real-world academic evaluations or guarantees of future academic standing.

---

## 1. Run Execution Metadata

| Attribute | Details |
|---|---|
| **Execution Date & Time** | `2026-10-17 05:32:43 UTC` |
| **Model Version** | `v1` |
| **Model Artifacts Used** | `decision_tree_classifier.joblib, random_forest_classifier.joblib, random_forest_regressor.joblib` |
| **Database & Collection** | `edupredict_db.student_predictions` |
| **Input Features Evaluated** | **30 features** (Grade columns G1, G2, G3 strictly excluded) |

---

## 2. Processing Summary

| Processing Metric | Count |
|---|---|
| **Total Records Attempted** | **5** |
| **Successful Predictions** | **5** (100.0%) |
| **Failed Records** | **0** |
| **MongoDB Upserted Records** | **0** |
| **MongoDB Matched / Updated Records** | **5** |

---

## 3. Sample Scored Predictions (First 10 Records)

| Student ID | Row ID | Decision Tree Risk | Random Forest Risk | RF Risk Probability | Estimated G3 (0–20) |
|---|---|---|---|---|---|
| `6ad2e6b67a5f508fb2593a06` | 1 | Passing | Passing | 28.0% | 9.75 |
| `6ad2e6b67a5f508fb2593a07` | 2 | At-Risk | Passing | 39.3% | 9.89 |
| `6ad2e6b67a5f508fb2593a08` | 3 | Passing | Passing | 41.0% | 9.48 |
| `6ad2e6b67a5f508fb2593a09` | 4 | At-Risk | At-Risk | 62.6% | 9.81 |
| `6ad2e6b67a5f508fb2593a0a` | 5 | Passing | Passing | 28.3% | 9.19 |

---

## 4. Error & Validation Diagnostics

None (All records scored successfully).

---

## 5. Architectural Safeguards

- **Zero Target Leakage:** Features `G1`, `G2`, `G3`, and `at_risk` were stripped prior to prediction; inference was conducted strictly on non-grade attributes.
- **Idempotency Confirmed:** MongoDB predictions collection uses a compound upsert filter on `(student_id, model_version)`, ensuring duplicate executions update records cleanly in-place without data duplication.
- **Source Integrity Preserved:** Original documents in `edupredict_db.students` were accessed read-only with zero modifications.
