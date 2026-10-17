# EduPredict — Exploratory Data Analysis & Data Quality Report

> **Dataset Nature:** **Synthetic Data**.  
> **Notice:** The dataset evaluated in this report (`data/student-mat-synthetic.csv` stored in MongoDB `edupredict_db.students`) is an artificial synthetic dataset generated for educational machine learning benchmarking. It does **not** represent real, personally identifiable student populations.

---

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Total Records Retrieved** | **5,000** documents |
| **Total Features / Columns** | **35** columns |
| **Missing Values** | **0** (100% complete) |
| **Full Duplicate Records** | **0** |
| **Duplicate Source Identifiers** | **0** |
| **At-Risk Students (`G3 < 10`)** | **2,369** (47.38%) |
| **Not At-Risk Students (`G3 >= 10`)** | **2,631** (52.62%) |
| **Data Quality Warnings** | **0** |

---

## 2. Academic Grade Distribution (Target Analysis)

In the standard Portuguese academic evaluation scale modeled here, grades range from **0 to 20**, with **10** serving as the minimum passing score.

| Grade Metric | G1 (Period 1) | G2 (Period 2) | G3 (Final Exam) |
|---|---|---|---|
| **Minimum** | 0.0 | 0.0 | 0.0 |
| **Maximum** | 20.0 | 20.0 | 20.0 |
| **Mean** | 9.74 | 9.76 | 9.76 |
| **Median** | 10.0 | 10.0 | 10.0 |
| **Standard Deviation** | 3.97 | 4.06 | 4.12 |
| **Out-of-Bounds Values** | 0 | 0 | 0 |

### Key Observations:
1. **Target Balance:** The classification target (`at_risk`) shows a healthy balance: **47.38% at risk** vs. **52.62% passing**. This means models will not require synthetic oversampling (e.g. SMOTE) or severe class-weight adjustments in Phase 4.
2. **Grade Stability:** Mean grades across G1 (9.74), G2 (9.76), and G3 (9.76) remain consistent, reflecting steady progression across the academic period.

---

## 3. Key Findings from Exploratory Charts

Visualizations were rendered and saved under `results/charts/`:

1. **Final Grade Distribution (`results/charts/g3_distribution.png`):**
   - Displays a bell-shaped distribution centered around 9.75.
   - Distinct red coloring marks students scoring below 10, highlighting early intervention targets.
2. **Weekly Study Time (`results/charts/studytime_distribution.png`):**
   - The majority of students report study times in categories 1 (<2 hrs) and 2 (2–5 hrs), while smaller cohorts engage in intensive study (>10 hrs).
3. **Absences vs. Final Grade (`results/charts/absences_vs_g3.png`):**
   - Higher absence frequencies exhibit negative drift on final performance.
   - *Note on Causality:* While absences correlate with lower grades, correlation does not establish direct causation; absences may reflect underlying personal, health, or motivational factors.
4. **Previous Course Failures (`results/charts/failures_vs_g3.png`):**
   - Shows a strong negative gradient: students with 0 past failures display significantly higher median final grades compared to students with 1, 2, or 3 past failures.
5. **Academic Risk Categorization (`results/charts/at_risk_distribution.png`):**
   - Quantifies the cohort into 2,369 at-risk students and 2,631 passing students.
6. **Correlation Matrix (`results/charts/numeric_correlation_heatmap.png`):**
   - High positive correlations exist among the grade series (`G1`, `G2`, `G3`).
   - Weak to moderate correlations appear between alcohol consumption (`Dalc`, `Walc`) and social outing frequency (`goout`).
   - Parental education (`Medu`, `Fedu`) shows positive association with student achievement.

---

## 4. Feature Selection Strategy for Machine Learning (Phase 4)

To prevent **target leakage** and build an authentic early warning model:
- **Excluded Features:**
  - `G1` and `G2`: Because interim exam scores occur late in the semester, excluding them forces the model to evaluate demographic, behavioral, and engagement factors prior to exam sittings.
  - `_id`, `source_row_id`, `dataset_source`: Database keys carry no behavioral signal.
- **Retained Features (30 Predictors):**
  - **13 Numeric Features:** `age`, `Medu`, `Fedu`, `traveltime`, `studytime`, `failures`, `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
  - **17 Categorical Features:** `school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.

---

## 5. Artifact Verification

- **Cleaned Dataset:** `results/cleaned_students.csv` (5000 rows, 36 columns including `at_risk`).
- **Data Quality JSON:** `results/data_quality_report.json`.
- **Chart Files:** All 6 PNG visualizations generated at 150 DPI in `results/charts/`.
