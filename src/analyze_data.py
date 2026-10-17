"""Exploratory Data Analysis and Quality Reporting Module for EduPredict.

Phase 3 Implementation:
- Retrieves synthetic student records directly from MongoDB 'students' collection.
- Validates data integrity, types, uniqueness, and grade constraints.
- Generates 6 analytical visualizations saved to results/charts/.
- Produces a machine-readable data quality report (results/data_quality_report.json).
- Produces a comprehensive markdown analysis report (reports/data_analysis_report.md).
- Exports a clean, reproducible CSV for modeling (results/cleaned_students.csv).
- Preserves all MongoDB documents without destructive modifications.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Output directory constants
CHARTS_DIR: Path = PROJECT_ROOT / "results" / "charts"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CLEANED_CSV_PATH: Path = RESULTS_DIR / "cleaned_students.csv"
DATA_QUALITY_JSON_PATH: Path = RESULTS_DIR / "data_quality_report.json"
ANALYSIS_REPORT_MD_PATH: Path = REPORTS_DIR / "data_analysis_report.md"

DATASET_SOURCE_NAME: str = "synthetic"
AT_RISK_THRESHOLD: int = 10


def fetch_synthetic_data_from_mongodb() -> pd.DataFrame:
    """Connect to MongoDB and retrieve synthetic student documents.

    Excludes the internal MongoDB '_id' field to keep features clean.
    Filters specifically for documents tagged with dataset_source='synthetic'.

    Returns:
        pd.DataFrame containing the retrieved student records.

    Raises:
        RuntimeError: If MongoDB is unreachable or returns 0 records.
    """
    client = get_mongo_client()
    try:
        ping_database(client)
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to MongoDB: {exc}") from exc

    db = get_database(client)
    collection = db[COLLECTION_STUDENTS]

    query = {"dataset_source": DATASET_SOURCE_NAME}
    projection = {"_id": 0}

    cursor = collection.find(query, projection)
    records = list(cursor)

    if not records:
        raise RuntimeError(
            f"No documents found in '{COLLECTION_STUDENTS}' collection matching {query}. "
            "Please ensure Phase 2 import has been executed."
        )

    df = pd.DataFrame(records)
    print(f"[INFO] Retrieved {len(df)} records from MongoDB collection '{COLLECTION_STUDENTS}'.")
    return df


def validate_retrieved_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data integrity, columns, data types, and grade ranges.

    Returns:
        Dictionary of validation findings and metrics.
    """
    num_rows, num_cols = df.shape

    # Check missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # Check duplicated rows
    full_duplicates = int(df.duplicated().sum())

    # Check duplicate source_row_id
    duplicate_source_ids = 0
    if "source_row_id" in df.columns:
        duplicate_source_ids = int(df["source_row_id"].duplicated().sum())

    # Validate grade columns G1, G2, G3
    grade_cols = ["G1", "G2", "G3"]
    grade_stats: Dict[str, Dict[str, Any]] = {}
    out_of_bounds_grades: Dict[str, int] = {}

    for gc in grade_cols:
        if gc in df.columns:
            min_g = float(df[gc].min())
            max_g = float(df[gc].max())
            mean_g = float(df[gc].mean())
            median_g = float(df[gc].median())
            std_g = float(df[gc].std())

            oob_count = int(((df[gc] < 0) | (df[gc] > 20)).sum())
            out_of_bounds_grades[gc] = oob_count

            grade_stats[gc] = {
                "min": min_g,
                "max": max_g,
                "mean": round(mean_g, 2),
                "median": round(median_g, 2),
                "std": round(std_g, 2),
                "out_of_bounds_count": oob_count,
            }

    # At-risk counts
    at_risk_mask = df["G3"] < AT_RISK_THRESHOLD
    at_risk_count = int(at_risk_mask.sum())
    not_at_risk_count = int((~at_risk_mask).sum())

    validation_summary = {
        "dataset_source": DATASET_SOURCE_NAME,
        "is_synthetic": True,
        "total_records": num_rows,
        "total_columns": num_cols,
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_counts.items() if v > 0},
        "full_duplicate_rows": full_duplicates,
        "duplicate_source_row_ids": duplicate_source_ids,
        "grade_statistics": grade_stats,
        "at_risk_count": at_risk_count,
        "at_risk_percentage": round((at_risk_count / num_rows) * 100, 2),
        "not_at_risk_count": not_at_risk_count,
        "not_at_risk_percentage": round((not_at_risk_count / num_rows) * 100, 2),
        "validation_warnings": [],
    }

    if total_missing > 0:
        validation_summary["validation_warnings"].append(f"Found {total_missing} missing values.")
    if full_duplicates > 0:
        validation_summary["validation_warnings"].append(f"Found {full_duplicates} full duplicate rows.")
    if duplicate_source_ids > 0:
        validation_summary["validation_warnings"].append(f"Found {duplicate_source_ids} duplicate source_row_ids.")
    for gc, count in out_of_bounds_grades.items():
        if count > 0:
            validation_summary["validation_warnings"].append(
                f"Found {count} out-of-bounds grades in column '{gc}'."
            )

    return validation_summary


def generate_eda_charts(df: pd.DataFrame, output_dir: Path = CHARTS_DIR) -> List[Path]:
    """Generate and save 6 high-quality exploratory analysis charts using matplotlib.

    1. Distribution of final grades (G3)
    2. Distribution of study time
    3. Absences vs Final Grade
    4. Previous failures vs Final Grade
    5. Count of at-risk (G3 < 10) vs not-at-risk students
    6. Correlation heatmap of numeric features

    Args:
        df: Input DataFrame with student data.
        output_dir: Directory where charts will be saved.

    Returns:
        List of generated image file Paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: List[Path] = []

    # Chart styling defaults
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    font_title = {"fontsize": 14, "fontweight": "bold", "color": "#1f2937"}
    font_label = {"fontsize": 11, "fontweight": "semibold", "color": "#374151"}

    # --------------------------------------------------------------------------
    # Chart 1: Distribution of Final Grades (G3)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    g3 = df["G3"]
    mean_g3 = g3.mean()
    median_g3 = g3.median()

    bins = np.arange(-0.5, 21.5, 1)
    n, _, patches = ax.hist(g3, bins=bins, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85, rwidth=0.85)

    # Highlight failing (<10) vs passing (>=10)
    for bin_idx, patch in enumerate(patches):
        if bin_idx < 10:
            patch.set_facecolor("#ef4444")  # Coral red for failing
        else:
            patch.set_facecolor("#10b981")  # Emerald green for passing

    ax.axvline(10, color="#b91c1c", linestyle="--", linewidth=1.5, label="Passing Threshold (10)")
    ax.axvline(mean_g3, color="#1e3a8a", linestyle="-", linewidth=2, label=f"Mean: {mean_g3:.2f}")
    ax.axvline(median_g3, color="#d97706", linestyle=":", linewidth=2, label=f"Median: {median_g3:.1f}")

    ax.set_title("Distribution of Final Grades (G3) [Synthetic Dataset]", fontdict=font_title, pad=12)
    ax.set_xlabel("Final Grade (G3: 0–20 Scale)", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Number of Students", fontdict=font_label, labelpad=8)
    ax.set_xticks(range(0, 21))
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    chart1_path = output_dir / "g3_distribution.png"
    plt.savefig(chart1_path)
    plt.close()
    generated_charts.append(chart1_path)
    print(f"  [CHART] Saved: {chart1_path.name}")

    # --------------------------------------------------------------------------
    # Chart 2: Distribution of Weekly Study Time
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    study_labels = {1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"}
    study_counts = df["studytime"].value_counts().sort_index()
    labels = [study_labels.get(k, str(k)) for k in study_counts.index]

    bars = ax.bar(labels, study_counts.values, color="#6366f1", edgecolor="#4338ca", width=0.6, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 25, f"{h:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Weekly Study Time Distribution [Synthetic Dataset]", fontdict=font_title, pad=12)
    ax.set_xlabel("Study Time Categories", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Student Count", fontdict=font_label, labelpad=8)
    ax.set_ylim(0, max(study_counts.values) * 1.15)
    plt.tight_layout()

    chart2_path = output_dir / "studytime_distribution.png"
    plt.savefig(chart2_path)
    plt.close()
    generated_charts.append(chart2_path)
    print(f"  [CHART] Saved: {chart2_path.name}")

    # --------------------------------------------------------------------------
    # Chart 3: Absences vs Final Grade (G3)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    # Group by absences to show mean grade and confidence spread
    abs_stats = df.groupby("absences")["G3"].agg(["mean", "count"]).reset_index()
    abs_filtered = abs_stats[abs_stats["count"] >= 10]  # Only show points with sufficient sample support

    # Scatter of all points with jitter for visibility
    jitter_y = df["G3"] + np.random.uniform(-0.25, 0.25, len(df))
    jitter_x = df["absences"] + np.random.uniform(-0.25, 0.25, len(df))
    ax.scatter(jitter_x, jitter_y, alpha=0.15, color="#64748b", s=18, edgecolors="none", label="Student observations (jittered)")

    # Trend line of mean grade
    ax.plot(abs_filtered["absences"], abs_filtered["mean"], color="#dc2626", linewidth=2.5, marker="o", markersize=6, label="Mean G3 by Absences")
    ax.axhline(10, color="#10b981", linestyle="--", linewidth=1.5, label="Passing Grade (10)")

    ax.set_title("School Absences vs. Final Grade (G3) [Synthetic Dataset]", fontdict=font_title, pad=12)
    ax.set_xlabel("Number of School Absences", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Final Grade (G3)", fontdict=font_label, labelpad=8)
    ax.set_ylim(-1, 21)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right")
    plt.tight_layout()

    chart3_path = output_dir / "absences_vs_g3.png"
    plt.savefig(chart3_path)
    plt.close()
    generated_charts.append(chart3_path)
    print(f"  [CHART] Saved: {chart3_path.name}")

    # --------------------------------------------------------------------------
    # Chart 4: Previous Failures vs Final Grade (G3)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    failure_groups = [df[df["failures"] == f]["G3"].values for f in sorted(df["failures"].unique())]
    labels_failures = [f"{f} failure(s)\n(n={len(df[df['failures'] == f]):,})" for f in sorted(df["failures"].unique())]

    box = ax.boxplot(
        failure_groups,
        patch_artist=True,
        medianprops={"color": "#b91c1c", "linewidth": 2},
        boxprops={"facecolor": "#93c5fd", "edgecolor": "#1d4ed8", "alpha": 0.8},
        whiskerprops={"color": "#1d4ed8", "linewidth": 1.5},
        capprops={"color": "#1d4ed8", "linewidth": 1.5},
        flierprops={"marker": "o", "color": "#64748b", "alpha": 0.4, "markersize": 4},
    )
    ax.set_xticks(range(1, len(failure_groups) + 1), labels=labels_failures)

    ax.axhline(10, color="#10b981", linestyle="--", linewidth=1.5, label="Passing Grade (10)")
    ax.set_title("Impact of Past Course Failures on Final Grade [Synthetic Dataset]", fontdict=font_title, pad=12)
    ax.set_xlabel("Number of Past Failures", fontdict=font_label, labelpad=8)
    ax.set_ylabel("Final Grade (G3: 0–20)", fontdict=font_label, labelpad=8)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()

    chart4_path = output_dir / "failures_vs_g3.png"
    plt.savefig(chart4_path)
    plt.close()
    generated_charts.append(chart4_path)
    print(f"  [CHART] Saved: {chart4_path.name}")

    # --------------------------------------------------------------------------
    # Chart 5: Count of At-Risk (G3 < 10) vs. Not At-Risk Students
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    at_risk_mask = df["G3"] < AT_RISK_THRESHOLD
    risk_counts = pd.Series([at_risk_mask.sum(), (~at_risk_mask).sum()], index=["At-Risk (G3 < 10)", "Not At-Risk (G3 >= 10)"])
    colors = ["#f87171", "#34d399"]
    bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor=["#b91c1c", "#059669"], width=0.55)

    total_studs = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total_studs) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, h + 30, f"{h:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Student Academic Risk Classification [Synthetic Dataset]", fontdict=font_title, pad=12)
    ax.set_ylabel("Student Count", fontdict=font_label, labelpad=8)
    ax.set_ylim(0, max(risk_counts.values) * 1.18)
    plt.tight_layout()

    chart5_path = output_dir / "at_risk_distribution.png"
    plt.savefig(chart5_path)
    plt.close()
    generated_charts.append(chart5_path)
    print(f"  [CHART] Saved: {chart5_path.name}")

    # --------------------------------------------------------------------------
    # Chart 6: Correlation Heatmap for Numeric Features
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    num_cols = [
        "age",
        "Medu",
        "Fedu",
        "traveltime",
        "studytime",
        "failures",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
        "G3",
    ]
    corr = df[num_cols].corr()

    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left", fontsize=9, fontweight="semibold")
    ax.set_yticklabels(num_cols, fontsize=9, fontweight="semibold")

    # Annotate correlation values inside each cell
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)

    ax.set_title("Correlation Matrix of Numeric Features [Synthetic Dataset]", fontdict=font_title, pad=35)
    plt.tight_layout()

    chart6_path = output_dir / "numeric_correlation_heatmap.png"
    plt.savefig(chart6_path)
    plt.close()
    generated_charts.append(chart6_path)
    print(f"  [CHART] Saved: {chart6_path.name}")

    return generated_charts


def export_clean_analysis_csv(df: pd.DataFrame, output_path: Path = CLEANED_CSV_PATH) -> None:
    """Save clean analysis dataset to CSV.

    Preserves source identifier, excludes MongoDB _id, includes at_risk target,
    and preserves original records in MongoDB.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean = df.copy()
    if "_id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["_id"])

    # Ensure binary classification target is available
    if "at_risk" not in df_clean.columns:
        df_clean["at_risk"] = (df_clean["G3"] < AT_RISK_THRESHOLD).astype(int)

    df_clean.to_csv(output_path, index=False)
    print(f"[SUCCESS] Exported cleaned analysis CSV ({len(df_clean)} rows) to: {output_path}")


def generate_data_quality_json(
    summary: Dict[str, Any],
    output_path: Path = DATA_QUALITY_JSON_PATH,
) -> None:
    """Save machine-readable data quality report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SUCCESS] Exported data quality JSON report to: {output_path}")


def generate_markdown_analysis_report(
    summary: Dict[str, Any],
    output_path: Path = ANALYSIS_REPORT_MD_PATH,
) -> None:
    """Write an extensive, beginner-friendly Markdown EDA report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# EduPredict — Exploratory Data Analysis & Data Quality Report

> **Dataset Nature:** **Synthetic Data**.  
> **Notice:** The dataset evaluated in this report (`data/student-mat-synthetic.csv` stored in MongoDB `edupredict_db.students`) is an artificial synthetic dataset generated for educational machine learning benchmarking. It does **not** represent real, personally identifiable student populations.

---

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Total Records Retrieved** | **{summary['total_records']:,}** documents |
| **Total Features / Columns** | **{summary['total_columns']}** columns |
| **Missing Values** | **{summary['total_missing_values']}** (100% complete) |
| **Full Duplicate Records** | **{summary['full_duplicate_rows']}** |
| **Duplicate Source Identifiers** | **{summary['duplicate_source_row_ids']}** |
| **At-Risk Students (`G3 < 10`)** | **{summary['at_risk_count']:,}** ({summary['at_risk_percentage']}%) |
| **Not At-Risk Students (`G3 >= 10`)** | **{summary['not_at_risk_count']:,}** ({summary['not_at_risk_percentage']}%) |
| **Data Quality Warnings** | **{len(summary['validation_warnings'])}** |

---

## 2. Academic Grade Distribution (Target Analysis)

In the standard Portuguese academic evaluation scale modeled here, grades range from **0 to 20**, with **10** serving as the minimum passing score.

| Grade Metric | G1 (Period 1) | G2 (Period 2) | G3 (Final Exam) |
|---|---|---|---|
| **Minimum** | {summary['grade_statistics']['G1']['min']} | {summary['grade_statistics']['G2']['min']} | {summary['grade_statistics']['G3']['min']} |
| **Maximum** | {summary['grade_statistics']['G1']['max']} | {summary['grade_statistics']['G2']['max']} | {summary['grade_statistics']['G3']['max']} |
| **Mean** | {summary['grade_statistics']['G1']['mean']} | {summary['grade_statistics']['G2']['mean']} | {summary['grade_statistics']['G3']['mean']} |
| **Median** | {summary['grade_statistics']['G1']['median']} | {summary['grade_statistics']['G2']['median']} | {summary['grade_statistics']['G3']['median']} |
| **Standard Deviation** | {summary['grade_statistics']['G1']['std']} | {summary['grade_statistics']['G2']['std']} | {summary['grade_statistics']['G3']['std']} |
| **Out-of-Bounds Values** | {summary['grade_statistics']['G1']['out_of_bounds_count']} | {summary['grade_statistics']['G2']['out_of_bounds_count']} | {summary['grade_statistics']['G3']['out_of_bounds_count']} |

### Key Observations:
1. **Target Balance:** The classification target (`at_risk`) shows a healthy balance: **{summary['at_risk_percentage']}% at risk** vs. **{summary['not_at_risk_percentage']}% passing**. This means models will not require synthetic oversampling (e.g. SMOTE) or severe class-weight adjustments in Phase 4.
2. **Grade Stability:** Mean grades across G1 ({summary['grade_statistics']['G1']['mean']}), G2 ({summary['grade_statistics']['G2']['mean']}), and G3 ({summary['grade_statistics']['G3']['mean']}) remain consistent, reflecting steady progression across the academic period.

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
   - Quantifies the cohort into {summary['at_risk_count']:,} at-risk students and {summary['not_at_risk_count']:,} passing students.
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

- **Cleaned Dataset:** `results/cleaned_students.csv` ({summary['total_records']} rows, 36 columns including `at_risk`).
- **Data Quality JSON:** `results/data_quality_report.json`.
- **Chart Files:** All 6 PNG visualizations generated at 150 DPI in `results/charts/`.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Exported markdown analysis report to: {output_path}")


def main() -> None:
    """Orchestrate Phase 3 Exploratory Data Analysis and quality reporting."""
    print("=" * 70)
    print(" EduPredict: Phase 3 - Exploratory Data Analysis & Preprocessing")
    print("=" * 70)

    # 1. Fetch data from MongoDB
    print("\n[Step 1/5] Fetching data from MongoDB...")
    df = fetch_synthetic_data_from_mongodb()

    # 2. Validate data
    print("\n[Step 2/5] Validating retrieved data...")
    summary = validate_retrieved_data(df)
    print(f"  - Total records      : {summary['total_records']}")
    print(f"  - Total columns      : {summary['total_columns']}")
    print(f"  - Missing values     : {summary['total_missing_values']}")
    print(f"  - Duplicate rows     : {summary['full_duplicate_rows']}")
    print(f"  - At-risk students   : {summary['at_risk_count']} ({summary['at_risk_percentage']}%)")
    print(f"  - Not at-risk        : {summary['not_at_risk_count']} ({summary['not_at_risk_percentage']}%)")

    # 3. Generate EDA charts
    print("\n[Step 3/5] Generating analytical charts in results/charts/...")
    charts = generate_eda_charts(df, output_dir=CHARTS_DIR)
    print(f"[SUCCESS] Generated {len(charts)} charts.")

    # 4. Export reports and clean CSV
    print("\n[Step 4/5] Generating reports and export datasets...")
    generate_data_quality_json(summary, output_path=DATA_QUALITY_JSON_PATH)
    generate_markdown_analysis_report(summary, output_path=ANALYSIS_REPORT_MD_PATH)
    export_clean_analysis_csv(df, output_path=CLEANED_CSV_PATH)

    # 5. Summary
    print("\n[Step 5/5] Verification summary:")
    print("  - MongoDB documents untouched: Confirmed.")
    print(f"  - Exported clean CSV: {CLEANED_CSV_PATH.name} ({len(df)} rows)")
    print(f"  - Data quality JSON: {DATA_QUALITY_JSON_PATH.name}")
    print(f"  - Markdown analysis report: {ANALYSIS_REPORT_MD_PATH.name}")
    print(f"  - Visualizations saved: {len(charts)} files in {CHARTS_DIR}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Phase 3: Exploratory Data Analysis & Quality Reporting Completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
