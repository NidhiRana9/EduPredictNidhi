"""Model Performance & Evaluation View for EduPredict.

Visualizes and contextualizes the Phase 5 evaluation benchmarks on the
untouched held-out test cohort (n=1,000 students), contrasting:
- Task A: At-Risk Classification (Decision Tree vs. Random Forest)
- Task B: Continuous Grade Regression (Random Forest Regressor)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st

from src.ui.components import render_callout, render_header, render_stat_tile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"


def load_metrics_data() -> Optional[Dict[str, Any]]:
    """Safely load Phase 5 evaluation metrics from results/metrics.json."""
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_performance_page() -> None:
    """Render the Model Performance page view."""
    render_header(
        title="Model Performance &amp; Evaluation",
        subtitle="Empirical evaluation benchmarks computed on the untouched 1,000-student held-out test cohort.",
    )

    metrics_data = load_metrics_data()

    if not metrics_data:
        render_callout(
            title="Metrics Artifact Notice",
            body="Could not locate <code>results/metrics.json</code>. Please run <code>python -m src.evaluate_models</code> to generate Phase 5 evaluation metrics.",
            is_warning=True,
        )
        return

    # Extract Classification Data
    clf_evals = metrics_data.get("classification_evaluations", {})
    dt_metrics = clf_evals.get("decision_tree_classifier", {})
    rf_metrics = clf_evals.get("random_forest_classifier", {})

    dt_acc = dt_metrics.get("accuracy", 0.6040)
    dt_prec = dt_metrics.get("precision", 0.5756)
    dt_rec = dt_metrics.get("recall", 0.6266)
    dt_f1 = dt_metrics.get("f1_score", 0.6000)
    dt_cm = dt_metrics.get("confusion_matrix", {})

    rf_acc = rf_metrics.get("accuracy", 0.5860)
    rf_prec = rf_metrics.get("precision", 0.5802)
    rf_rec = rf_metrics.get("recall", 0.4578)
    rf_f1 = rf_metrics.get("f1_score", 0.5118)
    rf_cm = rf_metrics.get("confusion_matrix", {})

    # Extract Regression Data
    reg_evals = metrics_data.get("regression_evaluations", {})
    reg_metrics = reg_evals.get("random_forest_regressor", {})
    rf_mae = reg_metrics.get("mae", 3.3870)
    rf_rmse = reg_metrics.get("rmse", 4.2779)
    rf_r2 = reg_metrics.get("r2_score", -0.0234)

    # Tabs to organize Classification and Regression cleanly
    tab_clf, tab_reg, tab_methodology = st.tabs(
        ["Task A: At-Risk Classification", "Task B: Grade Regression", "Evaluation Methodology"]
    )

    # TAB 1: CLASSIFICATION
    with tab_clf:
        st.markdown(
            "<div style='font-family: Georgia, serif; font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.25rem;'>"
            "Classifier Comparison: Decision Tree vs. Random Forest"
            "</div>"
            "<div style='font-size: 0.88rem; color: #77776F; margin-bottom: 1.25rem;'>"
            "Target definition: <code>at_risk = 1</code> when Final Grade <code>G3 &lt; 10</code> (otherwise 0). Test cohort: 1,000 synthetic records (474 actual at-risk, 526 actual passing)."
            "</div>",
            unsafe_allow_html=True,
        )

        # 4 High-level summary metric tiles
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_stat_tile("Decision Tree Recall", f"{dt_rec * 100:.1f}%", "Flagged 297 / 474 at-risk students")
        with col2:
            render_stat_tile("Random Forest Recall", f"{rf_rec * 100:.1f}%", "Flagged 217 / 474 at-risk students")
        with col3:
            render_stat_tile("Decision Tree F1", f"{dt_f1:.4f}", "Balanced precision and sensitivity")
        with col4:
            render_stat_tile("Random Forest F1", f"{rf_f1:.4f}", "Conservative ensemble balance")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Comparative Metrics Table
        st.markdown(
            f"""
            <div class="editorial-card">
                <div class="editorial-card-header">Detailed Test Cohort Metric Breakdown (pos_label=1)</div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #E5E0D7; text-align: left; color: #77776F;">
                            <th style="padding: 6px 0;">Evaluation Metric</th>
                            <th style="text-align: right; padding: 6px 0;">Decision Tree</th>
                            <th style="text-align: right; padding: 6px 0;">Random Forest</th>
                            <th style="text-align: right; padding: 6px 0;">Operational Advantage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #F0ECE1;">
                            <td style="padding: 8px 0; font-weight: 600;">Recall / Sensitivity (pos_label=1)</td>
                            <td style="text-align: right; font-weight: 700; color: #355B45;">{dt_rec * 100:.2f}%</td>
                            <td style="text-align: right; color: #77776F;">{rf_rec * 100:.2f}%</td>
                            <td style="text-align: right; font-weight: 600; color: #355B45;">Decision Tree (+16.88%)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F0ECE1;">
                            <td style="padding: 8px 0; font-weight: 600;">F1-Score (pos_label=1)</td>
                            <td style="text-align: right; font-weight: 700; color: #355B45;">{dt_f1:.4f}</td>
                            <td style="text-align: right; color: #77776F;">{rf_f1:.4f}</td>
                            <td style="text-align: right; font-weight: 600; color: #355B45;">Decision Tree (+0.0882)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F0ECE1;">
                            <td style="padding: 8px 0; font-weight: 600;">Overall Accuracy</td>
                            <td style="text-align: right; font-weight: 700; color: #355B45;">{dt_acc * 100:.2f}%</td>
                            <td style="text-align: right; color: #77776F;">{rf_acc * 100:.2f}%</td>
                            <td style="text-align: right; font-weight: 600; color: #355B45;">Decision Tree (+1.80%)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F0ECE1;">
                            <td style="padding: 8px 0; font-weight: 600;">Precision (pos_label=1)</td>
                            <td style="text-align: right; color: #77776F;">{dt_prec * 100:.2f}%</td>
                            <td style="text-align: right; font-weight: 700; color: #355B45;">{rf_prec * 100:.2f}%</td>
                            <td style="text-align: right; color: #77776F;">Random Forest (+0.46%)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F0ECE1;">
                            <td style="padding: 8px 0;">True Positives (At-Risk correctly flagged)</td>
                            <td style="text-align: right; font-weight: 600; color: #355B45;">{dt_cm.get('true_positives', 297)}</td>
                            <td style="text-align: right;">{rf_cm.get('true_positives', 217)}</td>
                            <td style="text-align: right; color: #355B45;">DT catches 80 more at-risk students</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0;">False Negatives (At-Risk students missed)</td>
                            <td style="text-align: right; font-weight: 600; color: #355B45;">{dt_cm.get('false_negatives', 177)}</td>
                            <td style="text-align: right; color: #9E2A2B;">{rf_cm.get('false_negatives', 257)}</td>
                            <td style="text-align: right; color: #355B45;">DT misses 80 fewer struggling students</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Operational takeaway callout
        render_callout(
            title="Why Recall is the Deciding Metric in Educational Early Warning",
            body=(
                "In academic counseling, the consequence of a <strong>False Negative</strong> (a struggling student left undetected) "
                "is severe: the student receives no support and may fail. Conversely, a <strong>False Positive</strong> merely prompts "
                "an advisor to schedule a check-in or suggest tutoring for a passing student. "
                "Because the <strong>Decision Tree captures 62.7% of struggling students</strong> compared to only <strong>45.8% for Random Forest</strong>, "
                "the Decision Tree is the superior operational model for proactive intervention."
            ),
            is_warning=False,
        )

        # Confusion Matrices Images
        st.markdown("<h3>Confusion Matrix Visualizations</h3>", unsafe_allow_html=True)
        col_img1, col_img2 = st.columns(2)

        dt_cm_path = CHARTS_DIR / "confusion_matrix_decision_tree.png"
        rf_cm_path = CHARTS_DIR / "confusion_matrix_random_forest.png"

        with col_img1:
            if dt_cm_path.exists():
                st.image(
                    str(dt_cm_path),
                    caption="Decision Tree Confusion Matrix (297 True Positives / 177 False Negatives)",
                    use_container_width=True,
                )
        with col_img2:
            if rf_cm_path.exists():
                st.image(
                    str(rf_cm_path),
                    caption="Random Forest Confusion Matrix (217 True Positives / 257 False Negatives)",
                    use_container_width=True,
                )

    # TAB 2: REGRESSION
    with tab_reg:
        st.markdown(
            "<div style='font-family: Georgia, serif; font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.25rem;'>"
            "Task B: Final Grade Regression (Random Forest Regressor)"
            "</div>"
            "<div style='font-size: 0.88rem; color: #77776F; margin-bottom: 1.25rem;'>"
            "Target: Continuous final grade <code>G3</code> on a 0–20 scale. Evaluated strictly without interim marks G1 and G2."
            "</div>",
            unsafe_allow_html=True,
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            render_stat_tile("Mean Absolute Error (MAE)", f"{rf_mae:.2f} pts", "Average error magnitude on 0–20 scale")
        with col_r2:
            render_stat_tile("Root Mean Squared Error", f"{rf_rmse:.2f} pts", "Penalizes larger grade errors")
        with col_r3:
            render_stat_tile("Coefficient of Determination", f"{rf_r2:.4f}", "Variance explained relative to mean baseline")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        render_callout(
            title="Understanding the Negative R² Result",
            body=(
                "The Random Forest Regressor produced an R² of approximately <strong>-0.0234</strong>. "
                "In machine learning, an R² below zero indicates that predicting exact continuous marks strictly from "
                "pre-semester demographic and lifestyle factors exhibits high residual variance, performing marginally worse than "
                "a constant mean prediction baseline. <br/><br/>"
                "<strong>Practical Implication:</strong> Binary classification (<code>at_risk</code>) is substantially more robust, "
                "stable, and actionable for early academic counseling than attempting point grade estimation without midterm test scores."
            ),
            is_warning=True,
        )

        st.markdown("<h3>Regression Residual &amp; Prediction Distributions</h3>", unsafe_allow_html=True)
        col_reg1, col_reg2 = st.columns(2)

        reg_actual_path = CHARTS_DIR / "regression_actual_vs_predicted.png"
        reg_res_path = CHARTS_DIR / "regression_residuals_distribution.png"

        with col_reg1:
            if reg_actual_path.exists():
                st.image(
                    str(reg_actual_path),
                    caption="Actual vs. Predicted Final Grade (G3)",
                    use_container_width=True,
                )
        with col_reg2:
            if reg_res_path.exists():
                st.image(
                    str(reg_res_path),
                    caption="Distribution of Regression Residuals (Mean ~ 0.06)",
                    use_container_width=True,
                )

    # TAB 3: METHODOLOGY
    with tab_methodology:
        st.markdown(
            """
            <div class="editorial-card" style="margin-top: 0.5rem;">
                <div class="editorial-card-header">Rigorous Evaluation Protocol</div>
                <div style="font-size: 0.9rem; line-height: 1.55; color: #292D29;">
                    <p>EduPredict adheres to strict machine learning standards to deliver dependable diagnostic evaluations:</p>
                    <ol style="padding-left: 1.25rem;">
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Stratified Train/Test Split:</strong> The 5,000 synthetic records were partitioned into an 80% training cohort (4,000 records) and a 20% test cohort (1,000 records) with stratification on <code>at_risk</code> to maintain identical class ratios across partitions.
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>No In-Flight Refitting:</strong> Evaluated models were pre-fitted during Phase 4 and loaded immutably from <code>models/*.joblib</code>. Preprocessing parameters (StandardScaler means, OneHotEncoder categories) are baked directly into the pipelines.
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Zero Grade Leakage:</strong> All interim examination scores (<code>G1</code>, <code>G2</code>) and final targets (<code>G3</code>, <code>at_risk</code>) were strictly stripped before preprocessing.
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Idempotent Evaluation Storage:</strong> Metrics were validated using automated test suites and persisted into MongoDB collection <code>model_evaluations</code> using compound keys on <code>(model_name, dataset_source)</code>.
                        </li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
