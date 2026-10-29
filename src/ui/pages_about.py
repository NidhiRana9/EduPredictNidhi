"""About the Project View for EduPredict.

Provides comprehensive project context, data lineage documentation,
anti-leakage architectural principles, model descriptions, and ethical
limitations in academic analytics.
"""

import streamlit as st
from src.ui.components import clean_html, render_callout, render_header


def render_about_page() -> None:
    """Render the About the Project page view in Green Dashboard style."""
    render_header(
        title="About EduPredict Platform",
        subtitle="End-to-end leak-proof machine learning system for early academic performance diagnostics.",
        category_label="SYSTEM ARCHITECTURE & METHODOLOGY",
    )

    # Workflow architecture container
    arch_html = """
    <div class="dashboard-card">
        <div class="dashboard-card-header">
            <span class="dashboard-card-title">System Architecture &amp; Data Pipeline</span>
            <span class="status-pill status-pill-green">Zero Leakage Architecture</span>
        </div>
        <div style="padding: 0.5rem 0 0.75rem 0; font-size: 0.92rem; color: #12281D; line-height: 1.55;">
            EduPredict operates as an end-to-end, leak-proof diagnostic prediction pipeline:
        </div>
        <div style="background-color: #F8FAF9; border: 1px solid #E2ECE5; border-radius: 8px; padding: 1.15rem 1.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #0A3E26; line-height: 1.85;">
            <span style="color: #15803D; font-weight: 700;">1. Student Documents (MongoDB)</span><br/>
            &nbsp;&nbsp;&darr; (Sanitize &amp; actively strip _id, G1, G2, G3, at_risk)<br/>
            <span style="color: #15803D; font-weight: 700;">2. Feature Validation &amp; Schema Enforcement</span> (30 non-grade variables)<br/>
            &nbsp;&nbsp;&darr; (Pass through pre-fitted scikit-learn ColumnTransformer)<br/>
            <span style="color: #15803D; font-weight: 700;">3. Persisted Model Artifacts</span> (.joblib pipelines: DT, RF Classifier, RF Regressor)<br/>
            &nbsp;&nbsp;&darr; (Multi-model inference without in-flight refitting)<br/>
            <span style="color: #15803D; font-weight: 700;">4. Structured Prediction Output</span> (Risk flags, risk probability, estimated grade)<br/>
            &nbsp;&nbsp;&darr; (Idempotent upsert on student_id + model_version)<br/>
            <span style="color: #15803D; font-weight: 700;">5. MongoDB Storage (student_predictions)</span>
        </div>
    </div>
    """
    st.markdown(clean_html(arch_html), unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        target_html = """
        <div class="dashboard-card">
            <div class="dashboard-card-header">
                <span class="dashboard-card-title">Target Variables &amp; Formulation</span>
                <span class="status-pill status-pill-green">Dual Objective</span>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #12281D;">
                <p>EduPredict evaluates two distinct analytical objectives:</p>
                <ul style="padding-left: 1.15rem;">
                    <li style="margin-bottom: 0.65rem;">
                        <strong style="color: #0A3E26;">Task A — Binary Classification (<code>at_risk</code>):</strong><br/>
                        Flags whether a student is projected to fall below the passing mark on the standard 0–20 Portuguese academic scale:
                        <br/><code style="color: #15803D; background: #DCFCE7; padding: 2px 6px; border-radius: 4px;">at_risk = 1 if G3 &lt; 10 else 0</code>
                    </li>
                    <li style="margin-bottom: 0.65rem;">
                        <strong style="color: #0A3E26;">Task B — Continuous Regression (<code>G3</code>):</strong><br/>
                        Estimates the continuous final examination mark (0.0 to 20.0 points) using an ensemble regressor.
                    </li>
                </ul>
            </div>
        </div>
        """
        st.markdown(clean_html(target_html), unsafe_allow_html=True)

    with col2:
        leakage_html = """
        <div class="dashboard-card">
            <div class="dashboard-card-header">
                <span class="dashboard-card-title">Anti-Leakage Engineering</span>
                <span class="status-pill status-pill-green">Strict Isolation</span>
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #12281D;">
                <p>Target leakage occurs when future or post-enrollment test marks leak into early-warning indicators:</p>
                <ul style="padding-left: 1.15rem;">
                    <li style="margin-bottom: 0.65rem;">
                        <strong>No Grade Predictors:</strong> <code>G1</code> (1st Period) and <code>G2</code> (2nd Period) are excluded, ensuring the model functions before exams occur.
                    </li>
                    <li style="margin-bottom: 0.65rem;">
                        <strong>No Preprocessor Contamination:</strong> The ColumnTransformer fits exclusively on training folds, storing scaling medians and category lists directly in joblib artifacts.
                    </li>
                </ul>
            </div>
        </div>
        """
        st.markdown(clean_html(leakage_html), unsafe_allow_html=True)

    # Ethical limitations callout
    render_callout(
        title="Ethical Guidance & Academic Advisor Responsibility",
        body=(
            "EduPredict is an early-warning decision support system designed exclusively to help academic counselors "
            "identify students who may benefit from proactive tutoring, mentoring, or check-ins. "
            "Model outputs are statistical likelihoods derived from cohort patterns, not definitive forecasts of student capability. "
            "Under no circumstances should automated model predictions be used to deny academic opportunities, gate courses, "
            "or replace human evaluation."
        ),
        is_warning=False,
    )
