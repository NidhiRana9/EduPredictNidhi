"""About the Project View for EduPredict.

Provides comprehensive project context, data lineage documentation,
anti-leakage architectural principles, model descriptions, and ethical
limitations in academic analytics.
"""

import streamlit as st
from src.ui.components import render_callout, render_header


def render_about_page() -> None:
    """Render the About the Project page view."""
    render_header(
        title="About EduPredict",
        subtitle="A modular, leak-proof Machine Learning system for early academic performance analysis.",
    )

    # Workflow architecture container
    st.markdown(
        """
        <div class="editorial-card">
            <div class="editorial-card-header">System Architecture &amp; Data Flow</div>
            <div style="padding: 0.75rem 0; font-size: 0.92rem; color: #292D29; line-height: 1.6;">
                The EduPredict architecture operates as an end-to-end, leak-proof prediction and analysis pipeline:
            </div>
            <div style="background-color: #FAF8F3; border: 1px solid #E5E0D7; border-radius: 6px; padding: 1.1rem 1.4rem; font-family: monospace; font-size: 0.88rem; color: #355B45; line-height: 1.8;">
                <strong>Student Records (MongoDB)</strong><br/>
                &nbsp;&nbsp;&darr; (Sanitize &amp; actively strip _id, G1, G2, G3, at_risk)<br/>
                <strong>Feature Validation &amp; Schema Enforcement</strong> (30 non-grade variables)<br/>
                &nbsp;&nbsp;&darr; (Pass through pre-fitted scikit-learn ColumnTransformer)<br/>
                <strong>Persisted Model Artifacts</strong> (.joblib pipelines: DT, RF Classifier, RF Regressor)<br/>
                &nbsp;&nbsp;&darr; (Multi-model inference without in-flight refitting)<br/>
                <strong>Structured Prediction Output</strong> (Risk flags, risk probability, estimated grade)<br/>
                &nbsp;&nbsp;&darr; (Idempotent upsert on student_id + model_version)<br/>
                <strong>MongoDB Storage (student_predictions)</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-header">Target Variables &amp; Formulation</div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: #292D29;">
                    <p>EduPredict evaluates two distinct machine learning targets:</p>
                    <ul style="padding-left: 1.15rem;">
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Task A — Binary Classification (<code>at_risk</code>):</strong><br/>
                            Flags whether a student is projected to fall below the passing mark on the standard 0–20 Portuguese academic scale:
                            <br/><code>at_risk = 1 if G3 &lt; 10 else 0</code>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Task B — Continuous Regression (<code>G3</code>):</strong><br/>
                            Estimates the numerical final grade (0.0 to 20.0 points) using an ensemble regressor.
                        </li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-header">Role of Local MongoDB</div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: #292D29;">
                    <p>EduPredict utilizes a local MongoDB Community instance (<code>edupredict_db</code>) structured into dedicated collections:</p>
                    <ul style="padding-left: 1.15rem;">
                        <li style="margin-bottom: 0.4rem;">
                            <code>students</code>: Stores imported baseline student demographic, familial, and academic profiles.
                        </li>
                        <li style="margin-bottom: 0.4rem;">
                            <code>student_predictions</code>: Stores multi-model assessment outputs using idempotent upserts on <code>(student_id, model_version)</code>.
                        </li>
                        <li style="margin-bottom: 0.4rem;">
                            <code>model_evaluations</code>: Stores versioned performance logs and confusion matrix data.
                        </li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-header">The Three Persisted ML Models</div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: #292D29;">
                    <p>Model artifacts are stored in <code>models/*.joblib</code> and loaded immutably for inference:</p>
                    <ul style="padding-left: 1.15rem;">
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Decision Tree Classifier:</strong><br/>
                            Trained with <code>class_weight='balanced'</code>. Achieved <strong>62.7% recall</strong> on the held-out test set, making it the primary operational flag for early student support.
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Random Forest Classifier:</strong><br/>
                            100-tree ensemble classifier offering conservative risk predictions and calibrated class probabilities (<code>predict_proba</code>).
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <strong>Random Forest Regressor:</strong><br/>
                            Estimates expected final marks (MAE &approx; 3.39 points) to provide a continuous reference benchmark.
                        </li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-card-header">Target Leakage Prevention</div>
                <div style="font-size: 0.88rem; line-height: 1.55; color: #292D29;">
                    <p>A fundamental vulnerability in academic analytics is grade leakage. If models have access to midterm test scores (<code>G1</code>, <code>G2</code>), predicting the final exam (<code>G3</code>) becomes trivial but provides zero early-warning utility.</p>
                    <p style="color: #355B45; font-weight: 600; margin-bottom: 0;">
                        EduPredict enforces strict feature exclusion across all stages, ensuring only pre-semester non-grade features are used.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Synthetic Dataset & Ethical Disclaimer Callouts
    render_callout(
        title="Dataset Notice & Synthetic Data Provenance",
        body=(
            "The data used throughout this project (<code>data/student-mat-synthetic.csv</code> and MongoDB collection <code>students</code>) "
            "is entirely synthetic. It was generated to mirror the statistical distributions, feature schemas, and target relationships of the benchmark "
            "Student Performance dataset published by P. Cortez and A. Silva (2008). "
            "It contains no genuine, personally identifiable records from any school, university, or educational institution."
        ),
        is_warning=False,
    )

    render_callout(
        title="Ethical Considerations & Advisory Scope",
        body=(
            "Model predictions produced by EduPredict are statistical estimations designed to assist academic advisors in offering early tutoring, "
            "mentorship, and study habit support. Predictions must never be treated as deterministic assessments of a student's intellect, "
            "nor used for high-stakes decisions such as admissions, tracking, or punitive measures."
        ),
        is_warning=True,
    )
