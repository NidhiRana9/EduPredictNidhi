"""Assess a Student View for EduPredict.

Provides an intuitive green dashboard form to collect the 30 non-grade predictor features,
execute inference through the existing PredictionService, save the result
idempotently into MongoDB, and present structured multi-model results.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import streamlit as st
from pymongo import UpdateOne

from src.database import (
    COLLECTION_STUDENT_PREDICTIONS,
    COLLECTION_STUDENTS,
    get_database,
    get_mongo_client,
    ping_database,
)
from src.prediction_service import PredictionService
from src.ui.components import clean_html, render_callout, render_header, render_risk_gauge, render_stat_tile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource
def get_cached_prediction_service() -> PredictionService:
    """Instantiate and cache the PredictionService across user interactions.

    Loads the 3 pre-trained joblib pipelines once, preventing redundant disk I/O.
    """
    return PredictionService()


def save_prediction_to_db(prediction_doc: Dict[str, Any]) -> Tuple[bool, str]:
    """Save prediction document into MongoDB student_predictions collection.

    Uses an idempotent upsert on (student_id, model_version).
    """
    client = None
    try:
        client = get_mongo_client(timeout_ms=2000)
        ping_database(client)
        db = get_database(client)
        col = db[COLLECTION_STUDENT_PREDICTIONS]

        filter_doc = {
            "student_id": prediction_doc["student_id"],
            "model_version": prediction_doc["model_version"],
        }
        res = col.update_one(filter_doc, {"$set": prediction_doc}, upsert=True)
        is_new = res.upserted_id is not None
        return True, "Saved to MongoDB collection 'student_predictions' (idempotent upsert)."
    except Exception as exc:
        return False, f"Prediction computed, but MongoDB write failed: {exc}"
    finally:
        if client is not None:
            client.close()


def render_predict_page() -> None:
    """Render the Assess a Student form and results panel."""
    render_header(
        title="Student Risk Assessment",
        subtitle="Evaluate individual student academic risk and estimated final grade using 30 pre-semester indicators.",
        category_label="STUDENT INFERENCE ENGINE",
    )

    # Educational limitation notice
    render_callout(
        title="Anti-Leakage Protocol Active",
        body=(
            "In strict compliance with academic prediction guidelines, <strong>G1 (First Period)</strong>, "
            "<strong>G2 (Second Period)</strong>, and <strong>G3 (Final Exam Mark)</strong> are completely excluded. "
            "All indicators are derived from pre-semester demographic, familial, attendance, and study variables."
        ),
        is_warning=False,
    )

    # Initialize PredictionService
    try:
        service = get_cached_prediction_service()
    except Exception as exc:
        st.error(
            f"Failed to initialize prediction models. Ensure model artifacts exist in models/. Error: {exc}"
        )
        return

    # Profile pre-fill presets to facilitate evaluation testing
    st.markdown(clean_html("<h3>Assessment Presets &amp; Student Identifier</h3>"), unsafe_allow_html=True)
    col_preset, col_id = st.columns([0.6, 0.4])

    with col_preset:
        preset_option = st.selectbox(
            "Quick Profile Presets",
            options=["Custom Form Input", "High-Risk Student Profile", "Standard Passing Student Profile"],
            help="Select a representative profile to quickly inspect model behavior without manually configuring all 30 features.",
        )

    with col_id:
        default_id = f"STU-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
        student_id_input = st.text_input(
            "Student Identifier",
            value=default_id,
            help="Unique student ID or registration code. Used as the idempotent storage key in MongoDB.",
        )

    # Configure defaults according to selected preset
    if preset_option == "High-Risk Student Profile":
        p_age, p_failures, p_absences = 18, 2, 16
        p_studytime, p_freetime, p_goout = 1, 4, 4
        p_dalc, p_walc, p_health = 3, 4, 2
        p_higher, p_schoolsup, p_famsup, p_paid = "no", "yes", "no", "no"
        p_medu, p_fedu = 1, 1
        p_famrel, p_internet, p_romantic = 2, "no", "yes"
    elif preset_option == "Standard Passing Student Profile":
        p_age, p_failures, p_absences = 16, 0, 2
        p_studytime, p_freetime, p_goout = 3, 3, 2
        p_dalc, p_walc, p_health = 1, 1, 4
        p_higher, p_schoolsup, p_famsup, p_paid = "yes", "no", "yes", "yes"
        p_medu, p_fedu = 4, 3
        p_famrel, p_internet, p_romantic = 4, "yes", "no"
    else:
        # Balanced defaults from dataset medians
        p_age, p_failures, p_absences = 17, 0, 4
        p_studytime, p_freetime, p_goout = 2, 3, 3
        p_dalc, p_walc, p_health = 1, 2, 4
        p_higher, p_schoolsup, p_famsup, p_paid = "yes", "no", "no", "no"
        p_medu, p_fedu = 3, 3
        p_famrel, p_internet, p_romantic = 4, "yes", "no"

    st.markdown(clean_html("<div style='height: 0.5rem;'></div>"), unsafe_allow_html=True)

    with st.form("student_assessment_form"):
        # SECTION 1: Student and School Information (5 features)
        sec1_html = """
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.05rem; font-weight: 700; color: #0A3E26; border-bottom: 2px solid #E2ECE5; padding-bottom: 6px; margin-bottom: 14px;">
            <span style="background: #15803D; color: #FFFFFF; border-radius: 6px; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800;">1</span>
            <span>Student &amp; School Demographics</span>
        </div>
        """
        st.markdown(clean_html(sec1_html), unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            school = st.selectbox(
                "School",
                options=["GP", "MS"],
                format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)",
            )
        with c2:
            sex = st.selectbox(
                "Sex",
                options=["F", "M"],
                format_func=lambda x: "Female" if x == "F" else "Male",
            )
        with c3:
            age = st.number_input("Age", min_value=15, max_value=22, value=p_age, step=1)
        with c4:
            address = st.selectbox(
                "Home Address",
                options=["U", "R"],
                format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)",
            )
        with c5:
            reason = st.selectbox(
                "School Choice Reason",
                options=["course", "home", "reputation", "other"],
                format_func=lambda x: {
                    "course": "Course Preference",
                    "home": "Close to Home",
                    "reputation": "School Reputation",
                    "other": "Other",
                }.get(x, x),
            )

        # SECTION 2: Family and Educational Background (7 features)
        sec2_html = """
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.05rem; font-weight: 700; color: #0A3E26; border-bottom: 2px solid #E2ECE5; padding-bottom: 6px; margin-top: 18px; margin-bottom: 14px;">
            <span style="background: #15803D; color: #FFFFFF; border-radius: 6px; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800;">2</span>
            <span>Family &amp; Educational Background</span>
        </div>
        """
        st.markdown(clean_html(sec2_html), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            famsize = st.selectbox(
                "Family Size",
                options=["GT3", "LE3"],
                format_func=lambda x: "> 3 Members (GT3)" if x == "GT3" else "≤ 3 Members (LE3)",
            )
            guardian = st.selectbox(
                "Legal Guardian",
                options=["mother", "father", "other"],
                format_func=lambda x: x.capitalize(),
            )
        with c2:
            Pstatus = st.selectbox(
                "Parent Cohabitation",
                options=["T", "A"],
                format_func=lambda x: "Living Together (T)" if x == "T" else "Apart (A)",
            )
            Medu = st.selectbox(
                "Mother's Education",
                options=[0, 1, 2, 3, 4],
                index=p_medu,
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary (4th gr.)",
                    2: "2 - 5th to 9th gr.",
                    3: "3 - Secondary",
                    4: "4 - Higher education",
                }[x],
            )
        with c3:
            Mjob = st.selectbox(
                "Mother's Job",
                options=["other", "services", "at_home", "teacher", "health"],
                format_func=lambda x: {
                    "services": "Civil Services",
                    "at_home": "At Home",
                    "teacher": "Teacher",
                    "health": "Healthcare",
                    "other": "Other",
                }.get(x, x),
            )
            Fedu = st.selectbox(
                "Father's Education",
                options=[0, 1, 2, 3, 4],
                index=p_fedu,
                format_func=lambda x: {
                    0: "0 - None",
                    1: "1 - Primary (4th gr.)",
                    2: "2 - 5th to 9th gr.",
                    3: "3 - Secondary",
                    4: "4 - Higher education",
                }[x],
            )
        with c4:
            Fjob = st.selectbox(
                "Father's Job",
                options=["other", "services", "at_home", "teacher", "health"],
                format_func=lambda x: {
                    "services": "Civil Services",
                    "at_home": "At Home",
                    "teacher": "Teacher",
                    "health": "Healthcare",
                    "other": "Other",
                }.get(x, x),
            )

        # SECTION 3: Study Habits and Educational Support (10 features)
        sec3_html = """
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.05rem; font-weight: 700; color: #0A3E26; border-bottom: 2px solid #E2ECE5; padding-bottom: 6px; margin-top: 18px; margin-bottom: 14px;">
            <span style="background: #15803D; color: #FFFFFF; border-radius: 6px; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800;">3</span>
            <span>Study Habits &amp; Support Systems</span>
        </div>
        """
        st.markdown(clean_html(sec3_html), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            traveltime = st.selectbox(
                "Travel Time to School",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1: < 15 min",
                    2: "2: 15 to 30 min",
                    3: "3: 30 to 60 min",
                    4: "4: > 60 min",
                }[x],
            )
            schoolsup = st.selectbox(
                "Extra School Academic Support",
                options=["no", "yes"],
                index=0 if p_schoolsup == "no" else 1,
                format_func=lambda x: x.capitalize(),
            )
            activities = st.selectbox(
                "Extra-Curricular Activities",
                options=["no", "yes"],
                format_func=lambda x: x.capitalize(),
            )
            internet = st.selectbox(
                "Internet Access at Home",
                options=["yes", "no"],
                index=0 if p_internet == "yes" else 1,
                format_func=lambda x: x.capitalize(),
            )

        with c2:
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                index=p_studytime - 1,
                format_func=lambda x: {
                    1: "1: < 2 hours",
                    2: "2: 2 to 5 hours",
                    3: "3: 5 to 10 hours",
                    4: "4: > 10 hours",
                }[x],
            )
            famsup = st.selectbox(
                "Family Educational Support",
                options=["no", "yes"],
                index=0 if p_famsup == "no" else 1,
                format_func=lambda x: x.capitalize(),
            )
            nursery = st.selectbox(
                "Attended Nursery School",
                options=["yes", "no"],
                format_func=lambda x: x.capitalize(),
            )

        with c3:
            failures = st.selectbox(
                "Past Class Failures",
                options=[0, 1, 2, 3],
                index=p_failures,
                format_func=lambda x: f"{x} failure{'s' if x != 1 else ''}",
            )
            paid = st.selectbox(
                "Extra Paid Tutoring Classes",
                options=["no", "yes"],
                index=0 if p_paid == "no" else 1,
                format_func=lambda x: x.capitalize(),
            )
            higher = st.selectbox(
                "Aims for Higher Education",
                options=["yes", "no"],
                index=0 if p_higher == "yes" else 1,
                format_func=lambda x: x.capitalize(),
            )

        # SECTION 4: Lifestyle, Health, and Attendance (8 features)
        sec4_html = """
        <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.05rem; font-weight: 700; color: #0A3E26; border-bottom: 2px solid #E2ECE5; padding-bottom: 6px; margin-top: 18px; margin-bottom: 14px;">
            <span style="background: #15803D; color: #FFFFFF; border-radius: 6px; width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 800;">4</span>
            <span>Lifestyle, Health &amp; Attendance</span>
        </div>
        """
        st.markdown(clean_html(sec4_html), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            romantic = st.selectbox(
                "Romantic Relationship",
                options=["no", "yes"],
                index=0 if p_romantic == "no" else 1,
                format_func=lambda x: x.capitalize(),
            )
            famrel = st.slider(
                "Family Relationship Quality",
                min_value=1,
                max_value=5,
                value=p_famrel,
                help="1: Very poor, 5: Excellent",
            )

        with c2:
            freetime = st.slider(
                "Free Time After School",
                min_value=1,
                max_value=5,
                value=p_freetime,
                help="1: Very low, 5: Very high",
            )
            goout = st.slider(
                "Going Out with Friends",
                min_value=1,
                max_value=5,
                value=p_goout,
                help="1: Very low, 5: Very high",
            )

        with c3:
            Dalc = st.slider(
                "Workday Alcohol Consumption",
                min_value=1,
                max_value=5,
                value=p_dalc,
                help="1: Very low, 5: Very high",
            )
            Walc = st.slider(
                "Weekend Alcohol Consumption",
                min_value=1,
                max_value=5,
                value=p_walc,
                help="1: Very low, 5: Very high",
            )

        with c4:
            health = st.slider(
                "Current Health Status",
                min_value=1,
                max_value=5,
                value=p_health,
                help="1: Very poor, 5: Very good",
            )
            absences = st.number_input(
                "School Absences (Days)",
                min_value=0,
                max_value=30,
                value=p_absences,
                step=1,
            )

        st.markdown(clean_html("<div style='height: 1rem;'></div>"), unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Run Academic Risk Assessment →",
            type="primary",
            use_container_width=True,
        )

    # PROCESS FORM SUBMISSION
    if submitted:
        # Build exact 30-feature dictionary
        feature_dict: Dict[str, Any] = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
        }

        # Resolve student identifier
        clean_sid = student_id_input.strip() or f"STU-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

        try:
            # Execute inference using existing PredictionService
            prediction_result = service.predict_single(feature_dict, student_id=clean_sid)

            # Persist to MongoDB student_predictions
            db_success, db_msg = save_prediction_to_db(prediction_result)

            # Display focused results panel
            render_assessment_results(clean_sid, prediction_result, db_success, db_msg)

        except Exception as exc:
            st.error(f"Inference execution failed: {exc}")


def render_assessment_results(
    student_id: str,
    result: Dict[str, Any],
    db_saved: bool,
    db_message: str,
) -> None:
    """Render the focused multi-model assessment panel in green dashboard style."""
    preds = result["predictions"]
    dt_at_risk = preds["decision_tree_at_risk"]
    rf_at_risk = preds["random_forest_at_risk"]
    rf_risk_prob = preds["random_forest_risk_probability"]
    estimated_g3 = preds["random_forest_estimated_g3"]

    dt_label = "At-Risk" if dt_at_risk == 1 else "Not At-Risk (Passing)"
    rf_label = "At-Risk" if rf_at_risk == 1 else "Not At-Risk (Passing)"

    dt_pill = "status-pill status-pill-danger" if dt_at_risk == 1 else "status-pill status-pill-green"
    rf_pill = "status-pill status-pill-danger" if rf_at_risk == 1 else "status-pill status-pill-green"

    st.markdown(clean_html("<hr/>"), unsafe_allow_html=True)
    st.markdown(clean_html("<h2>Assessment Results Summary</h2>"), unsafe_allow_html=True)

    # Risk Meter Progress Gauge
    gauge_card_html = """
    <div class="dashboard-card" style="margin-bottom: 1.25rem;">
        <div class="dashboard-card-header">
            <span class="dashboard-card-title">Ensemble Risk Level Gauge</span>
            <span class="status-pill status-pill-green">Continuous Risk Calibration</span>
        </div>
    """
    st.markdown(clean_html(gauge_card_html), unsafe_allow_html=True)
    render_risk_gauge(rf_risk_prob)
    st.markdown(clean_html("</div>"), unsafe_allow_html=True)

    # Result Tiles (4 Columns)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        tile1 = f"""
        <div class="stat-tile">
            <div class="stat-tile-top">
                <span class="stat-label">Decision Tree</span>
                <span class="stat-icon-badge">🎯</span>
            </div>
            <div style="margin: 0.6rem 0;"><span class="{dt_pill}">{dt_label}</span></div>
            <div class="stat-caption">Primary early-warning flag (62.7% recall)</div>
        </div>
        """
        st.markdown(clean_html(tile1), unsafe_allow_html=True)

    with c2:
        tile2 = f"""
        <div class="stat-tile">
            <div class="stat-tile-top">
                <span class="stat-label">Random Forest</span>
                <span class="stat-icon-badge">🌲</span>
            </div>
            <div style="margin: 0.6rem 0;"><span class="{rf_pill}">{rf_label}</span></div>
            <div class="stat-caption">Ensemble consensus classification</div>
        </div>
        """
        st.markdown(clean_html(tile2), unsafe_allow_html=True)

    with c3:
        tile3 = f"""
        <div class="stat-tile">
            <div class="stat-tile-top">
                <span class="stat-label">Risk Probability</span>
                <span class="stat-icon-badge">📊</span>
            </div>
            <div class="stat-value">{rf_risk_prob * 100:.1f}%</div>
            <div class="stat-caption">Estimated likelihood of final mark &lt; 10</div>
        </div>
        """
        st.markdown(clean_html(tile3), unsafe_allow_html=True)

    with c4:
        tile4 = f"""
        <div class="stat-tile">
            <div class="stat-tile-top">
                <span class="stat-label">Estimated Grade (G3)</span>
                <span class="stat-icon-badge">📝</span>
            </div>
            <div class="stat-value">{estimated_g3:.1f} <span style="font-size: 1rem; color: #71887C; font-weight: 500;">/ 20</span></div>
            <div class="stat-caption">Random Forest continuous regression</div>
        </div>
        """
        st.markdown(clean_html(tile4), unsafe_allow_html=True)

    # Check for model disagreement
    models_disagree = dt_at_risk != rf_at_risk
    reg_disagrees_dt = (estimated_g3 < 10.0 and dt_at_risk == 0) or (estimated_g3 >= 10.0 and dt_at_risk == 1)

    if models_disagree or reg_disagrees_dt:
        disagree_html = """
        <div class="dashboard-callout warning" style="margin-top: 1.25rem;">
            <div class="dashboard-callout-title">Model Disagreement Advisory</div>
            <div>
                The classification and regression models generated differing assessment indicators for this profile.
                This occurs because classification and continuous regression optimize different mathematical loss functions:
                <ul>
                    <li>The <strong>Decision Tree Classifier</strong> maximizes sensitivity (high recall) to avoid missing struggling students.</li>
                    <li>The <strong>Random Forest Regressor</strong> estimates conditional mean grade with an average margin of error (&plusmn;3.39 points).</li>
                </ul>
                These separate estimates should be evaluated together as a multi-perspective advisory signal.
            </div>
        </div>
        """
        st.markdown(clean_html(disagree_html), unsafe_allow_html=True)

    # MongoDB persistence status pill
    if db_saved:
        st.markdown(
            clean_html(
                f"<div style='font-size: 0.85rem; color: #15803D; margin-top: 0.75rem; font-weight: 600;'>"
                f"<strong>Database Status:</strong> Assessment record for student <code>{student_id}</code> saved to MongoDB "
                f"<code>student_predictions</code>.</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            clean_html(
                f"<div style='font-size: 0.85rem; color: #B91C1C; margin-top: 0.75rem; font-weight: 600;'>"
                f"<strong>Database Status:</strong> {db_message}</div>"
            ),
            unsafe_allow_html=True,
        )

    # Advisory limitation disclaimer
    st.markdown(
        clean_html(
            "<div style='margin-top: 0.85rem; font-size: 0.78rem; color: #71887C; line-height: 1.45;'>"
            "<strong>Advisory Status:</strong> All predictions are statistical estimates derived from pre-semester attributes "
            "and should be used as diagnostic guidance for academic advisors, not definitive or deterministic judgments of student capability."
            "</div>"
        ),
        unsafe_allow_html=True,
    )
