"""Prediction History View for EduPredict.

Retrieves and presents real historical student assessments from MongoDB
collection 'student_predictions', with search, category filtering, sorting,
and document inspection.
"""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from src.database import (
    COLLECTION_STUDENT_PREDICTIONS,
    get_database,
    get_mongo_client,
    ping_database,
)
from src.ui.components import render_callout, render_header, render_stat_tile


def fetch_prediction_history() -> List[Dict[str, Any]]:
    """Retrieve all prediction documents from MongoDB 'student_predictions'."""
    client = None
    try:
        client = get_mongo_client(timeout_ms=2000)
        ping_database(client)
        db = get_database(client)
        col = db[COLLECTION_STUDENT_PREDICTIONS]
        records = list(col.find({}).sort("created_at", -1))
        return records
    except Exception as exc:
        st.error(f"Unable to retrieve prediction history from MongoDB: {exc}")
        return []
    finally:
        if client is not None:
            client.close()


def render_history_page() -> None:
    """Render the Prediction History table and filter controls."""
    render_header(
        title="Prediction History",
        subtitle="Review, filter, and inspect previously saved student risk assessments stored in MongoDB.",
    )

    records = fetch_prediction_history()

    # Metric tile + Refresh row
    col_stat1, col_stat2, col_stat3, col_ref = st.columns([0.28, 0.28, 0.28, 0.16])

    total_count = len(records)
    at_risk_count = sum(
        1 for r in records if r.get("predictions", {}).get("decision_tree_at_risk") == 1
    )
    passing_count = total_count - at_risk_count

    with col_stat1:
        render_stat_tile("Total Stored", f"{total_count:,}", "Documents in student_predictions")
    with col_stat2:
        render_stat_tile("Flagged At-Risk", f"{at_risk_count:,}", "By Decision Tree Classifier")
    with col_stat3:
        render_stat_tile("Passing / Not At-Risk", f"{passing_count:,}", "By Decision Tree Classifier")
    with col_ref:
        st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()

    if total_count == 0:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="editorial-card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-family: Georgia, serif; font-size: 1.25rem; color: #292D29; margin-bottom: 0.5rem;">
                    No Historical Predictions in MongoDB
                </div>
                <div style="font-size: 0.92rem; color: #77776F; max-width: 520px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                    There are currently no saved prediction documents in the <code>student_predictions</code> collection.
                    Assess a student via the form or run the Phase 6 batch prediction workflow to record inferences.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_act1, col_act2, col_act3 = st.columns([0.35, 0.3, 0.35])
        with col_act2:
            if st.button("Assess a Student Now →", type="primary", use_container_width=True):
                st.session_state["nav_page"] = "Predict a Student"
                st.rerun()
        return

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Search & Filter Controls
    col_search, col_filter, col_sort = st.columns([0.45, 0.3, 0.25])

    with col_search:
        search_query = st.text_input(
            "Search by Student ID",
            placeholder="Type student identifier...",
            help="Case-insensitive filter across student_id strings.",
        ).strip().lower()

    with col_filter:
        risk_filter = st.selectbox(
            "Filter by Decision Tree Risk",
            options=["All Records", "At-Risk Only", "Passing Only"],
        )

    with col_sort:
        sort_order = st.selectbox(
            "Sort Order",
            options=["Newest First", "Oldest First"],
        )

    # Apply Filters
    filtered_records = records

    if search_query:
        filtered_records = [
            r for r in filtered_records if search_query in str(r.get("student_id", "")).lower()
        ]

    if risk_filter == "At-Risk Only":
        filtered_records = [
            r
            for r in filtered_records
            if r.get("predictions", {}).get("decision_tree_at_risk") == 1
        ]
    elif risk_filter == "Passing Only":
        filtered_records = [
            r
            for r in filtered_records
            if r.get("predictions", {}).get("decision_tree_at_risk") == 0
        ]

    if sort_order == "Oldest First":
        filtered_records = sorted(
            filtered_records, key=lambda x: str(x.get("created_at", ""))
        )
    else:
        filtered_records = sorted(
            filtered_records, key=lambda x: str(x.get("created_at", "")), reverse=True
        )

    st.markdown(
        f"<div style='font-size: 0.85rem; color: #77776F; margin-top: 0.5rem; margin-bottom: 0.75rem;'>"
        f"Showing <strong>{len(filtered_records)}</strong> of {total_count} saved prediction records."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Format table dataframe
    table_rows = []
    for r in filtered_records:
        preds = r.get("predictions", {})
        created_at = r.get("created_at")
        time_str = (
            created_at.strftime("%Y-%m-%d %H:%M UTC")
            if hasattr(created_at, "strftime")
            else str(created_at)[:16]
        )

        dt_flag = "At-Risk" if preds.get("decision_tree_at_risk") == 1 else "Passing"
        rf_flag = "At-Risk" if preds.get("random_forest_at_risk") == 1 else "Passing"
        prob_val = preds.get("random_forest_risk_probability", 0.0)
        g3_val = preds.get("random_forest_estimated_g3", 0.0)

        table_rows.append(
            {
                "Student ID": str(r.get("student_id", "Unknown")),
                "Timestamp": time_str,
                "Decision Tree": dt_flag,
                "Random Forest": rf_flag,
                "Risk Probability": f"{prob_val * 100:.1f}%",
                "Estimated Final Grade": f"{g3_val:.2f} / 20",
                "Model Version": str(r.get("model_version", "v1")),
            }
        )

    if table_rows:
        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction records match the active search or filter criteria.")

    # Record Inspection Details Section
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    with st.expander("Inspect Raw Prediction Document from MongoDB"):
        student_id_list = [str(r.get("student_id", "Unknown")) for r in filtered_records]
        if student_id_list:
            selected_id = st.selectbox("Select Student Record to Inspect", options=student_id_list)
            target_record = next(
                (r for r in filtered_records if str(r.get("student_id", "")) == selected_id),
                None,
            )
            if target_record:
                # Format MongoDB document for inspection
                clean_doc = dict(target_record)
                if "_id" in clean_doc:
                    clean_doc["_id"] = str(clean_doc["_id"])
                if "created_at" in clean_doc and hasattr(clean_doc["created_at"], "isoformat"):
                    clean_doc["created_at"] = clean_doc["created_at"].isoformat()

                st.json(clean_doc)
