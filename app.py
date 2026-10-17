"""EduPredict — Student Performance Analytics

Streamlit application entry point. Implements an editorial, warm education
analytics interface for exploring student performance distributions, evaluating
machine learning benchmarks, and executing multi-model academic risk predictions.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.styles import apply_editorial_styles
from src.ui.pages_overview import render_overview_page
from src.ui.pages_predict import render_predict_page
from src.ui.pages_history import render_history_page
from src.ui.pages_performance import render_performance_page
from src.ui.pages_about import render_about_page

# Page Configuration
st.set_page_config(
    page_title="EduPredict — Student Performance Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Editorial CSS (Warm Ivory, Soft White, Forest Green, Terracotta)
apply_editorial_styles()

# Navigation Pages
PAGES = [
    "Overview",
    "Predict a Student",
    "Prediction History",
    "Model Performance",
    "About the Project",
]

# Initialize or manage session state navigation
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Overview"

# SIDEBAR SHELL
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <h1 class="brand-title">EduPredict</h1>
            <div class="brand-subtitle">Student Performance Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Radio button navigation
    selected_page = st.radio(
        label="Application Navigation",
        options=PAGES,
        index=PAGES.index(st.session_state["nav_page"])
        if st.session_state["nav_page"] in PAGES
        else 0,
        label_visibility="collapsed",
    )

    # Synchronize session state
    if selected_page != st.session_state["nav_page"]:
        st.session_state["nav_page"] = selected_page

    st.markdown(
        """
        <div class="sidebar-footer">
            <strong>EduPredict System</strong><br/>
            Academic Machine Learning Project<br/>
            Synthetic Dataset Benchmark<br/>
            Python &middot; scikit-learn &middot; MongoDB
        </div>
        """,
        unsafe_allow_html=True,
    )

# ROUTE TO SELECTED PAGE VIEW
current_page = st.session_state["nav_page"]

if current_page == "Overview":
    render_overview_page()
elif current_page == "Predict a Student":
    render_predict_page()
elif current_page == "Prediction History":
    render_history_page()
elif current_page == "Model Performance":
    render_performance_page()
elif current_page == "About the Project":
    render_about_page()
else:
    render_overview_page()
