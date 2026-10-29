"""EduPredict — Student Performance Analytics Dashboard

Streamlit application entry point. Implements an executive Green Dashboard
interface for exploring student cohort analytics, evaluating scikit-learn
benchmarks, and executing multi-model academic risk predictions.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.styles import apply_editorial_styles
from src.ui.components import clean_html
from src.ui.pages_overview import render_overview_page
from src.ui.pages_predict import render_predict_page
from src.ui.pages_history import render_history_page
from src.ui.pages_performance import render_performance_page
from src.ui.pages_about import render_about_page

# Page Configuration
st.set_page_config(
    page_title="EduPredict — Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Custom Green Dashboard CSS
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
    sidebar_brand = """
    <div class="brand-container">
        <div class="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                <path d="M6 12v5c3 3 9 3 12 0v-5"/>
            </svg>
        </div>
        <div class="brand-text-wrap">
            <div class="brand-title">Edu<span>Predict</span></div>
            <div class="brand-subtitle">Student Performance Dashboard</div>
        </div>
    </div>
    """
    st.markdown(clean_html(sidebar_brand), unsafe_allow_html=True)

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

    sidebar_footer = """
    <div class="sidebar-footer-card">
        <div class="sidebar-status-indicator">
            <span class="status-dot-active"></span> Platform Online
        </div>
        <div style="font-weight: 700; color: #0A3E26; font-size: 0.8rem; margin-bottom: 2px;">EduPredict Core v2.4</div>
        <div style="color: #4A6355; font-size: 0.73rem; line-height: 1.4;">
            Cohort: 5,000 Synthetic Records<br/>
            Storage: MongoDB (Idempotent)<br/>
            Compliance: Zero Grade Leakage
        </div>
    </div>
    """
    st.markdown(clean_html(sidebar_footer), unsafe_allow_html=True)

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
