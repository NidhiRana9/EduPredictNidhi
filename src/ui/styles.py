"""Custom CSS styling for the EduPredict Streamlit application.

Implements an executive 'Green Dashboard' design system:
- Clean, crisp dashboard background (#F4F7F5) with pure white card surfaces (#FFFFFF)
- Modern sans-serif typography (Plus Jakarta Sans / Inter / system font)
- Professional emerald, forest, and mint green color hierarchy
- Real institutional analytics styling (KPI stat cards, status pills, visual gauges)
- Completely free of AI tropes (no purple/neon glowing gradients, no sci-fi robot motifs)
"""

MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ==========================================================================
   Global Variables & Color Palette (Green Dashboard)
   ========================================================================== */
:root {
    --bg-dashboard: #F4F7F5;
    --surface-card: #FFFFFF;
    --surface-card-subtle: #F8FAF9;
    --surface-sidebar: #FFFFFF;
    --surface-green-subtle: #F0FDF4;
    --surface-green-pill: #DCFCE7;
    --surface-green-border: #BBF7D0;
    
    --primary-green: #15803D;
    --primary-green-dark: #0A4D30;
    --primary-green-darker: #063722;
    --primary-green-hover: #166534;
    --primary-green-light: #22C55E;
    --emerald-accent: #059669;
    
    --text-primary: #12281D;
    --text-secondary: #4A6355;
    --text-muted: #71887C;
    --text-on-dark: #F0FDF4;
    
    --border-subtle: #E2ECE5;
    --border-medium: #CADCD0;
    --border-accent: #86EFAC;
    --border-focus: #15803D;
    
    --status-success: #15803D;
    --status-success-bg: #DCFCE7;
    --status-success-border: #86EFAC;
    
    --status-warning: #B45309;
    --status-warning-bg: #FEF3C7;
    --status-warning-border: #FDE68A;
    
    --status-danger: #B91C1C;
    --status-danger-bg: #FEE2E2;
    --status-danger-border: #FECACA;
    
    --status-neutral: #4A6355;
    --status-neutral-bg: #EAEFEA;
    --status-neutral-border: #D5DFD7;
    
    --card-shadow: 0 1px 3px 0 rgba(16, 50, 30, 0.05), 0 2px 8px -2px rgba(16, 50, 30, 0.04);
    --card-shadow-hover: 0 4px 14px -2px rgba(16, 50, 30, 0.08), 0 2px 6px -2px rgba(16, 50, 30, 0.04);
}

/* ==========================================================================
   Base Typography & Layout
   ========================================================================== */
html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: var(--text-primary);
    background-color: var(--bg-dashboard);
}

.main, section.main {
    background-color: var(--bg-dashboard) !important;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 1.75rem;
    padding-bottom: 4rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Crisp Modern Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: var(--primary-green-darker) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

h1 {
    font-size: 1.85rem !important;
    line-height: 1.25 !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    font-size: 1.35rem !important;
    line-height: 1.35 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 2px solid var(--border-subtle);
    padding-bottom: 0.4rem;
}

h3 {
    font-size: 1.05rem !important;
    line-height: 1.4 !important;
    margin-top: 1.15rem !important;
    margin-bottom: 0.5rem !important;
    color: var(--primary-green-dark) !important;
    font-weight: 600 !important;
}

/* Tabular Numerals for Analytical Values */
.tabular-nums, .stMetric, table, td, th {
    font-variant-numeric: tabular-nums;
}

/* ==========================================================================
   Modern Green Dashboard Sidebar
   ========================================================================== */
section[data-testid="stSidebar"] {
    background-color: var(--surface-sidebar) !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 2px 0 8px rgba(16, 50, 30, 0.02) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}

/* Brand Container */
.brand-container {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border-subtle);
}

.brand-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #15803D 0%, #0A4D30 100%);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 1.15rem;
    box-shadow: 0 2px 6px rgba(21, 128, 61, 0.28);
}

.brand-text-wrap {
    display: flex;
    flex-direction: column;
}

.brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    color: var(--primary-green-darker) !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.03em !important;
}

.brand-title span {
    color: var(--primary-green) !important;
}

.brand-subtitle {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    margin-top: 0.15rem !important;
    font-weight: 600 !important;
}

/* Navigation Radios as Dashboard Menu Items */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] .stRadio label {
    background-color: transparent !important;
    padding: 0.62rem 0.85rem !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    cursor: pointer !important;
    transition: all 0.18s ease-in-out !important;
    border: 1px solid transparent !important;
    display: flex !important;
    align-items: center !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: var(--surface-green-subtle) !important;
    color: var(--primary-green-dark) !important;
    border-color: var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background-color: var(--surface-green-pill) !important;
    color: var(--primary-green-dark) !important;
    font-weight: 700 !important;
    border-color: var(--surface-green-border) !important;
    border-left: 4px solid var(--primary-green) !important;
}

/* Sidebar Footer / System Badge */
.sidebar-footer-card {
    margin-top: 2rem;
    padding: 1rem;
    background-color: var(--surface-green-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: 9px;
    font-size: 0.76rem;
    line-height: 1.45;
    color: var(--text-secondary);
}

.sidebar-status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-weight: 700;
    color: var(--primary-green);
    margin-bottom: 0.35rem;
    font-size: 0.78rem;
}

.status-dot-active {
    width: 7px;
    height: 7px;
    background-color: var(--primary-green);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 0 2px rgba(21, 128, 61, 0.2);
}

/* ==========================================================================
   Dashboard Page Header
   ========================================================================== */
.dashboard-header-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
}

.dashboard-header-left {
    flex: 1;
}

.dashboard-header-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--primary-green-darker);
    letter-spacing: -0.025em;
    margin: 0;
    line-height: 1.2;
}

.dashboard-header-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
    margin-bottom: 0;
    line-height: 1.4;
}

.dashboard-header-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ==========================================================================
   Dashboard KPI Cards & Stat Tiles
   ========================================================================== */
.stat-tile {
    background-color: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.2rem 1.25rem;
    position: relative;
    box-shadow: var(--card-shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    height: 100%;
    overflow: hidden;
}

.stat-tile::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #15803D 0%, #10B981 100%);
}

.stat-tile:hover {
    transform: translateY(-2px);
    box-shadow: var(--card-shadow-hover);
    border-color: var(--border-medium);
}

.stat-tile-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    font-weight: 700;
}

.stat-icon-badge {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background-color: var(--surface-green-pill);
    color: var(--primary-green);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
}

.stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--primary-green-darker);
    line-height: 1.1;
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
    margin-bottom: 0.35rem;
}

.stat-caption {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.35;
}

.stat-delta-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    background-color: var(--surface-green-pill);
    color: var(--primary-green);
    margin-right: 0.35rem;
}

/* ==========================================================================
   Dashboard Surface Cards
   ========================================================================== */
.dashboard-card {
    background-color: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--card-shadow);
}

.dashboard-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--primary-green-darker);
    margin-bottom: 0.75rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid var(--border-subtle);
}

.dashboard-card-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ==========================================================================
   Status Pills & Badges
   ========================================================================== */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.25;
    letter-spacing: 0.01em;
}

.status-pill-green {
    background-color: var(--status-success-bg);
    color: var(--status-success);
    border: 1px solid var(--status-success-border);
}

.status-pill-sand, .status-pill-warning {
    background-color: var(--status-warning-bg);
    color: var(--status-warning);
    border: 1px solid var(--status-warning-border);
}

.status-pill-terracotta, .status-pill-danger {
    background-color: var(--status-danger-bg);
    color: var(--status-danger);
    border: 1px solid var(--status-danger-border);
}

.status-pill-neutral {
    background-color: var(--status-neutral-bg);
    color: var(--status-neutral);
    border: 1px solid var(--status-neutral-border);
}

/* ==========================================================================
   Callout Notices (Clean Institutional Style)
   ========================================================================== */
.editorial-callout, .dashboard-callout {
    background-color: var(--surface-card);
    border-left: 4px solid var(--primary-green);
    border-top: 1px solid var(--border-subtle);
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    border-radius: 0 8px 8px 0;
    padding: 0.95rem 1.25rem;
    margin: 1.15rem 0;
    font-size: 0.88rem;
    line-height: 1.5;
    color: var(--text-primary);
    box-shadow: var(--card-shadow);
}

.editorial-callout.warning, .dashboard-callout.warning {
    border-left-color: var(--status-warning);
    background-color: #FFFDF8;
}

.editorial-callout.danger, .dashboard-callout.danger {
    border-left-color: var(--status-danger);
    background-color: #FFFBFB;
}

.editorial-callout-title, .dashboard-callout-title {
    font-weight: 700;
    color: var(--primary-green-darker);
    margin-bottom: 0.25rem;
    font-size: 0.92rem;
}

.editorial-callout.warning .editorial-callout-title {
    color: var(--status-warning);
}

/* ==========================================================================
   Visual Progress & Risk Gauge Bars
   ========================================================================== */
.risk-meter-container {
    width: 100%;
    background-color: #E5EBE7;
    height: 10px;
    border-radius: 6px;
    overflow: hidden;
    margin: 0.5rem 0 0.25rem 0;
    position: relative;
}

.risk-meter-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

.risk-meter-fill.low {
    background-color: var(--primary-green);
}

.risk-meter-fill.medium {
    background-color: var(--status-warning);
}

.risk-meter-fill.high {
    background-color: var(--status-danger);
}

/* ==========================================================================
   Streamlit Native Overrides: Buttons, Inputs, Sliders (Green Dashboard)
   ========================================================================== */
/* Primary Buttons */
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(180deg, #16A34A 0%, #15803D 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #15803D !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 6px rgba(21, 128, 61, 0.25) !important;
    transition: all 0.16s ease-in-out !important;
}

div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(180deg, #15803D 0%, #0F5132 100%) !important;
    border-color: #0F5132 !important;
    box-shadow: 0 4px 12px rgba(21, 128, 61, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Secondary / Default Buttons */
div.stButton > button,
button[data-testid="baseButton-secondary"] {
    background-color: var(--surface-card) !important;
    color: var(--primary-green-dark) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.15rem !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.16s ease-in-out !important;
}

div.stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover {
    background-color: var(--surface-green-subtle) !important;
    border-color: var(--primary-green) !important;
    color: var(--primary-green) !important;
}

/* Form Inputs & Selectboxes */
input, select, textarea, div[data-baseweb="select"] > div {
    background-color: var(--surface-card) !important;
    border-color: var(--border-medium) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}

input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--primary-green) !important;
    box-shadow: 0 0 0 2px rgba(21, 128, 61, 0.15) !important;
}

/* Sliders */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: var(--primary-green) !important;
    border-color: var(--primary-green) !important;
    box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.2) !important;
}

div[data-testid="stSlider"] div[data-testid="stTickBar"] {
    color: var(--text-muted) !important;
}

/* Tab Bar */
div[data-baseweb="tab-list"] {
    border-bottom: 2px solid var(--border-subtle) !important;
    gap: 1.25rem !important;
    background-color: transparent !important;
}

div[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    padding: 0.75rem 0.4rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary-green) !important;
    font-weight: 700 !important;
    border-bottom: 3px solid var(--primary-green) !important;
}

/* Dataframe & Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 9px;
    background-color: var(--surface-card);
    box-shadow: var(--card-shadow);
}

/* Divider */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1.75rem 0 !important;
}

/* Clean Dashboard Tables */
.dashboard-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    margin-top: 0.5rem;
}

.dashboard-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    background-color: var(--surface-card-subtle);
    border-bottom: 1px solid var(--border-subtle);
    font-weight: 700;
}

.dashboard-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}

.dashboard-table tr:hover td {
    background-color: var(--surface-green-subtle);
}

/* ==========================================================================
   Hide Streamlit Deploy Button & Header Overrides
   ========================================================================== */
.stAppDeployButton,
[data-testid="stAppDeployButton"],
[data-testid="stDeployButton"],
button[title="Deploy this app"],
header [data-testid="stToolbar"] .stAppDeployButton,
header [data-testid="stToolbar"] [data-testid="stAppDeployButton"],
#MainMenu [data-testid="stAppDeployButton"],
div[data-testid="stToolbarActions"] .stAppDeployButton {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
"""


def apply_editorial_styles() -> None:
    """Inject the custom CSS into the active Streamlit app."""
    import streamlit as st

    st.markdown(MAIN_CSS, unsafe_allow_html=True)
