"""Custom CSS styling for the EduPredict Streamlit application.

Implements the 'warm editorial education analytics' design system:
- Warm ivory background (#F7F5EF)
- Soft white surfaces (#FFFDF9)
- Deep charcoal typography (#292D29) with system serif headings
- Forest green primary accents (#355B45)
- Muted terracotta secondary accents (#B96950)
- Warm gray borders (#E5E0D7)
- Zero blue theme, zero decorative emoji
"""

MAIN_CSS = """
<style>
/* ==========================================================================
   Typography & Global Variables
   ========================================================================== */
:root {
    --bg-main: #F7F5EF;
    --surface-white: #FFFDF9;
    --surface-subtle-green: #E8EEE7;
    --surface-warning: #F4E8D8;
    --surface-terracotta-subtle: #F7ECE8;
    --text-primary: #292D29;
    --text-secondary: #77776F;
    --accent-green: #355B45;
    --accent-green-hover: #294736;
    --accent-terracotta: #B96950;
    --status-success: #2E5A36;
    --status-error: #9E2A2B;
    --border-warm: #E5E0D7;
    --border-focus: #355B45;
}

html, body, [class*="st-"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: var(--text-primary);
    background-color: var(--bg-main);
}

/* Serif headings for editorial authority */
h1, h2, h3 {
    font-family: "Georgia", "Cambria", "Times New Roman", serif;
    color: var(--text-primary);
    font-weight: 600;
    letter-spacing: -0.01em;
}

h1 {
    font-size: 2.1rem;
    line-height: 1.25;
    margin-bottom: 0.35rem;
}

h2 {
    font-size: 1.45rem;
    line-height: 1.35;
    margin-top: 1.25rem;
    margin-bottom: 0.5rem;
}

h3 {
    font-size: 1.15rem;
    line-height: 1.4;
    margin-top: 1rem;
    margin-bottom: 0.35rem;
}

/* Tabular numerals for clean analytical figures */
.tabular-nums, .stMetric, table, td, th {
    font-variant-numeric: tabular-nums;
}

/* ==========================================================================
   Main Container & Spacing
   ========================================================================== */
.main .block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* ==========================================================================
   Sidebar Styling
   ========================================================================== */
section[data-testid="stSidebar"] {
    background-color: #FAF8F3;
    border-right: 1px solid var(--border-warm);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* Brand header in sidebar */
.brand-container {
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border-warm);
}

.brand-title {
    font-family: "Georgia", "Cambria", "Times New Roman", serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-top: 0.25rem;
    font-weight: 600;
}

.sidebar-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-warm);
    font-size: 0.75rem;
    line-height: 1.4;
    color: var(--text-secondary);
}

/* Sidebar navigation radios styled as editorial tabs */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.25rem;
}

section[data-testid="stSidebar"] .stRadio label {
    background-color: transparent;
    padding: 0.55rem 0.75rem;
    border-radius: 4px;
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--text-primary);
    cursor: pointer;
    transition: background-color 0.15s ease, color 0.15s ease;
    border: 1px solid transparent;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background-color: #F0ECE1;
}

section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background-color: var(--surface-subtle-green);
    color: var(--accent-green);
    font-weight: 600;
    border-color: #D3DFD2;
}

/* ==========================================================================
   Editorial Cards & Panels
   ========================================================================== */
.editorial-card {
    background-color: var(--surface-white);
    border: 1px solid var(--border-warm);
    border-radius: 6px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
}

.editorial-card-header {
    font-family: "Georgia", "Cambria", "Times New Roman", serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #F0EBE1;
}

.stat-tile {
    background-color: var(--surface-white);
    border: 1px solid var(--border-warm);
    border-radius: 6px;
    padding: 1.15rem 1.25rem;
    height: 100%;
}

.stat-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.stat-value {
    font-family: "Georgia", "Cambria", "Times New Roman", serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}

.stat-caption {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
    line-height: 1.3;
}

/* ==========================================================================
   Status Pills & Badges
   ========================================================================== */
.status-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.3;
    border: 1px solid transparent;
}

.status-pill-green {
    background-color: var(--surface-subtle-green);
    color: var(--status-success);
    border-color: #C7D7C5;
}

.status-pill-terracotta {
    background-color: var(--surface-terracotta-subtle);
    color: var(--status-error);
    border-color: #EAC9C0;
}

.status-pill-sand {
    background-color: var(--surface-warning);
    color: #8A5A2B;
    border-color: #E2D1BC;
}

.status-pill-neutral {
    background-color: #EFECE6;
    color: var(--text-secondary);
    border-color: var(--border-warm);
}

/* ==========================================================================
   Editorial Callouts & Limitation Notes
   ========================================================================== */
.editorial-callout {
    background-color: var(--surface-white);
    border-left: 3px solid var(--accent-green);
    border-top: 1px solid var(--border-warm);
    border-right: 1px solid var(--border-warm);
    border-bottom: 1px solid var(--border-warm);
    border-radius: 0 6px 6px 0;
    padding: 0.9rem 1.15rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    line-height: 1.45;
    color: var(--text-primary);
}

.editorial-callout.warning {
    border-left-color: var(--accent-terracotta);
    background-color: #FFFAF8;
}

.editorial-callout-title {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

/* ==========================================================================
   Streamlit Native Overrides: Buttons, Inputs, Sliders (NO BLUE THEME)
   ========================================================================== */
/* Primary buttons */
div.stButton > button[kind="primary"] {
    background-color: var(--accent-green) !important;
    color: #FFFDF9 !important;
    border: 1px solid var(--accent-green) !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: var(--accent-green-hover) !important;
    border-color: var(--accent-green-hover) !important;
}

/* Secondary / default buttons */
div.stButton > button {
    background-color: var(--surface-white) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-warm) !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

div.stButton > button:hover {
    background-color: #F2EFE8 !important;
    border-color: #D3CDC2 !important;
    color: var(--text-primary) !important;
}

/* Input fields & Selectboxes */
input, select, textarea, div[data-baseweb="select"] > div {
    background-color: var(--surface-white) !important;
    border-color: var(--border-warm) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}

input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 1px var(--accent-green) !important;
}

/* Sliders */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: var(--accent-green) !important;
    border-color: var(--accent-green) !important;
}

div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-secondary) !important;
}

/* Tab bar */
div[data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border-warm) !important;
    gap: 1.5rem !important;
}

div[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 0.65rem 0.2rem !important;
    border-bottom: 2px solid transparent !important;
}

div[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-green) !important;
    font-weight: 600 !important;
    border-bottom-color: var(--accent-green) !important;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-warm);
    border-radius: 6px;
    background-color: var(--surface-white);
}

/* Divider */
hr {
    border-color: var(--border-warm) !important;
    margin: 1.5rem 0 !important;
}
</style>
"""


def apply_editorial_styles() -> None:
    """Inject the custom CSS into the active Streamlit app."""
    import streamlit as st

    st.markdown(MAIN_CSS, unsafe_allow_html=True)
