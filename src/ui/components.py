"""Reusable UI components for the EduPredict Streamlit application.

These components implement the executive Green Dashboard aesthetic:
- Modern typography with clear visual hierarchy
- Polished KPI metric cards with accent stripes and status chips
- Executive status pills and real-time operational indicators
- Robust dedenting to guarantee HTML is always rendered properly and never as raw code blocks
"""

from typing import Optional
import streamlit as st


def clean_html(html_str: str) -> str:
    """Strip all leading whitespace and empty lines from HTML.
    
    Guarantees Streamlit's CommonMark parser never treats HTML as an indented code block.
    """
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return "\n".join(lines)


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    status_text: Optional[str] = None,
    status_type: str = "green",
    category_label: Optional[str] = "ACADEMIC PERFORMANCE DASHBOARD",
) -> None:
    """Render a modern green dashboard page header with status indicator.

    Args:
        title: Main page heading.
        subtitle: Optional explanatory subtext.
        status_text: Optional small status badge text.
        status_type: 'green', 'sand', 'warning', 'terracotta', 'danger', or 'neutral'.
        category_label: Small uppercase context breadcrumb.
    """
    pill_class = f"status-pill status-pill-{status_type}"
    dot_html = "<span class='status-dot-active'></span>" if status_type == "green" else ""

    col_title, col_status = st.columns([0.76, 0.24])

    with col_title:
        cat_html = (
            f"<div style='font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; color: #15803D; margin-bottom: 0.2rem;'>"
            f"{category_label}"
            f"</div>"
            if category_label
            else ""
        )
        html_title = f"""
        <div style="margin-bottom: 0.5rem;">
            {cat_html}
            <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #0A3E26; letter-spacing: -0.025em;">{title}</h1>
        </div>
        """
        st.markdown(clean_html(html_title), unsafe_allow_html=True)
        if subtitle:
            html_sub = f"<div style='color: #4A6355; font-size: 0.92rem; margin-top: -0.2rem; margin-bottom: 1.25rem; line-height: 1.45;'>{subtitle}</div>"
            st.markdown(clean_html(html_sub), unsafe_allow_html=True)

    with col_status:
        if status_text:
            html_status = f"""
            <div style="text-align: right; padding-top: 0.75rem;">
                <span class="{pill_class}">{dot_html}{status_text}</span>
            </div>
            """
            st.markdown(clean_html(html_status), unsafe_allow_html=True)


def render_stat_tile(
    label: str,
    value: str,
    caption: str,
    delta_text: Optional[str] = None,
    delta_type: str = "green",
    icon: Optional[str] = None,
) -> None:
    """Render a clean, bordered KPI dashboard tile with tabular numerals.

    Args:
        label: Uppercase short metric descriptor.
        value: Primary figure or percentage.
        caption: Contextual subtitle or basis.
        delta_text: Optional trend or comparative chip text.
        delta_type: 'green', 'warning', 'danger', or 'neutral'.
        icon: Optional icon or symbol for top right badge.
    """
    icon_html = (
        f"<div class='stat-icon-badge'>{icon}</div>"
        if icon
        else "<div class='stat-icon-badge'>&bull;</div>"
    )

    delta_html = ""
    if delta_text:
        bg_color = "#DCFCE7" if delta_type == "green" else ("#FEE2E2" if delta_type == "danger" else "#FEF3C7")
        text_color = "#15803D" if delta_type == "green" else ("#B91C1C" if delta_type == "danger" else "#B45309")
        delta_html = (
            f"<span style='display: inline-block; background-color: {bg_color}; color: {text_color}; "
            f"padding: 0.12rem 0.45rem; border-radius: 4px; font-size: 0.73rem; font-weight: 700; "
            f"margin-right: 0.35rem;'>{delta_text}</span>"
        )

    html = f"""
    <div class="stat-tile">
        <div class="stat-tile-top">
            <span class="stat-label">{label}</span>
            {icon_html}
        </div>
        <div class="stat-value">{value}</div>
        <div class="stat-caption">{delta_html}{caption}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_callout(
    title: str,
    body: str,
    is_warning: bool = False,
    is_danger: bool = False,
) -> None:
    """Render a left-bordered green dashboard advisory box.

    Args:
        title: Short title of the notice.
        body: Explanatory text.
        is_warning: If True, uses amber/gold border; otherwise forest green.
        is_danger: If True, uses alert red border.
    """
    callout_class = "dashboard-callout"
    if is_danger:
        callout_class += " danger"
    elif is_warning:
        callout_class += " warning"

    html = f"""
    <div class="{callout_class}">
        <div class="dashboard-callout-title">{title}</div>
        <div style="font-size: 0.88rem; color: #2A4034; line-height: 1.5;">{body}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)


def render_card_start(title: Optional[str] = None, badge_text: Optional[str] = None) -> None:
    """Begin a dashboard card surface."""
    badge_html = f"<span class='status-pill status-pill-green'>{badge_text}</span>" if badge_text else ""
    header_html = (
        f"<div class='dashboard-card-header'>"
        f"<span class='dashboard-card-title'>{title}</span>"
        f"{badge_html}"
        f"</div>"
        if title
        else ""
    )
    st.markdown(clean_html(f"<div class='dashboard-card'>{header_html}"), unsafe_allow_html=True)


def render_card_end() -> None:
    """Close a dashboard card surface."""
    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_gauge(probability: float) -> None:
    """Render a clean dashboard progress gauge for academic risk probability.

    Args:
        probability: Float between 0.0 and 1.0.
    """
    pct = max(0.0, min(100.0, probability * 100))
    if pct < 35:
        color_class = "low"
        status_label = "Low Academic Risk"
        badge_style = "color: #15803D; font-weight: 700;"
    elif pct < 60:
        color_class = "medium"
        status_label = "Moderate Academic Risk"
        badge_style = "color: #B45309; font-weight: 700;"
    else:
        color_class = "high"
        status_label = "Elevated Academic Risk"
        badge_style = "color: #B91C1C; font-weight: 700;"

    html = f"""
    <div style="margin: 0.5rem 0 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 0.84rem;">
            <span style="{badge_style}">{status_label}</span>
            <span style="font-weight: 800; font-size: 1.15rem; color: #0A3E26;">{pct:.1f}%</span>
        </div>
        <div class="risk-meter-container">
            <div class="risk-meter-fill {color_class}" style="width: {pct:.1f}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #71887C; margin-top: 2px;">
            <span>0% (Secure Pass)</span>
            <span>50% (Threshold)</span>
            <span>100% (High Risk)</span>
        </div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)
