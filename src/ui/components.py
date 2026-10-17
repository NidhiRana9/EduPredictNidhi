"""Reusable UI components for the EduPredict Streamlit application.

These components implement the warm editorial aesthetic:
- Clear typography hierarchy with serif headings
- Purposeful whitespace and subtle borders
- Zero decorative emojis
- Accessible contrast and tabular numerals
"""

from typing import Optional
import streamlit as st


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    status_text: Optional[str] = None,
    status_type: str = "green",
) -> None:
    """Render a clean, restrained editorial page header.

    Args:
        title: Main page heading (serif).
        subtitle: Optional explanatory subtext.
        status_text: Optional small status badge text.
        status_type: 'green', 'sand', 'terracotta', or 'neutral'.
    """
    col1, col2 = st.columns([0.78, 0.22])

    with col1:
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(
                f"<p style='color: #77776F; font-size: 0.98rem; margin-top: -0.2rem; margin-bottom: 1.25rem; line-height: 1.45;'>{subtitle}</p>",
                unsafe_allow_html=True,
            )

    with col2:
        if status_text:
            pill_class = f"status-pill status-pill-{status_type}"
            st.markdown(
                f"<div style='text-align: right; margin-top: 0.6rem;'><span class='{pill_class}'>{status_text}</span></div>",
                unsafe_allow_html=True,
            )


def render_stat_tile(label: str, value: str, caption: str) -> None:
    """Render a clean, bordered metric tile with tabular numerals.

    Args:
        label: Uppercase short metric descriptor.
        value: Primary figure or percentage.
        caption: Contextual subtitle or basis.
    """
    html = f"""
    <div class="stat-tile">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-caption">{caption}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_callout(title: str, body: str, is_warning: bool = False) -> None:
    """Render a left-bordered editorial notice box.

    Args:
        title: Short title of the notice.
        body: Explanatory text.
        is_warning: If True, uses terracotta border/tint; otherwise forest green.
    """
    warning_class = " warning" if is_warning else ""
    html = f"""
    <div class="editorial-callout{warning_class}">
        <div class="editorial-callout-title">{title}</div>
        <div style="font-size: 0.88rem; color: #3E423E; line-height: 1.45;">{body}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_card_start(title: Optional[str] = None) -> None:
    """Begin an editorial card surface."""
    header_html = f"<div class='editorial-card-header'>{title}</div>" if title else ""
    st.markdown(f"<div class='editorial-card'>{header_html}", unsafe_allow_html=True)


def render_card_end() -> None:
    """Close an editorial card surface."""
    st.markdown("</div>", unsafe_allow_html=True)
