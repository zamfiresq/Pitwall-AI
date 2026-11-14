"""Visual identity for the Streamlit dashboard.

Colours and layout follow Streamlit's own light/dark theme. Headings use
Formula 1 Bold; body text uses Formula 1 Regular. Both files live in
``assets/fonts`` and are embedded so the browser does not need them installed.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st


RED = "#E8002EC4"          # Ferrari F1 red
YELLOW = "#FFF200DB"       # Giallo Modena
PANEL = "#161618"
GRID = "rgba(128,128,128,0.28)"
MUTED = "#8A8A98"

COMPOUND_COLORS = {
    "SOFT": "#FF3B3B",
    "MEDIUM": "#F0C400",
    "HARD": "#C5C8D0",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067AD",
}
DRIVER_COLORS = {"LEC": RED, "HAM": "#E2B400"}
SEASON_COLORS = {"2025": "#8A8A98", "2026": RED}

FONT_FAMILY = "Formula1"
FONT_BODY = f"'{FONT_FAMILY}', sans-serif"
FONT_DISPLAY = FONT_BODY

_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _font_src(filename: str) -> str:
    path = _FONT_DIR / filename
    if not path.exists():
        return ""
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'url("data:font/ttf;base64,{payload}") format("truetype")'


def _font_face(weight: int, filename: str) -> str:
    src = _font_src(filename)
    if not src:
        return ""
    return f"""
@font-face {{
  font-family: "{FONT_FAMILY}";
  src: {src};
  font-weight: {weight};
  font-style: normal;
  font-display: swap;
}}
"""


def _css() -> str:
    faces = (
        _font_face(200, "Formula1-Regular_web_0.ttf")
        + _font_face(800, "Formula1-Bold_web_0.ttf")
    )
    return f"""
<style>
{faces}

html, body, .stApp,
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
[data-testid="stMarkdown"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"],
[data-baseweb="select"], [data-baseweb="popover"] {{
  font-family: {FONT_BODY};
}}

h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stMetricValue"],
.stTabs [data-baseweb="tab"],
.pw-title, .pw-driver, .pw-num {{
  font-family: {FONT_DISPLAY};
  font-weight: 700;
  font-style: normal;
}}

.block-container {{ padding-top: 0.8rem; padding-bottom: 1.6rem; max-width: 1280px; }}

/* Streamlit adds a chain-link control to headings; hide it in the dashboard. */
[data-testid="stHeaderLink"] {{ display: none !important; }}

h2, h3 {{
  position: relative;
  border-left: 0;
  padding-left: 0.72rem;
  font-size: 0.9rem;
  line-height: 1.25;
  margin-top: 1.25rem;
  margin-bottom: 0.7rem;
}}
h2::before, h3::before {{
  content: "";
  position: absolute;
  left: -0.67rem;
  top: 50%;
  transform: translateY(-50%);
  width: 4px;
  height: 1.15em;
  border-radius: 2px;
  background: {RED};
}}

.pw-hero {{
  display: flex; justify-content: space-between; align-items: flex-end; gap: 1.5rem; flex-wrap: wrap;
  padding: 0 0 1.1rem 0;
}}

.pw-title {{
  font-weight: 600; text-transform: uppercase;
  font-size: 2.65rem; line-height: 1.4; margin: 2.5rem 0 0.25rem 0; letter-spacing: 0.04em;
  color: inherit;
}}
.pw-title span {{ color: {RED}; margin-left: 0.5rem; }}
.pw-sub {{ color: light-dark(#5E5E6B, #9B9BA8); font-size: 1.02rem; font-weight: 400; }}
.pw-drivers {{ display: flex; gap: 0.7rem; flex-wrap: wrap; }}
.pw-driver {{
  display: flex; align-items: center; gap: 0.6rem;
  background: light-dark(rgba(0,0,0,0.04), rgba(255,255,255,0.06));
  border: 1px solid light-dark(rgba(0,0,0,0.12), rgba(255,255,255,0.14));
  border-bottom: 3px solid {RED};
  padding: 0.35rem 0.7rem; border-radius: 4px 12px 4px 4px;
  text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.78rem;
  color: inherit;
}}
.pw-driver.ham {{ border-bottom-color: light-dark(#C4A000, {YELLOW}); }}
.pw-num {{ font-size: 1.4rem; color: {RED}; }}
.pw-driver.ham .pw-num {{ color: light-dark(#A68400, {YELLOW}); }}

[data-testid="stMetric"] {{
  border-top: 2px solid {RED};
  padding-top: 0.25rem;
}}
[data-testid="stMetricLabel"] p {{
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.65rem; font-weight: 400;
}}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
[data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums;
}}

/* Give newer interactive sections the same restrained timing-screen treatment. */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid light-dark(rgba(0,0,0,0.10), rgba(255,255,255,0.12));
  border-radius: 5px;
  overflow: hidden;
}}
[data-testid="stChatMessage"] {{
  border: 1px solid light-dark(rgba(0,0,0,0.08), rgba(255,255,255,0.10));
  border-left: 3px solid {RED};
  border-radius: 4px 10px 4px 4px;
  background: light-dark(rgba(0,0,0,0.02), rgba(255,255,255,0.035));
}}
[data-testid="stChatInput"] textarea {{
  border-color: light-dark(rgba(0,0,0,0.18), rgba(255,255,255,0.20));
}}
[data-testid="stButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {{
  border-color: {RED};
  background: {RED};
  color: #FFFFFF;
  font-family: {FONT_DISPLAY};
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{
  border-radius: 4px 10px 4px 4px;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 0.2rem; }}
.stTabs [data-baseweb="tab"] {{
  text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.78rem;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {RED}; height: 3px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: inherit; }}

/* Keep dashboard text and controls compact without shrinking chart labels. */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"] {{ font-size: 0.88rem; }}

.pw-legend {{
  display: flex; gap: 0.35rem 0.7rem; flex-wrap: wrap;
  font-size: 0.72rem; font-weight: 400;
  margin-top: 0.15rem;
}}
.pw-dot {{
  display: inline-block; width: 0.65rem; height: 0.65rem; border-radius: 50%;
  margin-right: 0.35rem; vertical-align: -1px;
  border: 1px solid light-dark(rgba(0,0,0,0.35), rgba(255,255,255,0.45));
}}

.pw-note {{ color: light-dark(#5E5E6B, #9B9BA8); font-size: 0.82rem; font-weight: 400; }}
.pw-callout {{
  background: light-dark(rgba(0,0,0,0.03), rgba(255,255,255,0.05));
  border: 1px solid light-dark(rgba(0,0,0,0.08), rgba(255,255,255,0.12));
  border-left: 4px solid light-dark(#C4A000, {YELLOW});
  padding: 0.6rem 0.8rem; border-radius: 4px; margin: 0.45rem 0 0.7rem 0; font-size: 0.86rem;
  color: inherit;
}}
.pw-callout b {{ color: light-dark(#8A6A00, {YELLOW}); }}

/* Sidebar stays pinned to the top and has no internal scrollbar. */
section[data-testid="stSidebar"] {{ overflow: hidden !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  height: 100vh !important;
  overflow: auto;
   justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  # max-height: calc(100vh - 2rem) !important;
  overflow: hidden !important;
  padding-top: 0.35rem !important;
  margin-top: -2.75rem;
  padding-bottom: 2rem !important;
  scrollbar-width: none;
}}
section[data-testid="stSidebar"] .block-container {{
  padding-top: 0.35rem !important;
  padding-bottom: 0.35rem !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]::-webkit-scrollbar {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.82rem !important; }}
section[data-testid="stSidebar"] [data-testid="stHeading"] {{ margin-top: 0.45rem !important; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
  font-size: 0.92rem !important;
  line-height: 1.2 !important;
  margin-top: 0.85rem !important;
  margin-bottom: 0.72rem !important;
  padding-left: 0.8rem !important;
}}
section[data-testid="stSidebar"] h2::before,
section[data-testid="stSidebar"] h3::before {{
  background: {YELLOW};
  width: 3px;
  height: 1.18em;
}}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
  font-size: 0.76rem !important;
  line-height: 1.2 !important;
  margin-bottom: 0.12rem !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="select"] *,
section[data-testid="stSidebar"] input {{
  font-size: 0.78rem !important;
  line-height: 1.2 !important;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {{ min-height: 2.15rem !important; }}
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p {{ font-size: 0.78rem !important; }}
section[data-testid="stSidebar"] .pw-legend {{ font-size: 0.72rem; gap: 0.45rem 0.75rem; }}

@media (max-width: 900px) {{
  .pw-title {{ font-size: 2.35rem; }}
  .pw-driver {{ font-size: 0.7rem; }}
  .pw-num {{ font-size: 1.15rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_css(), unsafe_allow_html=True)


def hero() -> None:
    """Page header. Built as one line so Markdown does not treat indentation as code."""
    html = (
        '<div class="pw-hero"><div>'
        # '<div class="pw-kicker">Scuderia Ferrari &middot; Data analysis</div>'
        '<div class="pw-title">Pitwall<span>AI</span></div>'
        '<div class="pw-sub">Tyre degradation and race-pace analytics &middot; 2025&ndash;2026 seasons</div>'
        '</div><div class="pw-drivers">'
        '<div class="pw-driver"><span class="pw-num">16</span>Charles Leclerc</div>'
        '<div class="pw-driver ham"><span class="pw-num">44</span>Lewis Hamilton</div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def compound_legend() -> None:
    items = "".join(
        f'<span><span class="pw-dot" style="background:{color}"></span>{name.title()}</span>'
        for name, color in COMPOUND_COLORS.items() if name != "WET"
    )
    st.markdown(f'<div class="pw-legend">{items}</div>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<div class="pw-callout">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="pw-note">{text}</p>', unsafe_allow_html=True)


def _active_theme():
    context = getattr(st, "context", None)
    return getattr(context, "theme", None) if context is not None else None


def style_fig(fig, height: int | None = None):
    """Chart styling that follows Streamlit's light or dark theme."""
    ui = _active_theme()
    theme_type = getattr(ui, "type", None)
    text = getattr(ui, "textColor", None)
    hover_bg = getattr(ui, "secondaryBackgroundColor", None) or "rgba(0,0,0,0)"
    if theme_type == "dark":
        template = "plotly_dark"
    elif theme_type == "light":
        template = "plotly_white"
    else:
        template = "plotly"

    font = {"family": FONT_FAMILY, "size": 13}
    if text:
        font["color"] = text
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=font,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"family": FONT_FAMILY, "size": 12}},
        hoverlabel={
            "bgcolor": hover_bg,
            "bordercolor": RED,
            "font": {"family": FONT_FAMILY, "color": text or "#111111"},
        },
    )
    if height:
        fig.update_layout(height=height)
    axis = {"gridcolor": GRID, "zerolinecolor": GRID}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
