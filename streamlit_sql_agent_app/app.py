from __future__ import annotations

import os
import re
import sys
import tomllib
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for import_path in (APP_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from zain_jordan_sql_agent import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    connect_sqlite,
    create_chat_model,
    create_zain_sql_agent,
    list_tables,
    run_sql_agent,
)


APP_DB_PATH = APP_DIR / "zain_customer_360_ai_demo.db"
AGENT_PROMPT_VERSION = "customer-360-business-format-v2"

APP_PAGES = [
    ("AI Chat", "🗨︎", "Ask business questions in natural language."),
    ("Executive Analytics", "▥", "Dynamic dashboards and adjustable KPIs."),
    ("Evidence Search", "⌕", "Retrieve evidence from the database and summarize it."),
    ("Customer 360", "◎", "Lookup a customer and inspect their history."),
    ("Prompt Library", "☰", "Reusable telecom analysis prompts."),
    ("Help", "?", "Usage guide and examples."),
]

EXAMPLE_PROMPTS_BY_CATEGORY = {
    "Customer & Churn": [
        "Which cities have the most high-risk churn customers? Show the top 10 and explain the business risk.",
        "Compare churn risk by customer segment and recommend the best retention action.",
        "What are the main churn reasons among high-value customers?",
    ],
    "Revenue & Billing": [
        "Which customer value segments have the highest ARPU and six-month revenue?",
        "Show billing status distribution and total amount by status.",
        "Which customers or segments should collections prioritize based on overdue invoices?",
    ],
    "Campaigns": [
        "Which campaigns had the highest conversion rate and revenue generated?",
        "Compare campaign conversion by channel and target segment.",
        "Which campaign audience should we retarget next and why?",
    ],
    "Support & Experience": [
        "What are the most common complaint categories by severity?",
        "Which support channels have the highest negative sentiment?",
        "Find customer experience issues that could explain churn risk.",
    ],
    "Network": [
        "Which network event types affected the most customers?",
        "Show network tower capacity by city and technology.",
        "Find network-related patterns that could increase complaints or churn.",
    ],
}

FLAT_EXAMPLE_PROMPTS = [prompt for prompts in EXAMPLE_PROMPTS_BY_CATEGORY.values() for prompt in prompts]


st.set_page_config(
    page_title="Zain Customer 360 AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Theme and layout
# -----------------------------

def chart_palette() -> dict[str, str]:
    return {
        "accent": "#674c99",
        "accent2": "#2563eb",
        "accent3": "#8b5cf6",
        "danger": "#dc2626",
        "warning": "#d97706",
        "success": "#047857",
        "text": "#0f172a",
        "muted": "#64748b",
    }


def build_css() -> str:
    return f"""
<style>
:root {{
    color-scheme: light dark;
    --page: #f5f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --surface-3: #eef6f2;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #dde5ef;
    --soft-line: #edf2f7;
    --accent: #674c99;
    --accent-dark: #513b7d;
    --accent-soft: #f3effa;
    --danger: #dc2626;
    --warning: #d97706;
    --info: #2563eb;
    --purple: #7c3aed;
    --shadow: 0 22px 70px rgba(15,23,42,0.10);
    --shadow-sm: 0 12px 34px rgba(15,23,42,0.07);
    --sidebar: #0b1220;
    --sidebar-2: #111827;
    --input-bg: #ffffff;
    --chart-bg: #ffffff;
    --drawer-width: 348px;
    --rail-width: 74px;
    --toggle-left: calc(var(--drawer-width) - 1.15rem);
    --radius-xl: 28px;
    --radius-lg: 20px;
    --radius-md: 14px;
}}

@media (prefers-color-scheme: dark) {{
    :root {{
        --page: #070b13;
        --surface: #0f172a;
        --surface-2: #111827;
        --surface-3: #182235;
        --ink: #f8fafc;
        --muted: #a8b3c7;
        --line: rgba(255,255,255,0.10);
        --soft-line: rgba(255,255,255,0.07);
        --accent: #9b7bd6;
        --accent-dark: #c4b5fd;
        --accent-soft: rgba(103,76,153,0.22);
        --danger: #fb7185;
        --warning: #fbbf24;
        --info: #60a5fa;
        --purple: #a78bfa;
        --shadow: 0 22px 70px rgba(0,0,0,0.34);
        --shadow-sm: 0 12px 36px rgba(0,0,0,0.28);
        --input-bg: #0b1220;
        --chart-bg: #0f172a;
    }}
}}

#MainMenu,
footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
    background: radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 10%, transparent), transparent 32rem), var(--page) !important;
    color: var(--ink) !important;
}}

[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.main .block-container {{
    max-width: 1240px;
    padding-top: 0.8rem !important;
    padding-right: 1.8rem !important;
    padding-bottom: 2.4rem !important;
    padding-left: 1.8rem !important;
    transition: max-width 260ms ease, padding 260ms ease;
}}

section[data-testid="stSidebar"] {{
    width: var(--drawer-width) !important;
    min-width: var(--drawer-width) !important;
    background:
        radial-gradient(circle at 24px 12px, rgba(103,76,153,0.34), transparent 16rem),
        linear-gradient(180deg, var(--sidebar) 0%, var(--sidebar-2) 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 20px 0 70px rgba(0,0,0,0.28);
    transition: width 260ms ease, min-width 260ms ease, transform 260ms ease, margin-left 260ms ease, opacity 220ms ease;
    z-index: 999;
}}

section[data-testid="stSidebar"] > div {{
    padding: 0 0.95rem 1.35rem;
}}

.drawer-toggle {{
    position: fixed;
    top: 4.5rem;
    left: var(--toggle-left);
    z-index: 1200;
    width: 2.45rem;
    height: 2.45rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--accent) 38%, transparent);
    background: var(--surface);
    color: var(--accent-dark) !important;
    box-shadow: var(--shadow-sm);
    text-decoration: none !important;
    font-size: 1.35rem;
    font-weight: 900;
    line-height: 1;
    transition: left 260ms ease, transform 180ms ease, background 180ms ease, box-shadow 180ms ease;
}}

.drawer-toggle.open {{
    display: none;
}}

.drawer-toggle:hover {{
    transform: translateY(-1px) scale(1.03);
    background: var(--accent-soft);
}}

body:has(#drawer-closed:target) {{
    --toggle-left: calc(var(--rail-width) - 1.15rem);
}}

body:has(#drawer-closed:target) section[data-testid="stSidebar"] {{
    width: var(--rail-width) !important;
    min-width: var(--rail-width) !important;
    transform: translateX(0);
    margin-left: 0;
    opacity: 1;
    box-shadow: 10px 0 34px rgba(0,0,0,0.18);
}}

body:has(#drawer-closed:target) section[data-testid="stSidebar"] > div {{
    padding: 0 0.42rem 1rem;
}}

body:has(#drawer-closed:target) [data-testid="stMain"] .block-container,
body:has(#drawer-closed:target) [data-testid="stMainBlockContainer"],
body:has(#drawer-closed:target) [data-testid="stAppViewBlockContainer"],
body:has(#drawer-closed:target) .main .block-container {{
    max-width: 1320px;
}}

body:has(#drawer-closed:target) .drawer-toggle.close {{
    display: none;
}}

body:has(#drawer-closed:target) .drawer-toggle.open {{
    display: grid;
}}

body:has(#drawer-closed:target) .drawer-brand {{
    justify-content: center;
    padding: 0 0 1rem;
}}

body:has(#drawer-closed:target) .drawer-brand > div:not(.drawer-logo),
body:has(#drawer-closed:target) [data-testid="stSidebar"] .drawer-section-title,
body:has(#drawer-closed:target) [data-testid="stSidebar"] .drawer-note {{
    display: none !important;
}}

body:has(#drawer-closed:target) [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.drawer-section),
body:has(#drawer-closed:target) [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.drawer-section) ~ div {{
    display: none !important;
}}

body:has(#drawer-closed:target) .drawer-logo {{
    width: 2.55rem;
    height: 2.55rem;
    border-radius: 0.9rem;
}}

body:has(#drawer-closed:target) .drawer-logo::before {{
    width: 1.62rem;
    height: 1.62rem;
}}

body:has(#drawer-closed:target) [data-testid="stSidebar"] div.stButton > button {{
    width: 3rem !important;
    height: 3rem !important;
    min-height: 3rem !important;
    padding: 0 !important;
    border-radius: 14px !important;
    justify-content: center !important;
    text-align: center !important;
    margin: 0.28rem auto !important;
}}

body:has(#drawer-closed:target) [data-testid="stSidebar"] div.stButton > button p {{
    font-size: 0 !important;
    line-height: 1 !important;
    text-align: center !important;
}}

body:has(#drawer-closed:target) [data-testid="stSidebar"] div.stButton > button p::first-letter {{
    color: #f8fafc !important;
    font-size: 1.35rem !important;
    font-weight: 900 !important;
}}

body:has(#drawer-closed:target) [data-testid="stSidebar"] div.stButton > button[aria-label="Start a new chat"],
body:has(#drawer-closed:target) [data-testid="stSidebar"] div.stButton > button[aria-label="Open Evidence Search"] {{
    display: none !important;
}}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: #f8fafc !important;
}}

[data-testid="stSidebar"] label p {{
    color: rgba(248,250,252,0.76) !important;
    font-size: 0.75rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
}}

[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="textarea"] {{
    background: rgba(255,255,255,0.96) !important;
    border-color: rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    min-height: 2.35rem !important;
}}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span {{
    color: #111827 !important;
}}

[data-testid="stSidebar"] svg {{
    color: #111827 !important;
    fill: #111827 !important;
}}

.drawer-brand {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0 0.25rem 0.9rem;
}}

.drawer-logo {{
    width: 2.7rem;
    height: 2.7rem;
    border-radius: 1rem;
    display: grid;
    place-items: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 14px 34px rgba(103,76,153,0.30);
}}

.drawer-logo::before {{
    content: "";
    width: 1.72rem;
    height: 1.72rem;
    display: block;
    background: linear-gradient(135deg, #f8fafc, #aeb4bb);
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-12 -12 144 144'%3E%3Cpath d='M98 91C78 111 42 113 19 93C-4 73-4 35 17 17C39-2 77-5 99 16C120 36 121 72 101 91C83 108 50 108 30 91C10 75 10 46 28 30C46 14 76 12 94 29C112 46 112 73 96 88C81 102 56 102 41 89C26 76 27 54 41 42C55 30 77 29 90 42C103 55 102 74 90 86C79 96 61 96 50 87C40 78 40 62 50 53C60 44 76 44 84 54C91 63 88 77 78 85' fill='none' stroke='black' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='-12 -12 144 144'%3E%3Cpath d='M98 91C78 111 42 113 19 93C-4 73-4 35 17 17C39-2 77-5 99 16C120 36 121 72 101 91C83 108 50 108 30 91C10 75 10 46 28 30C46 14 76 12 94 29C112 46 112 73 96 88C81 102 56 102 41 89C26 76 27 54 41 42C55 30 77 29 90 42C103 55 102 74 90 86C79 96 61 96 50 87C40 78 40 62 50 53C60 44 76 44 84 54C91 63 88 77 78 85' fill='none' stroke='black' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") center / contain no-repeat;
}}

.drawer-brand h2 {{
    margin: 0;
    font-size: 1.05rem;
    line-height: 1.1;
}}

.drawer-brand p {{
    margin: 0.18rem 0 0;
    color: rgba(248,250,252,0.66) !important;
    font-size: 0.76rem;
    line-height: 1.32;
}}

.drawer-section {{
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid rgba(255,255,255,0.10);
}}

.drawer-section-title {{
    color: rgba(248,250,252,0.62);
    font-size: 0.68rem;
    font-weight: 900;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin: 0 0 0.55rem;
}}

.drawer-note {{
    color: rgba(248,250,252,0.68) !important;
    font-size: 0.78rem;
    line-height: 1.45;
    margin: 0.25rem 0 0.65rem;
}}

.nav-button-row {{
    display: grid;
    gap: 0.25rem;
}}

[data-testid="stSidebar"] div.stButton > button {{
    background: transparent;
    color: #f8fafc !important;
    border: 1px solid transparent;
    min-height: 2.35rem;
    height: auto;
    padding: 0.48rem 0.62rem;
    border-radius: 13px;
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Symbol", "Noto Sans Symbols 2", "Segoe UI", sans-serif;
    font-weight: 760;
    font-size: 0.84rem;
}}

[data-testid="stSidebar"] div.stButton > button *,
[data-testid="stSidebar"] div.stButton > button p {{
    color: #f8fafc !important;
    opacity: 1 !important;
    text-align: left !important;
}}

[data-testid="stSidebar"] div.stButton > button:hover {{
    border-color: rgba(103,76,153,0.42);
    background: rgba(255,255,255,0.08);
}}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{
    background: rgba(103,76,153,0.30) !important;
    border-color: rgba(196,181,253,0.48) !important;
    box-shadow: inset 4px 0 0 #c4b5fd !important;
}}

.sidebar-metrics {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.45rem;
}}

.sidebar-metric {{
    border: 1px solid rgba(255,255,255,0.11);
    background: rgba(255,255,255,0.055);
    border-radius: 14px;
    padding: 0.62rem 0.55rem;
    min-width: 0;
}}

.sidebar-metric-label {{
    color: rgba(248,250,252,0.58);
    font-size: 0.64rem;
    font-weight: 800;
}}

.sidebar-metric-value {{
    color: #f8fafc;
    font-size: 0.94rem;
    font-weight: 900;
    margin-top: 0.16rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.app-hero {{
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 1.1rem 1.25rem;
    margin: 0 0 1rem;
    background:
        radial-gradient(circle at 92% 6%, color-mix(in srgb, var(--accent) 36%, transparent) 0%, transparent 17rem),
        radial-gradient(circle at 8% 100%, color-mix(in srgb, var(--accent-dark) 16%, transparent) 0%, transparent 16rem),
        linear-gradient(135deg, color-mix(in srgb, var(--surface) 92%, var(--accent) 8%) 0%, var(--surface-2) 58%, color-mix(in srgb, var(--surface-2) 86%, var(--accent) 14%) 100%);
    box-shadow: var(--shadow-sm);
}}

.app-hero-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}}

.kicker {{
    color: var(--accent-dark);
    font-size: 0.72rem;
    font-weight: 950;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin: 0 0 0.3rem;
}}

.hero-title {{
    color: var(--ink);
    font-size: clamp(1.55rem, 3vw, 2.3rem);
    line-height: 1.08;
    margin: 0;
    letter-spacing: -0.045em;
    font-weight: 950;
}}

.hero-copy {{
    max-width: 760px;
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.95rem;
    margin: 0.55rem 0 0;
}}

.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    white-space: nowrap;
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    background: var(--accent-soft);
    color: var(--accent-dark);
    padding: 0.52rem 0.76rem;
    border-radius: 999px;
    font-weight: 900;
    font-size: 0.78rem;
}}

.card {{
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--surface);
    box-shadow: var(--shadow-sm);
    padding: 1rem;
    margin-bottom: 0.9rem;
}}

.card-subtle {{
    border: 1px solid var(--soft-line);
    border-radius: var(--radius-lg);
    background: var(--surface-2);
    padding: 0.9rem;
}}

.section-title {{
    color: var(--ink);
    font-size: 1.02rem;
    font-weight: 920;
    margin: 0 0 0.5rem;
}}

.page-copy, .small-muted {{
    color: var(--muted);
    line-height: 1.55;
}}

.small-muted {{
    font-size: 0.82rem;
}}

.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.9rem;
}}

.metric-card {{
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--surface);
    box-shadow: var(--shadow-sm);
    padding: 0.95rem;
    min-height: 7rem;
}}

.metric-label {{
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 950;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}}

.metric-value {{
    color: var(--ink);
    font-size: 1.65rem;
    font-weight: 950;
    margin-top: 0.35rem;
    line-height: 1;
}}

.metric-caption {{
    color: var(--muted);
    font-size: 0.8rem;
    margin-top: 0.4rem;
    line-height: 1.42;
}}

.insight-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.1rem 0 0.9rem;
}}

.insight-card {{
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, var(--surface), var(--surface-2));
    padding: 0.92rem;
    min-height: 7.5rem;
}}

.insight-title {{
    color: var(--ink);
    font-weight: 920;
    margin-bottom: 0.34rem;
}}

.insight-copy {{
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.5;
}}

.prompt-chip {{
    display: inline-block;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--ink);
    border-radius: 999px;
    padding: 0.42rem 0.7rem;
    margin: 0.16rem;
    font-size: 0.8rem;
    box-shadow: 0 6px 16px color-mix(in srgb, var(--ink) 7%, transparent);
}}

.message-row {{
    display: flex;
    margin: 0.72rem 0;
}}

.message-row.user {{ justify-content: flex-end; }}
.message-row.assistant {{ justify-content: flex-start; }}

.bubble {{
    max-width: min(82%, 850px);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 0.95rem 1.05rem;
    line-height: 1.6;
    font-size: 0.94rem;
    box-shadow: var(--shadow-sm);
}}

.bubble.user {{
    background: color-mix(in srgb, var(--accent) 18%, var(--surface));
    border-color: color-mix(in srgb, var(--accent) 28%, transparent);
    border-top-right-radius: 8px;
    color: var(--ink);
}}

.bubble.assistant {{
    background: var(--surface);
    border-top-left-radius: 8px;
    color: var(--ink);
}}

.message-label {{
    display: block;
    color: var(--muted);
    font-size: 0.67rem;
    font-weight: 950;
    margin-bottom: 0.32rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

.agent-answer .section-heading {{
    display: flex;
    align-items: center;
    gap: 0.42rem;
    margin: 0.78rem 0 0.32rem;
    color: var(--ink);
    font-size: 0.9rem;
    font-weight: 950;
}}

.agent-answer .section-heading:first-child {{
    margin-top: 0;
}}

.agent-answer .section-heading::before {{
    content: "";
    width: 0.46rem;
    height: 0.46rem;
    border-radius: 999px;
    background: var(--accent);
    flex: 0 0 auto;
}}

.agent-answer p {{
    margin: 0.26rem 0;
}}

.agent-answer ul {{
    margin: 0.22rem 0 0.5rem 1.05rem;
    padding: 0;
}}

.agent-answer li {{
    margin: 0.22rem 0;
    padding-left: 0.15rem;
}}

.agent-answer strong {{
    color: var(--ink);
    font-weight: 900;
}}

.loading-row {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    color: var(--muted);
    font-weight: 800;
}}

.loading-dots {{ display: inline-flex; gap: 0.25rem; }}
.loading-dots span {{
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 999px;
    background: var(--accent);
    animation: agentPulse 1s infinite ease-in-out;
}}
.loading-dots span:nth-child(2) {{ animation-delay: 0.15s; }}
.loading-dots span:nth-child(3) {{ animation-delay: 0.3s; }}
@keyframes agentPulse {{
    0%, 80%, 100% {{ opacity: 0.30; transform: translateY(0); }}
    40% {{ opacity: 1; transform: translateY(-3px); }}
}}

.result-card {{
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    background: var(--surface);
    box-shadow: var(--shadow-sm);
    padding: 0.85rem;
    margin-bottom: 0.65rem;
}}

.result-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}}

.source-badge {{
    border: 1px solid color-mix(in srgb, var(--accent) 34%, transparent);
    background: var(--accent-soft);
    color: var(--accent-dark);
    padding: 0.2rem 0.42rem;
    border-radius: 999px;
}}

pre, code {{
    border-radius: 12px !important;
}}

div[data-testid="stForm"],
div[data-testid="stTextArea"] textarea,
div[data-baseweb="select"] > div,
div[data-baseweb="input"],
div[data-baseweb="textarea"] {{
    background: var(--input-bg) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
    border-radius: 14px !important;
}}

div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-testid="stTextArea"] textarea {{
    color: var(--ink) !important;
}}

textarea::placeholder, input::placeholder {{
    color: var(--muted) !important;
    opacity: 1 !important;
}}

div.stButton > button,
div[data-testid="stForm"] button,
div[data-testid="stDownloadButton"] button {{
    border-radius: 13px !important;
    min-height: 2.55rem;
    font-weight: 850 !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    box-shadow: 0 8px 20px color-mix(in srgb, var(--ink) 6%, transparent);
}}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] button:hover {{
    border-color: color-mix(in srgb, var(--accent) 42%, transparent) !important;
    background: var(--accent-soft) !important;
}}

button[kind="primary"] {{
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    color: white !important;
    border-color: color-mix(in srgb, var(--accent) 70%, black) !important;
}}

div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"],
div[data-testid="stForm"] button[kind="primary"],
div[data-testid="stForm"] button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, var(--accent), var(--accent-dark)) !important;
    color: #ffffff !important;
    border-color: color-mix(in srgb, var(--accent) 70%, black) !important;
}}

button[kind="primary"] *,
button[kind="primary"] p,
button[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primary"] p,
div.stButton > button[kind="primary"] *,
div.stButton > button[kind="primary"] p,
div.stButton > button[data-testid="stBaseButton-primary"] *,
div.stButton > button[data-testid="stBaseButton-primary"] p,
div[data-testid="stForm"] button[kind="primary"] *,
div[data-testid="stForm"] button[kind="primary"] p,
div[data-testid="stForm"] button[data-testid="stBaseButton-primary"] *,
div[data-testid="stForm"] button[data-testid="stBaseButton-primary"] p {{
    color: #ffffff !important;
    opacity: 1 !important;
}}

[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    color: #f8fafc !important;
}}

[data-testid="stSidebar"] div.stButton > button *,
[data-testid="stSidebar"] div.stButton > button p,
[data-testid="stSidebar"] div.stButton > button span,
[data-testid="stSidebar"] div.stButton > button div,
[data-testid="stSidebar"] div.stButton > button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] div.stButton > button [data-testid="stMarkdownContainer"] p {{
    color: #f8fafc !important;
    opacity: 1 !important;
    text-align: left !important;
}}

[data-testid="stDataFrame"] {{
    border-radius: var(--radius-md);
    overflow: hidden;
}}

hr {{
    border: 0;
    height: 1px;
    background: var(--line);
}}

@media (max-width: 980px) {{
    :root {{ --drawer-width: 318px; }}
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {{
        padding-right: 0.9rem !important;
        padding-left: 0.9rem !important;
    }}
    .app-hero-row {{ flex-direction: column; align-items: flex-start; }}
    .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .insight-grid {{ grid-template-columns: 1fr; }}
    .bubble {{ max-width: 94%; }}
}}
</style>
"""


def safe_html(text: Any) -> str:
    return escape("" if text is None else str(text))


def render_hero(page: str, title: str, description: str, status: str = "Read-only SQL") -> None:
    st.markdown(
        f"""
        <div class="app-hero">
            <div class="app-hero-row">
                <div>
                    <div class="kicker">{safe_html(page)}</div>
                    <h1 class="hero-title">{safe_html(title)}</h1>
                    <p class="hero-copy">{safe_html(description)}</p>
                </div>
                <div class="status-pill">● {safe_html(status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(metrics: list[tuple[str, Any, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value, caption) in zip(cols, metrics, strict=True):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="metric-label">{safe_html(label)}</div>
                    <div class="metric-value">{safe_html(value)}</div>
                    <div class="metric-caption">{safe_html(caption)}</div>
                    """,
                    unsafe_allow_html=True,
                )


def render_insight_cards(cards: list[tuple[str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (title, copy) in zip(cols, cards, strict=True):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="insight-title">{safe_html(title)}</div>
                    <div class="insight-copy">{safe_html(copy)}</div>
                    """,
                    unsafe_allow_html=True,
                )


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{safe_html(title)}</div>', unsafe_allow_html=True)


# -----------------------------
# Data access helpers
# -----------------------------

def get_api_key_from_sources() -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "")
    secret_key = ""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    local_secret_key = ""
    local_secrets_path = APP_DIR / ".streamlit" / "secrets.toml"
    if local_secrets_path.exists():
        try:
            local_secret_key = tomllib.loads(local_secrets_path.read_text()).get("OPENAI_API_KEY", "")
        except Exception:
            local_secret_key = ""
    return env_key or secret_key or local_secret_key


@st.cache_data(show_spinner=False)
def get_filter_options(db_path: str) -> dict[str, list[str]]:
    conn = connect_sqlite(db_path)
    options: dict[str, list[str]] = {}
    queries = {
        "cities": "SELECT DISTINCT city FROM customers WHERE city IS NOT NULL ORDER BY city",
        "customer_segments": "SELECT DISTINCT customer_segment FROM customers WHERE customer_segment IS NOT NULL ORDER BY customer_segment",
        "value_segments": "SELECT DISTINCT value_segment FROM customer_value_segments WHERE value_segment IS NOT NULL ORDER BY value_segment",
        "risk_levels": "SELECT DISTINCT risk_level FROM customer_churn_scores WHERE risk_level IS NOT NULL ORDER BY CASE risk_level WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 3 ELSE 4 END",
        "service_types": "SELECT DISTINCT service_type FROM subscriptions WHERE service_type IS NOT NULL ORDER BY service_type",
        "campaign_types": "SELECT DISTINCT campaign_type FROM campaigns WHERE campaign_type IS NOT NULL ORDER BY campaign_type",
        "payment_statuses": "SELECT DISTINCT payment_status FROM invoices WHERE payment_status IS NOT NULL ORDER BY payment_status",
        "channels": "SELECT DISTINCT channel FROM customer_campaign_responses WHERE channel IS NOT NULL ORDER BY channel",
    }
    for key, query in queries.items():
        options[key] = pd.read_sql_query(query, conn).iloc[:, 0].dropna().astype(str).tolist()
    date_bounds = pd.read_sql_query("SELECT MIN(signup_date) AS min_date, MAX(signup_date) AS max_date FROM customers", conn).iloc[0]
    options["signup_date_bounds"] = [str(date_bounds["min_date"]), str(date_bounds["max_date"])]
    return options


def placeholders(values: tuple[Any, ...]) -> str:
    return ",".join(["?"] * len(values))


def build_customer_scope_cte(
    cities: tuple[str, ...] = (),
    customer_segments: tuple[str, ...] = (),
    value_segments: tuple[str, ...] = (),
    risk_levels: tuple[str, ...] = (),
    service_types: tuple[str, ...] = (),
    campaign_types: tuple[str, ...] = (),
    payment_statuses: tuple[str, ...] = (),
    min_arpu: float = 0.0,
    signup_start: str | None = None,
    signup_end: str | None = None,
) -> tuple[str, list[Any]]:
    conditions = ["1=1"]
    params: list[Any] = []
    if cities:
        conditions.append(f"c.city IN ({placeholders(cities)})")
        params.extend(cities)
    if customer_segments:
        conditions.append(f"c.customer_segment IN ({placeholders(customer_segments)})")
        params.extend(customer_segments)
    if value_segments:
        conditions.append(f"cvs.value_segment IN ({placeholders(value_segments)})")
        params.extend(value_segments)
    if risk_levels:
        conditions.append(f"ch.risk_level IN ({placeholders(risk_levels)})")
        params.extend(risk_levels)
    if service_types:
        conditions.append(f"s.service_type IN ({placeholders(service_types)})")
        params.extend(service_types)
    if campaign_types:
        conditions.append(
            f"""
            EXISTS (
                SELECT 1
                FROM customer_campaign_responses r
                JOIN campaigns camp ON camp.campaign_id = r.campaign_id
                WHERE r.customer_id = c.customer_id
                  AND camp.campaign_type IN ({placeholders(campaign_types)})
            )
            """
        )
        params.extend(campaign_types)
    if payment_statuses:
        conditions.append(
            f"""
            EXISTS (
                SELECT 1
                FROM accounts a
                JOIN invoices i ON i.account_id = a.account_id
                WHERE a.customer_id = c.customer_id
                  AND i.payment_status IN ({placeholders(payment_statuses)})
            )
            """
        )
        params.extend(payment_statuses)
    if min_arpu and min_arpu > 0:
        conditions.append("COALESCE(cvs.arpu_jod, 0) >= ?")
        params.append(float(min_arpu))
    if signup_start:
        conditions.append("date(c.signup_date) >= date(?)")
        params.append(signup_start)
    if signup_end:
        conditions.append("date(c.signup_date) <= date(?)")
        params.append(signup_end)

    where_clause = " AND ".join(conditions)
    cte = f"""
        WITH customer_scope AS (
            SELECT DISTINCT c.customer_id
            FROM customers c
            LEFT JOIN customer_churn_scores ch ON c.customer_id = ch.customer_id
            LEFT JOIN customer_value_segments cvs ON c.customer_id = cvs.customer_id
            LEFT JOIN subscriptions s ON c.customer_id = s.customer_id
            WHERE {where_clause}
        )
    """
    return cte, params


@st.cache_data(show_spinner=False)
def get_dynamic_analytics_data(
    db_path: str,
    cities: tuple[str, ...],
    customer_segments: tuple[str, ...],
    value_segments: tuple[str, ...],
    risk_levels: tuple[str, ...],
    service_types: tuple[str, ...],
    min_arpu: float,
    signup_start: str | None,
    signup_end: str | None,
    top_n: int,
    campaign_types: tuple[str, ...],
    payment_statuses: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    conn = connect_sqlite(db_path)
    cte, params = build_customer_scope_cte(
        cities=cities,
        customer_segments=customer_segments,
        value_segments=value_segments,
        risk_levels=risk_levels,
        service_types=service_types,
        campaign_types=campaign_types,
        payment_statuses=payment_statuses,
        min_arpu=min_arpu,
        signup_start=signup_start,
        signup_end=signup_end,
    )

    payment_condition = ""
    payment_params: list[Any] = []
    if payment_statuses:
        payment_condition = f" AND i.payment_status IN ({placeholders(payment_statuses)})"
        payment_params.extend(payment_statuses)

    campaign_condition = ""
    campaign_params: list[Any] = []
    if campaign_types:
        campaign_condition = f" AND camp.campaign_type IN ({placeholders(campaign_types)})"
        campaign_params.extend(campaign_types)

    data: dict[str, pd.DataFrame] = {}

    data["kpis"] = pd.read_sql_query(
        cte
        + f"""
        SELECT
            (SELECT COUNT(*) FROM customer_scope) AS scoped_customers,
            (SELECT COUNT(DISTINCT ch.customer_id)
             FROM customer_churn_scores ch JOIN customer_scope cs ON cs.customer_id = ch.customer_id
             WHERE ch.risk_level = 'High') AS high_risk_customers,
            (SELECT ROUND(AVG(ch.churn_score), 3)
             FROM customer_churn_scores ch JOIN customer_scope cs ON cs.customer_id = ch.customer_id) AS avg_churn_score,
            (SELECT ROUND(AVG(cvs.arpu_jod), 2)
             FROM customer_value_segments cvs JOIN customer_scope cs ON cs.customer_id = cvs.customer_id) AS avg_arpu_jod,
            (SELECT ROUND(SUM(i.total_amount_jod), 2)
             FROM invoices i
             JOIN accounts a ON a.account_id = i.account_id
             JOIN customer_scope cs ON cs.customer_id = a.customer_id
             WHERE 1=1 {payment_condition}) AS total_invoiced_jod,
            (SELECT COUNT(*)
             FROM complaints co JOIN customer_scope cs ON cs.customer_id = co.customer_id) AS total_complaints,
            (SELECT COUNT(*)
             FROM support_interactions si JOIN customer_scope cs ON cs.customer_id = si.customer_id
             WHERE si.customer_sentiment = 'Negative') AS negative_support_cases
        """,
        conn,
        params=params + payment_params,
    )

    data["city_customers"] = pd.read_sql_query(
        cte
        + """
        SELECT c.city, COUNT(DISTINCT c.customer_id) AS total_customers
        FROM customers c JOIN customer_scope cs ON cs.customer_id = c.customer_id
        GROUP BY c.city
        ORDER BY total_customers DESC
        LIMIT ?
        """,
        conn,
        params=params + [int(top_n)],
    )

    data["churn_mix"] = pd.read_sql_query(
        cte
        + """
        SELECT ch.risk_level, COUNT(DISTINCT ch.customer_id) AS customers, ROUND(AVG(ch.churn_score), 3) AS avg_churn_score
        FROM customer_churn_scores ch JOIN customer_scope cs ON cs.customer_id = ch.customer_id
        GROUP BY ch.risk_level
        ORDER BY CASE ch.risk_level WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 3 ELSE 4 END
        """,
        conn,
        params=params,
    )

    data["high_risk_city"] = pd.read_sql_query(
        cte
        + """
        SELECT c.city, COUNT(DISTINCT c.customer_id) AS high_risk_customers
        FROM customers c
        JOIN customer_scope cs ON cs.customer_id = c.customer_id
        JOIN customer_churn_scores ch ON c.customer_id = ch.customer_id
        WHERE ch.risk_level = 'High'
        GROUP BY c.city
        ORDER BY high_risk_customers DESC
        LIMIT ?
        """,
        conn,
        params=params + [int(top_n)],
    )

    data["value_segments"] = pd.read_sql_query(
        cte
        + """
        SELECT
            cvs.value_segment,
            COUNT(DISTINCT cvs.customer_id) AS customers,
            ROUND(AVG(cvs.arpu_jod), 2) AS avg_arpu_jod,
            ROUND(AVG(cvs.total_revenue_6m_jod), 2) AS avg_revenue_6m_jod
        FROM customer_value_segments cvs JOIN customer_scope cs ON cs.customer_id = cvs.customer_id
        GROUP BY cvs.value_segment
        ORDER BY avg_revenue_6m_jod DESC
        """,
        conn,
        params=params,
    )

    data["complaints"] = pd.read_sql_query(
        cte
        + """
        SELECT co.complaint_category, co.severity, COUNT(*) AS complaints
        FROM complaints co JOIN customer_scope cs ON cs.customer_id = co.customer_id
        GROUP BY co.complaint_category, co.severity
        ORDER BY complaints DESC
        """,
        conn,
        params=params,
    )

    data["support_sentiment"] = pd.read_sql_query(
        cte
        + """
        SELECT si.channel, si.customer_sentiment, COUNT(*) AS interactions
        FROM support_interactions si JOIN customer_scope cs ON cs.customer_id = si.customer_id
        GROUP BY si.channel, si.customer_sentiment
        ORDER BY interactions DESC
        """,
        conn,
        params=params,
    )

    data["campaigns"] = pd.read_sql_query(
        cte
        + f"""
        SELECT
            camp.campaign_name,
            camp.campaign_type,
            camp.target_segment,
            COUNT(r.response_id) AS total_sent,
            SUM(r.converted_flag) AS total_converted,
            ROUND(100.0 * SUM(r.converted_flag) / NULLIF(COUNT(r.response_id), 0), 2) AS conversion_rate,
            ROUND(SUM(r.revenue_generated_jod), 2) AS revenue_generated_jod
        FROM campaigns camp
        JOIN customer_campaign_responses r ON camp.campaign_id = r.campaign_id
        JOIN customer_scope cs ON cs.customer_id = r.customer_id
        WHERE 1=1 {campaign_condition}
        GROUP BY camp.campaign_id, camp.campaign_name, camp.campaign_type, camp.target_segment
        ORDER BY conversion_rate DESC, revenue_generated_jod DESC
        LIMIT ?
        """,
        conn,
        params=params + campaign_params + [int(top_n)],
    )

    data["billing"] = pd.read_sql_query(
        cte
        + f"""
        SELECT i.payment_status, COUNT(*) AS invoices, ROUND(SUM(i.total_amount_jod), 2) AS total_amount_jod
        FROM invoices i
        JOIN accounts a ON a.account_id = i.account_id
        JOIN customer_scope cs ON cs.customer_id = a.customer_id
        WHERE 1=1 {payment_condition}
        GROUP BY i.payment_status
        ORDER BY invoices DESC
        """,
        conn,
        params=params + payment_params,
    )

    data["revenue_trend"] = pd.read_sql_query(
        cte
        + """
        SELECT cms.summary_month, ROUND(SUM(cms.total_revenue_jod), 2) AS total_revenue_jod, ROUND(AVG(cms.churn_score), 3) AS avg_churn_score
        FROM customer_monthly_summary cms JOIN customer_scope cs ON cs.customer_id = cms.customer_id
        GROUP BY cms.summary_month
        ORDER BY cms.summary_month
        """,
        conn,
        params=params,
    )

    data["scoped_customers"] = pd.read_sql_query(
        cte
        + """
        SELECT DISTINCT
            c.customer_id,
            c.full_name,
            c.city,
            c.customer_segment,
            ch.risk_level,
            ROUND(ch.churn_score, 3) AS churn_score,
            cvs.value_segment,
            ROUND(cvs.arpu_jod, 2) AS arpu_jod,
            c.status
        FROM customers c
        JOIN customer_scope cs ON cs.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments cvs ON cvs.customer_id = c.customer_id
        ORDER BY COALESCE(ch.churn_score, 0) DESC, COALESCE(cvs.arpu_jod, 0) DESC
        LIMIT 200
        """,
        conn,
        params=params,
    )

    return data


@st.cache_resource(show_spinner=False)
def get_agent(db_path: str, model_name: str, api_key: str, prompt_version: str):
    return create_zain_sql_agent(
        db_path=db_path,
        model_name=model_name,
        openai_api_key=api_key,
    )


# -----------------------------
# Chat helpers
# -----------------------------

def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = ""
    st.session_state.scroll_to_latest = False


def submit_prompt(prompt: str, db_path: str, model_name: str, api_key: str) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    loading_slot = st.empty()
    with loading_slot.container():
        render_loading_message()
    with st.spinner("Running guarded SQL analysis..."):
        try:
            agent = get_agent(db_path, model_name, api_key, AGENT_PROMPT_VERSION)
            answer = run_sql_agent(agent, clean_prompt)
        except Exception as exc:
            answer = (
                "I could not complete the request.\n\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
    loading_slot.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_prompt = ""
    st.session_state.scroll_to_latest = True


def format_inline_answer_text(text: str) -> str:
    safe = escape(text.strip())
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    return safe


def format_assistant_answer_html(content: str) -> str:
    lines = [line.strip() for line in content.strip().splitlines()]
    html_parts: list[str] = []
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            html_parts.append("</ul>")
            list_open = False

    heading_pattern = re.compile(r"^(?:#+\s*)?(?:\*\*)?(?:\d+\.\s*)?([A-Za-z][A-Za-z &/()-]{2,45})(?:\*\*)?\s*:\s*(.*)$")

    for line in lines:
        if not line:
            close_list()
            continue

        bullet_match = re.match(r"^(?:[-*•]\s+|\d+[.)]\s+)(.+)$", line)
        heading_match = heading_pattern.match(line)

        if heading_match and len(line) < 110:
            close_list()
            title, rest = heading_match.groups()
            html_parts.append(f'<div class="section-heading">{format_inline_answer_text(title)}</div>')
            if rest.strip():
                html_parts.append(f"<p>{format_inline_answer_text(rest)}</p>")
        elif bullet_match:
            if not list_open:
                html_parts.append("<ul>")
                list_open = True
            html_parts.append(f"<li>{format_inline_answer_text(bullet_match.group(1))}</li>")
        else:
            close_list()
            html_parts.append(f"<p>{format_inline_answer_text(line)}</p>")

    close_list()
    return '<div class="agent-answer">' + "".join(html_parts) + "</div>"


def format_message_html(content: str, role: str) -> str:
    if role == "assistant":
        return format_assistant_answer_html(content)
    return escape(content).replace("\n", "<br>")


def render_chat_autoscroll() -> None:
    st.markdown('<div id="chat-latest-response"></div>', unsafe_allow_html=True)
    if not st.session_state.get("scroll_to_latest"):
        return
    components.html(
        """
        <script>
        const anchor = window.parent.document.getElementById("chat-latest-response");
        if (anchor) {
            anchor.scrollIntoView({ behavior: "smooth", block: "end" });
        }
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_latest = False


def render_message(role: str, content: str) -> None:
    safe_role = "user" if role == "user" else "assistant"
    label = "You" if safe_role == "user" else "Customer 360 AI"
    st.markdown(
        f"""
        <div class="message-row {safe_role}">
            <div class="bubble {safe_role}">
                <span class="message-label">{label}</span>
                {format_message_html(content, safe_role)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_message() -> None:
    st.markdown(
        """
        <div class="message-row assistant">
            <div class="bubble assistant">
                <span class="message-label">Customer 360 AI</span>
                <div class="loading-row">
                    <span>Reviewing customer data</span>
                    <span class="loading-dots"><span></span><span></span><span></span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_composer() -> tuple[str, bool, bool]:
    with st.form("sql_agent_form", clear_on_submit=False):
        prompt = st.text_area(
            "Ask the Customer 360 database",
            value=st.session_state.pending_prompt,
            placeholder="Example: Which cities have the most high-risk churn customers? Show the top 10 and recommend next actions.",
            height=104,
        )
        col_submit, col_clear = st.columns([1.4, 1])
        submitted = col_submit.form_submit_button("Ask AI", use_container_width=True, type="primary")
        cleared = col_clear.form_submit_button("Clear conversation", use_container_width=True)
    return prompt, submitted, cleared


# -----------------------------
# RAG search helpers
# -----------------------------

def tokenize_query(query: str) -> list[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]{2,}", query.lower())
    stopwords = {"the", "and", "for", "with", "from", "this", "that", "show", "find", "what", "which", "how"}
    return [token for token in tokens if token not in stopwords][:12]


@st.cache_data(show_spinner=False)
def get_searchable_tables(db_path: str) -> dict[str, list[str]]:
    conn = connect_sqlite(db_path)
    tables = list_tables(conn)["name"].tolist()
    searchable: dict[str, list[str]] = {}
    for table in tables:
        info = pd.read_sql_query(f'PRAGMA table_info("{table.replace(chr(34), chr(34)*2)}")', conn)
        cols: list[str] = []
        for _, row in info.iterrows():
            col_type = str(row["type"]).upper()
            col_name = str(row["name"])
            if "TEXT" in col_type or col_name.endswith("_id") or col_name in {"customer_id", "subscription_id", "account_id"}:
                cols.append(col_name)
        if cols:
            searchable[table] = cols
    return searchable


def row_to_snippet(row: pd.Series, max_fields: int = 9, max_chars: int = 620) -> str:
    fields: list[str] = []
    for key, value in row.items():
        if key == "_rowid" or pd.isna(value):
            continue
        text = str(value).replace("\n", " ").strip()
        if not text:
            continue
        if len(text) > 120:
            text = text[:117] + "..."
        fields.append(f"{key}: {text}")
        if len(fields) >= max_fields:
            break
    snippet = " | ".join(fields)
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3] + "..."
    return snippet


@st.cache_data(show_spinner=False)
def rag_search_database(
    db_path: str,
    query: str,
    selected_tables: tuple[str, ...],
    limit_per_table: int = 8,
    max_results: int = 30,
    require_all_terms: bool = False,
) -> pd.DataFrame:
    tokens = tokenize_query(query)
    if not tokens:
        return pd.DataFrame(columns=["table", "rowid", "score", "snippet"])

    conn = connect_sqlite(db_path)
    searchable = get_searchable_tables(db_path)
    tables_to_search = selected_tables or tuple(searchable.keys())
    results: list[dict[str, Any]] = []

    for table in tables_to_search:
        columns = searchable.get(table, [])
        if not columns:
            continue
        safe_table = table.replace('"', '""')
        safe_cols = [col.replace('"', '""') for col in columns]
        select_cols = ", ".join([f'"{col}"' for col in safe_cols])

        token_clauses: list[str] = []
        params: list[Any] = []
        for token in tokens:
            col_clauses = [f'LOWER(CAST("{col}" AS TEXT)) LIKE ?' for col in safe_cols]
            token_clauses.append("(" + " OR ".join(col_clauses) + ")")
            params.extend([f"%{token.lower()}%"] * len(safe_cols))
        joiner = " AND " if require_all_terms else " OR "
        where_clause = joiner.join(token_clauses)
        query_sql = f'SELECT rowid AS _rowid, {select_cols} FROM "{safe_table}" WHERE {where_clause} LIMIT ?'
        try:
            table_df = pd.read_sql_query(query_sql, conn, params=params + [int(limit_per_table)])
        except Exception:
            continue

        for _, row in table_df.iterrows():
            haystack = " ".join(str(v).lower() for v in row.values if pd.notna(v))
            score = sum(haystack.count(token) for token in tokens)
            if query.lower() in haystack:
                score += 8
            results.append(
                {
                    "table": table,
                    "rowid": int(row.get("_rowid", 0) or 0),
                    "score": int(score),
                    "snippet": row_to_snippet(row),
                }
            )

    if not results:
        return pd.DataFrame(columns=["table", "rowid", "score", "snippet"])
    result_df = pd.DataFrame(results).sort_values(["score", "table"], ascending=[False, True]).head(max_results)
    return result_df.reset_index(drop=True)


def build_rag_context(results_df: pd.DataFrame, max_chars: int = 12000) -> str:
    lines: list[str] = []
    for index, row in results_df.iterrows():
        lines.append(f"[{index + 1}] Source table: {row['table']} | rowid: {row['rowid']} | score: {row['score']}\n{row['snippet']}")
    context = "\n\n".join(lines)
    if len(context) > max_chars:
        context = context[: max_chars - 3] + "..."
    return context


def extract_model_text(result: Any) -> str:
    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, str):
            return content
        return str(content)
    return str(result)


def answer_with_rag(question: str, context: str, model_name: str, api_key: str) -> str:
    model = create_chat_model(model_name=model_name, temperature=0.1, openai_api_key=api_key)
    system_prompt = (
        "You are a telecom Customer 360 analyst. Answer using only the retrieved context. "
        "If the context is not enough, say what is missing. Cite evidence by source numbers like [1], [2]. "
        "Be concise, business-focused, and include recommended next actions when useful."
    )
    user_prompt = f"Question:\n{question}\n\nRetrieved context:\n{context}"
    result = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return extract_model_text(result)


# -----------------------------
# Customer 360 helpers
# -----------------------------

@st.cache_data(show_spinner=False)
def search_customers(
    db_path: str,
    term: str,
    city: str | None,
    risk_level: str | None,
    limit: int = 25,
) -> pd.DataFrame:
    conn = connect_sqlite(db_path)
    conditions = ["1=1"]
    params: list[Any] = []
    if term.strip():
        like = f"%{term.strip().lower()}%"
        conditions.append(
            "("
            "LOWER(c.full_name) LIKE ? OR LOWER(c.email) LIKE ? OR LOWER(c.phone_number) LIKE ? "
            "OR LOWER(s.msisdn) LIKE ? OR CAST(c.customer_id AS TEXT) LIKE ?"
            ")"
        )
        params.extend([like, like, like, like, like])
    if city:
        conditions.append("c.city = ?")
        params.append(city)
    if risk_level:
        conditions.append("ch.risk_level = ?")
        params.append(risk_level)
    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT DISTINCT
            c.customer_id,
            c.full_name,
            c.city,
            c.customer_segment,
            c.status,
            s.msisdn,
            ch.risk_level,
            ROUND(ch.churn_score, 3) AS churn_score,
            cvs.value_segment,
            ROUND(cvs.arpu_jod, 2) AS arpu_jod
        FROM customers c
        LEFT JOIN subscriptions s ON s.customer_id = c.customer_id
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments cvs ON cvs.customer_id = c.customer_id
        WHERE {where_clause}
        ORDER BY COALESCE(ch.churn_score, 0) DESC, COALESCE(cvs.arpu_jod, 0) DESC
        LIMIT ?
    """
    return pd.read_sql_query(sql, conn, params=params + [int(limit)])


@st.cache_data(show_spinner=False)
def get_customer_profile(db_path: str, customer_id: int) -> dict[str, pd.DataFrame]:
    conn = connect_sqlite(db_path)
    profile: dict[str, pd.DataFrame] = {}
    profile["summary"] = pd.read_sql_query(
        """
        SELECT
            c.*,
            ch.risk_level,
            ROUND(ch.churn_score, 3) AS churn_score,
            ch.main_risk_reason,
            ch.recommended_action,
            cvs.value_segment,
            ROUND(cvs.arpu_jod, 2) AS arpu_jod,
            ROUND(cvs.total_revenue_6m_jod, 2) AS total_revenue_6m_jod,
            cvs.lifetime_months
        FROM customers c
        LEFT JOIN customer_churn_scores ch ON ch.customer_id = c.customer_id
        LEFT JOIN customer_value_segments cvs ON cvs.customer_id = c.customer_id
        WHERE c.customer_id = ?
        """,
        conn,
        params=(customer_id,),
    )
    profile["subscriptions"] = pd.read_sql_query(
        """
        SELECT s.subscription_id, s.msisdn, s.service_type, s.status, s.activation_date, s.contract_end_date,
               p.plan_name, p.plan_category, p.monthly_fee_jod, p.technology
        FROM subscriptions s
        LEFT JOIN plans p ON p.plan_id = s.plan_id
        WHERE s.customer_id = ?
        ORDER BY s.primary_subscription_flag DESC, s.activation_date DESC
        """,
        conn,
        params=(customer_id,),
    )
    profile["invoices"] = pd.read_sql_query(
        """
        SELECT i.invoice_id, i.issue_date, i.due_date, i.payment_status, i.days_overdue, ROUND(i.total_amount_jod, 2) AS total_amount_jod
        FROM invoices i
        JOIN accounts a ON a.account_id = i.account_id
        WHERE a.customer_id = ?
        ORDER BY i.issue_date DESC
        LIMIT 8
        """,
        conn,
        params=(customer_id,),
    )
    profile["complaints"] = pd.read_sql_query(
        """
        SELECT complaint_date, complaint_category, severity, status, complaint_description
        FROM complaints
        WHERE customer_id = ?
        ORDER BY complaint_date DESC
        LIMIT 8
        """,
        conn,
        params=(customer_id,),
    )
    profile["support"] = pd.read_sql_query(
        """
        SELECT interaction_datetime, channel, reason_category, issue_type, priority, resolution_status, customer_sentiment, resolution_time_minutes
        FROM support_interactions
        WHERE customer_id = ?
        ORDER BY interaction_datetime DESC
        LIMIT 8
        """,
        conn,
        params=(customer_id,),
    )
    profile["devices"] = pd.read_sql_query(
        """
        SELECT device_type, brand, model, os, device_5g_capable_flag, installment_flag, monthly_installment_jod
        FROM devices
        WHERE customer_id = ?
        ORDER BY purchase_date DESC
        LIMIT 8
        """,
        conn,
        params=(customer_id,),
    )
    return profile


# -----------------------------
# Page renderers
# -----------------------------

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="drawer-brand">
                <div class="drawer-logo" aria-hidden="true"></div>
                <div>
                    <h2>Zain Customer 360 AI</h2>
                    <p>Customer insights, evidence search, and decision dashboard.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="drawer-section-title">Workspace</div>', unsafe_allow_html=True)
        for page, icon, _desc in APP_PAGES:
            label = f"{page}  {icon}" if page == "Help" else f"{icon}  {page}" if icon else page
            is_active = st.session_state.app_page == page
            if st.button(label, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.app_page = page
                st.rerun()

        st.markdown('<div class="drawer-section"><div class="drawer-section-title">Quick action</div></div>', unsafe_allow_html=True)
        if st.button("Start a new chat", use_container_width=True):
            reset_chat()
            st.session_state.app_page = "AI Chat"
            st.rerun()
        if st.button("Open Evidence Search", use_container_width=True):
            st.session_state.app_page = "Evidence Search"
            st.rerun()
        st.markdown(
            '<p class="drawer-note">Tip: use Analytics filters first, then use Chat or Evidence Search to investigate the segment you discovered.</p>',
            unsafe_allow_html=True,
        )

    return st.session_state.app_page


def reset_analytics_filters(min_signup: str | None, max_signup: str | None) -> None:
    reset_values = {
        "analytics_cities": [],
        "analytics_risks": [],
        "analytics_top_n": 10,
        "analytics_customer_segments": [],
        "analytics_value_segments": [],
        "analytics_min_arpu": 0.0,
        "analytics_services": [],
        "analytics_campaign_types": [],
        "analytics_payment_statuses": [],
        "analytics_signup_start": min_signup or "",
        "analytics_signup_end": max_signup or "",
    }
    for key, value in reset_values.items():
        st.session_state[key] = value


def ensure_analytics_filter_defaults(min_signup: str | None, max_signup: str | None) -> None:
    defaults = {
        "analytics_cities": [],
        "analytics_risks": [],
        "analytics_top_n": 10,
        "analytics_customer_segments": [],
        "analytics_value_segments": [],
        "analytics_min_arpu": 0.0,
        "analytics_services": [],
        "analytics_campaign_types": [],
        "analytics_payment_statuses": [],
        "analytics_signup_start": min_signup or "",
        "analytics_signup_end": max_signup or "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_chat_page(api_key_input: str, db_path: str, model_name: str) -> None:
    render_hero(
        "AI Chat",
        "Ask the Customer 360 database like an analyst",
        "Ask about churn, revenue, campaigns, complaints, billing, or customer experience and get a plain-language business answer.",
        "Business assistant",
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Start with a precise business question</div>
                <div class="page-copy">Ask about churn, revenue, campaigns, complaints, support sentiment, customers, billing, or network activity. The agent will use the database rather than guessing.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="small-muted" style="font-weight:900; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.35rem;">Suggested prompts</div>', unsafe_allow_html=True)
        prompt_cols = st.columns(3)
        for idx, prompt in enumerate(FLAT_EXAMPLE_PROMPTS[:6]):
            with prompt_cols[idx % 3]:
                if st.button(prompt, key=f"chat_suggest_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
    else:
        for message in st.session_state.messages:
            render_message(message["role"], message["content"])
        render_chat_autoscroll()

    prompt, submitted, cleared = render_composer()
    if cleared:
        reset_chat()
        st.rerun()
    if submitted:
        if not api_key_input:
            st.error("AI answers are currently unavailable. Please ask the app administrator to configure access.")
        else:
            submit_prompt(prompt, db_path, model_name, api_key_input)
            st.rerun()


def render_dynamic_analytics_page(db_path: str, model_name: str, api_key: str) -> None:
    render_hero(
        "Executive Analytics",
        "Dynamic Customer 360 dashboard",
        "Adjust filters, top-N limits, segments, risk levels, ARPU thresholds, campaign types, and billing status to explore the business from different angles.",
        "Interactive controls",
    )

    options = get_filter_options(db_path)
    min_signup, max_signup = options.get("signup_date_bounds", [None, None])
    ensure_analytics_filter_defaults(min_signup, max_signup)

    with st.expander("Analytics filters", expanded=True):
        reset_col, _spacer = st.columns([1, 3])
        with reset_col:
            if st.button("Reset filters", use_container_width=True):
                reset_analytics_filters(min_signup, max_signup)
                st.rerun()

        customer_col, business_col, date_col = st.columns(3)
        with customer_col:
            st.markdown('<div class="section-title">Customer filters</div>', unsafe_allow_html=True)
            selected_cities = st.multiselect("Cities", options=options["cities"], key="analytics_cities")
            selected_customer_segments = st.multiselect("Customer segments", options=options["customer_segments"], key="analytics_customer_segments")
            selected_value_segments = st.multiselect("Value segments", options=options["value_segments"], key="analytics_value_segments")
            selected_risks = st.multiselect("Churn risk", options=options["risk_levels"], key="analytics_risks")
            min_arpu = st.slider("Minimum ARPU (JOD)", 0.0, 100.0, step=1.0, key="analytics_min_arpu")
        with business_col:
            st.markdown('<div class="section-title">Business filters</div>', unsafe_allow_html=True)
            selected_services = st.multiselect("Service type", options=options["service_types"], key="analytics_services")
            selected_campaign_types = st.multiselect("Campaign type", options=options["campaign_types"], key="analytics_campaign_types")
            selected_payment_statuses = st.multiselect("Billing status", options=options["payment_statuses"], key="analytics_payment_statuses")
            top_n = st.slider("Results to show", 5, 20, key="analytics_top_n")
        with date_col:
            st.markdown('<div class="section-title">Date range</div>', unsafe_allow_html=True)
            signup_start = st.text_input("Signup start date", help="Use YYYY-MM-DD. Leave blank for no lower bound.", key="analytics_signup_start")
            signup_end = st.text_input("Signup end date", help="Use YYYY-MM-DD. Leave blank for no upper bound.", key="analytics_signup_end")

    data = get_dynamic_analytics_data(
        db_path=db_path,
        cities=tuple(selected_cities),
        customer_segments=tuple(selected_customer_segments),
        value_segments=tuple(selected_value_segments),
        risk_levels=tuple(selected_risks),
        service_types=tuple(selected_services),
        min_arpu=float(min_arpu),
        signup_start=signup_start.strip() or None,
        signup_end=signup_end.strip() or None,
        top_n=int(top_n),
        campaign_types=tuple(selected_campaign_types),
        payment_statuses=tuple(selected_payment_statuses),
    )

    kpis = data["kpis"].iloc[0]
    scoped_customers = int(kpis.get("scoped_customers") or 0)
    high_risk = int(kpis.get("high_risk_customers") or 0)
    high_risk_rate = (high_risk / scoped_customers * 100) if scoped_customers else 0
    avg_arpu = float(kpis.get("avg_arpu_jod") or 0)
    total_invoiced = float(kpis.get("total_invoiced_jod") or 0)

    render_metric_grid(
        [
            ("Scoped customers", f"{scoped_customers:,}", "Customers matching the active filters"),
            ("High-risk churn", f"{high_risk:,}", f"{high_risk_rate:.1f}% of scoped customers"),
            ("Average ARPU", f"{avg_arpu:.2f} JOD", "Filtered customer value segments"),
            ("Invoiced amount", f"{total_invoiced:,.0f} JOD", "Invoices matching billing filters"),
        ]
    )

    top_city = data["city_customers"].iloc[0].to_dict() if not data["city_customers"].empty else {"city": "N/A", "total_customers": 0}
    top_campaign = data["campaigns"].iloc[0].to_dict() if not data["campaigns"].empty else {"campaign_name": "N/A", "conversion_rate": 0, "revenue_generated_jod": 0}
    top_complaint_df = data["complaints"].groupby("complaint_category", as_index=False)["complaints"].sum().sort_values("complaints", ascending=False) if not data["complaints"].empty else pd.DataFrame()
    top_complaint = top_complaint_df.iloc[0].to_dict() if not top_complaint_df.empty else {"complaint_category": "N/A", "complaints": 0}
    render_insight_cards(
        [
            ("Largest market", f"{top_city['city']} leads this view with {int(top_city['total_customers']):,} customers."),
            ("Campaign winner", f"{top_campaign['campaign_name']} converts at {float(top_campaign['conversion_rate'] or 0):.1f}% and generated {float(top_campaign.get('revenue_generated_jod') or 0):,.0f} JOD."),
            ("Experience issue", f"{top_complaint['complaint_category']} is the largest complaint category with {int(top_complaint['complaints']):,} complaints."),
        ]
    )

    palette = chart_palette()
    risk_scale = alt.Scale(domain=["High", "Medium", "Low"], range=[palette["danger"], palette["warning"], palette["success"]])

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_section_title("Top cities by customers")
            if data["city_customers"].empty:
                st.info("No matching customers for the selected filters.")
            else:
                city_chart = (
                    alt.Chart(data["city_customers"])
                    .mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7)
                    .encode(
                        y=alt.Y("city:N", sort="-x", title=None),
                        x=alt.X("total_customers:Q", title="Customers"),
                        color=alt.value(palette["accent"]),
                        tooltip=["city", "total_customers"],
                    )
                    .properties(height=310, background="transparent")
                )
                st.altair_chart(city_chart, use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_title("Churn risk mix")
            if data["churn_mix"].empty:
                st.info("No churn score rows matched the filters.")
            else:
                churn_chart = (
                    alt.Chart(data["churn_mix"])
                    .mark_arc(innerRadius=64, outerRadius=112)
                    .encode(
                        theta=alt.Theta("customers:Q"),
                        color=alt.Color("risk_level:N", scale=risk_scale, title="Risk"),
                        tooltip=["risk_level", "customers", "avg_churn_score"],
                    )
                    .properties(height=310, background="transparent")
                )
                st.altair_chart(churn_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_section_title("High-risk customers by city")
            if data["high_risk_city"].empty:
                st.info("No high-risk customers matched the filters.")
            else:
                high_risk_chart = (
                    alt.Chart(data["high_risk_city"])
                    .mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7)
                    .encode(
                        y=alt.Y("city:N", sort="-x", title=None),
                        x=alt.X("high_risk_customers:Q", title="High-risk customers"),
                        color=alt.value(palette["danger"]),
                        tooltip=["city", "high_risk_customers"],
                    )
                    .properties(height=310, background="transparent")
                )
                st.altair_chart(high_risk_chart, use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_title("Revenue trend and churn pressure")
            if data["revenue_trend"].empty:
                st.info("No monthly summary rows matched the filters.")
            else:
                revenue_chart = (
                    alt.Chart(data["revenue_trend"])
                    .mark_line(point=True, interpolate="monotone")
                    .encode(
                        x=alt.X("summary_month:N", title="Month"),
                        y=alt.Y("total_revenue_jod:Q", title="Revenue (JOD)"),
                        color=alt.value(palette["accent2"]),
                        tooltip=["summary_month", "total_revenue_jod", "avg_churn_score"],
                    )
                    .properties(height=310, background="transparent")
                )
                st.altair_chart(revenue_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_section_title("Complaints by category and severity")
            if data["complaints"].empty:
                st.info("No complaints matched the filters.")
            else:
                complaint_chart = (
                    alt.Chart(data["complaints"])
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                    .encode(
                        y=alt.Y("complaint_category:N", sort="-x", title=None),
                        x=alt.X("sum(complaints):Q", title="Complaints"),
                        color=alt.Color("severity:N", title="Severity"),
                        tooltip=["complaint_category", "severity", "complaints"],
                    )
                    .properties(height=330, background="transparent")
                )
                st.altair_chart(complaint_chart, use_container_width=True)

    with right:
        with st.container(border=True):
            render_section_title("Support sentiment by channel")
            if data["support_sentiment"].empty:
                st.info("No support interactions matched the filters.")
            else:
                support_chart = (
                    alt.Chart(data["support_sentiment"])
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                    .encode(
                        y=alt.Y("channel:N", sort="-x", title=None),
                        x=alt.X("sum(interactions):Q", title="Interactions"),
                        color=alt.Color("customer_sentiment:N", title="Sentiment"),
                        tooltip=["channel", "customer_sentiment", "interactions"],
                    )
                    .properties(height=330, background="transparent")
                )
                st.altair_chart(support_chart, use_container_width=True)

    with st.container(border=True):
        render_section_title("Campaign conversion leaders")
        if data["campaigns"].empty:
            st.info("No campaigns matched the selected campaign filters.")
        else:
            campaign_chart = (
                alt.Chart(data["campaigns"])
                .mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7)
                .encode(
                    y=alt.Y("campaign_name:N", sort="-x", title=None),
                    x=alt.X("conversion_rate:Q", title="Conversion rate (%)"),
                    color=alt.Color("campaign_type:N", title="Type"),
                    tooltip=["campaign_name", "campaign_type", "target_segment", "total_sent", "total_converted", "conversion_rate", "revenue_generated_jod"],
                )
                .properties(height=350, background="transparent")
            )
            st.altair_chart(campaign_chart, use_container_width=True)

    with st.container(border=True):
        render_section_title("Filtered customer export preview")
        st.dataframe(data["scoped_customers"], use_container_width=True, hide_index=True)
        csv = data["scoped_customers"].to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered customers CSV", csv, "filtered_customer_360.csv", "text/csv", use_container_width=True)

    if api_key:
        with st.expander("Generate AI executive brief from current filters"):
            brief_prompt = st.text_area(
                "Brief instruction",
                value="Summarize the most important business risks and recommended actions from this filtered dashboard.",
                height=90,
            )
            if st.button("Generate executive brief", type="primary", use_container_width=True):
                context = {
                    "kpis": data["kpis"].to_dict(orient="records"),
                    "top_cities": data["city_customers"].head(5).to_dict(orient="records"),
                    "churn_mix": data["churn_mix"].to_dict(orient="records"),
                    "campaigns": data["campaigns"].head(5).to_dict(orient="records"),
                    "complaints": data["complaints"].head(8).to_dict(orient="records"),
                    "billing": data["billing"].to_dict(orient="records"),
                }
                loading_slot = st.empty()
                with loading_slot.container():
                    st.markdown(
                        """
                        <div class="card-subtle">
                            <div class="loading-row">
                                <span>Generating executive brief</span>
                                <span class="loading-dots"><span></span><span></span><span></span></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                try:
                    with st.spinner("Generating executive brief..."):
                        model = create_chat_model(model_name=model_name, temperature=0.1, openai_api_key=api_key)
                        response = model.invoke([
                            {"role": "system", "content": "You are an executive telecom analytics advisor. Use only the provided dashboard data. Be concise and action-oriented."},
                            {"role": "user", "content": f"Instruction: {brief_prompt}\n\nDashboard data:\n{context}"},
                        ])
                    loading_slot.empty()
                    st.success("Executive brief generated.")
                    st.markdown(extract_model_text(response))
                except Exception as exc:
                    loading_slot.empty()
                    st.error(f"Could not generate brief: {type(exc).__name__}: {exc}")


def render_rag_search_page(db_path: str, model_name: str, api_key: str) -> None:
    render_hero(
        "Evidence Search",
        "Find supporting customer evidence",
        "Search across Customer 360 records, inspect the matching details, then optionally generate an evidence-grounded answer with source numbers.",
        "Evidence retrieval",
    )

    searchable = get_searchable_tables(db_path)
    table_options = list(searchable.keys())
    with st.expander("Search controls", expanded=True):
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            rag_query = st.text_input("Search question or keywords", value=st.session_state.get("rag_query", "high risk churn complaints irbid"))
            st.session_state.rag_query = rag_query
        with c2:
            selected_tables = st.multiselect("Search categories", options=table_options, default=[])
            require_all = st.checkbox("Match every keyword", value=False)
        with c3:
            limit_per_table = st.slider("Results per category", 3, 20, 8)
            max_results = st.slider("Maximum results", 10, 60, 30)

    search_clicked = st.button("Search database evidence", type="primary", use_container_width=True)
    if search_clicked or "rag_results" not in st.session_state:
        results_df = rag_search_database(
            db_path=db_path,
            query=rag_query,
            selected_tables=tuple(selected_tables),
            limit_per_table=int(limit_per_table),
            max_results=int(max_results),
            require_all_terms=bool(require_all),
        )
        st.session_state.rag_results = results_df
    else:
        results_df = st.session_state.rag_results

    if results_df.empty:
        st.info("No evidence found. Try fewer keywords, turn off 'Match every keyword', or search more categories.")
        return

    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">Retrieved evidence</div>
            <div class="page-copy">Found {len(results_df):,} matching records. Results are ranked by keyword and phrase matches, then used as context for the optional answer.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if api_key:
        with st.expander("Generate answer from retrieved evidence", expanded=False):
            rag_question = st.text_area("Question to answer", value=rag_query, height=90)
            if st.button("Generate grounded answer", type="primary", use_container_width=True):
                context = build_rag_context(results_df)
                loading_slot = st.empty()
                with loading_slot.container():
                    st.markdown(
                        """
                        <div class="card-subtle">
                            <div class="loading-row">
                                <span>Generating grounded answer</span>
                                <span class="loading-dots"><span></span><span></span><span></span></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                try:
                    with st.spinner("Generating grounded answer..."):
                        answer = answer_with_rag(rag_question, context, model_name, api_key)
                    loading_slot.empty()
                    st.markdown("#### Grounded answer")
                    st.markdown(answer)
                except Exception as exc:
                    loading_slot.empty()
                    st.error(f"Answer generation failed: {type(exc).__name__}: {exc}")
    else:
        st.warning("AI answers are currently unavailable, so this page is showing search results only.")

    for idx, row in results_df.iterrows():
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-meta"><span class="source-badge">[{idx + 1}] {safe_html(row['table'])}</span><span>rowid {int(row['rowid'])}</span><span>score {int(row['score'])}</span></div>
                <div class="page-copy">{safe_html(row['snippet'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download retrieved evidence CSV", csv, "rag_evidence.csv", "text/csv", use_container_width=True)


def render_customer_360_page(db_path: str) -> None:
    render_hero(
        "Customer 360",
        "Find a customer and inspect the full context",
        "Search by name, email, phone, MSISDN, customer ID, city, or risk level. Use this page before contacting a customer or investigating churn.",
        "Customer lookup",
    )

    options = get_filter_options(db_path)
    with st.expander("Lookup filters", expanded=True):
        c1, c2, c3 = st.columns([1.6, 1, 1])
        with c1:
            term = st.text_input("Search term", placeholder="Name, email, phone, MSISDN, or customer ID")
        with c2:
            city = st.selectbox("City", options=[""] + options["cities"])
        with c3:
            risk = st.selectbox("Risk level", options=[""] + options["risk_levels"])

    results = search_customers(db_path, term=term, city=city or None, risk_level=risk or None, limit=30)
    if results.empty:
        st.info("No customers matched the selected filters.")
        return

    with st.container(border=True):
        render_section_title("Matching customers")
        st.dataframe(results, use_container_width=True, hide_index=True)

    labels = [
        f"{row.full_name} · ID {row.customer_id} · {row.city} · {row.risk_level or 'No risk'} · {row.msisdn or 'No MSISDN'}"
        for row in results.itertuples()
    ]
    selected_label = st.selectbox("Open customer profile", options=labels)
    selected_index = labels.index(selected_label)
    customer_id = int(results.iloc[selected_index]["customer_id"])
    profile = get_customer_profile(db_path, customer_id)
    summary = profile["summary"].iloc[0]

    render_metric_grid(
        [
            ("Customer", summary.get("full_name", "N/A"), f"ID {customer_id} · {summary.get('city', 'N/A')}"),
            ("Risk", summary.get("risk_level", "N/A"), f"Score {summary.get('churn_score', 'N/A')} · {summary.get('main_risk_reason', 'No reason')}"),
            ("Value", summary.get("value_segment", "N/A"), f"ARPU {summary.get('arpu_jod', 0)} JOD · 6M revenue {summary.get('total_revenue_6m_jod', 0)} JOD"),
            ("Recommended action", summary.get("recommended_action", "N/A"), "Churn model recommendation"),
        ]
    )

    tab_overview, tab_billing, tab_support, tab_devices = st.tabs(["Overview", "Billing", "Support & complaints", "Devices"])
    with tab_overview:
        with st.container(border=True):
            render_section_title("Customer profile")
            profile_fields = [
                "customer_id", "full_name", "nationality", "gender", "age_group", "city", "governorate", "customer_type",
                "customer_segment", "preferred_language", "email", "phone_number", "signup_date", "status",
            ]
            profile_df = pd.DataFrame([{"field": field, "value": summary.get(field)} for field in profile_fields])
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
        with st.container(border=True):
            render_section_title("Subscriptions")
            st.dataframe(profile["subscriptions"], use_container_width=True, hide_index=True)
    with tab_billing:
        with st.container(border=True):
            render_section_title("Recent invoices")
            st.dataframe(profile["invoices"], use_container_width=True, hide_index=True)
    with tab_support:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                render_section_title("Complaints")
                st.dataframe(profile["complaints"], use_container_width=True, hide_index=True)
        with right:
            with st.container(border=True):
                render_section_title("Support interactions")
                st.dataframe(profile["support"], use_container_width=True, hide_index=True)
    with tab_devices:
        with st.container(border=True):
            render_section_title("Devices")
            st.dataframe(profile["devices"], use_container_width=True, hide_index=True)


def render_prompt_library_page() -> None:
    render_hero(
        "Prompt Library",
        "Reusable telecom business questions",
        "Use these prompts as starting points. Clicking a prompt sends it to the AI Chat composer.",
        "Prompt accelerator",
    )
    for category, prompts in EXAMPLE_PROMPTS_BY_CATEGORY.items():
        with st.container(border=True):
            render_section_title(category)
            cols = st.columns(3)
            for idx, prompt in enumerate(prompts):
                with cols[idx % 3]:
                    if st.button(prompt, key=f"library_{category}_{idx}", use_container_width=True):
                        st.session_state.pending_prompt = prompt
                        st.session_state.app_page = "AI Chat"
                        st.rerun()


def render_help_page() -> None:
    render_hero(
        "Help",
        "How to use Customer 360 AI",
        "Use this guide to pick the right page, ask better questions, and interpret the main customer and revenue signals.",
        "User guide",
    )
    st.markdown(
        """
        <div class="card">
            <div class="section-title">Recommended workflow</div>
            <div class="page-copy">
                1. Start with <strong>Executive Analytics</strong> to narrow down a segment, city, risk level, or campaign.<br>
                2. Use <strong>Evidence Search</strong> to retrieve supporting records before asking for a grounded answer.<br>
                3. Use <strong>AI Chat</strong> for broader analysis and interpretation.<br>
                4. Use <strong>Customer 360</strong> when you need to investigate one customer before a retention or support action.<br>
                5. Use <strong>Prompt Library</strong> for ready-made questions when you need a starting point.
            </div>
        </div>
        <div class="card">
            <div class="section-title">What each page is for</div>
            <div class="page-copy">
                <strong>AI Chat</strong> answers business questions in natural language.<br>
                <strong>Executive Analytics</strong> shows trends, risks, campaign results, billing status, and customer slices.<br>
                <strong>Evidence Search</strong> finds matching records before generating a grounded answer.<br>
                <strong>Customer 360</strong> opens a single customer's profile, billing, support, and device history.<br>
                <strong>Prompt Library</strong> gives reusable questions for churn, revenue, campaigns, support, and network analysis.
            </div>
        </div>
        <div class="card">
            <div class="section-title">How to interpret key signals</div>
            <div class="page-copy">
                <strong>Churn risk</strong> highlights customers who may need retention attention.<br>
                <strong>ARPU</strong> shows average monthly revenue and helps prioritize value segments.<br>
                <strong>Billing status</strong> helps identify collection pressure and overdue exposure.<br>
                <strong>Campaign conversion</strong> shows which offers are working best for each audience.<br>
                <strong>Support sentiment and complaints</strong> point to experience issues that can affect loyalty.
            </div>
        </div>
        <div class="card">
            <div class="section-title">Example questions</div>
            <div class="page-copy">
                Which cities have the most high-risk churn customers?<br>
                Which campaign audience should we retarget next?<br>
                Which billing statuses need immediate attention?<br>
                What complaints are most common among high-value customers?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Main app
# -----------------------------

def init_session() -> None:
    defaults = {
        "messages": [],
        "pending_prompt": "",
        "app_page": "AI Chat",
        "scroll_to_latest": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    valid_pages = {page for page, _icon, _desc in APP_PAGES}
    if st.session_state.app_page not in valid_pages:
        st.session_state.app_page = "AI Chat"


def main() -> None:
    init_session()

    st.markdown(build_css(), unsafe_allow_html=True)
    st.markdown(
        """
        <span id="drawer-open"></span>
        <span id="drawer-closed"></span>
        <a class="drawer-toggle close" href="#drawer-closed" title="Close workspace" aria-label="Close workspace">‹</a>
        <a class="drawer-toggle open" href="#drawer-open" title="Open workspace" aria-label="Open workspace">›</a>
        """,
        unsafe_allow_html=True,
    )

    default_db = str(APP_DB_PATH.resolve())
    api_key = get_api_key_from_sources()
    model_name = DEFAULT_MODEL_NAME
    active_page = render_sidebar()

    if active_page == "AI Chat":
        render_chat_page(api_key, default_db, model_name)
    elif active_page == "Executive Analytics":
        render_dynamic_analytics_page(default_db, model_name, api_key)
    elif active_page == "Evidence Search":
        render_rag_search_page(default_db, model_name, api_key)
    elif active_page == "Customer 360":
        render_customer_360_page(default_db)
    elif active_page == "Prompt Library":
        render_prompt_library_page()
    else:
        render_help_page()


if __name__ == "__main__":
    main()
