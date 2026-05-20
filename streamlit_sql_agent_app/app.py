from __future__ import annotations

import os
import re
import sys
import tomllib
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for import_path in (APP_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from zain_jordan_sql_agent import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    create_zain_sql_agent,
    list_tables,
    run_sql,
    run_sql_agent,
    table_row_counts,
    connect_sqlite,
)


APP_DB_PATH = APP_DIR / "zain_customer_360_ai_demo.db"


MODEL_OPTIONS = [
    DEFAULT_MODEL_NAME,
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

st.set_page_config(
    page_title="Zain Jordan SQL Agent",
    page_icon="Z",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    color-scheme: light;
    --page: #f8fafc;
    --canvas: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e2e8f0;
    --soft-line: #edf2f7;
    --brand: #059669;
    --brand-dark: #047857;
    --brand-soft: #ecfdf5;
    --drawer: #0f172a;
    --drawer-panel: rgba(255,255,255,0.06);
    --drawer-line: rgba(255,255,255,0.12);
    --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    --shadow-sm: 0 8px 22px rgba(15, 23, 42, 0.06);
    --radius: 18px;
    --radius-sm: 12px;
    --drawer-width: 340px;
}

#MainMenu,
footer,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

.stApp {
    background: var(--page);
    color: var(--ink);
    color-scheme: light;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"] {
    color-scheme: light !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stMain"] .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
.main .block-container {
    max-width: 1180px;
    padding-top: 0 !important;
    padding-right: 2rem !important;
    padding-bottom: 2.4rem !important;
    padding-left: 2rem !important;
    margin-top: 0 !important;
    transition: max-width 220ms ease, padding 220ms ease;
}

.main div[data-testid="stElementContainer"]:has(style),
.main div[data-testid="stElementContainer"]:has(.drawer-toggle) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

.main div[data-testid="stMarkdownContainer"]:has(style),
.main div[data-testid="stMarkdownContainer"]:has(.drawer-toggle) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

section[data-testid="stSidebar"] {
    width: var(--drawer-width) !important;
    min-width: var(--drawer-width) !important;
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 16px 0 40px rgba(15, 23, 42, 0.20);
    transition: transform 260ms ease, margin-left 260ms ease, opacity 220ms ease;
    z-index: 999;
}

section[data-testid="stSidebar"] > div {
    padding: 1.1rem 0.95rem 1.25rem;
}

.drawer-toggle {
    position: fixed;
    top: 5.6rem;
    left: var(--toggle-left);
    z-index: 1200;
    width: 2.35rem;
    height: 2.35rem;
    display: grid;
    place-items: center;
    border-radius: 999px;
    border: 1px solid rgba(5,150,105,0.24);
    background: #ffffff;
    color: var(--brand-dark) !important;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.18);
    text-decoration: none !important;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1;
    transition: left 260ms ease, transform 180ms ease, background 180ms ease, box-shadow 180ms ease;
}

.drawer-toggle:hover {
    background: var(--brand-soft);
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(16, 24, 39, 0.22);
}

[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #f8fafc;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(248,250,252,0.72);
}

.drawer-title {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0.1rem 0 0.15rem;
}

.drawer-mark {
    width: 2.15rem;
    height: 2.15rem;
    border-radius: 11px;
    display: grid;
    place-items: center;
    color: #ffffff;
    background: linear-gradient(135deg, #059669, #22c55e);
    font-weight: 800;
    box-shadow: 0 10px 24px rgba(5, 150, 105, 0.20);
}

.drawer-title h2 {
    margin: 0;
    font-size: 1.05rem;
    line-height: 1.1;
}

.drawer-subtitle {
    color: rgba(248,250,252,0.64);
    font-size: 0.78rem;
    line-height: 1.35;
    margin: 0.4rem 0 0.85rem;
}

.drawer-section {
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--drawer-line);
}

.drawer-section-title {
    color: rgba(248,250,252,0.72);
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin: 0 0 0.48rem;
}

.nav-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 0.12rem 0;
}

.drawer-page-note {
    color: rgba(248,250,252,0.66);
    font-size: 0.8rem;
    line-height: 1.45;
    margin: 0.2rem 0 0.8rem;
}

.sidebar-note {
    color: rgba(248,250,252,0.62);
    font-size: 0.74rem;
    margin-top: -0.1rem;
    margin-bottom: 0.65rem;
}

[data-testid="stSidebar"] label p {
    color: rgba(248,250,252,0.76) !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #182033 !important;
}

[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #f8fafc !important;
    border-color: rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    min-height: 2.35rem !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] svg {
    color: #182033 !important;
    fill: #182033 !important;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.45rem;
}

.mini-card {
    border: 1px solid var(--drawer-line);
    background: var(--drawer-panel);
    border-radius: 12px;
    padding: 0.62rem 0.55rem;
    min-width: 0;
}

.mini-label {
    font-size: 0.66rem;
    color: rgba(255,255,255,0.62);
    white-space: nowrap;
}

.mini-value {
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 780;
    margin-top: 0.18rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.table-list {
    border: 1px solid var(--drawer-line);
    border-radius: 13px;
    background: rgba(255,255,255,0.045);
    overflow: hidden;
}

.table-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.075);
}

.table-row:last-child {
    border-bottom: 0;
}

.table-name {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: rgba(248,250,252,0.90);
    font-size: 0.78rem;
    font-weight: 680;
}

.table-count {
    flex: 0 0 auto;
    color: rgba(248,250,252,0.62);
    font-size: 0.73rem;
    font-variant-numeric: tabular-nums;
}

[data-testid="stSidebar"] div.stButton > button {
    background: transparent;
    color: #f8fafc;
    border: 1px solid transparent;
    min-height: 2.22rem;
    height: auto;
    padding: 0.42rem 0.58rem;
    border-radius: 10px;
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    font-weight: 660;
    font-size: 0.86rem;
}

[data-testid="stSidebar"] div.stButton > button *,
[data-testid="stSidebar"] div.stButton > button p {
    color: #f8fafc !important;
    opacity: 1 !important;
    text-align: left !important;
    width: 100%;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    border-color: rgba(34,197,94,0.20);
    background: rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: rgba(5,150,105,0.22) !important;
    border-color: rgba(34,197,94,0.36) !important;
    color: #ffffff !important;
    box-shadow: inset 3px 0 0 #22c55e !important;
}

[data-testid="stSidebar"] div.stButton > button[kind="primary"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button {
    justify-content: flex-start;
    text-align: left;
    min-height: 2.2rem;
    padding: 0.42rem 0.58rem;
}

[data-testid="stSidebar"] div.stButton > button[data-testid="baseButton-secondary"]:focus {
    box-shadow: 0 0 0 3px rgba(54,196,147,0.16);
}

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.16rem 1.05rem;
    margin: 0 0 0.7rem;
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: rgba(255,255,255,0.92);
    box-shadow: var(--shadow-sm);
    text-align: left;
}

.brand-stack {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: 0.22rem;
    min-width: 0;
}

.brand-title {
    margin: 0;
    font-size: 1.5rem !important;
    line-height: 1.2 !important;
    font-weight: 820;
    letter-spacing: 0;
    color: var(--ink);
}

.brand-subtitle {
    color: var(--muted);
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.4;
}

.status-pill {
    border: 1px solid rgba(5,150,105,0.22);
    background: var(--brand-soft);
    color: var(--brand-dark);
    padding: 0.44rem 0.72rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    white-space: nowrap;
    margin-top: 0;
}

.content-card {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: #ffffff;
    box-shadow: var(--shadow-sm);
    padding: 1.15rem;
    margin-bottom: 0.9rem;
}

.welcome-copy {
    max-width: 760px;
}

.welcome-copy h2 {
    color: var(--ink);
    font-size: 1.35rem;
    margin: 0 0 0.24rem;
    letter-spacing: 0;
}

.welcome-copy p {
    color: var(--muted);
    margin: 0;
    font-size: 0.94rem;
    line-height: 1.5;
}

.chat-frame {
    min-height: 180px;
    padding: 0.1rem 0 0.3rem;
}

.message-row {
    display: flex;
    margin: 0.65rem 0;
}

.message-row.user {
    justify-content: flex-end;
}

.message-row.assistant {
    justify-content: flex-start;
}

.bubble {
    max-width: min(78%, 760px);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.86rem 0.98rem;
    line-height: 1.58;
    font-size: 0.94rem;
    box-shadow: var(--shadow-sm);
}

.bubble.user {
    background: #dcfce7;
    border-color: rgba(5,150,105,0.18);
    border-top-right-radius: 6px;
    color: #064e3b;
}

.bubble.assistant {
    background: #ffffff;
    border-top-left-radius: 6px;
}

.loading-bubble {
    min-width: 13rem;
}

.loading-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    color: #344054;
    font-weight: 650;
}

.loading-dots {
    display: inline-flex;
    gap: 0.25rem;
}

.loading-dots span {
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 999px;
    background: var(--brand);
    animation: agentPulse 1s infinite ease-in-out;
}

.loading-dots span:nth-child(2) {
    animation-delay: 0.15s;
}

.loading-dots span:nth-child(3) {
    animation-delay: 0.3s;
}

@keyframes agentPulse {
    0%, 80%, 100% {
        opacity: 0.28;
        transform: translateY(0);
    }
    40% {
        opacity: 1;
        transform: translateY(-3px);
    }
}

.message-label {
    display: block;
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 800;
    margin-bottom: 0.32rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.agent-answer {
    color: #273247;
}

.agent-answer .section-heading {
    display: block;
    margin: 0.2rem 0 0.22rem;
    color: var(--ink);
    font-weight: 780;
}

.agent-answer .bullet-line {
    display: block;
    margin-left: 0.55rem;
}

.composer-title {
    color: var(--ink);
    font-size: 0.82rem;
    font-weight: 800;
    margin: 0.4rem 0 0.42rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

div[data-testid="stForm"] {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: #ffffff;
    box-shadow: var(--shadow-sm);
    padding: 0.82rem;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 14px;
    border: 1px solid var(--line);
    min-height: 4.6rem;
    color: var(--ink) !important;
    background: #ffffff !important;
    padding: 0.9rem 1rem !important;
    line-height: 1.45;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(5,150,105,0.55) !important;
    box-shadow: 0 0 0 3px rgba(5,150,105,0.10) !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #667085 !important;
    opacity: 1 !important;
}

div.stButton > button,
div[data-testid="stForm"] button {
    border-radius: 12px;
    min-height: 2.6rem;
    font-weight: 720;
}

.main div.stButton > button,
div[data-testid="stForm"] button {
    border: 1px solid var(--line);
    background: #ffffff;
    color: var(--ink);
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035);
}

.main div.stButton > button:hover,
div[data-testid="stForm"] button:hover {
    border-color: rgba(22,138,98,0.42);
    color: var(--brand-dark);
    background: #f7fffb;
}

button[kind="primary"] {
    background: var(--brand) !important;
    color: #ffffff !important;
    border-color: var(--brand) !important;
}

button[kind="primary"]:hover {
    background: var(--brand-dark) !important;
    border-color: var(--brand-dark) !important;
}

.suggestion-title {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 800;
    margin: 0.9rem 0 0.48rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.prompt-chip {
    display: inline-block;
    border: 1px solid var(--line);
    background: #ffffff;
    color: #334155;
    border-radius: 999px;
    padding: 0.38rem 0.66rem;
    margin: 0.16rem;
    font-size: 0.8rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}

.empty-state {
    margin-top: 0.6rem;
}

.table-meta {
    color: rgba(248,250,252,0.66);
    font-size: 0.78rem;
    margin-top: 0.45rem;
}

.help-list {
    color: rgba(248,250,252,0.72);
    font-size: 0.82rem;
    line-height: 1.55;
}

.page-kicker {
    color: var(--brand-dark);
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.page-title {
    color: var(--ink);
    font-size: 1.32rem;
    font-weight: 800;
    margin: 0;
}

.page-copy {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.5;
    margin-top: 0.25rem;
}

.sql-warning {
    border: 1px solid rgba(22,138,98,0.18);
    background: var(--brand-soft);
    color: var(--brand-dark);
    border-radius: 12px;
    padding: 0.7rem 0.85rem;
    font-size: 0.86rem;
    margin: 0.75rem 0;
}

.analytics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin-bottom: 0.85rem;
}

.analytics-card {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: #ffffff;
    box-shadow: var(--shadow-sm);
    padding: 0.9rem;
}

.analytics-label {
    color: #667085;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.analytics-value {
    color: var(--ink);
    font-size: 1.45rem;
    font-weight: 820;
    margin-top: 0.2rem;
}

.analytics-caption {
    color: #667085;
    font-size: 0.8rem;
    margin-top: 0.16rem;
}

.chart-card {
    border: 1px solid var(--line);
    border-radius: var(--radius);
    background: #ffffff;
    box-shadow: var(--shadow-sm);
    padding: 0.85rem;
    margin-bottom: 0.85rem;
}

.chart-title {
    color: var(--ink);
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

[data-testid="stSidebar"] [data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

@media (max-width: 900px) {
    :root { --drawer-width: 316px; }
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {
        padding-top: 0 !important;
        padding-right: 0.85rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.85rem !important;
    }
    .app-header { align-items: flex-start; flex-direction: column; padding: 0.16rem 1.05rem; }
    .status-pill { align-self: flex-start; }
    .brand-title { font-size: 1.25rem !important; }
    .metric-strip { grid-template-columns: 1fr; }
    .analytics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .bubble { max-width: 92%; }
}
</style>
"""

EXAMPLE_PROMPTS = [
    "How many customers are in the database?",
    "Show the top 10 cities by number of customers.",
    "Which cities have the most high-risk churn customers?",
    "Which campaigns had the highest conversion rate?",
]

APP_PAGES = [
    "Chat Agent",
    "Analytics",
    "Manual SQL",
    "Data Browser",
    "Prompt Library",
    "Help",
]

PAGE_LABELS = {
    "Chat Agent": "Chat Agent",
    "Analytics": "Analytics",
    "Manual SQL": "Manual SQL",
    "Data Browser": "Data Browser",
    "Prompt Library": "Prompt Library",
    "Help": "Help",
}


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
def get_database_summary(db_path: str) -> tuple[int, int, pd.DataFrame]:
    conn = connect_sqlite(db_path)
    tables = list_tables(conn)
    counts = table_row_counts(conn)
    total_customers = 0
    customer_rows = counts.loc[counts["table_name"] == "customers", "row_count"]
    if not customer_rows.empty:
        total_customers = int(customer_rows.iloc[0])
    return len(tables), total_customers, counts.head(8)


@st.cache_data(show_spinner=False)
def get_table_counts_df(db_path: str):
    conn = connect_sqlite(db_path)
    return table_row_counts(conn)


@st.cache_data(show_spinner=False)
def get_table_preview(db_path: str, table_name: str, limit: int = 5):
    counts = get_table_counts_df(db_path)
    valid_tables = set(counts["table_name"])
    if table_name not in valid_tables:
        return None
    conn = connect_sqlite(db_path)
    safe_table = table_name.replace('"', '""')
    return pd.read_sql_query(f'SELECT * FROM "{safe_table}" LIMIT ?', conn, params=(limit,))


@st.cache_data(show_spinner=False)
def get_analytics_data(db_path: str) -> dict[str, pd.DataFrame]:
    conn = connect_sqlite(db_path)
    queries = {
        "kpis": """
            SELECT
                (SELECT COUNT(*) FROM customers) AS total_customers,
                (SELECT COUNT(*) FROM customer_churn_scores WHERE risk_level = 'High') AS high_risk_customers,
                (SELECT ROUND(AVG(arpu_jod), 2) FROM customer_value_segments) AS avg_arpu_jod,
                (SELECT ROUND(SUM(total_amount_jod), 2) FROM invoices) AS total_invoiced_jod,
                (SELECT COUNT(*) FROM complaints) AS total_complaints,
                (SELECT COUNT(*) FROM support_interactions WHERE customer_sentiment = 'Negative') AS negative_support_cases
        """,
        "city_customers": """
            SELECT city, COUNT(*) AS total_customers
            FROM customers
            GROUP BY city
            ORDER BY total_customers DESC
            LIMIT 10
        """,
        "churn_mix": """
            SELECT risk_level, COUNT(*) AS customers, ROUND(AVG(churn_score), 3) AS avg_churn_score
            FROM customer_churn_scores
            GROUP BY risk_level
            ORDER BY CASE risk_level WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
        """,
        "high_risk_city": """
            SELECT c.city, COUNT(*) AS high_risk_customers
            FROM customers c
            JOIN customer_churn_scores ch ON c.customer_id = ch.customer_id
            WHERE ch.risk_level = 'High'
            GROUP BY c.city
            ORDER BY high_risk_customers DESC
            LIMIT 10
        """,
        "value_segments": """
            SELECT
                value_segment,
                COUNT(*) AS customers,
                ROUND(AVG(arpu_jod), 2) AS avg_arpu_jod,
                ROUND(AVG(total_revenue_6m_jod), 2) AS avg_revenue_6m_jod
            FROM customer_value_segments
            GROUP BY value_segment
            ORDER BY avg_revenue_6m_jod DESC
        """,
        "complaints": """
            SELECT complaint_category, severity, COUNT(*) AS complaints
            FROM complaints
            GROUP BY complaint_category, severity
            ORDER BY complaints DESC
        """,
        "support_sentiment": """
            SELECT channel, customer_sentiment, COUNT(*) AS interactions
            FROM support_interactions
            GROUP BY channel, customer_sentiment
        """,
        "campaigns": """
            SELECT
                c.campaign_name,
                c.campaign_type,
                c.target_segment,
                COUNT(r.response_id) AS total_sent,
                SUM(r.converted_flag) AS total_converted,
                ROUND(100.0 * SUM(r.converted_flag) / COUNT(r.response_id), 2) AS conversion_rate
            FROM campaigns c
            JOIN customer_campaign_responses r ON c.campaign_id = r.campaign_id
            GROUP BY c.campaign_id, c.campaign_name, c.campaign_type, c.target_segment
            ORDER BY conversion_rate DESC
            LIMIT 10
        """,
        "billing": """
            SELECT payment_status, COUNT(*) AS invoices, ROUND(SUM(total_amount_jod), 2) AS total_amount_jod
            FROM invoices
            GROUP BY payment_status
            ORDER BY invoices DESC
        """,
    }
    return {name: pd.read_sql_query(query, conn) for name, query in queries.items()}


@st.cache_resource(show_spinner=False)
def get_agent(db_path: str, model_name: str, api_key: str):
    return create_zain_sql_agent(
        db_path=db_path,
        model_name=model_name,
        openai_api_key=api_key,
    )


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = ""


def submit_prompt(prompt: str, db_path: str, model_name: str, api_key: str) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    loading_slot = st.empty()
    with loading_slot.container():
        render_loading_message()
    with st.spinner("Running SQL analysis..."):
        try:
            agent = get_agent(db_path, model_name, api_key)
            answer = run_sql_agent(agent, clean_prompt)
        except Exception as exc:
            answer = (
                "I could not complete the request.\n\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
    loading_slot.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.pending_prompt = ""


def format_message_html(content: str, role: str) -> str:
    safe_content = escape(content).replace("\n", "<br>")
    if role == "assistant":
        safe_content = re.sub(
            r"(?:(?:<br>)?)(\\d+\\.\\s[^:<]+:)",
            r'<br><span class="section-heading">\\1</span>',
            safe_content,
        )
        safe_content = re.sub(
            r"(<br>[-•]\\s[^<]+)",
            r'<span class="bullet-line">\\1</span>',
            safe_content,
        )
        return f'<div class="agent-answer">{safe_content}</div>'
    return safe_content


def render_message(role: str, content: str) -> None:
    safe_role = "user" if role == "user" else "assistant"
    safe_content = format_message_html(content, safe_role)
    label = "You" if safe_role == "user" else "SQL Agent"
    st.markdown(
        f"""
        <div class="message-row {safe_role}">
            <div class="bubble {safe_role}">
                <span class="message-label">{label}</span>
                {safe_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_message() -> None:
    st.markdown(
        """
        <div class="message-row assistant">
            <div class="bubble assistant loading-bubble">
                <span class="message-label">SQL Agent</span>
                <div class="loading-row">
                    <span>Analyzing the database</span>
                    <span class="loading-dots"><span></span><span></span><span></span></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_composer(welcome: bool = False):
    st.markdown('<div class="composer-title">Ask the database</div>', unsafe_allow_html=True)
    with st.form("sql_agent_form", clear_on_submit=False):
        prompt = st.text_area(
            "Message",
            value=st.session_state.pending_prompt,
            placeholder="Example: Which cities have the most high-risk churn customers? Show the top 10.",
            label_visibility="collapsed",
        )
        col_submit, col_clear = st.columns([1, 1])
        submitted = col_submit.form_submit_button("Submit", use_container_width=True, type="primary")
        cleared = col_clear.form_submit_button("Clear", use_container_width=True)
    return prompt, submitted, cleared


def render_main_suggestions() -> None:
    st.markdown('<div class="suggestion-title">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[index % 2]:
            if st.button(prompt, key=f"main_prompt_{index}", use_container_width=True):
                st.session_state.pending_prompt = prompt
                st.rerun()


def render_app_nav() -> str:
    if "app_page" not in st.session_state:
        st.session_state.app_page = APP_PAGES[0]

    st.markdown('<div class="drawer-section-title">Navigation</div>', unsafe_allow_html=True)
    for index, page in enumerate(APP_PAGES):
        label = PAGE_LABELS[page]
        is_active = st.session_state.app_page == page
        if st.button(label, key=f"app_nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.app_page = page
            st.rerun()
        if index < len(APP_PAGES) - 1:
            st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    return st.session_state.app_page


def render_workspace_overview(table_count: int, customer_count: int, model_name: str, top_tables: pd.DataFrame) -> None:
    st.markdown('<div class="drawer-section-title">Database overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="mini-card"><div class="mini-label">Tables</div><div class="mini-value">{table_count}</div></div>
            <div class="mini-card"><div class="mini-label">Customers</div><div class="mini-value">{customer_count}</div></div>
            <div class="mini-card"><div class="mini-label">Model</div><div class="mini-value">{model_name}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_tables(db_path: str, counts_df: pd.DataFrame) -> None:
    st.markdown('<div class="drawer-section-title">Table explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="drawer-page-note">Inspect table sizes and preview records before asking the agent a question.</div>',
        unsafe_allow_html=True,
    )
    table_name = st.selectbox("Preview table", options=counts_df["table_name"].tolist())
    row_count = int(counts_df.loc[counts_df["table_name"] == table_name, "row_count"].iloc[0])
    st.markdown(f'<div class="table-meta">{row_count:,} rows · first 5 records</div>', unsafe_allow_html=True)
    preview_df = get_table_preview(db_path, table_name)
    if preview_df is not None:
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


def render_workspace_prompts() -> None:
    st.markdown('<div class="drawer-section-title">Quick prompts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="drawer-page-note">Click one to load it into the chat input, then submit when ready.</div>',
        unsafe_allow_html=True,
    )
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=f"workspace_prompt_{prompt}", use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.rerun()


def render_workspace_help() -> None:
    st.markdown('<div class="drawer-section-title">How to ask</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="help-list">
        <strong>Good questions:</strong><br>
        - Ask for counts, rankings, comparisons, and trends.<br>
        - Mention the segment, city, risk level, or table when known.<br>
        - Ask for a business interpretation after the numbers.<br><br>
        <strong>Examples:</strong><br>
        - Which city has the highest churn risk?<br>
        - Compare campaign conversion by segment.<br>
        - Show complaint severity by category.<br>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_page(api_key_input: str, db_path: str, model_name: str) -> None:
    prompt = ""
    submitted = False
    cleared = False

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="content-card">
            <div class="welcome-copy">
                <div class="page-kicker">Chat agent</div>
                <h2>Start with a business question</h2>
                <p>Ask about churn, revenue, campaigns, complaints, support sentiment, or any table in the bundled Customer 360 database.</p>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prompt, submitted, cleared = render_composer(welcome=True)
        render_main_suggestions()
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <span class="prompt-chip">churn by city</span>
                    <span class="prompt-chip">campaign conversion</span>
                    <span class="prompt-chip">complaint severity</span>
                    <span class="prompt-chip">customer segments</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-frame">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            render_message(message["role"], message["content"])
        st.markdown("</div>", unsafe_allow_html=True)
        prompt, submitted, cleared = render_composer()

    if cleared:
        reset_chat()
        st.rerun()

    if submitted:
        if not api_key_input:
            st.error("Add an OpenAI API key from the left sidebar configuration section or set OPENAI_API_KEY before starting the app.")
        else:
            submit_prompt(prompt, db_path, model_name, api_key_input)
            st.rerun()


def render_manual_sql_page(db_path: str) -> None:
    st.markdown(
        """
        <div class="content-card">
            <div class="page-kicker">Manual SQL</div>
            <h2 class="page-title">Run a read-only SQL query</h2>
            <div class="page-copy">Write SQLite SELECT queries directly against the bundled Customer 360 database.</div>
            <div class="sql-warning">Only SELECT and WITH queries are allowed. Database modification statements are blocked.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    default_query = "SELECT city, COUNT(*) AS total_customers\nFROM customers\nGROUP BY city\nORDER BY total_customers DESC\nLIMIT 10;"
    query = st.text_area("SQL query", value=st.session_state.get("manual_sql", default_query), height=180)
    st.session_state.manual_sql = query
    run_clicked = st.button("Run SQL", type="primary", use_container_width=True)
    if run_clicked:
        try:
            conn = connect_sqlite(db_path)
            result_df = run_sql(conn, query)
            st.success(f"Query returned {len(result_df):,} rows.")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Query failed: {type(exc).__name__}: {exc}")


def render_analytics_page(db_path: str) -> None:
    data = get_analytics_data(db_path)
    kpis = data["kpis"].iloc[0]
    high_risk_rate = (kpis["high_risk_customers"] / kpis["total_customers"]) * 100
    top_city = data["city_customers"].iloc[0]
    top_campaign = data["campaigns"].iloc[0]
    top_complaint = data["complaints"].groupby("complaint_category", as_index=False)["complaints"].sum().sort_values("complaints", ascending=False).iloc[0]
    negative_support = data["support_sentiment"][data["support_sentiment"]["customer_sentiment"] == "Negative"]
    top_negative_channel = negative_support.sort_values("interactions", ascending=False).iloc[0]

    st.markdown(
        """
        <div class="content-card">
            <div class="page-kicker">Analytics</div>
            <h2 class="page-title">Customer 360 executive dashboard</h2>
            <div class="page-copy">A curated view of customer distribution, churn exposure, revenue value, complaints, support sentiment, campaigns, and billing health.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="analytics-grid">
            <div class="analytics-card"><div class="analytics-label">Customers</div><div class="analytics-value">{int(kpis["total_customers"]):,}</div><div class="analytics-caption">Active customer records</div></div>
            <div class="analytics-card"><div class="analytics-label">High Risk</div><div class="analytics-value">{int(kpis["high_risk_customers"]):,}</div><div class="analytics-caption">{high_risk_rate:.1f}% of customers</div></div>
            <div class="analytics-card"><div class="analytics-label">Avg ARPU</div><div class="analytics-value">{float(kpis["avg_arpu_jod"]):.2f} JOD</div><div class="analytics-caption">Customer value segments</div></div>
            <div class="analytics-card"><div class="analytics-label">Invoiced</div><div class="analytics-value">{float(kpis["total_invoiced_jod"]):,.0f} JOD</div><div class="analytics-caption">All invoices</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="content-card">
            <div class="page-kicker">Key insights</div>
            <div class="page-copy">
                <strong>{top_city["city"]}</strong> is the largest customer market with <strong>{int(top_city["total_customers"]):,}</strong> customers.
                High-risk churn exposure is currently <strong>{int(kpis["high_risk_customers"]):,}</strong> customers, or <strong>{high_risk_rate:.1f}%</strong> of the base.
                The largest complaint category is <strong>{top_complaint["complaint_category"]}</strong> with <strong>{int(top_complaint["complaints"]):,}</strong> complaints.
                The top campaign by conversion is <strong>{top_campaign["campaign_name"]}</strong> at <strong>{float(top_campaign["conversion_rate"]):.1f}%</strong>.
                Negative support sentiment is highest in <strong>{top_negative_channel["channel"]}</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    green_scale = alt.Scale(range=["#0f6849", "#168a62", "#36c493", "#9be7c7"])
    risk_colors = alt.Scale(domain=["High", "Medium", "Low"], range=["#d64545", "#f2a93b", "#168a62"])

    left, right = st.columns(2)
    with left:
        st.markdown('#### Top Cities by Customers')
        city_chart = (
            alt.Chart(data["city_customers"])
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                y=alt.Y("city:N", sort="-x", title=None),
                x=alt.X("total_customers:Q", title="Customers"),
                color=alt.value("#168a62"),
                tooltip=["city", "total_customers"],
            )
            .properties(height=300)
        )
        st.altair_chart(city_chart, use_container_width=True)

    with right:
        st.markdown('#### Churn Risk Mix')
        churn_chart = (
            alt.Chart(data["churn_mix"])
            .mark_arc(innerRadius=58, outerRadius=108)
            .encode(
                theta=alt.Theta("customers:Q"),
                color=alt.Color("risk_level:N", scale=risk_colors, title="Risk"),
                tooltip=["risk_level", "customers", "avg_churn_score"],
            )
            .properties(height=300)
        )
        st.altair_chart(churn_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('#### High-Risk Customers by City')
        high_risk_chart = (
            alt.Chart(data["high_risk_city"])
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                y=alt.Y("city:N", sort="-x", title=None),
                x=alt.X("high_risk_customers:Q", title="High-risk customers"),
                color=alt.value("#d64545"),
                tooltip=["city", "high_risk_customers"],
            )
            .properties(height=300)
        )
        st.altair_chart(high_risk_chart, use_container_width=True)

    with right:
        st.markdown('#### Value Segments: ARPU vs Revenue')
        value_chart = (
            alt.Chart(data["value_segments"])
            .mark_circle(size=220, opacity=0.86)
            .encode(
                x=alt.X("avg_arpu_jod:Q", title="Avg ARPU (JOD)"),
                y=alt.Y("avg_revenue_6m_jod:Q", title="Avg 6M revenue (JOD)"),
                color=alt.Color("value_segment:N", scale=green_scale, title="Segment"),
                size=alt.Size("customers:Q", title="Customers"),
                tooltip=["value_segment", "customers", "avg_arpu_jod", "avg_revenue_6m_jod"],
            )
            .properties(height=300)
        )
        st.altair_chart(value_chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('#### Complaints by Category and Severity')
        complaint_chart = (
            alt.Chart(data["complaints"])
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("complaint_category:N", sort="-x", title=None),
                x=alt.X("sum(complaints):Q", title="Complaints"),
                color=alt.Color("severity:N", title="Severity"),
                tooltip=["complaint_category", "severity", "complaints"],
            )
            .properties(height=330)
        )
        st.altair_chart(complaint_chart, use_container_width=True)

    with right:
        st.markdown('#### Support Sentiment by Channel')
        support_chart = (
            alt.Chart(data["support_sentiment"])
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("channel:N", sort="-x", title=None),
                x=alt.X("sum(interactions):Q", title="Interactions"),
                color=alt.Color("customer_sentiment:N", title="Sentiment"),
                tooltip=["channel", "customer_sentiment", "interactions"],
            )
            .properties(height=330)
        )
        st.altair_chart(support_chart, use_container_width=True)

    st.markdown('#### Top Campaign Conversion Rates')
    campaign_chart = (
        alt.Chart(data["campaigns"])
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            y=alt.Y("campaign_name:N", sort="-x", title=None),
            x=alt.X("conversion_rate:Q", title="Conversion rate (%)"),
            color=alt.Color("campaign_type:N", title="Type"),
            tooltip=["campaign_name", "campaign_type", "target_segment", "total_sent", "total_converted", "conversion_rate"],
        )
        .properties(height=360)
    )
    st.altair_chart(campaign_chart, use_container_width=True)

    st.markdown('#### Billing Status')
    billing_chart = (
        alt.Chart(data["billing"])
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusTopLeft=6)
        .encode(
            x=alt.X("payment_status:N", title=None),
            y=alt.Y("invoices:Q", title="Invoices"),
            color=alt.value("#168a62"),
            tooltip=["payment_status", "invoices", "total_amount_jod"],
        )
        .properties(height=260)
    )
    st.altair_chart(billing_chart, use_container_width=True)
    st.dataframe(data["billing"], use_container_width=True, hide_index=True)


def render_data_browser_page(db_path: str) -> None:
    st.markdown(
        """
        <div class="content-card">
            <div class="page-kicker">Data browser</div>
            <h2 class="page-title">Explore database tables</h2>
            <div class="page-copy">Inspect row counts and preview records before asking the agent for analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    counts_df = get_table_counts_df(db_path)
    st.dataframe(counts_df, use_container_width=True, hide_index=True)
    table_name = st.selectbox("Preview table", options=counts_df["table_name"].tolist(), key="main_table_preview")
    preview_df = get_table_preview(db_path, table_name, limit=20)
    if preview_df is not None:
        st.dataframe(preview_df, use_container_width=True, hide_index=True)


def render_prompt_library_page() -> None:
    st.markdown(
        """
        <div class="content-card">
            <div class="page-kicker">Prompt library</div>
            <h2 class="page-title">Reusable business questions</h2>
            <div class="page-copy">Load a suggested prompt into the chat input, then switch to Chat Agent to run it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=f"library_prompt_{prompt}", use_container_width=True):
            st.session_state.pending_prompt = prompt
            st.session_state.app_page = "Chat Agent"
            st.rerun()


def render_help_page() -> None:
    st.markdown(
        """
        <div class="content-card">
            <div class="page-kicker">Help</div>
            <h2 class="page-title">How to use the SQL Agent</h2>
            <div class="page-copy">
            Ask for counts, rankings, comparisons, and business interpretation. For manual SQL, use SELECT or WITH queries only.
            Good prompts mention a city, segment, risk level, table, or time period when possible.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        - Use **Chat Agent** for natural-language analysis.
        - Use **Manual SQL** when you already know the exact query.
        - Use **Data Browser** to inspect table names and columns.
        - Use the left sidebar configuration section for API key, model, and database path.
        """
    )


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "settings_open" not in st.session_state:
        st.session_state.settings_open = True
    if "app_page" not in st.session_state:
        st.session_state.app_page = APP_PAGES[0]

    requested_drawer_state = st.query_params.get("settings")
    if requested_drawer_state == "open":
        st.session_state.settings_open = True
    elif requested_drawer_state == "closed":
        st.session_state.settings_open = False

    toggle_left = "calc(var(--drawer-width) - 1.08rem)" if st.session_state.settings_open else "0.7rem"
    toggle_icon = "‹" if st.session_state.settings_open else "›"
    toggle_label = "Close workspace" if st.session_state.settings_open else "Open workspace"
    next_state = "closed" if st.session_state.settings_open else "open"

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    if not st.session_state.settings_open:
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] {
                transform: translateX(calc(-1 * var(--drawer-width)));
                margin-left: calc(-1 * var(--drawer-width));
            }
            .main .block-container {
                max-width: 1240px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <style>
        :root {{ --toggle-left: {toggle_left}; }}
        </style>
        <a class="drawer-toggle" href="?settings={next_state}" target="_self" title="{toggle_label}" aria-label="{toggle_label}">
            {toggle_icon}
        </a>
        """,
        unsafe_allow_html=True,
    )

    default_db = APP_DB_PATH.resolve()
    api_key_from_sources = get_api_key_from_sources()
    api_key_input = api_key_from_sources
    db_path = str(default_db)
    st.markdown(
        """
        <div class="app-header">
            <div class="brand-stack">
                <h1 class="brand-title">Zain Jordan SQL Agent</h1>
                <div class="brand-subtitle">Chat with the Customer 360 database using a guarded SQL agent.</div>
            </div>
            <div class="status-pill">Read-only SQL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="drawer-title">
                <div class="drawer-mark">Z</div>
                <h2>SQL Agent</h2>
            </div>
            <div class="drawer-subtitle">Customer 360 assistant for analysis, manual SQL, and data exploration.</div>
            """,
            unsafe_allow_html=True,
        )
        active_page = render_app_nav()

        st.markdown('<div class="drawer-section"><div class="drawer-section-title">Configuration</div></div>', unsafe_allow_html=True)
        if api_key_from_sources:
            st.caption("API key loaded from saved configuration.")
        else:
            st.caption("No API key configured.")
        model_name = st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL_NAME),
        )

        try:
            table_count, customer_count, top_tables = get_database_summary(db_path)
            st.markdown('<div class="drawer-section"></div>', unsafe_allow_html=True)
            render_workspace_overview(table_count, customer_count, model_name, top_tables)
        except Exception as exc:
            st.error(f"Database check failed: {exc}")
    if active_page == "Chat Agent":
        render_chat_page(api_key_input, db_path, model_name)
    elif active_page == "Analytics":
        render_analytics_page(db_path)
    elif active_page == "Manual SQL":
        render_manual_sql_page(db_path)
    elif active_page == "Data Browser":
        render_data_browser_page(db_path)
    elif active_page == "Prompt Library":
        render_prompt_library_page()
    else:
        render_help_page()


if __name__ == "__main__":
    main()
