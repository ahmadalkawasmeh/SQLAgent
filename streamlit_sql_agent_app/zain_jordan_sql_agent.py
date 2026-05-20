"""Reusable SQL agent utilities for the Zain Jordan Customer 360 database.

This module mirrors the goal of ``zain_jordan_class_3_sql_agent.ipynb``:
connect to the SQLite Customer 360 database, inspect its schema, create a
LangChain SQL toolkit, and answer business questions through a guarded SQL
agent.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


DEFAULT_DB_NAME = "zain_customer_360_ai_demo.db"
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_TOP_K = 5


SQL_AGENT_SYSTEM_PROMPT = """
You are Customer 360 AI, a senior telecom business analyst for Zain Jordan.

You are connected to a SQLite telecom Customer 360 database. Your audience is business users, not engineers.

Your job:
- Answer business questions using database evidence.
- Inspect tables and schemas before writing SQL.
- Generate syntactically correct SQLite queries.
- Double-check SQL queries before execution.
- Execute queries only after checking them.
- Translate the result into crisp business language with practical next steps.

Important safety rules:
- Only use SELECT queries.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE.
- Do not modify the database.
- Do not query all columns from a table unless absolutely necessary.
- Unless the user requests a specific number, limit query results to at most {top_k} rows.
- Use relevant columns only.
- If the question is ambiguous, explain your assumption.
- If a table or column does not exist, inspect schema and correct the query.
- Do not guess facts that are not in the database.
- Do not expose tool logs, raw SQL, table internals, or debugging details unless the user asks for them.

Useful telecom context:
- Churn analysis usually involves customers and customer_churn_scores.
- Revenue analysis usually involves customer_value_segments, invoices, payments, or transactions.
- Complaint analysis usually involves complaints and support_interactions.
- Campaign analysis usually involves campaigns and customer_campaign_responses.
- Network analysis usually involves network_towers and network_events.

Default answer format:
Direct answer:
- Start with the answer in one or two plain sentences.

Key numbers:
- Use 2-5 bullets with the most important figures, rankings, percentages, or totals.
- Include units such as customers, JOD, percent, invoices, complaints, or interactions.

Business meaning:
- Explain what the result means for churn, revenue, customer experience, campaign performance, or operations.

Recommended action:
- Give one practical next step. If the data is insufficient for a recommendation, say what additional slice to check.

Keep the answer concise. Prefer bullets over long paragraphs. Use exact numbers from the query result.
"""


DEMO_QUESTIONS = [
    "How many customers are in the database?",
    "Show the top 10 cities by number of customers.",
    "How many customers are high, medium, and low churn risk?",
    "Which cities have the most high-risk churn customers? Show the top 10 cities.",
    "Which customer value segments have the highest average ARPU and six-month revenue?",
]


def configure_openai_api_key(openai_api_key: str | None = None) -> str:
    """Set and return the OpenAI API key used by LangChain.

    Pass ``openai_api_key`` explicitly or set ``OPENAI_API_KEY`` in the
    environment before importing/running this module.
    """
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment or pass "
            "openai_api_key to configure_openai_api_key/create_sql_agent."
        )
    return api_key


def locate_database(db_path: str | Path | None = None, search_dir: str | Path = ".") -> Path:
    """Return the SQLite database path used by the workshop notebooks."""
    if db_path:
        resolved = Path(db_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Database file not found: {resolved}")
        return resolved

    search_path = Path(search_dir).expanduser().resolve()
    preferred = search_path / DEFAULT_DB_NAME
    if preferred.exists():
        return preferred

    db_files = sorted(search_path.glob("*.db"))
    if db_files:
        return db_files[0].resolve()

    raise FileNotFoundError(f"No .db file found in {search_path}")


def connect_sqlite(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Create a SQLite connection suitable for notebook and agent usage."""
    resolved = locate_database(db_path)
    return sqlite3.connect(str(resolved), check_same_thread=False)


def list_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    """List database tables."""
    return pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """,
        conn,
    )


def table_row_counts(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return row counts for every table in the database."""
    rows = []
    for table_name in list_tables(conn)["name"]:
        count_query = f'SELECT COUNT(*) AS row_count FROM "{table_name}"'
        count = pd.read_sql_query(count_query, conn)["row_count"][0]
        rows.append({"table_name": table_name, "row_count": int(count)})
    return pd.DataFrame(rows).sort_values("row_count", ascending=False)


def run_sql(conn: sqlite3.Connection, query: str, params: Iterable[object] | None = None) -> pd.DataFrame:
    """Run a read-only SQL query and return a DataFrame."""
    normalized = query.strip().lower()
    if not normalized.startswith("select") and not normalized.startswith("with"):
        raise ValueError("Only SELECT/WITH read-only queries are allowed.")
    return pd.read_sql_query(query, conn, params=tuple(params or ()))


def create_sql_database(db_path: str | Path | None = None) -> SQLDatabase:
    """Create LangChain's SQLDatabase wrapper for the SQLite DB."""
    resolved = locate_database(db_path)
    return SQLDatabase.from_uri(f"sqlite:///{resolved}")


def create_chat_model(
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0,
    openai_api_key: str | None = None,
):
    """Create the OpenAI chat model used by the SQL agent."""
    configure_openai_api_key(openai_api_key)
    return init_chat_model(
        model_name,
        model_provider="openai",
        temperature=temperature,
    )


def create_sql_tools(db: SQLDatabase, model) -> list:
    """Create SQL inspection/query tools for the agent."""
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    return toolkit.get_tools()


def create_zain_sql_agent(
    db_path: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
    openai_api_key: str | None = None,
):
    """Create the guarded Zain Jordan SQL agent."""
    db = create_sql_database(db_path)
    model = create_chat_model(model_name=model_name, openai_api_key=openai_api_key)
    sql_tools = create_sql_tools(db, model)
    system_prompt = SQL_AGENT_SYSTEM_PROMPT.format(top_k=top_k)
    return create_agent(model=model, tools=sql_tools, system_prompt=system_prompt)


def extract_final_text(result) -> str:
    """Extract readable final text from a LangChain agent result."""
    last_message = result["messages"][-1]

    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        return last_message.content

    if hasattr(last_message, "content_blocks"):
        parts = []
        for block in last_message.content_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(block["text"])
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    return str(last_message)


def run_sql_agent(agent, question: str) -> str:
    """Run the SQL agent and return only the final answer text."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return extract_final_text(result)


def run_sql_agent_verbose(agent, question: str) -> None:
    """Stream and print agent messages for teaching/debugging."""
    print("USER QUESTION:")
    print(question)
    print("=" * 100)

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        message = step["messages"][-1]
        try:
            message.pretty_print()
        except Exception:
            print(message)
        print("-" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zain Jordan SQL agent.")
    parser.add_argument("--db", default=None, help="Path to the SQLite database.")
    parser.add_argument("--question", default=DEMO_QUESTIONS[0], help="Business question to ask the SQL agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="OpenAI model name.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key. Prefer OPENAI_API_KEY env var.")
    parser.add_argument("--schema-only", action="store_true", help="Only print table names and row counts.")
    args = parser.parse_args()

    db_path = locate_database(args.db)
    conn = connect_sqlite(db_path)
    print(f"Database path: {db_path}")
    print(f"Number of tables: {len(list_tables(conn))}")
    print(table_row_counts(conn).to_string(index=False))

    if args.schema_only:
        return

    agent = create_zain_sql_agent(
        db_path=db_path,
        model_name=args.model,
        openai_api_key=args.api_key,
    )
    print()
    print("Question:")
    print(args.question)
    print()
    print("Answer:")
    print(run_sql_agent(agent, args.question))


if __name__ == "__main__":
    main()
