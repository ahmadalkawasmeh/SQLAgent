# Zain Jordan SQL Agent Streamlit App

ChatGPT-style Streamlit UI for `zain_jordan_sql_agent.py`.

## Local Run

From the repo root:

```bash
export OPENAI_API_KEY="your-key"
pip install -r streamlit_sql_agent_app/requirements.txt
streamlit run streamlit_sql_agent_app/app.py
```

The app includes and defaults to its own database copy:

```text
streamlit_sql_agent_app/zain_customer_360_ai_demo.db
```

## Deployment Notes

For Streamlit Community Cloud or similar platforms, set the API key as a secret:

```toml
OPENAI_API_KEY = "your-key"
```

Do not commit real API keys to this folder.

For local use, this app can also read:

```text
streamlit_sql_agent_app/.streamlit/secrets.toml
```

That file is ignored by git in this app folder.
