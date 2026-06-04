# Multi-Agent Clinical Trial Analytics App (Demo Mirror)

A faithful but minimal mirror of the customer's `app-one-page-demo-v2/` layout,
runnable end-to-end in **mock mode** so we can iterate on improvements before
flipping the live toggles.

## What this is

A Streamlit app on Databricks Apps that shows a Subject Disposition dashboard
plus an AI-powered chat sidebar. The sidebar is driven by a Supervisor →
Sub-Agent pattern:

- **Supervisor** — routes questions (Genie / Vector / Hybrid) and fuses
  multi-source answers. Stubbed LLM call falls through to keyword routing +
  string concatenation in mock mode.
- **Genie agent** — structured/tabular questions. Mock mode dispatches against
  canned pandas queries over `data/study_subjects.csv`.
- **Vector agent** — protocol/CRF questions via RAG. Mock mode uses pure-Python
  TF-IDF over `data/docs/*.md`.

## Run locally

```bash
pip install -r requirements.txt
python data/mock_data.py     # (re)generate the CSV
streamlit run app.py
```

Everything works without any Databricks credentials in mock mode.

## Going live

1. Fill in `app.yaml` env vars (`DATABRICKS_WAREHOUSE_ID`,
   `DATABRICKS_GENIE_SPACE_ID`, `VECTOR_SEARCH_ENDPOINT`,
   `VECTOR_SEARCH_INDEX`).
2. Uncomment and complete the `resources:` block in `app.yaml`.
3. Implement the `_handle_live` methods in `agents/genie_agent.py` and
   `agents/vector_agent.py`, and the `_call_llm` body in
   `agents/supervisor.py`.

See `UPGRADE_PLAN.md` for the planned iterations.
