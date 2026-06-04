"""Centralized configuration read from environment variables."""
from __future__ import annotations

import os

PAGE_TITLE = "Subject Disposition — Demo Study"
TREATMENT_LABEL = "Treatment"
BIOMARKER_POSITIVE = "Positive"

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
GENIE_SPACE_ID = os.getenv(
    "DATABRICKS_GENIE_SPACE_ID", "your_genie_space_id"
)
LLM_ENDPOINT = os.getenv(
    "SUPERVISOR_LLM_ENDPOINT", "databricks-meta-llama-3-1-70b-instruct"
)

# Two-tier supervisor — a small/fast router and a stronger fusion model.
# Pay-per-token Foundation Model API endpoints by default; switch to a
# provisioned-throughput endpoint for production PHI workloads (HIPAA).
ROUTER_LLM_ENDPOINT = os.getenv(
    "ROUTER_LLM_ENDPOINT", "ai_gateway_for_databricks-meta-llama-3-3-70b-instruct"
)
FUSION_LLM_ENDPOINT = os.getenv(
    "FUSION_LLM_ENDPOINT", "ai_gateway_for_databricks-claude-sonnet-4-6"
)

VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "your_vs_endpoint")
VECTOR_SEARCH_INDEX = os.getenv(
    "VECTOR_SEARCH_INDEX",
    "your_catalog.your_schema.protocol_chunks_vs_index",
)

USE_LIVE_GENIE = bool(GENIE_SPACE_ID)
USE_LIVE_VECTOR_SEARCH = bool(VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX)
USE_MULTI_AGENT = os.getenv("USE_MULTI_AGENT", "true").lower() == "true"

LAKEBASE_FEEDBACK_URL = os.getenv("LAKEBASE_FEEDBACK_URL", "")

# Lakebase short-term chat memory. All five must be set for ChatMemory to
# activate; otherwise it falls back to a no-op so the app keeps working
# with just Streamlit session state.
LAKEBASE_PROJECT = os.getenv("LAKEBASE_PROJECT", "")
LAKEBASE_BRANCH = os.getenv("LAKEBASE_BRANCH", "production")
LAKEBASE_ENDPOINT = os.getenv("LAKEBASE_ENDPOINT", "primary")
LAKEBASE_HOST = os.getenv("LAKEBASE_HOST", "")
LAKEBASE_DATABASE = os.getenv("LAKEBASE_DATABASE", "chat_memory")
CHAT_CONTEXT_TURNS = int(os.getenv("CHAT_CONTEXT_TURNS", "6"))

# App-level decision log destination. Mock mode (writer prints structured
# JSON to stdout) when DATABRICKS_WAREHOUSE_ID is empty. When set together
# with a warehouse id, the writer INSERTs to this UC table.
DECISION_LOG_TABLE = os.getenv(
    "DECISION_LOG_TABLE", "your_catalog.your_schema.supervisor_decisions"
)
