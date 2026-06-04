"""Centralized configuration read from environment variables."""
from __future__ import annotations

import os

PAGE_TITLE = "Subject Disposition — Demo Study"
TREATMENT_LABEL = "Treatment"
BIOMARKER_POSITIVE = "Positive"

WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
GENIE_SPACE_ID = os.getenv("DATABRICKS_GENIE_SPACE_ID", "")
LLM_ENDPOINT = os.getenv(
    "SUPERVISOR_LLM_ENDPOINT", "databricks-meta-llama-3-1-70b-instruct"
)
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "")
VECTOR_SEARCH_INDEX = os.getenv("VECTOR_SEARCH_INDEX", "")

USE_LIVE_GENIE = bool(GENIE_SPACE_ID)
USE_LIVE_VECTOR_SEARCH = bool(VECTOR_SEARCH_ENDPOINT and VECTOR_SEARCH_INDEX)
USE_MULTI_AGENT = os.getenv("USE_MULTI_AGENT", "true").lower() == "true"

LAKEBASE_FEEDBACK_URL = os.getenv("LAKEBASE_FEEDBACK_URL", "")
