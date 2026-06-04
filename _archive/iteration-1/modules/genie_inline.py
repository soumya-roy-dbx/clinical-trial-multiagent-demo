"""Legacy live-Genie sidebar.

In production this would connect to a Genie Space via databricks-sdk, poll
conversation status, and log feedback to Lakebase. This demo file is a thin
stub — wired so the import path matches the customer's layout. The mock app
path uses `genie_mock` instead.
"""
from __future__ import annotations

from modules.config import GENIE_SPACE_ID


def ask_live_genie(question: str) -> dict:
    """Placeholder for the live Genie call.

    In live mode, use databricks.sdk.WorkspaceClient().genie.start_conversation
    or create_message + poll get_message_query_result.
    """
    if not GENIE_SPACE_ID:
        return {"answer": "GENIE_SPACE_ID not configured.", "confidence": 0.0}
    raise NotImplementedError(
        "Live Genie path is stubbed in the demo; supply GENIE_SPACE_ID and "
        "implement via databricks-sdk in agents/genie_agent.py."
    )
