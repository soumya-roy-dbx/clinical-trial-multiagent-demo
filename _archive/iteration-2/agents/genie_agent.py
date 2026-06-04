"""Genie agent — structured/tabular questions.

Live mode: calls a Databricks Genie Space via databricks-sdk.
Mock mode: dispatches against canned pandas queries in `modules.genie_mock`.
"""
from __future__ import annotations

from agents.base import AgentResponse, BaseAgent
from modules import genie_mock
from modules.config import GENIE_SPACE_ID, USE_LIVE_GENIE


class GenieAgent(BaseAgent):
    name = "genie"

    def __init__(self) -> None:
        self.live = USE_LIVE_GENIE
        self._client = None
        if self.live:
            try:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
            except Exception:
                self.live = False

    def handle(self, question: str) -> AgentResponse:
        if self.live and self._client is not None:
            return self._handle_live(question)
        return self._handle_mock(question)

    def _handle_mock(self, question: str) -> AgentResponse:
        out = genie_mock.answer(question)
        return AgentResponse(
            answer=out["answer"],
            source="genie",
            confidence=out.get("confidence", 0.5),
            data=out.get("value"),
        )

    def _handle_live(self, question: str) -> AgentResponse:
        # Sketch only — fill in for live mode.
        # 1. self._client.genie.start_conversation(space_id=GENIE_SPACE_ID, content=question)
        # 2. poll get_message_query_result for the conversation/message ids
        # 3. format result rows into `answer` and capture sql
        # NOTE: rotate conversation_id per question to avoid REQUEST_LIMIT_EXCEEDED.
        raise NotImplementedError(
            f"Live Genie call against space {GENIE_SPACE_ID} not implemented in demo."
        )
