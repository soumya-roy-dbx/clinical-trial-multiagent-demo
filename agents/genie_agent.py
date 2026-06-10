"""Genie agent — structured/tabular questions.

Live mode: calls a Databricks Genie Space via the SDK with
**per-question conversation rotation** — every turn opens a fresh
conversation, so context never accumulates and the call never hits
REQUEST_LIMIT_EXCEEDED / CONTEXT_EXCEEDED_EXCEPTION. Tradeoff: no
follow-up conversational context, which fits the supervisor pattern
(each user turn is treated independently anyway).

Mock mode: dispatches against canned pandas queries in `modules.genie_mock`
(retained as fallback when env vars are unset or the live call fails).
"""
from __future__ import annotations

import json
import sys

from agents.base import AgentResponse, BaseAgent
from modules import genie_mock
from modules.config import GENIE_SPACE_ID, USE_LIVE_GENIE
from modules.observability import trace


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

    @trace(span_type="AGENT", name="genie_agent")
    def handle(self, question: str) -> AgentResponse:
        if self.live and self._client is not None:
            try:
                return self._handle_live(question)
            except Exception as exc:
                # Log to stderr (lands in `databricks apps logs`) then fall
                # back to mock so the app never returns blank.
                print(
                    json.dumps({
                        "genie_agent_fallback": {
                            "exception": type(exc).__name__,
                            "message": str(exc)[:300],
                        }
                    }),
                    file=sys.stderr, flush=True,
                )
        return self._handle_mock(question)

    # ---------- Live: per-question conversation, no context buildup ----------

    def _handle_live(self, question: str) -> AgentResponse:
        # Open a fresh conversation per question. This is the conversation
        # rotation strategy: simple, robust, never hits the context cap.
        conv = self._client.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID, content=question
        )
        status = conv.status.value if hasattr(conv.status, "value") else str(conv.status)
        if status != "COMPLETED":
            raise RuntimeError(f"Genie message status: {status}")

        # Prefer the text attachment for natural-language answers; capture SQL
        # and tabular result when present.
        answer_text = ""
        sql = None
        data_rows = None
        for att in (conv.attachments or []):
            if att.text and not answer_text:
                answer_text = att.text.content
            if att.query and att.query.query and not sql:
                sql = att.query.query

        # If Genie returned only a query attachment (no text), fetch the
        # result and turn the first row into a one-liner.
        if not answer_text and sql:
            result = self._client.genie.get_message_query_result(
                space_id=GENIE_SPACE_ID,
                conversation_id=conv.conversation_id,
                message_id=conv.message_id,
            )
            if result.statement_response and result.statement_response.result:
                data_rows = result.statement_response.result.data_array
                if data_rows:
                    answer_text = f"Query result: {data_rows[0]}"

        if not answer_text:
            answer_text = "Genie returned no answer."

        return AgentResponse(
            answer=answer_text,
            source="genie",
            confidence=0.9,
            data=data_rows,
            sql=sql,
        )

    # ---------- Mock: pandas over CSV ----------

    def _handle_mock(self, question: str) -> AgentResponse:
        out = genie_mock.answer(question)
        return AgentResponse(
            answer=out["answer"],
            source="genie",
            confidence=out.get("confidence", 0.5),
            data=out.get("value"),
        )
