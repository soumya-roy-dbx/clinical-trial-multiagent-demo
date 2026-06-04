"""App-level decision log for supervisor turns.

Captures app concepts that the AI Gateway inference table can't know:
which route the LLM picked, which route actually ran after fallbacks, why a
fallback fired, end-to-end app latency. JOIN-able to the AI Gateway
inference table on `request_id` for the full picture.

Behavior:
- Mock mode (no warehouse / no table configured): prints structured JSON to
  stdout. Visible via `databricks apps logs app-one-page-demo-v2`.
- Live mode (warehouse + table both set): INSERTs to UC via the SQL
  warehouse Statement Execution API.

Failures inside the logger are swallowed — logging must not break the app.
"""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from modules.config import DECISION_LOG_TABLE, WAREHOUSE_ID

# DDL for the target UC table. Run once when standing up live mode:
#
#   CREATE TABLE IF NOT EXISTS your_catalog.your_schema.supervisor_decisions (
#       request_id      STRING,
#       question        STRING,
#       llm_route       STRING,
#       final_route     STRING,
#       fallback_reason STRING,
#       confidence      DOUBLE,
#       response_chars  INT,
#       app_latency_ms  INT,
#       ts              TIMESTAMP
#   ) USING DELTA;
#
DDL = """\
CREATE TABLE IF NOT EXISTS {table} (
    request_id      STRING,
    question        STRING,
    llm_route       STRING,
    final_route     STRING,
    fallback_reason STRING,
    confidence      DOUBLE,
    response_chars  INT,
    app_latency_ms  INT,
    ts              TIMESTAMP
) USING DELTA
"""


@dataclass
class DecisionRecord:
    request_id: str
    question: str
    llm_route: Optional[str]
    final_route: str
    fallback_reason: Optional[str]
    confidence: float
    response_chars: int
    app_latency_ms: int
    ts: str  # ISO-8601 UTC


class DecisionLog:
    """Singleton-style logger. Construct once at app startup, call .log(...) per turn."""

    def __init__(self) -> None:
        self.live = bool(WAREHOUSE_ID and DECISION_LOG_TABLE)
        self._client = None
        if self.live:
            try:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
            except Exception:
                # WorkspaceClient unavailable — drop back to mock mode silently.
                self.live = False

    def log(self, record: DecisionRecord) -> None:
        if self.live and self._client is not None:
            self._log_live(record)
        else:
            self._log_mock(record)

    # ---------- Mock mode ----------

    def _log_mock(self, record: DecisionRecord) -> None:
        try:
            print(
                json.dumps({"decision_log": asdict(record)}),
                file=sys.stdout,
                flush=True,
            )
        except Exception:
            # Never let logging break the app.
            pass

    # ---------- Live mode ----------

    def _log_live(self, record: DecisionRecord) -> None:
        try:
            from databricks.sdk.service.sql import StatementParameterListItem

            params = [
                StatementParameterListItem(name=k, value=str(v) if v is not None else None)
                for k, v in asdict(record).items()
            ]
            stmt = (
                f"INSERT INTO {DECISION_LOG_TABLE} ("
                "request_id, question, llm_route, final_route, fallback_reason, "
                "confidence, response_chars, app_latency_ms, ts) VALUES ("
                ":request_id, :question, :llm_route, :final_route, :fallback_reason, "
                "CAST(:confidence AS DOUBLE), CAST(:response_chars AS INT), "
                "CAST(:app_latency_ms AS INT), CAST(:ts AS TIMESTAMP))"
            )
            self._client.statement_execution.execute_statement(
                statement=stmt,
                warehouse_id=WAREHOUSE_ID,
                parameters=params,
                wait_timeout="10s",
            )
        except Exception:
            # If the INSERT fails, fall back to mock logging so we don't lose the record.
            print(
                json.dumps({
                    "decision_log_insert_failed": asdict(record),
                    "error": traceback.format_exc(limit=2),
                }),
                file=sys.stderr,
                flush=True,
            )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
