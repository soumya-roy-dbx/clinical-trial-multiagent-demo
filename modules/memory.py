"""Lakebase-backed short-term chat memory.

Persists chat turns to a Lakebase Autoscaling Postgres database so
context survives page reloads, then loads the last N turns at the
start of each new turn so the supervisor can answer follow-ups
coherently.

Connection model
----------------
- Project / branch / endpoint identifiers come from env vars
  (LAKEBASE_PROJECT, LAKEBASE_BRANCH, LAKEBASE_ENDPOINT) so we can
  rebuild the endpoint resource path without an extra API call.
- LAKEBASE_HOST is the cached endpoint hostname (captured once during
  provisioning and pinned in `app.yaml` so we don't pay an API
  roundtrip on every cold start).
- LAKEBASE_DATABASE is the target Postgres database within the
  endpoint (we use a dedicated DB named `chat_memory`, not the
  default `postgres` which has a restricted public schema).
- Auth: short-lived OAuth tokens minted via the Lakebase
  `generateDatabaseCredential` REST endpoint. Tokens last 1 hour;
  this module refreshes the connection if more than 50 minutes have
  elapsed since the last mint.

Fallback
--------
- If any env var is unset, psycopg2 is missing, or the initial connect
  fails, `is_active` stays False and `load_recent_turns` /
  `store_turn` become no-ops. The app stays functional with just
  `st.session_state` (in-tab history pre-reload).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from dataclasses import dataclass

try:
    import psycopg2  # type: ignore
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False

from modules.config import (
    LAKEBASE_BRANCH,
    LAKEBASE_DATABASE,
    LAKEBASE_ENDPOINT,
    LAKEBASE_HOST,
    LAKEBASE_PROJECT,
)

# Tokens last 60 min; refresh the connection at 50 min to avoid mid-query expiry.
_TOKEN_TTL_SAFETY_S = 50 * 60


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  UUID PRIMARY KEY,
    started_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    last_turn_at     TIMESTAMP NOT NULL DEFAULT NOW(),
    session_meta     JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chat_turns (
    turn_id          BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    turn_index       INT NOT NULL,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    route            TEXT,
    confidence       REAL,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (conversation_id, turn_index, role)
);

CREATE INDEX IF NOT EXISTS chat_turns_conv_idx
    ON chat_turns (conversation_id, turn_index);
"""


@dataclass
class ChatTurn:
    role: str               # 'user' | 'assistant'
    content: str
    route: str | None = None
    confidence: float | None = None


class ChatMemory:
    """Short-term chat memory backed by Lakebase Postgres.

    Lifecycle: construct once per Streamlit session (cache via
    `@st.cache_resource`). The instance holds a single psycopg2
    connection that we refresh when the OAuth token nears expiry.
    """

    def __init__(self) -> None:
        self.is_active: bool = False
        self._conn = None
        self._token_obtained_at: float = 0.0
        if not _PSYCOPG2_AVAILABLE or not _all_env_set():
            return
        try:
            self._conn = self._open_connection()
            self.is_active = True
        except Exception as exc:
            self._log_failure("init", exc)

    # ---------- Public API ----------

    def ensure_conversation(self, conversation_id: str) -> None:
        """Idempotently register a conversation row before storing turns."""
        if not self.is_active:
            return
        try:
            self._exec(
                "INSERT INTO conversations (conversation_id) VALUES (%s) "
                "ON CONFLICT DO NOTHING",
                (conversation_id,),
            )
        except Exception as exc:
            self._log_failure("ensure_conversation", exc)

    def load_recent_turns(
        self, conversation_id: str, k: int = 6,
    ) -> list[ChatTurn]:
        """Return the most recent k turns for the conversation, oldest first."""
        if not self.is_active:
            return []
        try:
            rows = self._fetch(
                "SELECT role, content, route, confidence FROM chat_turns "
                "WHERE conversation_id = %s "
                "ORDER BY turn_index DESC LIMIT %s",
                (conversation_id, k),
            )
        except Exception as exc:
            self._log_failure("load_recent_turns", exc)
            return []
        return [ChatTurn(*r) for r in reversed(rows)]

    def store_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        route: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Append one turn to the conversation. Computes turn_index server-side."""
        if not self.is_active:
            return
        try:
            next_idx = self._fetch(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM chat_turns "
                "WHERE conversation_id = %s",
                (conversation_id,),
            )[0][0]
            self._exec(
                "INSERT INTO chat_turns "
                "(conversation_id, turn_index, role, content, route, confidence) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (conversation_id, next_idx, role, content, route, confidence),
            )
            self._exec(
                "UPDATE conversations SET last_turn_at = NOW() "
                "WHERE conversation_id = %s",
                (conversation_id,),
            )
        except Exception as exc:
            self._log_failure("store_turn", exc)

    def init_schema(self) -> None:
        """One-shot DDL apply. Run once during workspace provisioning."""
        if not self.is_active:
            raise RuntimeError(
                "ChatMemory inactive — verify LAKEBASE_* env vars and that "
                "psycopg2-binary is installed."
            )
        with self._conn.cursor() as cur:
            cur.execute(SCHEMA_DDL)

    # ---------- Internals ----------

    def _open_connection(self):
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        token = self._generate_token(w)
        email = w.current_user.me().user_name
        self._token_obtained_at = time.time()
        conn = psycopg2.connect(
            host=LAKEBASE_HOST,
            port=5432,
            database=LAKEBASE_DATABASE,
            user=email,
            password=token,
            sslmode="require",
        )
        conn.autocommit = True
        return conn

    def _generate_token(self, w) -> str:
        path = (
            f"/api/2.0/postgres/projects/{LAKEBASE_PROJECT}"
            f"/branches/{LAKEBASE_BRANCH}"
            f"/endpoints/{LAKEBASE_ENDPOINT}"
            ":generateDatabaseCredential"
        )
        req = urllib.request.Request(
            f"{w.config.host}{path}",
            data=b"{}",
            method="POST",
            headers={
                **w.config.authenticate(),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
        return payload["token"]

    def _maybe_refresh(self) -> None:
        if time.time() - self._token_obtained_at > _TOKEN_TTL_SAFETY_S:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._open_connection()

    def _exec(self, sql: str, params: tuple) -> None:
        self._maybe_refresh()
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _fetch(self, sql: str, params: tuple) -> list[tuple]:
        self._maybe_refresh()
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _log_failure(self, op: str, exc: Exception) -> None:
        print(
            json.dumps({
                f"chat_memory_{op}_failed": {
                    "exception": type(exc).__name__,
                    "message": str(exc)[:200],
                }
            }),
            file=sys.stderr,
            flush=True,
        )


def _all_env_set() -> bool:
    return all([
        LAKEBASE_PROJECT, LAKEBASE_BRANCH, LAKEBASE_ENDPOINT,
        LAKEBASE_HOST, LAKEBASE_DATABASE,
    ])


def format_context_for_prompt(turns: list[ChatTurn]) -> str:
    """Render loaded turns as plain text for injection into the supervisor's
    router and fusion prompts. Returns empty string when there's no history
    so the caller can conditionally skip the section."""
    if not turns:
        return ""
    lines = ["Prior conversation context (oldest first):"]
    for t in turns:
        speaker = "User" if t.role == "user" else "Assistant"
        lines.append(f"  {speaker}: {t.content}")
    return "\n".join(lines)
