"""Vector Search agent — unstructured protocol/CRF questions via RAG.

Iteration 2 architecture:
- Live mode: POST to the Databricks Vector Search REST API for top-k chunks,
  then synthesize an answer via the fusion LLM (Claude Sonnet 4.6) using
  the retrieved chunks as grounding context.
- Mock mode: TF-IDF over data/docs/*.md, returns the top chunk as-is.
  Triggered when env vars are unset OR any live call fails — the app
  always returns something.
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from agents.base import AgentResponse, BaseAgent
from modules.config import (
    FUSION_LLM_ENDPOINT,
    USE_LIVE_VECTOR_SEARCH,
    VECTOR_SEARCH_ENDPOINT,
    VECTOR_SEARCH_INDEX,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

_token = re.compile(r"[a-zA-Z]{2,}")

RAG_SYSTEM_PROMPT = (
    "You answer clinical-trial questions using ONLY the provided protocol "
    "and CRF excerpts. If the excerpts don't contain the answer, say so. "
    "Be concise — one short paragraph. Preserve clinical terminology."
)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _token.findall(text)]


class MockDocStore:
    def __init__(self, docs_dir: Path = DOCS_DIR) -> None:
        self.docs: list[dict] = []
        for p in sorted(docs_dir.glob("*.md")):
            text = p.read_text()
            self.docs.append({"id": p.stem, "text": text, "tokens": _tokenize(text)})
        self._build_idf()

    def _build_idf(self) -> None:
        n = max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d["tokens"]))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        for d in self.docs:
            tf = Counter(d["tokens"])
            length = max(sum(tf.values()), 1)
            d["tfidf"] = {t: (c / length) * self.idf.get(t, 0.0) for t, c in tf.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        q_tokens = _tokenize(query)
        q_tf = Counter(q_tokens)
        q_len = max(sum(q_tf.values()), 1)
        q_vec = {t: (c / q_len) * self.idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored: list[tuple[float, dict]] = []
        for d in self.docs:
            dot = sum(q_vec.get(t, 0.0) * v for t, v in d["tfidf"].items())
            d_norm = math.sqrt(sum(v * v for v in d["tfidf"].values())) or 1.0
            scored.append((dot / (q_norm * d_norm), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": d["id"], "score": float(score), "text": d["text"]}
            for score, d in scored[:k]
        ]


class VectorAgent(BaseAgent):
    name = "vector"

    def __init__(self) -> None:
        self.live = USE_LIVE_VECTOR_SEARCH
        self._store = None if self.live else MockDocStore()
        self._client = None
        if self.live:
            try:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
            except Exception:
                # Fall back to mock if SDK init fails for any reason.
                self.live = False
                self._store = MockDocStore()

    # ---------- Entry point ----------

    def handle(self, question: str) -> AgentResponse:
        if self.live and self._client is not None:
            try:
                return self._handle_live(question)
            except Exception as exc:
                # Surface the failure for debugging but never break the app.
                print(
                    json.dumps({
                        "vector_agent_fallback": {
                            "exception": type(exc).__name__,
                            "message": str(exc)[:300],
                        }
                    }),
                    file=sys.stderr, flush=True,
                )
                if self._store is None:
                    self._store = MockDocStore()
        return self._handle_mock(question)

    # ---------- Live: VS query + RAG synthesis ----------

    def _handle_live(self, question: str) -> AgentResponse:
        chunks = self._vs_search(question, k=3)
        if not chunks:
            return AgentResponse(
                answer="No relevant protocol/CRF passage found.",
                source="vector", confidence=0.0,
            )
        # Synthesize an answer from the retrieved chunks via the fusion LLM.
        answer = self._rag_synthesize(question, chunks)
        # Use the top retrieval score (clipped to [0, 1]) as a rough confidence.
        top_score = max(min(chunks[0]["score"], 1.0), 0.0)
        return AgentResponse(
            answer=answer,
            source="vector",
            confidence=top_score,
            chunks=chunks,
        )

    def _vs_search(self, question: str, k: int) -> list[dict]:
        host = self._client.config.host
        url = f"{host}/api/2.0/vector-search/indexes/{VECTOR_SEARCH_INDEX}/query"
        body = json.dumps({
            "query_text": question,
            "columns": ["chunk_id", "doc_type", "section", "chunk_text"],
            "num_results": k,
        }).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={**self._client.config.authenticate(), "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
        chunks: list[dict] = []
        for row in payload.get("result", {}).get("data_array", []):
            chunk_id, doc_type, section, text, score = row
            chunks.append({
                "id": chunk_id, "doc_type": doc_type, "section": section,
                "text": text, "score": float(score),
            })
        return chunks

    def _rag_synthesize(self, question: str, chunks: list[dict]) -> str:
        host = self._client.config.host
        url = f"{host}/ai-gateway/mlflow/v1/chat/completions"
        context = "\n\n".join(
            f"[{c['doc_type']} :: {c['section']}]\n{c['text']}" for c in chunks
        )
        user_msg = f"Question: {question}\n\nExcerpts:\n{context}"
        body = json.dumps({
            "model": FUSION_LLM_ENDPOINT,
            "messages": [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={**self._client.config.authenticate(), "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # If synthesis fails, return the top chunk text — partial result
            # is better than no result.
            return chunks[0]["text"][:800]

    # ---------- Mock: TF-IDF over local markdown ----------

    def _handle_mock(self, question: str) -> AgentResponse:
        chunks = self._store.search(question, k=3) if self._store else []
        if not chunks:
            return AgentResponse(
                answer="No relevant protocol/CRF passage found.",
                source="vector", confidence=0.0,
            )
        top = chunks[0]
        return AgentResponse(
            answer=top["text"].strip()[:800],
            source="vector",
            confidence=min(top["score"] * 2.0, 1.0),
            chunks=chunks,
        )
