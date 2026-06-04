"""Vector Search agent — unstructured protocol/CRF questions via RAG.

Live mode: queries a Databricks Vector Search index, then synthesizes via LLM.
Mock mode: loads data/docs/*.md, builds TF-IDF vectors, returns top-k chunks
(and either synthesizes via LLM if available, or returns the raw top chunk).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from agents.base import AgentResponse, BaseAgent
from modules.config import (
    USE_LIVE_VECTOR_SEARCH,
    VECTOR_SEARCH_ENDPOINT,
    VECTOR_SEARCH_INDEX,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

_token = re.compile(r"[a-zA-Z]{2,}")


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

    def handle(self, question: str) -> AgentResponse:
        if self.live:
            return self._handle_live(question)
        return self._handle_mock(question)

    def _handle_mock(self, question: str) -> AgentResponse:
        chunks = self._store.search(question, k=3) if self._store else []
        if not chunks:
            return AgentResponse(
                answer="No relevant protocol/CRF passage found.",
                source="vector",
                confidence=0.0,
            )
        top = chunks[0]
        return AgentResponse(
            answer=top["text"].strip()[:800],
            source="vector",
            confidence=min(top["score"] * 2.0, 1.0),
            chunks=chunks,
        )

    def _handle_live(self, question: str) -> AgentResponse:
        # Sketch only — fill in for live mode.
        # from databricks.vector_search.client import VectorSearchClient
        # vsc = VectorSearchClient()
        # idx = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VECTOR_SEARCH_INDEX)
        # results = idx.similarity_search(query_text=question, columns=["id","text"], num_results=3)
        raise NotImplementedError(
            f"Live VS call against {VECTOR_SEARCH_ENDPOINT}/{VECTOR_SEARCH_INDEX} not implemented in demo."
        )
