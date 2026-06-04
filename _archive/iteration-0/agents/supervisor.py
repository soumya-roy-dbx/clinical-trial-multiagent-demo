"""Supervisor agent — intent classification + multi-source response fusion.

Live behavior uses Llama 3.1 70B via the Foundation Model API for two LLM
calls: (1) route the question, (2) fuse sub-agent answers into one paragraph.
If the FMAPI client is unavailable, falls back to keyword routing and string
concatenation (no synthesis).
"""
from __future__ import annotations

import json
from typing import Iterable

from agents.base import AgentResponse, BaseAgent
from agents.genie_agent import GenieAgent
from agents.vector_agent import VectorAgent
from modules.config import LLM_ENDPOINT

GENIE_KEYWORDS = {
    "subjects", "count", "incidence", "rate", "region", "biomarker",
    "treatment", "control", "arm", "how many",
}
VECTOR_KEYWORDS = {
    "protocol", "criteria", "inclusion", "exclusion", "endpoint", "endpoints",
    "crf", "definition", "defined", "synopsis",
}


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self) -> None:
        self.genie = GenieAgent()
        self.vector = VectorAgent()
        self._client = None
        try:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        except Exception:
            self._client = None

    # ---------- Routing ----------

    def _keyword_classify(self, question: str) -> str:
        q = question.lower()
        g = sum(1 for kw in GENIE_KEYWORDS if kw in q)
        v = sum(1 for kw in VECTOR_KEYWORDS if kw in q)
        if g and v:
            return "hybrid"
        if v > g:
            return "vector"
        return "genie"

    def _llm_classify(self, question: str) -> dict:
        if self._client is None:
            raise RuntimeError("LLM client unavailable")
        prompt = (
            "Classify the user's clinical-trial question into one of: "
            "'genie' (structured/tabular), 'vector' (protocol/CRF text), "
            "or 'hybrid' (needs both). Respond with JSON: "
            '{"route": "...", "reasoning": "..."}.\n\nQuestion: ' + question
        )
        raw = self._call_llm(prompt, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"route": self._keyword_classify(question), "reasoning": "json parse fallback"}

    # ---------- Fusion ----------

    def _llm_fuse(self, question: str, parts: Iterable[AgentResponse]) -> str:
        if self._client is None:
            raise RuntimeError("LLM client unavailable")
        parts = list(parts)
        prompt = (
            "You are synthesizing answers from two clinical-trial assistants. "
            "Produce one concise paragraph that combines them.\n\n"
            f"Question: {question}\n\n"
            + "\n\n".join(f"[{p.source}] {p.answer}" for p in parts)
        )
        return self._call_llm(prompt, temperature=0.2)

    # ---------- LLM call ----------

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        # Live path would post to: self._client.serving_endpoints.query(
        #   name=LLM_ENDPOINT, messages=[...], temperature=temperature)
        # Stubbed in the demo to keep mock mode runnable without creds.
        raise RuntimeError(f"LLM endpoint {LLM_ENDPOINT} call stubbed in demo.")

    # ---------- Entry point ----------

    def handle(self, question: str) -> AgentResponse:
        try:
            decision = self._llm_classify(question)
            route = decision.get("route", "genie")
        except Exception:
            route = self._keyword_classify(question)

        if route == "genie":
            return self.genie.handle(question)
        if route == "vector":
            return self.vector.handle(question)

        # hybrid
        g = self.genie.handle(question)
        v = self.vector.handle(question)
        try:
            fused = self._llm_fuse(question, [g, v])
            return AgentResponse(answer=fused, source="hybrid",
                                 confidence=(g.confidence + v.confidence) / 2,
                                 data=g.data, chunks=v.chunks)
        except Exception:
            stacked = f"[genie] {g.answer}\n\n[vector] {v.answer}"
            return AgentResponse(answer=stacked, source="hybrid",
                                 confidence=(g.confidence + v.confidence) / 2,
                                 data=g.data, chunks=v.chunks)
