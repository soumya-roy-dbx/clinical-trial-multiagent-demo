"""Supervisor agent — two-tier LLM routing + multi-source response fusion.

- ROUTER tier: small/fast model at temp=0 returns strict JSON {route, reasoning}.
  Validated by Pydantic; one corrective retry on parse/validation failure;
  falls through to the keyword classifier on second failure.
- FUSION tier: stronger model at temp=0.2 synthesizes Genie + Vector answers
  into one clinical paragraph for hybrid questions. Falls through to string
  concatenation on failure.

When the router/fusion endpoint env vars are unset, `_call_llm` raises and
the app runs end-to-end via the structured fallback paths (keyword routing
and string concatenation), so mock-mode development needs no credentials.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from agents.base import AgentResponse, BaseAgent
from agents.genie_agent import GenieAgent
from agents.vector_agent import VectorAgent
from modules.config import FUSION_LLM_ENDPOINT, ROUTER_LLM_ENDPOINT
from modules.decision_log import DecisionLog, DecisionRecord, now_iso
from modules.observability import set_trace_tags, trace

# ---------- Routing prompt & schema ----------

Route = Literal["genie", "vector", "hybrid"]


class RouterDecision(BaseModel):
    """Schema the router LLM must return. Pydantic validates the route enum."""

    route: Route = Field(description="Which agent should answer the question.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


# Few-shot examples are the single biggest accuracy lever after temp=0. One
# example per route class keeps the prompt short and unambiguous.
ROUTER_FEW_SHOT = [
    {
        "question": "How many subjects are in the Treatment group?",
        "route": "genie",
        "reasoning": "Structured count over the subject table.",
    },
    {
        "question": "What are the inclusion criteria?",
        "route": "vector",
        "reasoning": "Answer lives in the protocol document text.",
    },
    {
        "question": "What is the incidence of Event A in Treatment and how is Event A defined in the protocol?",
        "route": "hybrid",
        "reasoning": "Needs both a tabular incidence rate and the protocol definition.",
    },
]

ROUTER_SYSTEM_PROMPT = (
    "You route clinical-trial questions to one of three downstream agents.\n\n"
    "Return ONLY a JSON object matching this schema (no preamble, no markdown):\n"
    '{"route": "<genie|vector|hybrid>", "reasoning": "<one sentence>"}\n\n'
    "Definitions:\n"
    "- genie:  structured/tabular questions (counts, rates, breakdowns by group).\n"
    "- vector: protocol/CRF questions (definitions, criteria, endpoints).\n"
    "- hybrid: needs both a tabular answer AND a protocol explanation.\n\n"
    "Examples:\n"
    + "\n".join(
        f'Q: {ex["question"]}\nA: {{"route": "{ex["route"]}", "reasoning": "{ex["reasoning"]}"}}'
        for ex in ROUTER_FEW_SHOT
    )
)

FUSION_SYSTEM_PROMPT = (
    "You are a clinical-trial assistant. Combine the structured (Genie) and "
    "document (Vector) answers below into a single concise paragraph. Preserve "
    "exact numbers from Genie; keep clinical terminology from Vector. No "
    "preamble; output the paragraph only."
)


# ---------- Fallback keyword classifier ----------

GENIE_KEYWORDS = {
    "subjects", "count", "incidence", "rate", "region", "biomarker",
    "treatment", "control", "arm", "how many",
}
VECTOR_KEYWORDS = {
    "protocol", "criteria", "inclusion", "exclusion", "endpoint", "endpoints",
    "crf", "definition", "defined", "synopsis",
}


# ---------- Supervisor ----------


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self) -> None:
        self.genie = GenieAgent()
        self.vector = VectorAgent()
        self.decision_log = DecisionLog()
        self._client = None
        try:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        except Exception:
            self._client = None

    # ---------- Routing ----------

    def _keyword_classify(self, question: str) -> Route:
        q = question.lower()
        g = sum(1 for kw in GENIE_KEYWORDS if kw in q)
        v = sum(1 for kw in VECTOR_KEYWORDS if kw in q)
        if g and v:
            return "hybrid"
        if v > g:
            return "vector"
        return "genie"

    @trace(span_type="LLM", name="router")
    def _call_router(self, question: str) -> RouterDecision:
        """Call the router LLM with strict JSON output + Pydantic validation.

        One corrective retry on parse/validation failure; raises on second
        failure so `handle()` can fall through to the keyword classifier.
        """
        # Llama's JSON mode requires the literal word "json" in the messages;
        # append it to the user content so the gateway accepts response_format.
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\nRespond in JSON."},
        ]
        raw = self._call_llm(
            endpoint=ROUTER_LLM_ENDPOINT,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            return RouterDecision.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError):
            # Corrective retry: tell the model exactly what went wrong.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON matching the "
                        'schema {"route": "<genie|vector|hybrid>", "reasoning": "..."}. '
                        "Return only the JSON object."
                    ),
                }
            )
            raw = self._call_llm(
                endpoint=ROUTER_LLM_ENDPOINT,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return RouterDecision.model_validate_json(raw)

    # ---------- Fusion ----------

    @trace(span_type="LLM", name="fusion")
    def _call_fusion(
        self, question: str, parts: Iterable[AgentResponse]
    ) -> str:
        """Synthesize sub-agent answers into one paragraph via the fusion LLM."""
        parts = list(parts)
        user_msg = (
            f"Question: {question}\n\n"
            + "\n\n".join(f"[{p.source}] {p.answer}" for p in parts)
        )
        messages = [
            {"role": "system", "content": FUSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        return self._call_llm(
            endpoint=FUSION_LLM_ENDPOINT,
            messages=messages,
            temperature=0.2,
        )

    # ---------- LLM call (stubbed; one-line change to go live) ----------

    def _call_llm(
        self,
        endpoint: str,
        messages: list[dict],
        temperature: float = 0.0,
        response_format: dict | None = None,
        max_tokens: int = 512,
        timeout: int = 20,
    ) -> str:
        """POST to a Mosaic AI Gateway endpoint (OpenAI-compatible).

        Auth is read from the WorkspaceClient (works locally with profiles and
        in Databricks Apps with the SP token). The endpoint name is the AI
        Gateway resource name (e.g. ai_gateway_for_<model>), not the
        underlying serving endpoint.
        """
        if self._client is None:
            raise RuntimeError("WorkspaceClient unavailable")
        host = self._client.config.host
        url = f"{host}/ai-gateway/mlflow/v1/chat/completions"
        body = {
            "model": endpoint,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        auth_headers = self._client.config.authenticate()
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={**auth_headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:500]
            # Log to stderr so failures land in `databricks apps logs` even
            # when the keyword fallback masks them downstream.
            print(
                json.dumps({
                    "gateway_error": {
                        "endpoint": endpoint,
                        "url": url,
                        "status": e.code,
                        "body": detail,
                    }
                }),
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError(f"AI Gateway HTTP {e.code}: {detail}") from None
        except Exception as e:
            print(
                json.dumps({
                    "gateway_error": {
                        "endpoint": endpoint,
                        "url": url,
                        "exception": type(e).__name__,
                        "message": str(e)[:300],
                        "trace": traceback.format_exc(limit=3),
                    }
                }),
                file=sys.stderr,
                flush=True,
            )
            raise
        return payload["choices"][0]["message"]["content"]

    # ---------- Entry point ----------

    @trace(span_type="AGENT", name="supervisor")
    def handle(self, question: str) -> AgentResponse:
        request_id = str(uuid.uuid4())
        t0 = time.monotonic()
        llm_route: str | None = None
        fallback_reason: str | None = None

        # Route — LLM first, keyword fallback on any failure.
        try:
            decision = self._call_router(question)
            llm_route = decision.route
            route: Route = decision.route
        except Exception as exc:
            fallback_reason = self._classify_failure(exc)
            route = self._keyword_classify(question)

        if route == "genie":
            resp = self.genie.handle(question)
        elif route == "vector":
            resp = self.vector.handle(question)
        else:
            # hybrid — call both sub-agents, then fuse.
            g = self.genie.handle(question)
            v = self.vector.handle(question)
            try:
                fused = self._call_fusion(question, [g, v])
                resp = AgentResponse(
                    answer=fused,
                    source="hybrid",
                    confidence=(g.confidence + v.confidence) / 2,
                    data=g.data,
                    chunks=v.chunks,
                )
            except Exception as exc:
                if fallback_reason is None:
                    fallback_reason = f"fusion_{self._classify_failure(exc)}"
                stacked = f"[genie] {g.answer}\n\n[vector] {v.answer}"
                resp = AgentResponse(
                    answer=stacked,
                    source="hybrid",
                    confidence=(g.confidence + v.confidence) / 2,
                    data=g.data,
                    chunks=v.chunks,
                )

        # Tag the trace so the Traces UI can be filtered by route / fallback.
        set_trace_tags({
            "llm_route": llm_route,
            "final_route": route,
            "fallback_reason": fallback_reason or "none",
        })

        # Emit the decision record (mock=stdout JSON, live=UC INSERT). Never raises.
        self.decision_log.log(
            DecisionRecord(
                request_id=request_id,
                question=question,
                llm_route=llm_route,
                final_route=route,
                fallback_reason=fallback_reason,
                confidence=resp.confidence,
                response_chars=len(resp.answer),
                app_latency_ms=int((time.monotonic() - t0) * 1000),
                ts=now_iso(),
            )
        )
        return resp

    @staticmethod
    def _classify_failure(exc: BaseException) -> str:
        """Map an exception class to a short reason code for the decision log."""
        if isinstance(exc, ValidationError):
            return "validation_failure"
        if isinstance(exc, json.JSONDecodeError):
            return "parse_failure"
        if isinstance(exc, RuntimeError) and "stubbed" in str(exc):
            return "llm_stubbed"
        if isinstance(exc, RuntimeError) and "unavailable" in str(exc):
            return "client_unavailable"
        return "llm_error"
