# Upgrade Plan

We will improve the customer's design across iterations so each step is
reviewable on its own.

## Iteration 0 — Baseline mirror (completed)
Faithful structural mirror of the customer's project, runnable in mock mode.
No improvements yet — this is the comparison point.

## Iteration 1 — Two-tier Supervisor + AI Gateway + Eval

Implements the combined recommendation for Q1 (LLM selection and consistency
for the Supervisor agent), with **AI Gateway** doing the heavy lifting on
logging and evaluation.

### Split-model strategy

Classification and fusion have different latency, cost, and quality profiles —
separate them across two endpoints.

**Routing tier** (fast, deterministic, cheap)
- Model: `databricks-meta-llama-3-3-70b-instruct` at `temperature=0` as the
  default. If routing latency matters more than nuance, drop to
  `databricks-meta-llama-3-1-8b-instruct` — for a 3-class enum task with
  few-shot examples, the 8B handles it well.
- Endpoint type: pay-per-token Foundation Model API. No `app.yaml` resource
  declaration required (the app SP can query these by default).

**Fusion tier** (quality matters)
- Model: `databricks-claude-sonnet-4-7` for clinical synthesis. Step up to a
  Claude Opus 4.x endpoint for higher-stakes turns where reasoning depth pays.
- Endpoint type: pay-per-token FMAPI for the demo. **For production with PHI,
  switch to a provisioned-throughput endpoint** to get HIPAA compliance,
  predictable latency, and a CAN_QUERY resource declaration in `app.yaml`.

### Consistency & accuracy practices

1. **Strict structured output.** `response_format={"type":"json_object"}` plus
   an explicit JSON schema in the router system prompt. Instruct: "Return only
   valid JSON, no preamble."
2. **Temperature 0** on the router; 0.2 on fusion (some variability helps
   natural prose).
3. **Few-shot examples** — 3–5 question/route pairs in the router system
   prompt covering each route class. Single biggest accuracy lever after
   temp=0.
4. **Pydantic validation layer** between the LLM call and the dispatcher.
   Validate `route` against `{"genie", "vector", "hybrid"}`; on parse or
   validation failure, retry once with a corrective message; on second
   failure, fall through to the existing keyword classifier.
5. **Keep the keyword fallback.** Right safety net for client init failures
   and for routing when the LLM is unreachable.
6. **Log routing decisions** — see "Logging via AI Gateway" below.
7. **MLflow eval harness** — see "Eval via AI Gateway + MLflow" below.

### Logging via AI Gateway

AI Gateway sits on the model serving endpoints and gives us most of practice 6
for free:

- **Inference tables** — every request + response (prompt, completion, tokens,
  latency, cost, request ID) auto-logged to a Delta table in UC. No app code.
- **Payload logging** — full JSON for both directions, SQL-queryable.
- **PII guardrails** — block/mask PII before it leaves the perimeter; a useful
  extra layer alongside HIPAA-certified provisioned-throughput endpoints.
- **Rate limiting + fallback endpoints** — orthogonal to our in-app keyword
  fallback. If the primary endpoint flaps, AI Gateway transparently fails
  over to a secondary endpoint.

The app still writes a small **decision log** for app-level concepts AI
Gateway can't know
(`{request_id, llm_route, final_route, fallback_reason, app_latency_ms}`).
JOIN to the AI Gateway inference table on `request_id` for the full picture.
~80% of the logging volume comes from AI Gateway for free.

### Eval via AI Gateway + MLflow

- **MLflow Tracing** auto-captures the supervisor's chain (router LLM call →
  sub-agent calls → fusion LLM call). Each span has latency, inputs, outputs.
- **`mlflow.evaluate()`** reads either from the AI Gateway inference table or
  from a hand-labeled CSV — same grading API for both:
  - Offline evals on a 50-question gold set (regression gate before deploy).
  - Online evals sampling real production traffic from the inference table
    (drift detection).
- Built-in LLM judges (`relevance`, `groundedness`, `correctness`) plus a
  custom grader for routing accuracy.
- If we later migrate to **Mosaic AI Agent Framework**, Agent Evaluation
  adds task-completion, tool-call-correctness, and instruction-following
  graders out of the box.

### Implementation checklist

- [ ] `agents/supervisor.py` — split router and fusion into two methods, each
      calling a different endpoint env var.
- [ ] `agents/supervisor.py` — JSON schema + few-shot examples in the router
      system prompt; Pydantic model for the parsed response with a single
      corrective retry.
- [ ] `modules/config.py` — add `ROUTER_LLM_ENDPOINT` and
      `FUSION_LLM_ENDPOINT` env vars (defaults reflect the recommendation).
- [ ] `app.yaml` — env var entries for both endpoints. Still no `resources:`
      block (deferred to iteration 3) unless using provisioned throughput.
- [ ] Workspace step (not app code): enable **AI Gateway** on both endpoints —
      usage tracking, inference tables (to a UC table we name), and PII
      guardrails.
- [ ] `modules/decision_log.py` — thin writer for the app-level decision log
      (insert into a UC table on each turn).
- [ ] `eval/gold_questions.csv` — ~50 hand-labeled questions (correct route +
      gold answer text).
- [ ] `eval/run_eval.py` — MLflow eval harness reading the gold set; routing
      accuracy + answer quality metrics logged as an MLflow run.

## Iteration 2 — Section-aware chunker + Vector Search wiring
- Replace the TF-IDF mock with a markdown-heading-aware chunker.
- Stand up a self-managed Delta-Sync VS index with `databricks-gte-large-en`
  embeddings; document chunk size (~800 tokens) and overlap (~150).
- Add metadata filters (`doc_type`, `section`) on retrieval.

## Iteration 3 — Hardened `app.yaml` + Genie conversation rotation
- Fill the `resources:` block (warehouse, genie_space, uc_securable for the
  volume, vector_search_endpoint, provisioned-throughput LLM endpoint if used).
- Per-question or sliding-window Genie conversation rotation to prevent
  `REQUEST_LIMIT_EXCEEDED`; typed exception handler that retries on a fresh
  conversation id.

## Iteration 4 — Optional: AI Agent Bricks (Knowledge Assistant) variant
- Swap `agents/vector_agent.py` for a managed Knowledge Assistant endpoint.
- Compare the two paths on the same eval set.
