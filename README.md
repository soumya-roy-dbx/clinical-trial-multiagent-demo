# Multi-Agent Clinical Trial Analytics App

A Databricks reference application demonstrating a multi-agent
supervisor pattern over clinical-trial data. Streamlit dashboard plus
a chat sidebar that routes each question to the right backend — Genie
for structured queries, Vector Search for protocol/CRF retrieval, or
a hybrid path that fuses both — and answers via Claude Sonnet 4.6
through AI Gateway.

> **Headline result.** Judge mean **3.33 → 4.06 / 5** and judge
> pass@4 **55.6% → 77.8%** across three measured iterations on a fixed
> 18-question gold set. Routing accuracy held at 100% throughout.

The app is built up in four self-contained iterations, each measured
against the previous so the contribution of every architectural choice
is visible in the metrics.

---

## Table of contents

- [Design questions this app answers](#design-questions-this-app-answers)
- [Architecture at a glance](#architecture-at-a-glance)
- [What's in this repo](#whats-in-this-repo)
- [How the app builds up, iteration by iteration](#how-the-app-builds-up-iteration-by-iteration)
  - [Iteration 0 — Local-runnable baseline](#iteration-0--local-runnable-baseline)
  - [Iteration 1 — Two-tier supervisor + AI Gateway + eval harness](#iteration-1--two-tier-supervisor--ai-gateway--eval-harness)
  - [Iteration 2 — Section-aware chunker + live Vector Search](#iteration-2--section-aware-chunker--live-vector-search)
  - [Iteration 3 — Resource bindings + UC Volume + live Genie](#iteration-3--resource-bindings--uc-volume--live-genie)
- [Eval results — full comparison](#eval-results--full-comparison)
- [Observability and monitoring](#observability-and-monitoring)
- [Operational gotchas to know](#operational-gotchas-to-know)
- [Adapt to your workspace](#adapt-to-your-workspace)
- [Reproduce step-by-step](#reproduce-step-by-step)
- [Extending this demo](#extending-this-demo)
- [Repo conventions](#repo-conventions)

---

## Design questions this app answers

Three design questions shape the architecture below:

1. **LLM selection for the supervisor.** Which model(s) should drive
   the router? How should routing decisions stay consistent across
   runs? How is routing quality measured?
2. **Vector Search best practices.** How should protocol / CRF
   markdown be chunked? Which embedding model? How is retrieval wired
   into the agent path, and how is an answer synthesized from the
   chunks?
3. **`app.yaml` and `databricks.yml` boundaries.** What lives in
   which file? What does the Apps platform honor today, and what
   still needs manual permission management?

Each iteration below addresses one of these. The eval harness turns the
answers into numbers.

---

## Architecture at a glance

```
                    ┌─────────────────────────────────────────┐
                    │   Streamlit (Databricks Apps)           │
                    │   • Subject Disposition KPIs            │
                    │   • Chat sidebar                        │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────────┐
              │   Supervisor (agents/supervisor.py)           │
              │   ┌─────────────────────────────────────┐     │
              │   │ Router LLM   Llama 3.3 70B • temp 0 │     │
              │   │              JSON + few-shot        │ ──┐ │
              │   └─────────────────────────────────────┘   │ │
              └───────────────────────────────────────┬─────┘ │
                                                     │       │
              ┌─────────────────────┐  ┌─────────────▼─────┐ │
              │ Genie agent         │  │ Vector agent      │ │
              │ start_conversation_ │  │ POST /vector-     │ │
              │ and_wait per turn   │  │   search/.../query│ │
              │   ▼                 │  │   ▼               │ │
              │ Genie Space         │  │ VS Index (Delta   │ │
              │ → SQL warehouse     │  │   Sync, gte-large)│ │
              │ → study_subjects    │  │ → protocol_chunks │ │
              └─────────┬───────────┘  └───────────┬───────┘ │
                        │                          │         │
                        └──────────┬───────────────┘         │
                                   ▼                         │
                    ┌────────────────────────────┐           │
                    │ Fusion LLM                 │           │
                    │ Claude Sonnet 4.6 • temp 0.2◀──────────┘
                    │ (via AI Gateway)           │
                    └────────────┬───────────────┘
                                 ▼
                       answer to user
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │ Telemetry written on every turn      │
              │ • AI Gateway → inference table       │
              │ • App decision log → UC table        │
              │ • MLflow trace                       │
              └──────────────────────────────────────┘
```

**Why two LLMs?** Routing is a deterministic 3-class classification —
fast and cheap (Llama, `temperature=0`). Fusion is clinical prose
synthesis — quality matters (Claude, `temperature=0.2`). Splitting
the two lets each tier sit at the right point on the cost / quality /
latency curve.

**Why AI Gateway?** Free inference-table logging (full prompt +
completion, tokens, latency, cost) plus PII guardrails before the
request leaves the perimeter — without writing any app code for it.
For production with PHI, swap pay-per-token for a provisioned-
throughput endpoint to get HIPAA-compliant serving.

---

## What's in this repo

```
app.py                              # Streamlit entry point
app.yaml                            # Databricks Apps runtime config (command + env)
databricks.yml                      # Asset Bundle: app + resource bindings + grants
requirements.txt                    # streamlit, pandas, databricks-sdk, pydantic, mlflow, psycopg2-binary
README.md                           # this file

agents/
  base.py                           # AgentResponse dataclass + BaseAgent interface
  supervisor.py                     # Two-tier router (Llama) + fusion (Claude) + decision log
  genie_agent.py                    # Live Genie via start_conversation_and_wait per turn
  vector_agent.py                   # Live VS query + Claude RAG synthesis (TF-IDF fallback)

modules/
  config.py                         # Centralized env-var configuration
  chat_ui.py                        # Streamlit chat sidebar component
  db.py                             # KPI queries for the dashboard
  decision_log.py                   # App-level decision writer (UC table or stdout)
  memory.py                         # Lakebase short-term chat memory (no-op when unconfigured)
  genie_mock.py                     # Local pandas dispatcher used as Genie fallback
  genie_inline.py                   # (deprecated; superseded by genie_agent)

data/
  docs/*.md                         # 6 source protocol + CRF markdown files
  chunking.py                       # Section-aware chunker (H2 split, ~800 tok, ~150 overlap)
  build_chunks_table.py             # One-shot loader → protocol_chunks Delta
  build_subjects_table.py           # One-shot loader → study_subjects Delta
  mock_data.py                      # Regenerate the synthetic subject CSV
  study_subjects.csv                # 100 synthetic subjects, used by Genie and the mock

eval/
  gold_questions.csv                # 18 hand-labeled questions (6 each: genie/vector/hybrid)
  run_eval.py                       # MLflow harness — routes, latency, Claude-as-judge
  README.md                         # Eval usage + metric definitions

_archive/iteration-N/                   # Frozen full-tree snapshots at the end of each iteration,
                                    # browseable side-by-side with the live tree at the root
```

---

## How the app builds up, iteration by iteration

Each iteration adds one self-contained capability and is measured against
the iteration below it on the same 18-question gold set. The
`_archive/iteration-N/` folders preserve the full file tree at each
checkpoint.

### Iteration 0 — Local-runnable baseline

**Capability:** end-to-end runnable demo **without any Databricks
credentials**, so the architecture and eval loop are reviewable in
isolation.

**What's in it:**
- Subject Disposition dashboard rendered from the synthetic CSV
- Single-LLM supervisor stub falling through to keyword routing
- TF-IDF over `data/docs/*.md` as a vector-search stand-in
- Pandas dispatcher over `data/study_subjects.csv` as a Genie stand-in
- No AI Gateway, no eval, no inference tables, no UC writes

This is the comparison point. Everything that follows is a delta on
top.

---

### Iteration 1 — Two-tier supervisor + AI Gateway + eval harness

**Capability:** answers design question #1 (LLM selection +
consistency for the supervisor), and adds the measurement scaffold
every later iteration depends on.

**Code at this iteration:**
- `agents/supervisor.py` — router and fusion split into separate
  methods on separate endpoints. Pydantic schema validates the
  routing JSON; one-shot corrective retry on parse failure; keyword
  classifier as a final safety net if both the LLM and the retry fail.
- `modules/decision_log.py` — thin writer that emits one row per turn
  with `request_id`, `llm_route`, `final_route`, `fallback_reason`,
  `confidence`, `response_chars`, `app_latency_ms`. Mock mode prints
  JSON to stdout; live mode INSERTs to UC (wired in Iteration 3).
- `eval/run_eval.py` + `eval/gold_questions.csv` — 18 hand-labeled
  questions (6 each across genie / vector / hybrid). Runs every
  question through `SupervisorAgent.handle()`, captures router LLM
  decisions separately, then asks Claude to judge each answer 1–5
  versus the gold. Logs one MLflow run per execution with routing
  accuracy, fallback rate, p50 / p95 latency, and judge mean /
  pass@4.

**Workspace assets used at this iteration:**

| Asset | Identifier | Purpose |
|---|---|---|
| AI Gateway endpoint (router) | `ai_gateway_for_databricks-meta-llama-3-3-70b-instruct` | Llama 3.3 70B for routing |
| AI Gateway endpoint (fusion) | `ai_gateway_for_databricks-claude-sonnet-4-6` | Claude Sonnet 4.6 for synthesis |
| Inference table (router) | `<catalog>.<schema>.gateway_inference_payload` | Auto-logged by AI Gateway |
| Inference table (fusion) | `<catalog>.<schema>.gateway_inference_for_databricks-claude-sonnet-4-6_payload` | Auto-logged by AI Gateway |
| MLflow experiment | `/Users/<you>/supervisor-eval` | One run per `python eval/run_eval.py` |

**Consistency techniques applied:**
1. `temperature=0` on the router
2. `response_format={"type":"json_object"}` + explicit schema in the system prompt
3. 3–5 few-shot examples per route class
4. Pydantic validation + one-shot corrective retry
5. Keyword fallback as the final safety net

**Iteration 1 baseline:** 100% routing accuracy, judge_mean 3.33,
judge_pass@4 55.6%, latency p50 2.9s / p95 13.4s. The judge score is
dragged down by the still-mock vector path (returns raw chunks) and
mock-dispatcher gaps in the Genie path — both addressed in Iterations 2
and 3.

---

### Iteration 2 — Section-aware chunker + live Vector Search

**Capability:** answers design question #2 (Vector Search best
practices), swaps the TF-IDF mock for a real RAG pipeline, and shows
what that buys on the vector slice of the eval.

**Code at this iteration:**
- `data/chunking.py` — section-aware markdown chunker. Splits on H2
  headings, targets ~800 tokens with ~150-token overlap, attaches
  `doc_type`, `section`, `chunk_index` metadata so retrieval can
  filter later.
- `data/build_chunks_table.py` — one-shot loader that runs the
  chunker over `data/docs/*.md` and writes to a Delta table via the
  SQL warehouse Statement Execution API. Re-run any time the corpus
  changes; the VS Delta Sync index picks up changes automatically.
- `agents/vector_agent.py` — `_handle_live` POSTs to
  `/api/2.0/vector-search/indexes/<index>/query` for the top-3
  chunks, then calls the Claude fusion gateway with a clinical RAG
  system prompt over the chunk text. TF-IDF MockDocStore retained as
  a fallback (handles SDK init failure and any 4xx/5xx on the VS
  call).

**Workspace assets used at this iteration:**

| Asset | Identifier | Notes |
|---|---|---|
| Delta table | `<catalog>.<schema>.protocol_chunks` | 26 rows, CDF enabled |
| VS index | `<catalog>.<schema>.protocol_chunks_vs_index` | DELTA_SYNC, CONTINUOUS |
| VS endpoint | `<your-vs-endpoint>` | Reused — no dedicated endpoint required |
| Embedding model | `databricks-gte-large-en` | Picked by Databricks-managed embeddings |

**Iteration 2 baseline:** judge_mean **3.33 → 3.56** (+0.23), routing
accuracy still 100%, latency p50 2.9s → 9.5s (real RAG costs ~7s per
vector turn). The headline win is hidden inside the average — the
**vector category** moved from ~3.2 to ~4.0 (raw chunk text →
synthesized clinical answers). The overall judge_mean is held back by
the Genie mock's missing canned queries (g05, g06) — fixed in
Iteration 3.

---

### Iteration 3 — Resource bindings + UC Volume + live Genie

**Capability:** answers design question #3 (where do resource
bindings and permissions actually live), moves the corpus to a UC
Volume, lights up live Genie, and reaches the "production-shaped"
version of the app.

**Code at this iteration:**
- `databricks.yml` — Asset Bundle declaring the app + its resource
  bindings + permission grants in one file: SQL warehouse, Genie
  Space, UC tables, UC Volume. `databricks bundle deploy` applies
  every grant atomically. (Three grants the Apps bundle schema
  doesn't yet cover are documented in [gotcha 2](#2-three-grants-the-apps-bundle-schema-does-not-yet-support).)
- `app.yaml` — runtime config only. Env vars that reference bundle
  resources use `valueFrom: <resource-key>` so the Apps runtime
  injects warehouse ID, Genie space ID, and table full names at
  startup without duplicating those identifiers.
- `modules/decision_log.py` — flips from stdout to UC INSERT when
  the warehouse env var is populated. Decisions live in
  `<catalog>.<schema>.supervisor_decisions`, queryable like any UC
  table.
- `agents/genie_agent.py` — `_handle_live` calls
  `WorkspaceClient.genie.start_conversation_and_wait()` **per
  question**. This per-turn conversation rotation is the trick to
  staying under Genie's context limits indefinitely — no single
  conversation accumulates. Falls back to the pandas mock dispatcher
  on any live failure (logged to stderr) so the app always returns.
- `data/build_subjects_table.py` — one-shot loader for the 100-row
  synthetic subject table. Column comments included for Genie.

**Workspace assets used at this iteration:**

| Asset | Identifier | Purpose |
|---|---|---|
| UC Volume (managed) | `<catalog>.<schema>.protocol_docs` | 6 source markdown docs (production-shaped corpus location) |
| Delta table | `<catalog>.<schema>.study_subjects` | 100 rows, column comments for Genie |
| Delta table | `<catalog>.<schema>.supervisor_decisions` | App-level decision log |
| Genie Space | `Clinical Trial Subject Disposition` | Over `study_subjects`, sample questions seeded |
| SQL warehouse | Serverless | Backs Genie, decision log writes, builder scripts |

**Iteration 3 baseline:** judge_mean **3.56 → 4.06** (+0.50, the
biggest single-iteration jump). judge_pass@4 **55.6% → 77.8%**. Routing
accuracy still 100%. Latency p50 13.7s / p95 35.4s (live Genie adds
~10s per genie / hybrid turn).

**Where Iteration 3 moves the needle:** the hybrid category goes from
mixed (`5,2,4,3,3,1`) to consistently strong (`5,5,4,5,4,4`) — both
sub-agents are now real so multi-source synthesis has good inputs on
both sides. g05 (indicated population count) finally passes because
Genie answers what the mock dispatcher couldn't.

**Known gap remaining:** g04 (Event A incidence) drops 5 → 1 because
live Genie defaults to strict equality on the adverse-event column
where the gold answer uses LIKE-style matching. Fixable with a Genie
Space instruction; documented for follow-up.

---

## Eval results — full comparison

Same 18-question gold set across all three iterations. All metrics
captured by `eval/run_eval.py` and logged to MLflow.

| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Δ (1→3) |
|---|---:|---:|---:|---:|
| Routing accuracy | 100% | 100% | 100% | flat |
| Fallback rate | 0% | 0% | 0% | flat |
| Judge mean (Claude, /5) | 3.33 | 3.56 | **4.06** | **+0.73** |
| Judge pass@4 | 55.6% | 55.6% | **77.8%** | **+22.2 pp** |
| Latency p50 | 2.9s | 9.5s | 13.7s | +10.8s |
| Latency p95 | 13.4s | 26.0s | 35.4s | +22.0s |

**How to read latency:** the p50 climbs as more of the pipeline
becomes real. A `vector` turn does router LLM → VS query → Claude RAG
= 3 serial network hops; a `hybrid` turn does router → Genie + VS in
parallel → Claude fusion = 4 hops, and Genie typically dominates. The
cost is honest — these are real round trips, not stubs.

**Per-category trend (judge mean):**

| Category | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---:|---:|---:|
| genie | 3.5 | 3.5 | 3.7 |
| vector | 3.2 | 4.0 | 4.5 |
| hybrid | 3.0 | 3.2 | 4.5 |

Vector lifts in Iteration 2 (real RAG), hybrid lifts in Iteration 3 (both
sub-agents now real), genie inches up across all three.

---

## Observability and monitoring

The app emits structured telemetry on every chat turn across three
independent layers of telemetry, so a single conversation is debuggable end-to-end
and quality / cost metrics can be aggregated across production
traffic.

### Tables and assets used for observability

| # | Asset | Type | What it captures | Written by |
|---|---|---|---|---|
| 1 | `<catalog>.<schema>.gateway_inference_payload` | Delta table (auto) | Every router LLM request + response: prompt, completion, tokens in/out, latency, cost, request_id, timestamp | AI Gateway (router endpoint) |
| 2 | `<catalog>.<schema>.gateway_inference_for_databricks-claude-sonnet-4-6_payload` | Delta table (auto) | Every fusion LLM request + response with the same shape as above | AI Gateway (fusion endpoint) |
| 3 | `<catalog>.<schema>.supervisor_decisions` | Delta table (app-managed) | One row per chat turn with the app-level decision: `request_id`, `llm_route`, `final_route`, `fallback_reason`, `confidence`, `response_chars`, `app_latency_ms`, `ts` | `modules/decision_log.py` |
| 4 | MLflow experiment at `/Users/<you>/supervisor-eval` | MLflow experiment | One run per `python eval/run_eval.py`: routing accuracy, fallback rate, p50/p95 latency, judge mean, judge pass@4, plus per-row results as an artifact | `eval/run_eval.py` |
| 5 | MLflow Traces (auto-captured) | MLflow tracing | Per-turn span tree: router LLM call → sub-agent calls → fusion LLM call with inputs, outputs, and latencies for each span | MLflow autolog |

Tables 1 and 2 are **free** — AI Gateway populates them on every call
with no app code. Table 3 is written by `modules/decision_log.py`
(stdout in mock mode, UC INSERT in live mode when
`DATABRICKS_WAREHOUSE_ID` is set). Table 4 is populated explicitly by
the eval harness; everything else is implicit telemetry.

### Joining decisions to gateway calls

Every supervisor turn produces 1 row in table 3 and 1–3 rows in
tables 1+2 (depending on the route: 1 row for `genie`-only, 3 rows
for `hybrid`). They share `request_id`, so the join is direct:

```sql
SELECT
    d.ts,
    d.llm_route,
    d.final_route,
    d.app_latency_ms,
    g.input_token_count + g.output_token_count AS total_tokens,
    g.latency_ms                                AS gateway_latency_ms,
    g.cost_usd
FROM   <catalog>.<schema>.supervisor_decisions d
JOIN   <catalog>.<schema>.gateway_inference_payload g
       ON d.request_id = g.request_id
WHERE  d.ts > current_timestamp() - INTERVAL 1 DAY
ORDER  BY d.ts DESC;
```

### Useful one-off queries

**Routing distribution over the last day:**
```sql
SELECT llm_route, final_route, COUNT(*) AS n
FROM   <catalog>.<schema>.supervisor_decisions
WHERE  ts > current_timestamp() - INTERVAL 1 DAY
GROUP  BY 1, 2 ORDER BY n DESC;
```

**Fallback rate (smoke test for live-path health):**
```sql
SELECT
    COUNT(*) FILTER (WHERE fallback_reason IS NOT NULL) * 1.0 / COUNT(*) AS fallback_rate,
    COUNT(*) AS turns
FROM <catalog>.<schema>.supervisor_decisions
WHERE ts > current_timestamp() - INTERVAL 1 HOUR;
```

**Latency p50/p95 per route:**
```sql
SELECT
    final_route,
    APPROX_PERCENTILE(app_latency_ms, 0.50) AS p50_ms,
    APPROX_PERCENTILE(app_latency_ms, 0.95) AS p95_ms,
    COUNT(*) AS turns
FROM <catalog>.<schema>.supervisor_decisions
WHERE ts > current_timestamp() - INTERVAL 1 DAY
GROUP BY 1;
```

**Top 10 most expensive turns (token-weighted):**
```sql
SELECT
    request_id,
    SUM(input_token_count + output_token_count) AS total_tokens,
    SUM(cost_usd) AS total_cost
FROM <catalog>.<schema>.gateway_inference_payload
WHERE timestamp > current_timestamp() - INTERVAL 1 DAY
GROUP BY 1 ORDER BY total_cost DESC LIMIT 10;
```

### Online evaluation recipe

The same `mlflow.evaluate()` call that grades the offline gold set
can be pointed at table 1 or table 2 to grade a sliding window of
production traffic. The eval harness in `eval/run_eval.py` is the
starting point; swap the input from `gold_questions.csv` to a SQL
query against the inference table and that becomes a drift detector.

---

## Operational gotchas to know

The behaviors below aren't in the docs but matter for any production
deploy.

### 1. Resources + permissions belong in `databricks.yml`, not `app.yaml`

The current Databricks Apps file contract:

| File | Purpose |
|---|---|
| `app.yaml` | **Runtime config only** — `command:` + `env:`. Read by the Apps runtime on container start. |
| `databricks.yml` | **Asset Bundle** — app metadata, resource bindings, permission grants. Read by `databricks bundle deploy`. |
| Apps UI → Resources tab | Manual click-through alternative if Asset Bundles aren't an option. |

A `resources:` block inside `app.yaml` looks plausible but isn't
acted on by the Apps runtime — permissions never get applied through
it. Apps may appear to work because the service principal inherits
permissions from group memberships (e.g., `users`), masking that
nothing was actually granted by the declaration. Production UC
isolation expects explicit grants.

The correct pattern: define everything in `databricks.yml` (see this
repo's file as a working template) and deploy with
`databricks bundle deploy`. The bundle applies `CAN_USE` / `CAN_RUN`
/ `SELECT` / `READ_VOLUME` grants atomically to the app's service
principal.

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev
```

To reference resource bindings from env vars without duplicating IDs,
use `valueFrom: <resource-key>` in `app.yaml` — the Apps runtime
injects the bound identifier (warehouse ID, Genie space ID, table
full name) at startup. Resource keys are defined under
`resources.apps.<app_name>.resources[].name` in `databricks.yml`.

### 2. Three grants the Apps bundle schema does NOT yet support

The Asset Bundle covers most resource grants atomically on
`bundle deploy`, but its `resources` schema for an `app` block has
real gaps. These three grants still need a one-time manual step
post-deploy:

**A. AI Gateway endpoints (`CAN_QUERY`)** — AI Gateway endpoints
live in a separate resource namespace from regular serving endpoints
(under `/api/2.0/permissions/ai-gateway-endpoints/...`). The
bundle's `serving_endpoint` sub-type only sees foundation-model
endpoints (`databricks-claude-sonnet-4-6`, etc.) and Knowledge
Assistant endpoints — not AI Gateway wrappers. Apply the grant
manually:

```bash
SP="<service_principal_client_id from `databricks apps get <app-name>`>"
for ep in <your-router-gateway> <your-fusion-gateway> ; do
  databricks api patch "/api/2.0/permissions/ai-gateway-endpoints/$ep" \
    --json "{\"access_control_list\":[{\"service_principal_name\":\"$SP\",\"permission_level\":\"CAN_QUERY\"}]}"
done
```

**B. Vector Search endpoint (`CAN_USE`)** — the app-resource schema
has no `vector_search_endpoint` sub-type today. Apply manually:

```bash
databricks api patch "/api/2.0/permissions/vector-search-endpoints/<endpoint-name>" \
  --json "{\"access_control_list\":[{\"service_principal_name\":\"$SP\",\"permission_level\":\"CAN_USE\"}]}"
```

**C. `MODIFY` on UC tables the app writes to** — the
`uc_securable.permission` enum allows only `EXECUTE` /
`READ_VOLUME` / `SELECT` / `USE_CONNECTION` / `WRITE_VOLUME`. There
is no `MODIFY` value, so the bundle can only grant read access to
tables. For the `supervisor_decisions` table (which the app INSERTs
into via the SQL warehouse), apply `MODIFY` via SQL:

```sql
GRANT MODIFY ON TABLE <catalog>.<schema>.supervisor_decisions
  TO `<service_principal_client_id>`;
```

Symptom if any of these are missing: app returns 4xx, the affected
feature silently falls back (or fails to write the decision log).

### 2b. Adopting an existing app into a bundle

If an app was previously deployed outside a bundle (via
`databricks apps create <name>`), the first `bundle deploy` fails
with "App with the same name already exists" — Terraform won't
implicitly adopt resources. Bind first:

```bash
databricks bundle deployment bind clinical_assistant <app-name> --auto-approve
databricks bundle deploy --target dev
databricks apps deploy <app-name> \
  --source-code-path /Workspace/Users/<you>/.bundle/<bundle-name>/dev/files
```

The bind step writes the existing app's ID into the bundle's
deployment state so subsequent deploys UPDATE the app rather than
trying to CREATE.

Also: avoid `mode: development` on a target used for production
deploys — that mode prefixes resource names with `[dev <user>]`,
which creates a parallel app instead of updating the bound one.

### 2c. Bundle's bundled Terraform may have an expired GPG key

`databricks bundle deploy` downloads a copy of Terraform under the
hood and verifies its checksum signature. If you see:

```
Error: error downloading Terraform: unable to verify checksums signature:
openpgp: key expired
```

…the embedded HashiCorp signing key has rotated. Workaround: point
the CLI at any working local Terraform binary (or install one via
`brew install terraform`):

```bash
DATABRICKS_TF_EXEC_PATH=/path/to/terraform \
  databricks bundle deploy --target dev
```

### 3. Genie context limits — rotate the conversation per turn

A single Genie conversation accumulates context. After ~20–30 turns
the next turn fails with `REQUEST_LIMIT_EXCEEDED` or
`CONTEXT_EXCEEDED`. The fix is structural: do not reuse a
conversation across turns. `agents/genie_agent.py` calls
`start_conversation_and_wait()` for every single user question, gets
the answer + SQL out of `conv.attachments`, falls back to
`get_message_query_result` if Genie returned only a query, and
discards the conversation. The Genie Space stays clean indefinitely.

### 4. PyPI access from Databricks Apps

Apps containers install from `requirements.txt` on every deploy.
This repo keeps the dependency list small (streamlit, pandas,
databricks-sdk, pydantic, mlflow, psycopg2-binary) — all broadly
available. If you add a niche library, verify it reaches your serving
cluster before depending on it; some workspaces have a private PyPI
mirror with a different package set.

---

## Adapt to your workspace

Every placeholder in the repo, what it represents, and where to find
your real value. Each placeholder string maps 1:1 to a single concept;
a global search-and-replace is safe.

| Placeholder | What it represents | Where to source your value | Files affected |
|---|---|---|---|
| `your_catalog` | Unity Catalog name | `databricks catalogs list` or your governance owner | `app.yaml`, `modules/config.py`, `modules/decision_log.py`, `data/build_*_table.py` |
| `your_schema` | Unity Catalog schema inside `your_catalog` | Pick — `CREATE SCHEMA IF NOT EXISTS your_catalog.<name>` | Same files |
| `your_warehouse_id` | Serverless SQL warehouse ID | `databricks warehouses list` (column `id`) | `databricks.yml` variables, `data/build_*_table.py` |
| `your_genie_space_id` | Genie Space ID | Genie Space URL or `databricks api get /api/2.0/genie/spaces` | `databricks.yml` variables, `modules/config.py` |
| `your_vs_endpoint` | Vector Search endpoint name | `databricks vector-search-endpoints list`, or create one | `app.yaml`, `modules/config.py` |
| `your_app_sp_id` | Service principal of the deployed Databricks App | `databricks apps get <app-name>` → `service_principal_client_id` | Gotcha snippets only |
| `your-workspace.cloud.databricks.com` | Workspace host | CLI profile / workspace URL | `databricks.yml` targets |
| `your-app-name.your-workspace.databricksapps.com` | Deployed Apps URL | Generated on first deploy | n/a |

**Recommended overrides (no source edits needed):** almost
everything runtime-critical flows through `os.getenv()` in
`modules/config.py` and the data scripts. For local dev or one-off
runs, set:

```bash
export DATABRICKS_WAREHOUSE_ID=<your warehouse id>
export DATABRICKS_GENIE_SPACE_ID=<your genie space id>
export VECTOR_SEARCH_ENDPOINT=<your vs endpoint>
export VECTOR_SEARCH_INDEX=<your catalog>.<your schema>.protocol_chunks_vs_index
export PROTOCOL_CHUNKS_TABLE=<your catalog>.<your schema>.protocol_chunks
export STUDY_SUBJECTS_TABLE=<your catalog>.<your schema>.study_subjects
export DECISION_LOG_TABLE=<your catalog>.<your schema>.supervisor_decisions
```

For deployed Apps, edit `databricks.yml` variables (catalog, schema,
warehouse_id, genie_space_id, app_name) and the
`targets.dev.workspace.host`. `app.yaml` literal values (AI Gateway
endpoint names, VS endpoint name, VS index) need to match your
workspace too.

---

## Reproduce step-by-step

Targeting a fresh Databricks workspace, AWS or Azure. A local dev
box is optional — its only role is `streamlit run app.py` in mock
mode for UI iteration.

**Prereqs:** Databricks workspace, [Databricks
CLI](https://docs.databricks.com/dev-tools/cli/install.html) authed
against it, a Unity Catalog you have write access to, and a
serverless SQL warehouse.

### Step 1 — Local clone + mock-mode smoke test

```bash
git clone <this-repo>
cd clinical-trial-multiagent-demo
pip install -r requirements.txt
python data/mock_data.py        # regenerates data/study_subjects.csv
streamlit run app.py            # opens http://localhost:8501
```

Mock mode works without any Databricks credentials. The chat sidebar
falls through to keyword routing + the TF-IDF doc store + the pandas
dispatcher.

### Step 2 — Catalog / schema / warehouse

```sql
-- In the Databricks SQL editor:
CREATE CATALOG IF NOT EXISTS your_catalog;
CREATE SCHEMA IF NOT EXISTS your_catalog.your_schema;
```

Capture your serverless SQL warehouse ID — get it from
`databricks warehouses list` or the Warehouse settings page.

### Step 3 — AI Gateway endpoints (router + fusion)

Create two AI Gateway endpoints — one for routing (Llama 3.3 70B),
one for fusion (Claude Sonnet 4.6). Enable the inference table (any
catalog / schema you own) and PII guardrails on each.

UI path: Serving → Create AI Gateway → wrap an existing FMAPI
endpoint → enable inference tables.

After creation, **grant the Apps SP `CAN_QUERY` on each** (see
[gotcha 2A](#2-three-grants-the-apps-bundle-schema-does-not-yet-support)).

### Step 4 — Protocol corpus → UC Volume

```bash
# Create the managed Volume
databricks api post /api/2.1/unity-catalog/volumes --json '{
  "catalog_name": "your_catalog",
  "schema_name": "your_schema",
  "name": "protocol_docs",
  "volume_type": "MANAGED"
}'

# Upload all 6 markdown files
for f in data/docs/*.md; do
  databricks fs cp "$f" "dbfs:/Volumes/your_catalog/your_schema/protocol_docs/$(basename $f)"
done
```

### Step 5 — Chunks Delta table + Vector Search index

```bash
# Build the chunks table from data/docs/
export DATABRICKS_WAREHOUSE_ID=<your warehouse id>
export PROTOCOL_CHUNKS_TABLE=your_catalog.your_schema.protocol_chunks
python data/build_chunks_table.py
```

Create the VS index over the table:

- UI: Compute → Vector Search → Create index → Delta Sync,
  CONTINUOUS pipeline, embedding model `databricks-gte-large-en`,
  source column `chunk_text`
- Or via REST: `POST /api/2.0/vector-search/indexes`

If you don't already have a VS endpoint, create one — or reuse a
shared one.

### Step 6 — Subjects table + Genie Space

```bash
export STUDY_SUBJECTS_TABLE=your_catalog.your_schema.study_subjects
python data/build_subjects_table.py
```

Create a Genie Space over the table:

- UI: Genie → New Space → select `study_subjects` → seed with 3–5
  sample questions covering counts, by-region, and
  by-arm-with-biomarker patterns (mirror the questions in
  `eval/gold_questions.csv` for a quick start)

Note the Genie Space ID from the URL or
`databricks api get /api/2.0/genie/spaces`.

### Step 7 — Supervisor decisions table

```sql
CREATE TABLE IF NOT EXISTS your_catalog.your_schema.supervisor_decisions (
    ts                TIMESTAMP NOT NULL,
    request_id        STRING,
    question_text     STRING,
    llm_route         STRING,
    final_route       STRING,
    fallback_reason   STRING,
    confidence        DOUBLE,
    response_chars    INT,
    app_latency_ms    INT
)
USING DELTA;
```

### Step 8 — Configure `app.yaml` (runtime) and `databricks.yml` (resources)

- **`app.yaml`** — runtime config only. Update the literal
  `value:` entries for AI Gateway endpoints, VS endpoint name, and
  VS index. The `valueFrom:` entries reference bundle resource keys
  and don't need changing if you keep the same key names in
  `databricks.yml`.
- **`databricks.yml`** — Asset Bundle. Update the `variables:`
  defaults (or pass them via `--var`) for `catalog`, `schema`,
  `warehouse_id`, `genie_space_id`, and `app_name`. Update
  `targets.dev.workspace.host` to your workspace URL.

### Step 9 — Deploy the bundle

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev

# If the bundle's embedded Terraform GPG key has expired
# (see gotcha 2c), point at a local terraform binary:
#   DATABRICKS_TF_EXEC_PATH=/path/to/terraform databricks bundle deploy ...
```

**First-time deploy when the app already exists:** bind first (see
[gotcha 2b](#2b-adopting-an-existing-app-into-a-bundle)):

```bash
databricks bundle deployment bind clinical_assistant <app-name> --auto-approve
databricks bundle deploy --target dev
```

**Activate the new source code:** `bundle deploy` uploads source
files and applies resource bindings, but it does NOT automatically
redeploy the app's container from the new source path. Trigger that
once explicitly:

```bash
databricks apps deploy <app-name> \
  --source-code-path /Workspace/Users/<you>/.bundle/<bundle-name>/dev/files
```

The bundle applies these grants atomically:

- `CAN_USE` on the SQL warehouse
- `CAN_RUN` on the Genie Space
- `SELECT` on each UC table
- `READ_VOLUME` on the protocol_docs Volume

**Then apply the three grants the bundle can't (see
[gotcha 2](#2-three-grants-the-apps-bundle-schema-does-not-yet-support)):**
AI Gateway `CAN_QUERY`, VS endpoint `CAN_USE`, and `MODIFY` on
`supervisor_decisions` via SQL.

After deploy, fetch the Apps URL:

```bash
databricks apps get <app-name>
```

If you prefer the Apps UI to Asset Bundles: create the app via
`databricks apps create <name>`, sync the source, then add each
resource binding manually under Apps → Resources tab. The bundle
path is recommended because it's reproducible — the UI path is not.

### Step 10 — Run the eval

```bash
# In a Databricks workspace terminal (or wherever your CLI is authed):
pip install -r requirements.txt
python eval/run_eval.py
```

The run shows up in MLflow at `/Users/<you>/supervisor-eval`
(default) with per-row results attached as an artifact
(`_last_results.csv`). Compare against the [baselines in this
README](#eval-results--full-comparison).

---

## Extending this demo

Natural next iterations to add on top of the four already shipped:

**Agent Bricks Knowledge Assistant A/B.** Provision a managed
Knowledge Assistant pointed at the same `protocol_docs` UC Volume,
wire it as an alternative `VectorAgent` behind a
`VECTOR_AGENT_VARIANT` env var, and run both variants against the
same 18-question gold set. Answers the customer-facing question
"managed vs. custom — what's the actual tradeoff" on real data.

**Lakebase-backed chat memory.** `modules/memory.py` is already
present as a dormant scaffold. Activating it requires provisioning a
Lakebase Postgres project, declaring it as a `postgres` bundle
resource, and populating the `LAKEBASE_*` env vars in `app.yaml`.
Short-term memory (last N turns per conversation) drops in without
schema changes; long-term semantic memory can use Lakebase's
`pgvector` extension.

**Genie Space instruction tuning.** The g04 gap (Event A incidence)
is fixable with a Genie Space instruction nudging the SQL toward
LIKE-style matching on the adverse-event column where the gold
convention uses it.

---

## Repo conventions

- **`_archive/iteration-N/` snapshots ship with the repo.** Browse the
  frozen state of each iteration side-by-side with the live tree at the
  root. Snapshots are not on the Python path so editing them has no
  runtime effect.
- **Mock-mode fallback on every live call.** Router LLM, fusion
  LLM, Vector Search, Genie, and the decision-log writer each have a
  structured fallback path. The app always returns an answer;
  failures are logged to stderr and (when the decision log is live)
  to the `supervisor_decisions` UC table for follow-up.
- **Two-file deploy contract.** `app.yaml` is runtime config only;
  `databricks.yml` is the Asset Bundle. Resource bindings and
  permission grants live in the bundle; the runtime references them
  via `valueFrom:`.
