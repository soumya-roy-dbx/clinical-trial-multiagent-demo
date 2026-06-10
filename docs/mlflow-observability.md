# MLflow: observability and evaluation

This app uses MLflow for two things:

1. **Tracing (observability)** — every chat turn is recorded as a *trace*: a
   timeline of the steps the supervisor took, with timings and inputs/outputs.
2. **Evaluation** — a repeatable scorecard that runs a fixed set of questions
   through the app and grades routing, answer quality, and latency, using
   `mlflow.genai.evaluate(...)`.

Both write to a **Databricks MLflow experiment**, so you view everything in the
workspace under **Experiments** — no extra infrastructure to stand up.

---

## What tracing captures

When tracing is on, each user question produces one trace containing nested
spans that mirror the agent flow:

```
supervisor (AGENT)
├─ router (LLM)              ← which route: genie | vector | hybrid
├─ genie_agent (AGENT)       ← if structured
├─ vector_agent (AGENT)      ← if document lookup
│   ├─ vector_search (RETRIEVER)
│   └─ rag_synthesize (LLM)
└─ fusion (LLM)              ← if hybrid, merges both answers
```

For each span you get: the input, the output, how long it took, and any error.
Each trace is also **tagged** with `llm_route`, `final_route`, and
`fallback_reason`, so in the **Traces** tab you can filter to, say, every turn
that fell back to the keyword router, or every `hybrid` turn that was slow.

This answers day-to-day questions like *"why was that answer wrong?"*,
*"which step is slow?"*, and *"how often does routing fall back?"* — without
adding print statements.

## What evaluation captures

`eval/run_eval.py` runs every question in `eval/gold_questions.csv` through the
real app and applies three scorers:

| Scorer | What it measures | How |
| --- | --- | --- |
| `routing_correctness` | Did the router pick the expected agent? | Exact match vs the gold label (deterministic, no LLM) |
| `answer_correctness` | Is the answer factually right vs the gold answer? | LLM judge — Claude via the AI Gateway, scored 1–5 |
| `latency_ms` | How fast was each turn? | App-measured `handle()` duration |

You get aggregated metrics (routing accuracy, mean judge score, latency
mean/median/p90) **and** a per-row table linked to each row's trace, all in the
**Evaluation** tab of the run. Re-run after any change to see the metrics move.

> **Why a custom judge instead of MLflow's built-in `Correctness` scorer?**
> The built-in LLM judges depend on Databricks-managed judge endpoints (Agent
> Evaluation), which aren't enabled in every workspace. This repo already calls
> Claude through the AI Gateway, so reusing it as the judge keeps the eval
> self-contained and adds inference-table logging for free. If your workspace
> has Agent Evaluation, you can swap in the built-ins — see [below](#optional-use-mlflows-built-in-judges).

---

## Where to look in the code

| File | Role |
| --- | --- |
| `modules/observability.py` | The `@trace` decorator (safe no-op if MLflow is absent) and `enable_tracing()` |
| `agents/supervisor.py`, `agents/vector_agent.py`, `agents/genie_agent.py` | `@trace(...)` on the hot-path methods — this is what creates the spans |
| `app.py` | Calls `enable_tracing()` once at startup |
| `eval/run_eval.py` | The `mlflow.genai.evaluate(...)` harness, scorers, and the Claude judge (`llm_judge`) |
| `eval/gold_questions.csv` | The fixed question set: `question`, `expected_route`, `gold_answer` |
| `app.yaml` | Tracing toggles: `MLFLOW_TRACING_ENABLED`, `MLFLOW_EXPERIMENT` |

---

## How to run the evaluation

From the repo root, with a Databricks profile configured:

```bash
# Full run: routing + latency + LLM judge, logged to Databricks MLflow
python eval/run_eval.py

# Faster: skip the LLM judge (routing + latency only)
python eval/run_eval.py --no-judge

# Keep results local instead of Databricks
python eval/run_eval.py --tracking-uri ./mlruns

# Choose the experiment
python eval/run_eval.py --experiment /Users/you@example.com/supervisor-eval
```

The script prints a summary and a link to the run. Open it and go to the
**Evaluation** and **Traces** tabs for the per-row detail.

> The eval calls live Genie + Vector Search + the LLMs, so a full 18-question
> run takes several minutes. Use `--no-judge` for a quick routing/latency check.

## Required setup

- **Dependencies:** `pip install -r requirements.txt` (needs `mlflow>=3.1` for
  `mlflow.genai`).
- **Auth:** a Databricks profile / `databricks auth login`. If more than one
  profile matches your host, set `DATABRICKS_CONFIG_PROFILE`.
- **App-side tracing (in Databricks Apps):**
  - `MLFLOW_TRACING_ENABLED=true` (default) and `MLFLOW_EXPERIMENT=<experiment path>` in `app.yaml`.
  - Grant the **app service principal `CAN_EDIT`** on that experiment so it can write traces. (Same kind of one-time grant as the other resources in the README.)
  - Set `MLFLOW_TRACING_ENABLED=false` to turn tracing off entirely.

Tracing is best-effort: if MLflow isn't installed or the experiment isn't
writable, the app logs a one-line note and runs normally — tracing never breaks
the app.

---

## How this helps

**In development**
- See exactly which step produced a bad answer and how long each took.
- The eval gives you a number to optimize: change a prompt, the chunker, or a
  model, re-run, and watch routing accuracy / judge score / latency move.
- Filter traces by route or fallback to find regressions fast.

**In production**
- Tracing is your audit trail: every answer keeps the steps, sources, and
  timings behind it — valuable in a regulated (clinical) setting.
- Run the same eval as a release gate before promoting a change.
- Trace tags + the AI Gateway inference table together give routing mix,
  fallback rate, latency percentiles, token use, and cost.

---

## Optional: use MLflow's built-in judges

On a workspace with Agent Evaluation, you can add MLflow's built-in scorers
alongside (or instead of) the custom judge. They read the `expected_response`
already in the dataset:

```python
from mlflow.genai.scorers import Correctness, Safety

scorers = [routing_correctness, latency_ms, Correctness(), Safety()]
result = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
```

If those scorers error with a judge/endpoint message, your workspace doesn't
have managed judges enabled — stick with the custom `answer_correctness` scorer.
