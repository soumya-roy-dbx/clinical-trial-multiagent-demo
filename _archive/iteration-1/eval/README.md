# Eval Harness

Regression gate for the supervisor: routing accuracy + latency + (optional)
LLM-judged answer quality, all logged as a single MLflow run.

## Run

```bash
# Default: logs to Databricks MLflow at /Users/<you>/supervisor-eval
python eval/run_eval.py

# Skip the judge — fast routing + latency only
python eval/run_eval.py --no-judge

# Local file backend instead (./mlruns), browse with: mlflow ui
python eval/run_eval.py --tracking-uri ./mlruns

# Custom experiment name
python eval/run_eval.py --experiment /Users/me@databricks.com/supervisor-eval-iteration-1
```

The script needs the same authentication as the deployed app — a valid
Databricks profile that can call the AI Gateway endpoints. When run
locally, it reads from `~/.databrickscfg`; in a Databricks notebook or
in Apps it picks up the SP token automatically.

## What gets graded

| Metric | How |
|---|---|
| `routing_accuracy` | Exact match between the router LLM's chosen route and `expected_route` in the gold set. |
| `fallback_rate` | Fraction of turns where any part of the LLM pipeline failed and a fallback path fired. |
| `latency_p50_ms`, `latency_p95_ms` | App-measured `handle()` duration (matches what the decision log captures in production). |
| `judge_mean`, `judge_pass_at_4` | Claude (the fusion endpoint) scores each candidate answer 1–5 against the gold answer; pass = score ≥ 4. |

## When to run

- Before any change that touches `agents/supervisor.py` or the gold set —
  capture a baseline run for comparison.
- After any prompt change, model swap, or chunker/retrieval change.
- Periodically as a regression check.

## Where results land

- **Databricks Experiments UI** (default): runs appear under
  `/Users/<you>/supervisor-eval` — open the workspace Experiments page,
  filter by that path, click into a run to see metrics + the per-row CSV
  artifact. Use the "Compare" button across multiple runs to track
  iteration-over-iteration progress.
- **Local file backend** (when `--tracking-uri ./mlruns`): browse with
  `mlflow ui` then open <http://localhost:5000>.
- **Per-row CSV** is also written to `eval/_last_results.csv` locally on
  every run (gitignored — overwritten each time).

## Editing the gold set

`eval/gold_questions.csv` has columns `id, expected_route, question,
gold_answer`. Add new rows for coverage gaps you find in production
traffic (see the AI Gateway inference table —
`your_catalog.your_schema.gateway_inference_payload` — for real
prompts users have sent).
