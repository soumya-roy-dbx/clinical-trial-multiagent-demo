# Archive

Frozen snapshots of each iteration the app builds up through. Browse
the state of any earlier iteration side-by-side with the current code
at the root of the repo.

## Convention

- `_archive/iteration-N/` is a complete copy of all project files at
  the end of iteration N.
- Snapshots are **not on the Python path** — the live app always runs
  from the root of the repo. Editing or deleting files under
  `_archive/` has no effect on the deployed app.
- Each snapshot was captured before the next iteration's work began.

## Snapshots present

- `iteration-0/` — local-runnable baseline: Subject Disposition
  dashboard rendered from a synthetic CSV, single-LLM supervisor stub
  falling through to keyword routing, TF-IDF over markdown for
  vector-search stand-in, pandas dispatcher for Genie stand-in. No AI
  Gateway, no eval, no inference tables, no UC writes. The
  comparison point for all later iterations.

- `iteration-1/` — two-tier supervisor (Llama 3.3 70B router + Claude
  Sonnet 4.6 fusion) calling AI Gateway endpoints with PII
  guardrails, usage tracking, and inference tables. Adds
  Pydantic-validated routing with few-shot prompting and a one-shot
  corrective retry; adds an app-level decision log
  (`modules/decision_log.py`); adds the MLflow eval harness
  (`eval/`) with 18 gold questions, routing-accuracy + latency +
  Claude-as-judge metrics. Baseline: 100% routing accuracy, 3.33 / 5
  judge mean, judge_pass@4 55.6%.

- `iteration-2/` — TF-IDF mock replaced with a Databricks Vector
  Search pipeline. Adds `data/chunking.py` (section-aware markdown
  chunker, ~800 tokens / ~150 overlap, doc_type + section metadata)
  and `data/build_chunks_table.py` (one-shot loader to a Delta
  table). `agents/vector_agent.py` `_handle_live` queries the VS
  index for top-3 chunks then synthesizes via the Claude fusion
  gateway. Vector judge score jumps 3.17 → 4.00; latency p50 rises
  2.9s → 9.5s as the cost of real RAG.

- `iteration-3/` — production-shaped iteration: UC Volume for
  protocol docs, Asset Bundle declaring app + resource bindings +
  permission grants in `databricks.yml`, decision log flipped to UC
  INSERT, and live Genie wiring via per-question conversation
  rotation. `genie_agent.py` `_handle_live` calls
  `start_conversation_and_wait` on the Clinical Trial Subject
  Disposition Genie Space. Hybrid category jumps from mixed
  (5,2,4,3,3,1) to consistently strong (5,5,4,5,4,4); judge_mean
  rises 3.56 → 4.06, judge_pass@4 55.6% → 77.8%. Latency p50 13.7s
  (Genie adds ~10s per genie / hybrid turn).

> **Note on the iteration-3 snapshot:** the earlier capture of this
> iteration declared resources inside `app.yaml`'s `resources:`
> block. That is the wrong Databricks file boundary — `app.yaml` is
> runtime-only and resources + permissions belong in `databricks.yml`
> (Asset Bundle) or the Apps UI Resources tab. The corrected
> structure lives at the root of the repo; gotcha #1 in the main
> `README.md` covers the details.
