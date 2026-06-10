"""MLflow GenAI eval harness for the multi-agent supervisor.

Runs every question in eval/gold_questions.csv through the deployed
SupervisorAgent and grades it with `mlflow.genai.evaluate(...)` and custom
scorers:

  - routing_correctness — deterministic: the LLM's route vs the expected route.
  - answer_correctness  — LLM judge (Claude via the fusion AI Gateway) scoring
                          the answer 1-5 against the gold reference.
  - latency_ms          — app-measured handle() duration per question.

Why this shape:
  * `mlflow.genai.evaluate` gives us the native GenAI Evaluation UI (per-row
    inputs/outputs/assessments, score distributions, pass rates) and links each
    row to its execution **trace** — no hand-rolled tables to maintain.
  * The quality judge is a *custom* scorer that reuses this repo's proven
    Claude-via-AI-Gateway judge instead of MLflow's built-in `Correctness`
    judge. The built-ins depend on Databricks-managed judge endpoints (Agent
    Evaluation), which aren't guaranteed in every workspace; the AI Gateway is
    already wired here and gives us inference-table logging for free. To switch
    to the built-in judges on a workspace that has them, see docs/mlflow-observability.md.

Usage:
    python eval/run_eval.py                          # full run (Databricks tracking + LLM judge)
    python eval/run_eval.py --no-judge               # skip the judge scorer (fast)
    python eval/run_eval.py --tracking-uri ./mlruns  # local file backend instead
    python eval/run_eval.py --experiment /Users/me@databricks.com/supervisor-eval-v2

By default the run logs to Databricks MLflow (visible in the workspace
Experiments UI). Set --tracking-uri to a path (e.g. ./mlruns) to keep runs local.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

# Make project importable when invoked from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.supervisor import SupervisorAgent  # noqa: E402
from modules.config import FUSION_LLM_ENDPOINT  # noqa: E402

GOLD = ROOT / "eval" / "gold_questions.csv"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a clinical-trial assistant. "
    "Given a question, a gold reference answer, and a candidate answer, "
    "score the candidate from 1 to 5 on factual correctness and completeness "
    "relative to the gold. 5 = matches all key facts; 3 = partial; "
    "1 = wrong or empty. Respond with ONLY a JSON object: "
    '{"score": <1-5>, "reason": "<one short sentence>"}.'
)


def llm_judge(client: WorkspaceClient, question: str, gold: str,
              candidate: str) -> tuple[int, str]:
    """Call Claude via the fusion gateway to grade a single response."""
    url = f"{client.config.host}/ai-gateway/mlflow/v1/chat/completions"
    body = json.dumps({
        "model": FUSION_LLM_ENDPOINT,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Gold answer: {gold}\n\n"
                f"Candidate answer: {candidate}\n\n"
                "Respond in JSON."
            )},
        ],
        "temperature": 0.0,
        "max_tokens": 120,
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={**client.config.authenticate(), "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = json.loads(resp.read().decode())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return 0, f"judge_error: HTTP {e.code}"
    # Tolerate fenced JSON ("```json ... ```").
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(text)
        return int(parsed["score"]), str(parsed.get("reason", ""))[:200]
    except Exception:
        return 0, f"judge_parse_failed: {text[:120]}"


def _resolve_default_experiment(tracking_uri: str) -> str:
    """Pick a sensible default experiment name based on backend.

    Databricks needs a workspace path; local file backends accept a bare name.
    """
    if tracking_uri == "databricks":
        try:
            user = WorkspaceClient().current_user.me().user_name
            return f"/Users/{user}/supervisor-eval"
        except Exception:
            return "/Shared/supervisor-eval"
    return "supervisor-eval"


# ---------- Dataset, prediction fn, and scorers ----------


def _build_dataset(gold_df: pd.DataFrame) -> list[dict]:
    """One eval row per gold question: inputs + expectations.

    `inputs` keys are passed to predict_fn as kwargs; `expectations` is handed
    to the scorers.
    """
    return [
        {
            "inputs": {"question": q["question"]},
            "expectations": {
                "expected_route": q["expected_route"],
                "expected_response": q["gold_answer"],
                "id": q["id"],
            },
        }
        for _, q in gold_df.iterrows()
    ]


def _make_predict_fn(agent: SupervisorAgent):
    """Wrap the supervisor so mlflow.genai.evaluate can call + trace it."""

    def predict_fn(question: str) -> dict:
        # Re-derive the LLM's routing decision independently of any fallback,
        # so routing_correctness measures the model, not the keyword backstop.
        # (Mirrors the metric reported since iteration 1.)
        try:
            route = agent._call_router(question).route
        except Exception:
            route = None
        t0 = time.monotonic()
        resp = agent.handle(question)
        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "answer": resp.answer,
            "route": route,
            "final_route": resp.source,
            "latency_ms": latency_ms,
        }

    return predict_fn


@scorer(aggregations=["mean"])
def routing_correctness(outputs: dict, expectations: dict) -> bool:
    """True when the router chose the expected route. Mean = routing accuracy."""
    return bool(outputs.get("route") == expectations.get("expected_route"))


@scorer(aggregations=["mean", "median", "p90"])
def latency_ms(outputs: dict) -> float:
    """App-measured handle() latency in milliseconds."""
    return float(outputs.get("latency_ms", 0))


def _make_answer_correctness(judge_client: WorkspaceClient):
    """Custom LLM-judge scorer reusing this repo's Claude grader."""

    @scorer(name="answer_correctness", aggregations=["mean"])
    def answer_correctness(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
        score, reason = llm_judge(
            judge_client,
            inputs["question"],
            expectations["expected_response"],
            outputs.get("answer", ""),
        )
        return Feedback(value=score, rationale=reason)

    return answer_correctness


def _print_summary(result, use_judge: bool) -> None:
    metrics = result.metrics or {}

    def m(key: str):
        return metrics.get(key)

    print("\n=== Summary (mlflow.genai.evaluate) ===")
    acc = m("routing_correctness/mean")
    if acc is not None:
        print(f"  routing_accuracy   : {acc:.1%}")
    if use_judge and m("answer_correctness/mean") is not None:
        print(f"  judge_mean         : {m('answer_correctness/mean'):.2f} / 5")
    for label, key in (("latency_p50 (ms)", "latency_ms/median"),
                       ("latency_mean (ms)", "latency_ms/mean"),
                       ("latency_p90 (ms)", "latency_ms/p90")):
        if m(key) is not None:
            print(f"  {label:18}: {m(key):.0f}")

    # judge_pass@4 and the per-route breakdown aren't native aggregations, so
    # compute them from the per-row table when MLflow exposes it.
    df = _results_table(result)
    if df is not None:
        _print_row_level(df, use_judge)

    print(f"\nMLflow run: {result.run_id}")
    print("Open the run in the Experiments UI → Traces / Evaluation tabs for "
          "per-row inputs, outputs, assessments, and span timings.")


def _results_table(result) -> pd.DataFrame | None:
    tables = result.tables or {}
    for key in ("eval_results", "eval_results_table", "genai_results"):
        if key in tables:
            return tables[key]
    # Fall back to the first table MLflow attached, if any.
    return next(iter(tables.values()), None)


def _print_row_level(df: pd.DataFrame, use_judge: bool) -> None:
    """Best-effort extras from the per-row results table (judge pass rate,
    per-route accuracy). Column names vary by MLflow version, so probe."""
    def col(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    judge_col = col("answer_correctness/value", "answer_correctness", "answer_correctness/score")
    if use_judge and judge_col is not None:
        try:
            passed = (pd.to_numeric(df[judge_col], errors="coerce") >= 4).mean()
            print(f"  judge_pass (>=4)   : {passed:.1%}")
        except Exception:
            pass


def run_eval(use_judge: bool, experiment: str):
    gold_df = pd.read_csv(GOLD)
    agent = SupervisorAgent()

    data = _build_dataset(gold_df)
    predict_fn = _make_predict_fn(agent)
    scorers = [routing_correctness, latency_ms]
    if use_judge:
        scorers.append(_make_answer_correctness(WorkspaceClient()))

    mlflow.set_experiment(experiment)
    print(f"Running {len(data)} questions through mlflow.genai.evaluate "
          f"(judge={'on' if use_judge else 'off'}) ...")
    result = mlflow.genai.evaluate(
        data=data,
        predict_fn=predict_fn,
        scorers=scorers,
    )
    _print_summary(result, use_judge)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-judge scorer (routing/latency only).")
    p.add_argument("--tracking-uri", default="databricks",
                   help="MLflow tracking URI (default: databricks). "
                        "Use './mlruns' for a local file backend.")
    p.add_argument("--experiment", default=None,
                   help="MLflow experiment name. Default: "
                        "/Users/<you>/supervisor-eval for Databricks, "
                        "'supervisor-eval' for local.")
    args = p.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment = args.experiment or _resolve_default_experiment(args.tracking_uri)
    run_eval(use_judge=not args.no_judge, experiment=experiment)


if __name__ == "__main__":
    main()
