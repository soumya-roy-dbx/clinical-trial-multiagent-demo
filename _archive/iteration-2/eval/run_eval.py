"""MLflow eval harness for the multi-agent supervisor.

Loads eval/gold_questions.csv, runs every question through the deployed
SupervisorAgent against the live AI Gateway endpoints, and grades:
  - Routing accuracy (exact match: llm_route vs expected_route)
  - Fallback rate (% of turns where any fallback path fired)
  - Latency p50 / p95 (from app-measured handle() duration)
  - Answer quality (LLM judge: Claude scores each response 1-5 vs gold)

Results are logged as a single MLflow run with summary metrics and a
per-row results table.

Usage:
    python eval/run_eval.py                          # full run (Databricks tracking + LLM judge)
    python eval/run_eval.py --no-judge               # skip the judge step (fast)
    python eval/run_eval.py --tracking-uri ./mlruns  # local file backend instead
    python eval/run_eval.py --experiment /Users/me@databricks.com/supervisor-eval-v2

By default the run logs to Databricks MLflow (visible in the workspace
Experiments UI). Set --tracking-uri to a path (e.g. ./mlruns) to keep
runs local.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient

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


def run_eval(use_judge: bool, experiment: str) -> pd.DataFrame:
    gold_df = pd.read_csv(GOLD)
    agent = SupervisorAgent()
    judge_client = WorkspaceClient() if use_judge else None

    rows: list[dict] = []
    for _, q in gold_df.iterrows():
        t0 = time.monotonic()
        resp = agent.handle(q["question"])
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Re-derive routing decision by calling the router separately. This
        # tells us exactly what the LLM decided, independent of whether the
        # final route came from the LLM or the keyword fallback.
        try:
            decision = agent._call_router(q["question"])
            llm_route = decision.route
            fallback = None
        except Exception as e:
            llm_route = None
            fallback = type(e).__name__

        score, reason = (0, "")
        if use_judge:
            score, reason = llm_judge(
                judge_client, q["question"], q["gold_answer"], resp.answer
            )

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "expected_route": q["expected_route"],
            "llm_route": llm_route,
            "final_route": resp.source,
            "routing_correct": llm_route == q["expected_route"],
            "fallback_reason": fallback,
            "latency_ms": latency_ms,
            "answer_chars": len(resp.answer),
            "judge_score": score,
            "judge_reason": reason,
            "answer_head": resp.answer[:240],
        })
        print(f"  [{q['id']}] expected={q['expected_route']:6} "
              f"llm={str(llm_route):6} latency={latency_ms}ms "
              f"judge={score}")

    results = pd.DataFrame(rows)

    # Summary metrics.
    routing_acc = float(results["routing_correct"].mean())
    fallback_rate = float(results["fallback_reason"].notna().mean())
    p50 = float(statistics.median(results["latency_ms"]))
    p95 = float(statistics.quantiles(results["latency_ms"], n=20)[-1])
    judge_mean = float(results["judge_score"].mean()) if use_judge else None
    judge_pass = float((results["judge_score"] >= 4).mean()) if use_judge else None

    print("\n=== Summary ===")
    print(f"  routing_accuracy : {routing_acc:.1%} ({results['routing_correct'].sum()}/{len(results)})")
    print(f"  fallback_rate    : {fallback_rate:.1%}")
    print(f"  latency_p50 (ms) : {p50:.0f}")
    print(f"  latency_p95 (ms) : {p95:.0f}")
    if use_judge:
        print(f"  judge_mean       : {judge_mean:.2f} / 5")
        print(f"  judge_pass (>=4) : {judge_pass:.1%}")

    # Per-route breakdown.
    print("\n=== Routing accuracy by expected route ===")
    print(results.groupby("expected_route")["routing_correct"].agg(["mean", "count"]).to_string())

    # Log to MLflow.
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_metric("routing_accuracy", routing_acc)
        mlflow.log_metric("fallback_rate", fallback_rate)
        mlflow.log_metric("latency_p50_ms", p50)
        mlflow.log_metric("latency_p95_ms", p95)
        if use_judge:
            mlflow.log_metric("judge_mean", judge_mean)
            mlflow.log_metric("judge_pass_at_4", judge_pass)
        mlflow.log_metric("n_questions", len(results))
        mlflow.log_param("router_endpoint",
                         agent.genie.__class__.__name__ + "+supervisor")
        mlflow.log_param("use_judge", use_judge)
        # Persist the per-row table.
        out_csv = ROOT / "eval" / "_last_results.csv"
        results.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))
        print(f"\nMLflow run: {run.info.run_id}")
        print(f"Per-row results: {out_csv}")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM-judge step (routing/latency only).")
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
