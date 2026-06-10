"""MLflow tracing setup — the app's observability layer.

Two responsibilities, both designed to *never break the app*:

1. `trace` — a decorator the agents put on their hot-path methods. If MLflow
   is importable it is `mlflow.trace` (each call becomes a span in a trace);
   if MLflow is missing it degrades to a no-op decorator, so the agents stay
   importable in any environment.

2. `enable_tracing()` — called once at app startup. Points MLflow at the
   Databricks tracking backend and an experiment so traces are exported and
   visible in the workspace Experiments UI. If anything is unavailable
   (no MLflow, no Databricks auth, no experiment permission) it disables
   tracing and returns False instead of raising.

Toggle with env vars (see app.yaml):
  MLFLOW_TRACING_ENABLED  "true" (default) | "false"
  MLFLOW_EXPERIMENT       explicit experiment path the app SP can write to
  MLFLOW_TRACKING_URI      defaults to "databricks"
"""
from __future__ import annotations

import json
import os
import sys

try:
    import mlflow

    trace = mlflow.trace  # real spans
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover - exercised only where mlflow is absent
    _HAS_MLFLOW = False

    def trace(func=None, *, name=None, span_type=None, attributes=None, **_kwargs):
        """No-op stand-in for `mlflow.trace` when MLflow isn't installed."""
        def _decorate(f):
            return f

        return _decorate(func) if callable(func) else _decorate


_DEFAULT_EXPERIMENT_LEAF = "clinical-assistant-traces"
_tracing_on = False


def enable_tracing(experiment: str | None = None) -> bool:
    """Configure MLflow tracing against Databricks. Safe to call repeatedly.

    Returns True if tracing is active, False if it was disabled or unavailable.
    """
    global _tracing_on
    if _tracing_on:
        return True
    if not _HAS_MLFLOW:
        return False
    if os.getenv("MLFLOW_TRACING_ENABLED", "true").lower() == "false":
        _safe_disable()
        return False
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
        exp = experiment or os.getenv("MLFLOW_EXPERIMENT") or _derive_experiment()
        if exp:
            mlflow.set_experiment(exp)
        _tracing_on = True
        return True
    except Exception as exc:
        # Tracing is best-effort observability; never let it break the app.
        print(json.dumps({"tracing_setup_skipped": str(exc)[:200]}),
              file=sys.stderr, flush=True)
        _safe_disable()
        return False


def set_trace_tags(tags: dict) -> None:
    """Tag the active trace (e.g. with the chosen route) for easy filtering.

    No-op when MLflow is absent or no trace is active. Never raises.
    """
    if not _HAS_MLFLOW:
        return
    try:
        # Only tag when a span is actually active (avoids a noisy warning when
        # handle() is called outside a trace, e.g. MLflow's eval validation probe).
        if mlflow.get_current_active_span() is None:
            return
        mlflow.update_current_trace(tags={k: str(v) for k, v in tags.items()})
    except Exception:
        pass


def _derive_experiment() -> str | None:
    """Default experiment under the current identity's home folder."""
    try:
        from databricks.sdk import WorkspaceClient

        user = WorkspaceClient().current_user.me().user_name
        return f"/Users/{user}/{_DEFAULT_EXPERIMENT_LEAF}"
    except Exception:
        return None


def _safe_disable() -> None:
    try:
        mlflow.tracing.disable()  # makes the @trace decorators inert
    except Exception:
        pass
