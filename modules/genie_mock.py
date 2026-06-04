"""Mock Genie behavior + the query dispatcher shared with the multi-agent path.

Each function returns a small dict with the answer payload. The dispatcher is
also imported by `agents.genie_agent` for its mock mode.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable

import pandas as pd

from modules.db import _load


def _treatment_count() -> dict:
    df = _load()
    n = int((df["arm"] == "Treatment").sum())
    return {"answer": f"There are {n} subjects in the Treatment group.", "value": n}


def _biomarker_distribution() -> dict:
    df = _load()
    dist = df["biomarker_status"].value_counts().to_dict()
    return {
        "answer": ", ".join(f"{k}: {v}" for k, v in dist.items()),
        "value": dist,
    }


def _region_counts() -> dict:
    df = _load()
    sub = df[df["arm"] == "Treatment"]
    by_region = sub.groupby("region").size().to_dict()
    return {
        "answer": ", ".join(f"{k}: {v}" for k, v in by_region.items()),
        "value": by_region,
    }


def _event_incidence(event: str) -> dict:
    df = _load()
    tr = df[df["arm"] == "Treatment"]
    n = int(tr["adverse_event"].str.contains(event, na=False).sum())
    total = int(len(tr))
    rate = n / total if total else 0.0
    return {
        "answer": f"Incidence of {event} in Treatment: {n}/{total} ({rate:.1%}).",
        "value": rate,
    }


_QUERY_DISPATCH: dict[str, Callable[[], dict]] = {
    "how many subjects are in the treatment group": _treatment_count,
    "what is the subject count by region": _region_counts,
    "what is the biomarker status distribution": _biomarker_distribution,
    "what is the incidence of event a in the treatment arm": lambda: _event_incidence("Event A"),
    "what is the incidence of event b by sub-type": lambda: _event_incidence("Event B"),
}


def answer(question: str) -> dict:
    """Fuzzy-match question against canned queries; return a result + confidence."""
    q = question.lower().strip().rstrip("?")
    best_key, best_score = None, 0.0
    for key in _QUERY_DISPATCH:
        score = SequenceMatcher(None, q, key).ratio()
        if score > best_score:
            best_key, best_score = key, score
    if best_key and best_score >= 0.55:
        out = _QUERY_DISPATCH[best_key]()
        confidence = 0.9 if best_score >= 0.8 else 0.6
        return {**out, "confidence": confidence, "matched": best_key}
    return {
        "answer": "I don't have data for that question in the mock dataset.",
        "confidence": 0.1,
        "matched": None,
    }
