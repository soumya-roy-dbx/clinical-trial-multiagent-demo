"""Data-loading layer.

Reads the mocked study CSV and exposes cached KPI functions that mirror the
shape of the production SQL Warehouse queries.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from modules.config import BIOMARKER_POSITIVE, TREATMENT_LABEL

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "study_subjects.csv"


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def get_overall_kpi() -> int:
    df = _load()
    return int((df["arm"] == TREATMENT_LABEL).sum())


def get_indicated_kpi() -> int:
    df = _load()
    mask = (df["arm"] == TREATMENT_LABEL) & (
        df["biomarker_status"] == BIOMARKER_POSITIVE
    )
    return int(mask.sum())


def get_region_table(indicated: bool = False) -> pd.DataFrame:
    df = _load()
    mask = df["arm"] == TREATMENT_LABEL
    if indicated:
        mask &= df["biomarker_status"] == BIOMARKER_POSITIVE
    sub = df[mask]
    grand_total = int(mask.sum())
    by_region = (
        sub.groupby("region", as_index=False)
        .size()
        .rename(columns={"size": "Subject count in Treatment"})
    )
    by_region["Grand Total"] = grand_total
    by_region = by_region.rename(columns={"region": "Region"})
    return by_region
