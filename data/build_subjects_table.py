"""(Re)build the Delta table that backs the Genie Space.

Reads `data/study_subjects.csv` and writes to
`your_catalog.your_schema.study_subjects` via the SQL warehouse
Statement Execution API. Run once after the CSV is regenerated.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "study_subjects.csv"
TABLE = os.getenv(
    "STUDY_SUBJECTS_TABLE", "your_catalog.your_schema.study_subjects"
)
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "your_warehouse_id")

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    subject_id        STRING NOT NULL COMMENT 'Site-assigned unique identifier (S####)',
    age               INT    COMMENT 'Years at consent (50-85)',
    sex               STRING COMMENT 'F or M',
    region            STRING COMMENT 'Eastland / Northland / Westland',
    arm               STRING COMMENT 'Treatment or Control',
    biomarker_status  STRING COMMENT 'Positive or Negative at screening',
    adverse_event     STRING COMMENT 'None / Event A / Event B / Event A+B',
    serious_ae        INT    COMMENT '0 or 1 flag for serious AE criteria',
    disposition       STRING COMMENT 'Completed / Discontinued / Ongoing'
)
USING DELTA
COMMENT 'Synthetic clinical-trial subject roster for demo Genie Space.'
""".strip()

INSERT = f"""
INSERT INTO {TABLE}
    (subject_id, age, sex, region, arm, biomarker_status, adverse_event, serious_ae, disposition)
VALUES
    (:subject_id, CAST(:age AS INT), :sex, :region, :arm, :biomarker_status,
     :adverse_event, CAST(:serious_ae AS INT), :disposition)
""".strip()


def _exec(w: WorkspaceClient, statement: str, params: list | None = None) -> None:
    w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        parameters=params or [],
        wait_timeout="30s",
    )


def main() -> None:
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    w = WorkspaceClient()
    print(f"Ensuring table {TABLE} exists...")
    _exec(w, DDL)

    print(f"Truncating {TABLE}...")
    _exec(w, f"TRUNCATE TABLE {TABLE}")

    print(f"Inserting {len(rows)} rows...")
    t0 = time.monotonic()
    for i, row in enumerate(rows, 1):
        params = [StatementParameterListItem(name=k, value=v) for k, v in row.items()]
        _exec(w, INSERT, params)
        if i % 25 == 0:
            print(f"  inserted {i}/{len(rows)}")
    print(f"  done ({int((time.monotonic()-t0)*1000)}ms total)")

    print("\nVerify:")
    print(f"  SELECT COUNT(*) FROM {TABLE};")
    print(f"  SELECT arm, COUNT(*) FROM {TABLE} GROUP BY arm;")


if __name__ == "__main__":
    main()
