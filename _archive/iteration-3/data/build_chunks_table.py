"""(Re)build the Delta table that backs the Vector Search index.

Reads `data/docs/*.md`, runs `data.chunking.chunk_all`, then writes the
resulting rows to `your_catalog.your_schema.protocol_chunks` via
the SQL warehouse Statement Execution API. The VS Delta Sync index
configured on top of this table will pick up changes on its next sync.

Run any time the markdown corpus changes:

    python data/build_chunks_table.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.chunking import chunk_all  # noqa: E402

# Source corpus. Default reads the repo-local data/docs/ directory; in
# production the same files live in the UC Volume
# /Volumes/your_catalog/your_schema/protocol_docs/ — point DOCS_DIR
# at that path (with appropriate Databricks SDK file IO) when running from
# a workspace job.
DOCS_DIR = ROOT / "data" / "docs"
TABLE = "your_catalog.your_schema.protocol_chunks"
WAREHOUSE_ID = "your_warehouse_id"  # Serverless Starter Warehouse

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    chunk_id     STRING NOT NULL,
    doc_type     STRING NOT NULL,
    section      STRING,
    chunk_index  INT,
    chunk_text   STRING,
    token_count  INT,
    source_path  STRING
)
USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""".strip()

INSERT = f"""
INSERT INTO {TABLE}
    (chunk_id, doc_type, section, chunk_index, chunk_text, token_count, source_path)
VALUES
    (:chunk_id, :doc_type, :section,
     CAST(:chunk_index AS INT), :chunk_text,
     CAST(:token_count AS INT), :source_path)
""".strip()


def _exec(w: WorkspaceClient, statement: str, params: list | None = None) -> None:
    w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        parameters=params or [],
        wait_timeout="30s",
    )


def main() -> None:
    chunks = chunk_all(DOCS_DIR)
    print(f"Chunked {len(chunks)} rows from {len(list(DOCS_DIR.glob('*.md')))} docs")

    w = WorkspaceClient()

    print(f"Ensuring table {TABLE} exists (warehouse starts on first call)...")
    t0 = time.monotonic()
    _exec(w, DDL)
    print(f"  ok ({int((time.monotonic()-t0)*1000)}ms)")

    print(f"Truncating {TABLE}...")
    _exec(w, f"TRUNCATE TABLE {TABLE}")

    print(f"Inserting {len(chunks)} rows...")
    t0 = time.monotonic()
    for i, c in enumerate(chunks, 1):
        params = [
            StatementParameterListItem(name=k, value=str(v))
            for k, v in asdict(c).items()
        ]
        _exec(w, INSERT, params)
        if i % 10 == 0:
            print(f"  inserted {i}/{len(chunks)}")
    print(f"  done ({int((time.monotonic()-t0)*1000)}ms total)")

    print(f"\nVerify in SQL editor:")
    print(f"  SELECT COUNT(*) FROM {TABLE};")
    print(f"  SELECT doc_type, COUNT(*) FROM {TABLE} GROUP BY doc_type;")


if __name__ == "__main__":
    main()
