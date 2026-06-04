"""Generate a synthetic clinical-trial subject dataset for the demo.

Run: `python data/mock_data.py` to (re)produce data/study_subjects.csv.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

REGIONS = ["Eastland", "Northland", "Westland"]
ARMS = ["Treatment", "Control"]
SEX = ["F", "M"]
BIOMARKER = ["Positive", "Negative"]
DISPOSITION = ["Completed", "Discontinued", "Ongoing"]
AE_TYPES = ["None", "Event A", "Event B", "Event A+B"]

N_SUBJECTS = 100
SEED = 42

OUT = Path(__file__).parent / "study_subjects.csv"


def generate() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for i in range(1, N_SUBJECTS + 1):
        arm = rng.choice(ARMS)
        biomarker = rng.choices(BIOMARKER, weights=[0.55, 0.45])[0]
        ae = rng.choices(AE_TYPES, weights=[0.55, 0.20, 0.15, 0.10])[0]
        serious_ae = ae != "None" and rng.random() < 0.25
        rows.append(
            {
                "subject_id": f"S{i:04d}",
                "age": rng.randint(50, 85),
                "sex": rng.choice(SEX),
                "region": rng.choice(REGIONS),
                "arm": arm,
                "biomarker_status": biomarker,
                "adverse_event": ae,
                "serious_ae": int(serious_ae),
                "disposition": rng.choices(
                    DISPOSITION, weights=[0.6, 0.15, 0.25]
                )[0],
            }
        )
    return rows


def main() -> None:
    rows = generate()
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
