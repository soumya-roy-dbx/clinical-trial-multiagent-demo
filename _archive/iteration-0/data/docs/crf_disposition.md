# CRF — Subject Disposition (MOCK)

## Purpose
Track each subject's status through the study (Completed / Discontinued /
Ongoing) and the reason for any early discontinuation.

## Fields
- `subject_id`
- `disposition` — Completed / Discontinued / Ongoing.
- (Free-text reason captured in production CRF; omitted from the mock dataset.)

## Populations
- **Overall population** — all enrolled subjects.
- **Indicated population** — biomarker-positive subjects in the Treatment arm,
  used for the primary efficacy analysis in the sub-study.

## Reporting
The dashboard reports Treatment-arm subject counts overall, for the indicated
population, and by region.
