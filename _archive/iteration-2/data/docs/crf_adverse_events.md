# CRF — Adverse Events (MOCK)

## Purpose
Record all adverse events (AEs) reported during the study, with seriousness
flag and protocol-defined event-of-interest classification.

## Fields
- `subject_id` — links to demographics.
- `adverse_event` — None / Event A / Event B / Event A+B.
- `serious_ae` — 0/1 flag indicating whether the event met serious criteria
  (death, hospitalization, life-threatening, persistent disability, congenital
  anomaly, or important medical event).

## Event Definitions
- **Event A** — protocol-defined adverse event of interest, symptomatic
  presentation.
- **Event B** — protocol-defined adverse event of interest, laboratory
  presentation.

## Incidence Computation
Incidence rate is computed as `count(subjects with event) / count(subjects in
arm)`, reported separately for Treatment and Control and overall.
