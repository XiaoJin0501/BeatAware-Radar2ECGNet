---
name: metrics-audit
description: Audit metric definitions, units, aggregation, and manuscript table consistency.
---

# Metrics Audit

Use when changing evaluation code, generating tables, or interpreting results.

Audit:

- metric formula;
- sampling rate;
- units;
- event tolerance;
- macro vs pooled aggregation;
- subject/fold/state aggregation;
- NaN handling;
- confidence interval or standard deviation definition;
- consistency between code, manifest, and manuscript table.

Flag any metric that lacks a contract in `docs/METRIC_CONTRACT.md`.
