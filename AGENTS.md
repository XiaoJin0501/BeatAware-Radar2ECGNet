# Codex Project Instructions

This file is the Codex entry point. It should stay thin. Long-term project
facts, scientific rules, and current state live in shared documents.

Before substantial work, read in order:

1. `docs/agent/SHARED_CONTEXT.md`
2. `docs/PROJECT_STATE.md`
3. `docs/RESEARCH_CONTRACT.md`
4. `docs/DATA_PROTOCOL.md`
5. `docs/METRIC_CONTRACT.md`
6. `docs/EXPERIMENT_POLICY.md`
7. The active experiment specification or GitHub issue

Codex primary responsibilities:

- controlled implementation;
- tests and smoke validation;
- experiment preflight;
- run manifest generation;
- metric and result auditing;
- claim-to-artifact verification.

Before running an experiment:

- validate its specification;
- confirm the baseline and changed variable;
- confirm output uniqueness;
- confirm checkpoint selection;
- confirm the run tier.

After each change:

- run relevant tests;
- create or update a change note when scientific behavior changes;
- report files, commands, results, and remaining risks.

Do not launch formal experiments without an approved experiment specification.
