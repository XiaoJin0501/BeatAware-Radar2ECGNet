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
7. `docs/AGENT_DEVELOPMENT_GUIDE.md`
8. The active experiment specification or GitHub issue

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

## Repository sanitation and experiment governance

- Do not create independent scripts or result directories whose names begin
  with `probe_`, `smoke_`, `pilot_`, or `gate_`. Auxiliary checks must use an
  existing registered entry point, in-memory analysis, or temporary parameters.
- Do not create an unregistered directory under `experiments_mmecg/`. Only an
  approved, registered formal experiment may persist there. Smoke output must
  be isolated inside its registered experiment directory and is never evidence.
- Failed and historical work remains traceable under `archive/`; move it there
  rather than deleting it. Update `EXPERIMENT_REGISTRY.csv` to say `Archived`
  and retain the destination.
- The canonical paper roadmap is `docs/Q1_JOURNAL_MASTER_PLAN.md`. The model
  name is REM-Net; do not revive REM-AlignNet or invent a replacement name.
- The `q1_loss_gate` completed 9/9 runs on 2026-09-17 and its detached session
  exited naturally. Its immutable audit is
  `docs/audits/Q1_LOSS_GATING_AUDIT.md`; do not overwrite its artifacts or
  `queue.log`.
- The read-only shared-gradient audit is complete and did not authorize PCGrad
  or branch isolation. The capacity-controlled B2a-wide/B2b gate completed 6/6
  runs and failed its promotion rule. B2b direct native-RCG fusion is closed;
  do not tune or scale it. Its immutable audit is
  `docs/audits/Q1_NATIVE_RCG_GATE_AUDIT.md`.
- The WSST-driven continuous REM-Net B1 with locked L3 MR-STFT has completed
  all 11 repaired `loso_v2` folds. Stop REM-Net tuning and all new-mechanism
  exploration. The only active path is matched-protocol baseline alignment,
  beginning with a read-only compatibility audit and registered specs.
- The 60-second paired-ECG calibration is closed Discussion-only evidence. Its
  evaluator is archived under `archive/legacy_scripts/audits/`; do not tune,
  promote, or restore it to the active script surface.
- `scripts/audit_shared_gradient_conflict.py` is the explicitly requested,
  registered exception to the legacy-script naming rule; it is read-only and
  must not be copied into additional one-off audit scripts.
