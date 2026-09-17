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
- The only active task is the detached `q1_loss_gate` session, waiting for GPU
  release to execute the registered L1/L2/L3 loss gate on folds 1/5/9. Never
  stop, restart, overwrite, or interfere with that session or its `queue.log`.
- Temporary compatibility exception until that session completes:
  `scripts/run_continuous_loss_ablation_queue.sh`,
  `scripts/train_continuous_ecg_pilot.py`, and
  `scripts/train_anchor_head.py` must remain at their current paths because the
  live queue references them. Archive/rename them only after verified completion.
