# Claude Code Entry Point

This file is the Claude Code entry point. It should stay thin. Long-term
project facts, scientific rules, and current state live in shared documents.

Read these files before substantial work:

1. `docs/agent/SHARED_CONTEXT.md`
2. `docs/PROJECT_STATE.md`
3. `docs/RESEARCH_CONTRACT.md`
4. `docs/DATA_PROTOCOL.md`
5. `docs/METRIC_CONTRACT.md`
6. `docs/EXPERIMENT_POLICY.md`
7. `docs/AGENT_DEVELOPMENT_GUIDE.md`
8. The active experiment specification or GitHub issue

Claude Code primary responsibilities:

- research design and hypothesis decomposition;
- cross-file architecture analysis;
- scientific interpretation;
- protocol review;
- manuscript drafting from verified evidence.

Before editing:

- identify the active issue and branch;
- classify the change as implementation, protocol, experiment, or paper;
- report scientific impact.

Do not launch formal experiments without an approved experiment specification.

## Repository sanitation and experiment governance

- Never generate independent scripts or result directories prefixed with
  `probe_`, `smoke_`, `pilot_`, or `gate_`. Use registered core entry points,
  in-memory analysis, or temporary arguments instead.
- Never create an unregistered directory in `experiments_mmecg/`. Persistent
  output is reserved for approved registered experiments.
- Preserve negative and historical evidence by moving it under `archive/` and
  marking its registry row `Archived`; do not physically delete it.
- `docs/Q1_JOURNAL_MASTER_PLAN.md` is the roadmap and REM-Net is the locked
  model identity. Do not revive REM-AlignNet or change the core idea to chase a
  metric.
- The only active handoff is the detached `q1_loss_gate` session. It is waiting
  for GPU release and will execute Step 2 (L1/L2/L3, folds 1/5/9). Do not stop,
  restart, overwrite, or interfere with it or its `queue.log`.
- Until the queue completes, its three referenced compatibility files must stay
  at their current paths: `run_continuous_loss_ablation_queue.sh`,
  `train_continuous_ecg_pilot.py`, and `train_anchor_head.py`.
