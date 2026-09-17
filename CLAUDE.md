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
- The L1/L2/L3 loss gate and the subsequent capacity-matched B2a-wide/B2b
  native-RCG gate are complete. The latter failed its waveform/mechanism rule;
  B2b direct fusion is closed and must not be rescued with post-hoc variants.
- The next authorized candidate is WSST-driven continuous REM-Net B1 with
  locked L3 MR-STFT on all 11 repaired `loso_v2` folds. Validation selects the
  checkpoint and test is opened exactly once afterward. Treat the first
  seed-42 result as a final candidate, not a headline claim without multi-seed
  confirmation.
