# Codex-Claude RQ1 Handoff

Last updated: 2026-08-13

## Current Goal

Complete a three-day Q1 journal draft for RQ1 using the frozen supervised
REM-Net baseline. The target venue tier is J-BHI / IEEE Sensors / BSPC. TBME is
treated as a higher-risk stretch target, not the immediate writing constraint.

## Frozen Modeling Boundary

- Paper model: REM-Net, Representation-Event-Morphology Network.
- Implementation class may remain `REMAlignMTLNet`.
- Final model uses `use_alignment=false`.
- No model redesign, loss redesign, retraining, or new seed sweep during the
  three-day draft sprint unless explicitly approved by the user.
- New numbers should be evaluation-only on frozen seed-42 final-test
  checkpoints/predictions under `experiments_mmecg/q1_final_remnet_incumbent/`.

## Claude Code Role

Claude owns manuscript prose and narrative flow:

- `docs/paper/RQ1.tex` Abstract, Introduction, Related Work, Methodology,
  Results, Discussion, Limitations, and Conclusion.
- Claim framing: morphology-first supervised radar-to-ECG, two radar modalities,
  fair comparison protocol, and continuous-gap discovery.
- Paper-level readability and reviewer-facing argumentation.

Claude should not promote a new metric unless it is traceable to a registered
run and consistent with `docs/PAPER_CLAIM_LEDGER.md`.

## Codex Role

Codex owns evidence integrity and technical audit:

- Verify final result files, registry rows, and commit traceability.
- Audit metric definitions, aggregation, and figure/table values.
- Generate or check evaluation-only scripts, tables, and publication figures
  when this does not conflict with Claude's active edits.
- Keep reported-only, adapted, calibrated, and common-protocol results separated.
- Commit small, scoped changes with clear messages.

Codex should avoid editing `docs/paper/RQ1.tex` while Claude is actively drafting
unless the user explicitly asks Codex to take over the manuscript file.

## Locked Claims

- MMECG Shape-200 morphology: seed-42 final-test PCC 0.882 +/- 0.061 over 11
  strict-LOSO folds; 3-seed stability 0.885 +/- 0.002.
- CR_RVS Shape-200 morphology: seed-42 final-test PCC 0.881 +/- 0.013 over 5
  subject-independent folds; 3-seed stability 0.886 +/- 0.001.
- Continuous ECG is not the main claim. Route-C evidence supports the discovery
  that the continuous gap is dominated by beat-anchor localization precision,
  not morphology capacity.

## Forbidden Claims

- Do not claim continuous-ECG SOTA.
- Do not claim identical-protocol superiority over VMD-CBNeT.
- Do not mix reported-only literature values into the strict fair-comparison
  block.
- Do not present Shape-200 as long-term continuous ECG evidence.
- Do not present route C as the final model.

## Immediate Three-Day Deliverables

1. Complete `docs/paper/RQ1.tex` with a coherent Q1 draft.
2. Produce final T1/T3/Table-D LaTeX-ready tables.
3. Produce publication-quality qualitative morphology figures.
4. Add evaluation-only rate metrics if traceable and honest.
5. Add evaluation-only robustness and efficiency results if time permits.
6. Run a final claim audit and metric audit before any paper-ready declaration.
