# Next Experiments: Paper-First Plan

## Goal

Build a Q1 manuscript around fair, output-formulation-aware comparison on
MMECG and CR_RVS. Do not force all methods into full-window `1 x 1600` ECG.

## Table A: MMECG Beat Morphology

Purpose: compare with radarODE-MTL and other cycle/beat-level methods under a
fair morphology target.

Current evidence:

| Method | Status | Current Key Result |
|---|---|---:|
| REM-AlignNet-MTL | complete pilot | Shape-200 PCC 0.890 |
| radarODE-MTL adapted | complete pilot | Shape-200 PCC 0.728 |
| ICONIP BeatAware | complete pilot | Shape-200 PCC 0.671 |
| WSST-TCN | complete pilot | Shape-200 PCC 0.273 |
| RSSRNet adapted | complete pilot | Shape-200 PCC 0.262 |
| AirECG adapted | complete pilot | Shape-200 PCC 0.180 |

Needed before final:

1. Lock aggregation: fold-macro or segment-macro.
2. Rebuild a final-style table with source labels:
   - proposed;
   - reproduced/adapted;
   - prior conference baseline.
3. Add compact event/fiducial timing if reliable in Shape-200 space.
4. Decide whether the current REM-AlignNet-MTL is final proposed or whether a
   very small final variant is still necessary.

## Table B: MMECG Continuous Segment

Purpose: show that the proposed family can also output continuous ECG segments,
but do not make this the only success criterion.

Current evidence:

| Method | Current Key Result |
|---|---:|
| ICONIP BeatAware | 8-s PCC 0.649 |
| REM-AlignNet-MTL-Refine fold01 only | fold01 long OLA PCC 0.420 |
| REM-AlignNet-MTL | 8-s PCC 0.235 |

Decision:

- Keep as secondary table.
- Do not spend the next main experiment cycle only optimizing full-window
  `1 x 1600` unless it also helps morphology/fiducial tables.

## Table C: CR_RVS / Nature External Validation

Purpose: show external generalization and compare with papers that use the
Nature/CR_RVS dataset.

Current evidence:

| Method | Status | Current Key Result |
|---|---|---:|
| CW-STFT + REM-AlignNet-MTL | complete pilot | PCC 0.887 |
| CW-STFT + TCN | complete pilot | PCC 0.550 |
| VMD-CBNeT published Nature result | reported reference | PCC 0.856, RMSE 0.112 |
| CNN+LSTM published Nature result | reported reference | PCC 0.647, RMSE 0.127 |

Needed before final:

1. Confirm CR_RVS split description as 5-fold subject-independent CV.
2. Align metrics to VMD-CBNeT where possible:
   - PCC;
   - RMSE;
   - RR error;
   - QT error;
   - PR error if labels support it;
   - QRS/QSR duration error.
3. Generate CR_RVS paper-style comparison figure/table.

## Table D: Literature Reported Results

Purpose: cite strong papers without pretending their private/sample-level
protocols are directly comparable.

Use `docs/LITERATURE_COMPARISON_LEDGER.csv`.

Rules:

- Same public dataset + subject-independent: may enter main comparison.
- Same public dataset + sample-level/random split: reported-only or
  supplementary.
- Private/custom dataset: reported-only context.
- Released code adapted to our protocol: reproduced/adapted baseline table.

## Minimum Ablation

Only keep ablations needed to support the Q1 story:

| Variant | Purpose |
|---|---|
| WSST-TCN | representation-only baseline |
| REM-Event | event structure value |
| REM-AlignNet-MTL | final morphology/event model |

Optional:

- REM-AlignNet-MTL without shape head if easy to run and if the reviewer may
  ask whether morphology supervision matters.

## Immediate Execution Order

1. Build `scripts/make_literature_comparison_report.py`.
2. Generate:
   - MMECG morphology final-style table;
   - CR_RVS reported-vs-ours table;
   - one combined supervisor figure.
3. Audit CR_RVS fiducial metric availability.
4. If CR_RVS fiducials are available, compute VMD-CBNeT-style RR/QT/QRS
   errors for our CR_RVS result.
5. Only after this, decide whether to run any new model training.

## Do Not Do Next

- Do not run more full-window-only pilots as the main path.
- Do not place sample-level reported papers in the same table as strict LOSO
  without a caveat.
- Do not call radarODE-MTL long ECG assembly reproduced unless an official
  assembly implementation is found or recreated with enough evidence.
