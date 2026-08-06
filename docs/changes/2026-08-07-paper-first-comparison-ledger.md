# 2026-08-07 Paper-First Comparison Ledger

## Why

The project direction was corrected from "force every method into full-window
`1 x 1600` ECG" to "compare each method under the closest fair output
formulation used in the literature." This is necessary for a Q1 manuscript
because many radar-to-ECG papers report beat morphology, fiducial timing, or
dataset-specific waveform metrics rather than the same 8-second output target.

## What Changed

- Added `docs/Q1_PAPER_COMPARISON_STRATEGY.md`.
- Added `docs/LITERATURE_COMPARISON_LEDGER.csv`.
- Added `docs/NEXT_EXPERIMENTS_PAPER_FIRST.md`.
- Added `scripts/make_paper_first_comparison_report.py`.
- Updated `docs/PROJECT_STATE.md` and `docs/PAPER_CLAIM_LEDGER.md`.

## Generated Report

Generated:

- `experiments_mmecg/q1_paper_first_comparison_report/mmecg_shape200_comparison.csv`
- `experiments_mmecg/q1_paper_first_comparison_report/mmecg_shape200_pcc.png`
- `experiments_mmecg/q1_paper_first_comparison_report/crrvs_reported_vs_ours_comparison.csv`
- `experiments_mmecg/q1_paper_first_comparison_report/crrvs_reported_vs_ours_pcc.png`
- `experiments_mmecg/q1_paper_first_comparison_report/README.md`

## Current Paper-Relevant Evidence

- MMECG Shape-200:
  - REM-AlignNet-MTL PCC `0.890`.
  - radarODE-MTL adapted ECG_shape/Shape-200 PCC `0.728`.
  - ICONIP BeatAware Shape-200 PCC `0.671`.
- CR_RVS / Nature:
  - CW-STFT + REM-AlignNet-MTL 5-fold PCC `0.887`.
  - VMD-CBNeT reported Nature PCC `0.856`, RMSE `0.112`.
  - CNN+LSTM reported Nature PCC `0.647`, RMSE `0.127`.

## Scientific Meaning

Shape-200 / beat morphology is a legitimate main comparison axis, especially
for radarODE-MTL and VMD-CBNeT-style methods. Full-window `1 x 1600` ECG should
remain a secondary continuous-monitoring capability, not the sole success
criterion.

## Next Agent Task

Audit CR_RVS fiducial labels and implement VMD-CBNeT-style RR/QT/PR/QRS error
metrics for our CR_RVS predictions if label quality supports it.
