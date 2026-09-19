# Q1 Tier-1 Template-Floor Audit

Date: 2026-09-19. Experiment: `q1_tier1_readout_from_continuous_seed42` (W1a of
the Q1 evidence plan). Evaluator: `scripts/evaluate_tier1_readout.py`.
Outputs: `experiments_mmecg/q1_tier1_readout_from_continuous_seed42/`.

## Finding

Shape-200 PCC (centre cycle cut by ground-truth R peaks, resampled to 200 samples,
min-max normalised on both sides) is **saturated by a radar-free predictor**. If
every test window is predicted with the training fold's mean beat -- no radar, no
test data -- the score equals the project's Tier-1 headline numbers.

| Setting | Radar-free template | Reported model | Paired vs template |
|---|---:|---:|---|
| MMECG, old `loso` split | **0.8791 ± 0.0701** | REM-Net incumbent 0.882 ± 0.061 (BLOCKED) | per-fold values not archived |
| MMECG, old `loso` split | 0.8791 ± 0.0701 | REM-AlignNet-MTL 0.8898 ± 0.0538 | Δ +0.011, 7/11 folds, Wilcoxon p = 0.413 |
| MMECG, old `loso` split | 0.8791 ± 0.0701 | radarODE-MTL (adapted) 0.7283 ± 0.1328 | Δ −0.151, 0/11 folds, p = 0.001 |
| MMECG, `loso_v2` | **0.8806 ± 0.0710** | — | — |
| CR_RVS, 5-fold | **0.8870 ± 0.0131** | REM-Net 0.881 ± 0.013 | model is below the template |

Template RMSE / MAE in normalised units: MMECG `loso_v2` 0.109 / 0.089, CR_RVS
0.102 / 0.087. Population std across folds throughout. Per-fold archived values for
REM-AlignNet-MTL and radarODE-MTL come from
`archive/legacy_experiments/pilots_and_gates/q1_paper_style_report/fold_level_all_methods.csv`;
the template was recomputed on the identical old split for that comparison.

## Tier-1 readout of the frozen continuous models

The same Tier-1 definition applied to the 66 frozen continuous test runs on
`loso_v2` (checked fold by fold against `shape200_pcc`):

| Model | Shape-200 PCC | RMSE | MAE |
|---|---:|---:|---:|
| Chen-TCN | 0.345 ± 0.176 | 0.255 | 0.185 |
| REM-Net | 0.321 ± 0.181 | 0.276 | 0.200 |
| radarODE-MTL | 0.308 ± 0.137 | 0.265 | 0.186 |
| Cross-modal Conformer | 0.308 ± 0.160 | 0.259 | 0.183 |
| WaveGRU-Net | 0.221 ± 0.145 | 0.279 | 0.208 |
| Attention-DCGAN | 0.140 ± 0.108 | 0.272 | 0.196 |
| **Radar-free template** | **0.881 ± 0.071** | **0.109** | **0.089** |

Cutting the cycle with ground-truth R peaks removes amplitude and baseline error but
not timing error: a predicted QRS displaced by tens of milliseconds still lands off
the cycle. Every continuous model is far below the template on PCC and on RMSE.
REM-Net versus the baselines repeats the Tier-2 picture: statistically
indistinguishable from Chen-TCN, radarODE-MTL and the Conformer (p ≥ 0.64 on PCC),
better than WaveGRU-Net and Attention-DCGAN (p = 0.001).

## Why this happens

ECG beats of one lead are highly stereotyped. After per-cycle min-max
normalisation and resampling to a fixed length, most of the variance a PCC can see
is the shared P-QRS-T template. A model that outputs a canonical beat is scored as
excellent whether or not it read anything from the radar. The metric measures
template-likeness, and the template is available from the training labels alone.

## Consequences

1. The Tier-1 headline numbers (MMECG 0.882, CR_RVS 0.881) are **not evidence
   that REM-Net extracted beat morphology from radar**. Claims M1, M2, S1 and S3 in
   `docs/PAPER_CLAIM_LEDGER.md` are suspended.
2. The "0.882 versus radarODE 0.728" comparison says radarODE-MTL fell below a
   radar-free template, not that REM-Net learned morphology.
3. The CR_RVS comparison with VMD-CBNeT's reported 0.856 is moot for the same
   reason, besides its protocol difference.
4. Any future Tier-1 claim must be **template-referenced**: report the template
   floor beside every model and test each model against it per fold. A skill score
   such as `(PCC_model − PCC_template) / (1 − PCC_template)`, or an RMSE ratio to
   the template, states the radar-derived component directly.
5. W1b (a trained Tier-1 rerun) is paused. Launched as designed, it would most
   likely reproduce a template-level number. Its design, and whether Tier 1 stays a
   positive result in the manuscript, is a decision for the authors.

## What still stands

- The matched continuous benchmark (Tier 2) and its statistics are unaffected.
- Tier 3 timing diagnostics are unaffected.
- The finding itself is a measurement contribution: beat-normalised PCC, the
  metric used for ECG_shape in radarODE-MTL and for Shape-200 here, is saturated
  by a label-only template on both an FMCW and a CW dataset.
