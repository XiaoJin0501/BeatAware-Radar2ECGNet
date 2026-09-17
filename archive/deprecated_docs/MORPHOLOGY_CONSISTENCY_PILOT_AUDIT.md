# Morphology-to-Waveform Consistency Pilot Audit

## Decision

**Do not promote B1 to the final model.** The candidate improved several MMECG
validation metrics but did not generalize that benefit to CR_RVS. This fails the
pre-registered requirement that B1 improve both datasets without a material
timing or amplitude regression. These are three-fold, seed-42 development
results, not publishable test results.

## Protocol

- B0: frozen REM-Net recipe without the candidate loss.
- B1: detached morphology-head to continuous-waveform cycle consistency,
  weight 0.05.
- B2: direct reference-cycle supervision at the same weight, included to test
  whether any gain is specific to cross-output consistency.
- MMECG folds: 1, 5, and 9 under strict LOSO development validation.
- CR_RVS folds: 1, 3, and 5 under subject-independent development validation.
- All arms used seed 42, the same architecture and optimizer, at most 80 epochs,
  patience 20, and `best_loss.pt` selected by the common validation base loss.
- Neither the runner nor the evaluator read a test split.

## Fold-Macro Results

| Dataset | Arm | Full-window PCC | RMSE | Shape-200 PCC | Waveform F1@50 ms |
|---|---:|---:|---:|---:|---:|
| MMECG | B0 | 0.3028 +/- 0.0133 | 0.1630 | 0.9152 | 0.6058 |
| MMECG | B1 | 0.3372 +/- 0.0376 | 0.1618 | 0.9157 | 0.6365 |
| MMECG | B2 | 0.3095 +/- 0.0243 | 0.1615 | 0.9143 | 0.6126 |
| CR_RVS | B0 | 0.4411 +/- 0.2056 | 0.1319 | 0.9088 | 0.3970 |
| CR_RVS | B1 | 0.4168 +/- 0.2578 | 0.1333 | 0.9213 | 0.3490 |
| CR_RVS | B2 | 0.3976 +/- 0.2364 | 0.1346 | 0.9208 | 0.4206 |

On MMECG, B1 changed full-window PCC by +0.0343 relative to B0 and won
2/3 folds; it exceeded B2 on PCC and waveform-derived F1 in all three folds.
On CR_RVS, B1 changed PCC by -0.0243 and waveform-derived F1@50 ms by
-0.0481 relative to B0, winning only 1/3 folds for each metric. The large
CR_RVS fold variance also rules out a robustness claim.

## Interpretation Boundaries

The pilot supports only a narrow observation: coupling the two outputs may help
MMECG development folds. It does not establish a modality-independent method,
a final-test gain, or a new Q1 contribution. B1 must not appear in the abstract,
contribution list, main result tables, or final-model description.

The evaluation also exposed an important distinction. Event-head F1 and peaks
derived from the reconstructed waveform can differ substantially. A weak event
head alone is therefore insufficient evidence that the reconstructed waveform
lacks useful timing. Existing manuscript language about the continuous-ECG gap
should distinguish direct event-head localization, waveform-derived timing, and
full-window waveform agreement.

## Handoff / Next Action

1. Keep the current frozen REM-Net incumbent as the supervised RQ1 model.
2. Do not run B1 on final test or expand it to all folds/seeds.
3. Audit `docs/paper/RQ1.tex` for claims that equate event-head failure with
   waveform timing failure; use the frozen-model rate audit as the source.
4. Complete the journal draft around verified additions: dual-modality
   subject-independent validation, multi-metric/region fidelity, robustness,
   efficiency, fair comparison tiers, and the carefully bounded continuous-gap
   analysis.
5. If model research resumes after the draft deadline, open a new registered
   development study aimed at cross-modality timing calibration. Do not iterate
   on the present consistency-loss weight or inspect test results for selection.

## Reproduction

```bash
/home/qhh2237/miniconda3/envs/cyberbrain/bin/python \
  scripts/aggregate_morphology_consistency_pilot.py
```

Machine-readable outputs are under
`experiments_mmecg/q1_morphology_consistency_pilot_summary/`. Per-fold metrics
remain under each registered run's `results_validation/metrics.json`.
