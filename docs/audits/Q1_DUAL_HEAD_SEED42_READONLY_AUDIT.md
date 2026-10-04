# Dual-head REM-Net seed-42 read-only audit

Date: 2026-10-04. Experiment:
`q1_dual_head_full_loso_seed42`. No checkpoint, prediction, threshold or test
metric was changed by this audit.

## Completion and morphology

All 11 folds reached `COMPLETE`; test was opened only after the validation
checkpoint was frozen. Fold-macro ECG Shape Recovery is PCC
**0.8859 +/- 0.0455**, RMSE **0.1106 +/- 0.0298**, and R2
**0.5471 +/- 0.1601** (population standard deviation over 11 held-out
subjects). This is known-cycle, independently min-max-normalised Shape-200
evidence, not continuous ECG reconstruction.

The mandatory radar-free training-fold template comparison changes the
interpretation. Its PCC is **0.8806 +/- 0.0710**. The dual-head model exceeds
the template on only **5/11** folds; mean paired margin is **+0.0053**, and the
two-sided paired Wilcoxon test gives **p=0.8984**. Therefore the current seed-42
Shape result does not demonstrate radar-derived morphology beyond the template
floor and cannot be used as a morphology-superiority headline.

## Structured event read-out

Across the 10 folds with at least one matched event, Q/R/S/T MedAE means are
37/37/41/39 ms. Fold-macro coverage over all 11 folds is only
0.562/0.564/0.563/0.571 and MDR@150 ms is
0.438/0.436/0.437/0.429. Physiological Q<R<S<T ordering has zero violations
where a complete beat exists. Fold 7 has no matched event and retains undefined
MedAE rather than a fabricated zero.

## Fold 3/7 failure mechanism

The collapse is checkpoint-selection conflict, not evidence that the event
targets were never learnable:

| fold | selected epoch | selected val Shape PCC | selected val event F1@100 | later best val event F1@100 | later epoch |
|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 0.932 | 0.0002 | 0.756 | 16 |
| 7 | 4 | 0.909 | 0.0000 | 0.359 | 19 |

The preregistered lexicographic rule gives absolute priority to Shape PCC, so
it froze very early checkpoints before the event head matured. Test outputs
confirm almost no probability crosses the fixed 0.5 threshold: fold 3 has R
peaks in 3.7% of windows and no Q/S/T peaks; fold 7 has no R/S/T peak and one Q
peak in 765 windows. This audit does not authorise post-test checkpoint
reselection. Any alternative multi-objective checkpoint criterion is a new,
prospectively registered experiment and must rerun all affected methods.

## Baseline compatibility

The existing matched continuous baselines are complete for seeds 42, 123 and
2024 and use repaired `loso_v2`, full 8-second support, validation-only
checkpoint selection and one-time test opening. They remain valid comparators
for the **continuous 1600-sample endpoint**.

They are not a matched primary comparison for the new direct Shape head:

| item | dual-head REM-Net | existing matched baselines | compatible? |
|---|---|---|---|
| split/support | loso_v2, 8 s | loso_v2, 8 s | yes |
| seed family | 42 (currently) | 42/123/2024 | seed 42 only now |
| supervised output | direct central-cycle 200-point shape + Q/R/S/T | continuous 1600-point waveform | no |
| checkpoint criterion | lexicographic Shape PCC/RMSE/event F1 | continuous composite PCC/R2/F1 | no |
| Shape read-out | native direct head | GT-cycle crop of continuous prediction | diagnostic only |
| event read-out | direct probabilities + structured decoder | waveform delineation | system-level comparison only; read-out differs |

Consequently, no retraining is needed merely to preserve the existing
continuous table, but a manuscript table claiming same-task Shape superiority
requires newly registered direct-Shape adaptations (at minimum Chen-TCN,
radarODE-MTL and Attention-DCGAN) trained and selected under the same
known-cycle target and checkpoint contract. All five continuous models may be
included only in a separately labelled diagnostic table.

## Decision

Do not launch seeds 123/2024 or CR_RVS yet. First freeze a defensible
multi-objective validation-only checkpoint contract and a matched direct-Shape
baseline specification. Otherwise additional seeds would repeat a known
selection failure and produce high Shape PCC that is not above the radar-free
template floor.
