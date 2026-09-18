# Q1 Zero-Label Scalar-Shift TTA Fold-2 Audit

## Status

**FAILED / route closed.** This is a nonpublishable, post-final-test pilot on
Fold 2. It does not authorize tuning on the test subject or scaling to other
folds.

## Frozen inputs and adaptation boundary

- Source checkpoint: `q1_remnet_continuous_strict_loso_seed42/L3/fold_02/checkpoints/best_composite.pt`, validation-selected epoch 24.
- Adaptation data: the first two held-out-subject WSST windows only.
- Adapted quantity: one scalar circular Fourier shift; all REM-Net parameters
  frozen.
- Optimizer: Adam, 20 steps, learning rate `1e-2`.
- Adaptation loss: event-to-waveform-gradient cosine inconsistency plus 0.5
  times circular autocorrelation prominence.
- The adaptation function has no ECG/target argument. The subject-pool loader
  restores WSST rows from split metadata without opening the ECG dataset.
- ECG was opened only after the scalar was frozen, under `torch.no_grad()`, to
  score the remaining 168 windows. The first two adaptation windows were
  excluded from evaluation.

## Result

| Quantity | Value |
|---|---:|
| Adaptation time | 1.365 s |
| Selected shift | +2.312 samples / +11.562 ms |
| Query windows | 168 |
| Original PCC@0 | 0.021232 |
| TTA PCC@0 | -0.040824 |
| Delta PCC | **-0.062056** |

The cyclic loss remained numerically constant (`-0.598440`) across all 20
steps, as required by the translation invariance of circular autocorrelation.
The event consistency term decreased only from `0.336458` to a minimum of
`0.327092`; it selected an internal event-waveform adjustment near 12 ms, not
the approximately 55 ms GT-assisted Fold-2 offset observed in the separate
failure diagnostic.

## Interpretation and decision

The frozen event and waveform outputs do not provide an external timing
reference for the subject-level ECG offset. Their self-consistency can optimize
their relative phase while both remain jointly displaced from the unobserved
ECG. Autocorrelation cannot identify an absolute shift. Consequently, this
specific zero-label scalar TTA mechanism worsens the deployment endpoint and is
closed. No learning-rate, step-count, K-window, sign, or loss-weight search is
authorized on Fold 2, because that would tune against a held-out test subject.

Machine-readable result:
`experiments_mmecg/q1_zero_label_tta_fold02/result.json`.

