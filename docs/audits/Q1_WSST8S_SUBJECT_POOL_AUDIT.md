# Q1 WSST8s Subject-Pool Audit

Date: 2026-09-18

## Outcome

The deduplicated full-window WSST cache is complete and ready for the registered
11-fold `loso_v2` benchmark. No training was launched during this task.

| Check | Result |
|---|---:|
| Subject files | 11 |
| Unique 8-second windows | 16,682 |
| Fold/split references resolved | 33,257 / 33,257 |
| Missing references | 0 |
| Temporary H5 files | 0 |
| Pool size | 25,717,820,453 bytes (23.95 GiB) |
| WSST tensor per window | `(50, 71, 240)`, `float16` |
| Aligned ECG per window | `(1, 1600)` |
| Support contract | `full_window_samples_0_1600` |

The original expectation of less than 7 GiB was not compatible with the
measured lossless tensor footprint. The implemented pool remains below the
hard 37 GiB cache limit and avoids the roughly twofold duplication of a
physical per-fold layout.

## Numerical checks

- The builder compared 64 windows reused from the prior formal 8-second cache:
  all were bit-identical.
- An independent audit indexed all 11 files and resolved every train,
  validation, and test occurrence across all 11 folds. It found 16,682 unique
  keys behind 33,257 fold/split occurrences and no duplicate pool key.
- Sampled WSST tensors were finite and within the frozen `[0, 1]` normalization
  range.
- The real fold-1 loader produced `(2, 50, 71, 240)` input and `(2, 1, 1600)`
  target tensors. A CPU B1 forward pass produced finite waveform
  `(2, 1, 1600)` and event `(2, 2, 1600)` outputs.
- The first loaded batch was bit-identical to the previous fold-1 L3
  `train_wsst8s.h5` cache.

## Provenance and use

The pool lives at
`/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool/`, with one
`subject_XX.h5` per subject. Fold membership is virtual: the loader resolves
the canonical `(subject_id, record_id, start_idx, end_idx)` key sequence from
each `loso_v2` split and reads the single subject copy. Four-second WSST is not
an allowed fallback.

This audit removes the data blocker only. The registered 11-fold experiment
remains **not launched** pending the normal clean-worktree/GPU preflight.
