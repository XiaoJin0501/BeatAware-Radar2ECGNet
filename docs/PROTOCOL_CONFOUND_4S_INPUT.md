# Protocol Confound: 4 s Radar Input vs 8 s ECG Target (MMECG)

Declared 2026-09-09 following `docs/RCG_PROVENANCE_AUDIT.md`. Binding.

## The defect

`scripts/precompute_wsst4s_mmecg.py` builds the MMECG WSST from `rcg[..., 400:1200]`
— the centre 4 s — while the ECG target spans all 1600 samples (8 s). **Half of every
output window therefore has no radar input support.**

Remediation: `scripts/precompute_wsst8s_mmecg.py` recomputes over the full
1600-sample support at the *same temporal hop* (6.667 samples = 33.3 ms), giving 240
bins rather than stretching 120 across 8 s. New files are `<split>_wsst8s.h5`;
existing `_wsst.h5` / `_wsst4s.h5` are left untouched.

## Scope — what is confounded

**Rule, not an enumeration.** A result is protocol-confounded if it is a
**time-resolved metric spanning the full 1600-sample MMECG window** produced by a
model whose input was the 4 s-crop WSST. That includes:

- every full-window / continuous / dense waveform metric (PCC, RMSE, NRMSE, MAE, R²)
- every overlap-add or assembly result
- the C0–C5 continuous-gap audit in its entirety
- event-head and waveform-derived R-peak F1, RR and HR error over the full window —
  roughly half of each predicted mask lies outside the supported region
- the retracted D1 material and the 0.394 / 0.323 dense full-window figures

These may **not** be used for model-quality claims, comparisons between models, or
mechanism attribution. They remain valid as records of what the *old pipeline* did.

## Scope — what is NOT confounded

- **Shape-200 morphology results.** The centre cycle lies inside samples 400–1200 in
  170/170 fold-01 test windows, so the morphology target is fully supported, and
  `0.882 ± 0.061` (MMECG) is unaffected by this defect. **This is survival, not
  promotion.** Under `docs/CONTINUOUS_ECG_TASK_CONTRACT.md` Shape-200 is auxiliary
  morphology evidence and is no longer the revised paper's main result; it may not be
  reported as the headline merely because the continuous numbers were withdrawn.
- **All CR_RVS results.** CR_RVS's `wsst` field is produced by `_stft_map` over the
  **full** window (`build_crrvs_wsst_shape_h5.py:169`), 93 bins across 1600 samples.
  It has no crop and therefore no support asymmetry. Its time resolution is coarser
  (86 ms/bin vs MMECG's 33.3 ms), which is a separate matter, not this confound.
  `0.881 ± 0.013` is unaffected — again auxiliary morphology evidence, not a headline.

## Consequence for the mechanism narrative

The manuscript attributes the Shape-200-vs-continuous dissociation to beat-anchor
localization. That attribution is now **unsafe**: the supported fraction of the target
differs between the two endpoints (100 % vs 50 %), which alone predicts the direction
and much of the size of the gap. No mechanism claim may be restated until the pilot
re-establishes the comparison on full-support input.

## Enforcement for new arms

- `topk_method != "corr"` — the ECG-GT range-bin selector is prohibited.
- `narrow_bandpass = False` for any native-rate RCG arm (the 0.8–3.5 Hz band holds
  only 2.3 % of MMECG `rcg` power).
- Any WSST consumed by a new arm must carry `support = full_window_samples_0_1600`
  and `supported_fraction_of_target = 1.0` in its file attributes.
