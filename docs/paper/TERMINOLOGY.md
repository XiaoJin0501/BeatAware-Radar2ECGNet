# Manuscript Terminology and Its Mapping to Code

Binding from 2026-09-21. `docs/METRIC_CONTRACT.md` requires every metric name in a
manuscript table to map to a code function and a manifest field. This file is that
mapping. Manuscript prose uses the left column; audits, specs and code keep the right
column unchanged.

| Manuscript term | Internal name | Code / field | Definition |
|---|---|---|---|
| **ECG Shape Recovery** | Shape-200 | `shape200_pcc`, `scripts/evaluate_tier1_readout.py` | Single-beat morphology: the cycle between two reference R peaks, resampled to 200 samples and min–max normalised on both sides. A known-cycle diagnostic, never long-term reconstruction. |
| **Long-term ECG Reconstruction** | continuous benchmark / Tier 2 | `pcc`, `r2`, `nrmse`, `rmse`, `mae` in `evaluate()` | The deployment endpoint: the full 8 s, 1600-sample waveform in stored units, no reference-derived alignment. |
| **ECG Anchor Estimation** | R-peak / event detection | see the two rows below | Placing cardiac anchors (R peaks) in time. Two read-outs exist and are never merged. |
| — waveform-derived anchors | waveform R-peak F1 | `f1_50ms`, `f1_100ms`, `f1_150ms` | Anchors detected **on the reconstructed waveform**. Available for every system. |
| — anchor-head estimates | event-head F1 | `event_f1_*ms` | Anchors from a dedicated anchor head. Only REM-Net has one; the column is left empty for systems without it, never scored as zero. |
| **Cycle Length Estimation** | RR interval error | `rr_error_ms`, `rr_coverage` | Error of the estimated beat-to-beat interval, aggregated only over entries with anchor coverage ≥ 0.80. |
| **Cardiac Intervals** | RR / HR errors together | `rr_error_ms`, `hr_error_bpm` | Interval-level clinical read-outs. |
| **Radar-free template floor** | template floor | `template_floor()` in `scripts/evaluate_tier1_readout.py` | Training-fold mean beat predicted for every test window; the reference level for ECG Shape Recovery. |
| **Oracle-timing bound** | T0 | `T0_gt` in `scripts/evaluate_template_timing.py` | Template placed at reference anchors; upper bound for Long-term ECG Reconstruction. |
| **Anchor timing jitter** | within-subject jitter | `within_subject_jitter_ms` | Robust within-subject spread of the signed anchor offset (1.4826 × MAD). |
| **Per-subject anchor offset** | per-subject bias | `per_subject_bias_ms`, `between_subject_bias_sd_ms` | Each subject's median signed offset; its across-subject spread. |
| **Strict LOSO** | `loso_v2` | split root `loso_v2` | Subject-disjoint folds with eleven distinct validation subjects, hashes recorded per fold. |

Rules that survive the rename:

1. ECG Shape Recovery may never be described as Long-term ECG Reconstruction, and
   every ECG Shape Recovery number is reported beside the radar-free template floor.
2. Waveform-derived anchors and anchor-head estimates keep separate columns.
3. The oracle-timing bound and any reference-derived offset correction are labelled as
   oracles wherever they appear.
4. Units stay explicit: amplitude errors are in per-record z-score units.
