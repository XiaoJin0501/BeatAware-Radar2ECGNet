# Morphology Fiducial Booster — No-Retraining Evidence

Status: completed diagnostic booster (2026-08-11). This is not a new experiment
or model change. It computes phase-region morphology metrics from existing G1
Shape-200 prediction files using `scripts/make_morphology_fiducial_booster.py`.
Artifacts are generated under
`experiments_mmecg/q1_morphology_fiducial_booster/` and are intentionally ignored
by git.

## Why this was needed

The G2 qualitative overlays showed a real weakness: REM-Net can have high
Shape-200 PCC while still showing visible amplitude/baseline mismatch in some
regions. A Q1 reviewer may reasonably ask whether the reported PCC is only a
smooth global curve metric. This booster adds region-wise morphology evidence
without changing the supervised RQ1 scope and without retraining.

## Metric definition

Shape-200 is a min-max normalized R-to-R cycle, so the metrics below are fixed
cycle-phase morphology metrics, not clinical ECG delineation. They must not be
reported as QT/QRS diagnostic interval errors.

- `whole`: full 200-point morphology.
- `qrs_edge`: early and late R/QRS-edge phase (`0:24` and `176:200`).
- `st_t`: post-QRS/ST/T phase (`24:96`).
- `diastolic`: low-amplitude mid-cycle phase (`96:176`).
- `t_wave_peak_phase_mae`: phase-point error of the dominant peak in `32:96`.
- `p_region_peak_phase_mae`: phase-point error of the dominant peak in `132:176`.

## Results

Summary file:
`experiments_mmecg/q1_morphology_fiducial_booster/summary.md`.

| Dataset | Method | Whole PCC | QRS-edge PCC | ST/T PCC | Diastolic PCC | T peak phase MAE | T amp MAE | P/pre-R phase MAE | P/pre-R amp MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMECG | REM-Net | **0.885** | **0.962** | **0.787** | **0.510** | **5.2** | **0.159** | **13.6** | **0.076** |
| MMECG | radarODE-MTL | 0.728 | 0.961 | 0.157 | 0.034 | 15.9 | 0.227 | 16.7 | 0.104 |
| MMECG | REM-Event | 0.279 | 0.123 | 0.431 | 0.347 | 11.4 | 0.287 | 14.7 | 0.191 |
| MMECG | WSST-TCN | 0.273 | 0.112 | 0.426 | 0.322 | 11.3 | 0.268 | 14.9 | 0.184 |
| CR_RVS | REM-Net | **0.886** | **0.941** | **0.793** | **0.380** | **4.9** | **0.120** | **6.4** | **0.094** |
| CR_RVS | CW-STFT+TCN | 0.550 | 0.464 | 0.744 | 0.358 | 6.2 | 0.172 | 7.8 | 0.124 |

## Interpretation

This booster strengthens the paper. REM-Net is not only winning by a single
global PCC: on MMECG it preserves both QRS-edge morphology (0.962 PCC) and ST/T
region morphology (0.787 PCC), while radarODE-MTL is competitive on QRS-edge
morphology (0.961) but collapses in the ST/T and diastolic regions. This gives a
more specific mechanism for the Shape-200 improvement: REM-Net retains non-QRS
beat structure rather than merely detecting the R/QRS edge.

On CR_RVS, REM-Net also improves QRS-edge and ST/T morphology over the local
CW-STFT+TCN baseline. The gain is largest in the QRS-edge region and smaller in
the ST/T and diastolic regions, which is consistent with the lower SNR of
low-amplitude morphology under CW radar.

The result also explains why the overlay figure does not look as perfect as some
published examples. The model captures dominant beat morphology and the T-region
phase reasonably well, but low-amplitude diastolic morphology remains weak
(MMECG 0.510, CR_RVS 0.380). This should be reported as a limitation rather than
hidden by cherry-picked overlays.

## Claim language

Allowed:

- "Region-wise morphology analysis shows that the proposed model improves not
  only global Shape-200 PCC but also QRS-edge and ST/T morphology fidelity."
- "On MMECG, radarODE-MTL is competitive in the QRS-edge phase but much weaker in
  ST/T and diastolic morphology, suggesting that REM-Net's gain comes from
  preserving broader beat structure."
- "The diastolic phase remains less reliable, indicating that low-amplitude
  morphology is still challenging."

Forbidden:

- "REM-Net has accurate QT/QRS clinical intervals in Shape-200."
- "The model solves continuous ECG morphology."
- "The overlay mismatch is only a plotting artifact."

## Recommendation

Keep this booster in the paper as either a compact table or a figure next to the
qualitative overlays. It directly answers the "easy PCC metric" critique and
should be included before drafting the full Results section.
