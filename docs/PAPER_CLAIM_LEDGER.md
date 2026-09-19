# Paper Claim Ledger

> **PROTOCOL-CONFOUNDED (2026-09-09).** All MMECG full-window / continuous /
> assembly / R-peak-timing results on this page were produced with a 4 s radar
> input against an 8 s ECG target, so 50% of each window had no input support.
> They are excluded from model-quality claims. Shape-200 results and all CR_RVS
> results are NOT affected. See `docs/PROTOCOL_CONFOUND_4S_INPUT.md`.


> **TIER-1 CLAIMS SUSPENDED (2026-09-19).** A radar-free predictor -- the training
> fold's mean beat -- scores Shape-200 PCC 0.879 +/- 0.070 on the old MMECG split,
> 0.881 +/- 0.071 on `loso_v2` and 0.887 +/- 0.013 on CR_RVS. The Tier-1 headline
> numbers (0.882, 0.881) sit at this floor; REM-AlignNet-MTL is indistinguishable from
> it per fold (p = 0.413) and radarODE-MTL falls below it on 11/11 folds. M1, M2, S1
> and S3 are suspended as radar evidence. See
> `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md`.

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

## Benchmark paper claims (governing, 2026-09-19)

These claims govern the IEEE TIM manuscript (`docs/paper/MANUSCRIPT_ARCHITECTURE_TIM.md`).
The older sections below are historical; where they conflict, this section wins.

| ID | Manuscript claim | Evidence | Status |
|---|---|---|---|
| C1 | The widely used MMECG LOSO construction leaks model selection across folds (two validation subjects across eleven folds); `loso_v2` repairs it with eleven distinct validation subjects. | `docs/MMECG_SPLIT_PROVENANCE_AUDIT.md`; per-fold split hashes in `experiments/manifests/q1_remnet_continuous_strict_loso_seed42.json` | **supported** |
| C2 | Beat-normalised PCC (Shape-200 / ECG_shape) is saturated by a radar-free training-label template: 0.879 ± 0.070 (MMECG old split), 0.881 ± 0.071 (`loso_v2`), 0.887 ± 0.013 (CR_RVS). Prior Tier-1 results are at this floor; radarODE-MTL is below it on 11/11 folds. | `q1_tier1_readout_from_continuous_seed42`; `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md` | **supported** |
| C3 | Under one leak-free protocol, six models from five families all reach R² ≈ 0 on strict blind continuous ECG; REM-Net is statistically indistinguishable from Chen-TCN, radarODE-MTL and the Conformer and better than WaveGRU-Net and Attention-DCGAN. | `q1_remnet_continuous_strict_loso_seed42`, `q1_matched_baselines_continuous_strict_loso_seed42` | **seed 42 supported; three-seed confirmation pending (W2)** |
| C4 | The continuous gap is dominated by beat timing: a fixed template at ground-truth R peaks reaches PCC 0.856 / R² 0.730, at radar-derived peaks PCC ≈ 0.20 / R² < 0; amplitude shrinkage is the loss-optimal hedge against timing error. | `q1_template_timing_diagnostics_seed42`; `docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md` | **supported on MMECG; CR_RVS pending (W3)** |
| C5 | C2-C4 hold on an external 24 GHz CW dataset. | W3: `q1_remnet_continuous_crrvs_seed42`, `q1_matched_baselines_continuous_crrvs_seed42` | **pending** |
| E1 | REM-Net: 1.79 M parameters, 0.98 ms per window. | `docs/EFFICIENCY_AUDIT.md` | supported; engineering characterisation only |

Forbidden under this positioning: any sentence that REM-Net outperforms the matched
baselines, recovers beat morphology from radar, or resolves amplitude collapse;
any Tier-1 number without its template floor; any CR_RVS comparison with reported
literature presented as a matched result.

## Main thesis (revised 2026-08-10, after route-C demotion)
> **RETRACTED 2026-09-09 (audit C).** The oracle-cycle OLA figure of 0.93-0.98 is
> NOT raw continuous-ECG evidence. `evaluate_mmecg_long_assembly.py:373` scores it
> against `_shape200_targets`, which min-max normalizes **every cycle of both the
> prediction and the ground truth**, and it covers only ~44% of samples (one center
> cycle per 400-sample hop). On the same run/fold it reproduces the Shape-200 score
> to within 0.002 (shape_head 0.9420 vs OLA 0.9436), i.e. it re-expresses the
> beat-normalized morphology metric on a time axis. The only raw-ECG, full-coverage
> number is the dense full-window OLA at PCC 0.394. A validation-only decomposition
> audit is registered; no claim may rest on 0.93-0.98 until it completes.


A new representation-aware radar-to-ECG model achieves strong **beat-morphology**
reconstruction on two radar modalities (FMCW MMECG, CW CR_RVS) under
subject-independent protocols, compares favorably with published/reproduced
baselines. The continuous-ECG gap is reported as a **bounded diagnostic** with at
least three components — morphology, beat timing, and per-cycle amplitude/baseline —
and no component is currently claimed as dominant. See
the archived historical designs under `archive/deprecated_docs/`.

> Demoted (2026-08-10): "anchor-guided assembly as the main continuous model" is
> NOT the main claim. 3-fold LOSO showed anchor detection recall generalizes but
> anchor PRECISION does not (fold_02/03 MAE 40–55 ms vs fold_01 16–20 ms;
> continuous OLA ~0 on 2/3 folds even after recall is restored). Route C is kept
> as a discovery + mechanism/future-direction section, not the Q1 main model.
> The earlier "electromechanical delay `δ(t)`" thesis was falsified before this
> (see `archive/deprecated_docs/ALIGNMENT_DESIGN.md` §7b).

## Main claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| M1 | Strong beat-morphology reconstruction on MMECG (Shape-200). | T1 final-test, G1 stability | **SUSPENDED 2026-09-19 (template floor)** | Shape-200 PCC **0.882 ± 0.061** (seed-42 final-test, 11 folds); G1 stability 0.885 ± 0.002 across 3 seeds; radarODE(adapted) 0.728; BeatAware 0.671 is Block B (calibration, not comparable). |
| M2 | External validation on CR_RVS (CW radar) matches/exceeds published results. | T3 final-test, G1 stability | **SUSPENDED 2026-09-19 (template floor)** | CW-STFT+REM-Net Shape-200 PCC **0.881 ± 0.013** (seed-42 final-test, 5 folds); G1 stability 0.886 ± 0.001; VMD-CBNeT *reported* 0.856; different-protocol caveat (morphology space). |
| M3 | Compares favorably with published/reproduced baselines under fair, output-formulation-aware protocols. | T1/T2/T3 tiers | partial | Needs final tier-labeled tables; sample-level results kept separate from subject-independent. |
| M4 | Generalizes across FMCW (MMECG) and CW (CR_RVS) under subject-independent protocols. | T1, T3 | partial | Morphology track; CR_RVS lag-aware refine 5-fold pilot (PCC 0.500 vs TCN 0.438). |

## Discovery / mechanism claims (route C, demoted from main model)

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| D1 | ~~The radar→ECG continuous gap is governed by beat-anchor localization, not morphology.~~ | ~~oracle-cycle OLA 0.93–0.98~~ | **RETRACTED 2026-09-09** | Evidence invalid: scored against per-cycle min-max normalized `_shape200_targets`, not raw ECG, at ~44% coverage; reproduces the Shape-200 score to within 0.002. Superseded by D1' pending audit. |
| D1' | The continuous-ECG gap decomposes into morphology, beat-timing, and per-cycle amplitude/baseline components; a beat-normalized target cannot supply the amplitude component. | dense full-window OLA vs raw ECG = 0.394 @ 100% coverage; `_shape200_targets` discards per-cycle gain/offset by construction | **pending** — validation-only decomposition audit registered | Diagnostic only. Not a model contribution. |
| D2 | Robust sub-50 ms anchor localization under LOSO is the open bottleneck: detection recall generalizes across subjects but precision does not. | 3-fold anchor validation + failure diagnosis + threshold re-eval; frozen event-head R-F1 (RATE_METRIC_AUDIT) | supported | fold_02/03 MAE 40–55 ms; OLA ~0; recovering recall does not recover OLA. MMECG frozen event head: F1@150/100/50 ms = **0.806/0.749/0.544** (sharp tolerance drop), RR 30.4 ms, HR 0.32 bpm (92.8% coverage). Frame as finding + future direction. |
| D3 (side) | A per-beat electromechanical offset (predicted anchor vs GT R) is observable but subject-dependent. | anchor signed-bias analysis | optional | Interpretability side-note; not globally correctable (LOSO val→test offset does not transfer). |
| D4 | Morphology transfers across radar modality but beat localization does not; CW localization largely fails while morphology stays high. | frozen event-head R-F1 (RATE_METRIC_AUDIT) | supported | CR_RVS (CW) event head F1@{50,100,150} ms ≈ 0.08/0.10/0.10 vs Shape-200 PCC 0.881; MMECG (FMCW) F1@150 ms 0.806. Do NOT report CR_RVS RR/HR (4.2% biased coverage). |

## Supporting / boundary claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| S1 | Strong beat morphology reconstruction on MMECG and CR_RVS. | T1, T3, G1 | **SUSPENDED 2026-09-19 (template floor)** | Shape-200 final-test PCC 0.882 ± 0.061 (MMECG), 0.881 ± 0.013 (CR_RVS); 3-seed stability 0.885 ± 0.002 / 0.886 ± 0.001; morphology evidence only. |
| S2 | Compares favorably with published/reproduced baselines under fair protocols. | T1/T2/T3 tiers | partial | Requires fair-tier labels; sample-level reported results separated from subject-independent main comparison. |
| S3 | CR_RVS external morphology exceeds VMD-CBNeT reported PCC. | T3 | **SUSPENDED 2026-09-19 (template floor)** | 0.881 ± 0.013 final-test vs *reported* 0.856; different-protocol/morphology-space caveat required. |
| S4 | Morphology degrades gracefully under input noise; FMCW is more noise-robust than CW. | ROBUSTNESS_AUDIT (eval-only, frozen) | supported | MMECG PCC 0.834 at 0 dB (−5.4%), 0.781 at −6 dB; CR_RVS drops ~2–5× faster (redundancy of 50 range bins vs single CW channel). Scope caveat: representation-space additive noise, NOT motion-artifact simulation. |

## Forbidden / caution

| ID | Statement | Rule |
|---|---|---|
| F1 | "Shape-200 results demonstrate long continuous ECG reliability." | Forbidden — Shape-200 cannot support this wording. |
| F2 | "We reproduce radarODE-MTL long-term assembly." | Forbidden unless a faithful assembly is recreated; currently compare only in Shape-200/ECG_shape space, labeled adapted. |
| F3 | "The method solves domain generalization." | Forbidden — claim external validity of the principle, not full domain generalization. |
| F4 | QT/QTc superiority claims. | Only on a confirmed-reliable delineation subset (see METRIC_CONTRACT). |

## Rules

- Every quantitative conclusion traces to an experiment ID + manifest.
- `δ` must be self-emergent in the main model; pseudo-GT delay is analysis-only.
- No causal claim without the controlled B0→B4 ablation.
- No RQ1 performance claim may rely on PCC alone. Morphology claims require
  RMSE/MAE/R² support where available; monitoring claims require QMR/R-F1/RR or
  documented coverage; component claims require the ablation plan in
  the current registered specification and `docs/Q1_JOURNAL_MASTER_PLAN.md`.
- Promotion from pilot→final only via the result-promotion workflow.
