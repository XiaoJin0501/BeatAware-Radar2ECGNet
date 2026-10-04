# Current Project State

> **PROTOCOL-CONFOUNDED (2026-09-09).** All MMECG full-window / continuous /
> assembly / R-peak-timing results on this page were produced with a 4 s radar
> input against an 8 s ECG target, so 50% of each window had no input support.
> They are excluded from model-quality claims. Shape-200 results and all CR_RVS
> results are NOT affected. See `docs/PROTOCOL_CONFOUND_4S_INPUT.md`.

Last updated: 2026-10-04
Active branch: exp/rem-shape-guided-continuous-decoder
Active GitHub issue: none

## Current Research Objective

Serve RQ1 with REM-Net and the factorized three-tier benchmark locked in
`docs/Q1_JOURNAL_MASTER_PLAN.md`: Shape-200 morphology, strict blind continuous
ECG, and GT-assisted temporal-tolerance diagnostics. Delay-warp and Route C
assembly are closed historical routes, not candidates for revival.

## Verified Evidence

- MMECG Shape-200 morphology: REM-Net seed-42 final-test PCC **0.882 ± 0.061**
  over 11 strict-LOSO folds; G1 stability **0.885 ± 0.002** across 3 seeds.
  Main-line strength; compare against radarODE adapted 0.728 and caveated
  BeatAware calibration 0.671.
- CR_RVS Shape-200: CW-STFT+REM-Net seed-42 final-test PCC **0.881 ± 0.013**
  over 5 subject-independent folds; G1 stability **0.886 ± 0.001** across 3
  seeds. This exceeds VMD-CBNeT reported PCC 0.856 only under our morphology
  protocol, with the reported-only/different-protocol caveat.
- **RETRACTED 2026-09-09 (audit C):** the route-C "oracle-cycle OLA 0.93–0.98 ⇒
  continuous gap is beat-anchor localization" claim (D1) is withdrawn. That OLA was
  scored against per-cycle min-max normalized `_shape200_targets` rather than raw
  ECG, at ~44% coverage, and reproduces the Shape-200 score to within 0.002. Only
  the dense full-window OLA vs raw ECG (0.394 @ 100% coverage) is valid. A
  validation-only decomposition audit (morphology / timing / amplitude) is
  registered; see `docs/PAPER_CLAIM_LEDGER.md` D1'.
- Route C robustness FAILED under 3-fold LOSO: fold_01 fusion F1@50 0.82 / OLA
  0.51 was a non-representative easy fold; 3-fold mean F1@50 0.56 / OLA 0.17,
  folds 02/03 OLA ~0. Diagnosis: recall collapse was mostly a threshold artifact,
  but threshold-tuned re-eval did NOT recover OLA ⇒ anchor **precision** does not
  generalize across subjects (fold_02/03 MAE 40–55 ms vs fold_01 16–20 ms) (D2).
- Falsified earlier: electromechanical-delay warp (latent/output) does not close
  the MMECG continuous gap (ALIGNMENT_DESIGN §7b).
- Rate/localization (eval-only, frozen; RATE_METRIC_AUDIT): MMECG event-head
  R-F1@{150,100,50}ms = **0.806/0.749/0.544** (sharp sub-50 ms drop), RR 30.4 ms,
  HR 0.32 bpm; CR_RVS localization **largely fails** (F1≈0.10) while morphology
  stays 0.881 ⇒ morphology transfers across modality, beat localization does not
  (D2/D4). CR_RVS RR/HR excluded (4.2% biased coverage).
- Noise robustness (eval-only, frozen; ROBUSTNESS_AUDIT): graceful Shape-200
  degradation; MMECG 0.834 at 0 dB (−5.4%), 0.781 at −6 dB; CW degrades ~2–5×
  faster (single-channel, no range-bin redundancy). Scope: input-corruption test,
  not motion artifacts (S4).
- Efficiency (EFFICIENCY_AUDIT): REM-Net **1.79M params**, 0.98 ms/window (RTX 4080
  SUPER), vs radarODE-MTL 26.8M and BeatAware 4.85M — smallest-but-one and most
  accurate.
- Manuscript `docs/paper/RQ1.tex`: added tab:rate, tab:noise, tab:efficiency;
  folded rate/robustness/efficiency into Intro contributions + Discussion +
  Limitations. All numbers traceable to audit docs; citations resolve; tables balanced.

## Current Blockers

- Final-test evidence chain is locked for the supervised morphology-first model,
  but advisor feedback requires that the next claims not rely on PCC alone.
- Controlled ablations are required before stating that event structure,
  morphology supervision, Shape-200 PCC loss, or alignment causes the observed
  improvements.
- Final manuscript tables/figures still need publication-grade formatting after
  the ablation results are known.
- Reported-literature table is design-locked but selected R4 papers can still be
  audited if needed.
- radarODE long-assembly not faithfully reproducible → compare in Shape-200 space,
  labeled adapted.
- Robust sub-50 ms anchor localization under LOSO remains unsolved; the former
  Route C implementation path is closed rather than retained as queued work.

## Only approved next task

**Author override, 2026-10-04:** after matched baselines and the registered
fixed-algorithm Q/R/S/T screen completed, the author authorised one bounded
model-rescue path rather than open-ended tuning. Gate 0 label learnability passed
and is audited in `docs/audits/Q1_EVENT_LABEL_LEARNABILITY_AUDIT.md`. The only
next model experiment that may be prepared is
`experiments/specs/q1_structured_event_gate_v1.json`: a folds 1/5/9 direct
Q/R/S/T event-head feasibility test using the incumbent WSST stem and TCN. It
completed and failed: two of four event medians improved, macro MedAE improved
6.85% rather than 10%, only 1/3 folds improved, and ordering safety failed on
folds 5/9. Event-conditioned waveform decoding is therefore not authorised;
range-attention, raw-RCG fusion, delay warp, calibration, broad sweeps and
11-fold expansion remain prohibited. See
`docs/audits/Q1_STRUCTURED_EVENT_GATE_AUDIT.md`.

The `q1_loss_gate` completed all 9/9 runs on 2026-09-17. L3 MR-STFT won the
waveform metrics (fold-macro PCC 0.293 versus L0 0.223; R² 0.012 versus -0.136)
but reduced F1@100 from 0.903 to 0.869 and did not improve the smoothing ratios
consistently. The original joint promotion gate therefore did not pass. The
full record is `docs/audits/Q1_LOSS_GATING_AUDIT.md`.

MR-STFT remains locked as the waveform core loss. The completed read-only
shared-gradient audit did not support a cross-fold MR-STFT/event-BCE conflict:
global cosine means were -0.137/+0.276/+0.088 on folds 1/5/9, and no module met
the preregistered strong-conflict threshold. PCGrad is not authorized. The full
record is `docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.md`.

The repaired-`loso_v2`, capacity-controlled B2a-wide versus B2b gate completed
6/6 runs. B2b improved tolerant event detection but reduced PCC on 3/3 folds,
did not improve R2 or QRS-region MAE consistently, and failed its preregistered
promotion rule. The native-RCG direct-fusion route is closed; see
`docs/audits/Q1_NATIVE_RCG_GATE_AUDIT.md`.

The frozen WSST-driven REM-Net B1 + L3 MR-STFT candidate completed all 11
`loso_v2` folds. Its seed-42 fold-macro continuous endpoint is PCC@0 0.305,
R² 0.018, NRMSE 2.151, and F1@100 ms 0.750. REM-Net tuning and new-mechanism
exploration are stopped.

The metric audit of that run is complete
(`docs/audits/Q1_REMNET_CONTINUOUS_11FOLD_METRIC_AUDIT.md`; manifest
`experiments/manifests/q1_remnet_continuous_strict_loso_seed42.json`). Permitted
interpretation: R² is **centred near zero across folds**, not reliably positive
— negative on folds 1, 2, 3, 10 and 11 with population std 0.181 — and the
amplitude variance ratio **0.312 ± 0.150** documents substantial residual
regression toward the mean. Amplitude collapse is mitigated, not resolved.
Units: RMSE/MAE/region MAE are in per-record z-score units, not millivolts.

Two reporting rules are binding from this audit (see the new "Continuous
benchmark reporting rules" section of `docs/METRIC_CONTRACT.md`):

- RR/HR aggregate only over folds with `rr_coverage >= 0.80`. Fold 7 collapsed
  to 0.0039 (waveform F1@50 0.003 while its event head still reached F1@100
  0.410) and is reported as a failure case, not averaged in. Gated values: RR
  **44.5 ± 30.1 ms**, HR **3.39 ± 1.83 bpm** over 10 folds; the ungated 218.9 ms
  / 6.09 bpm in the frozen summary JSON must not be tabulated.
- `above_10hz_energy_ratio` is reported as **median 0.122, IQR [0.083, 0.195]**
  with per-fold values, because fold 5 reaches 1.702; its mean ± std may not
  stand alone.

The validation composite is partially degenerate and must be described as such
in the methods: the NRMSE term is identically zero because NRMSE > 1 throughout,
so selection was effectively driven by PCC and waveform F1@100. It was frozen
before the run and is not edited retroactively.

The only approved next task is matched-protocol baseline alignment. Audit and
then adapt radarODE-MTL, AirECG, RSSRNet, and retained generic baselines to the
same repaired split, 8-second input/output support, validation-only checkpoint
selection, one-time test opening, evaluator, and fold-macro aggregation. Old
split results and reported literature values cannot enter the matched table.
No baseline training starts until its compatibility audit, registered spec,
and clean preflight are complete.

The 60-second paired-ECG calibration is a closed Discussion-only experiment:
PCC improved from 0.305 to 0.327 in the macro aggregate but only 5/11 folds
improved, while R², NRMSE, and F1 worsened. Its script is archived under
`archive/legacy_scripts/audits/`; do not tune or promote this route.
The data blocker was removed on 2026-09-18. The subject-centric 8-second WSST
pool contains 11 files, 16,682 unique windows, and resolves all 33,257 fold
references in 23.95 GiB. Bitwise reuse and a real-loader CPU forward smoke both
pass (`docs/audits/Q1_WSST8S_SUBJECT_POOL_AUDIT.md`). Do not substitute
4-second WSST. The 11-fold benchmark is complete.

The formal runner is hardened for epoch-level recovery with atomic artifacts
and a terminal test-access state machine. Its frozen budget is 100 maximum
epochs with patience 15. A controlled 1-epoch interruption resumed at epoch 2
without constructing or evaluating the test split.

## Do Not Do

- Do not promote pilots.
- Do not claim component causality before the current registered loss gate and
  its multi-metric promotion rule are satisfied.
- Do not optimize or narrate RQ1 around PCC alone.
- Do not revive or scale Route C assembly or the learned delay-warp design.
- Do not treat an archived roadmap, matrix, ADR, or change log as current
  authorization. The master plan is the sole roadmap.
- Do not describe Shape-200 as long-term continuous ECG evidence.
