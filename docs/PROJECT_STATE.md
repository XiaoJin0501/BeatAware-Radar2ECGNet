# Current Project State

> **PROTOCOL-CONFOUNDED (2026-09-09).** All MMECG full-window / continuous /
> assembly / R-peak-timing results on this page were produced with a 4 s radar
> input against an 8 s ECG target, so 50% of each window had no input support.
> They are excluded from model-quality claims. Shape-200 results and all CR_RVS
> results are NOT affected. See `docs/PROTOCOL_CONFOUND_4S_INPUT.md`.

Last updated: 2026-09-17
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

The only approved next task is to specify and preflight a repaired-`loso_v2`,
capacity-controlled B2a (WSST only) versus B2b (WSST + native RCG) gate using
the locked MR-STFT waveform loss. No training is currently active.

## Do Not Do

- Do not promote pilots.
- Do not claim component causality before the current registered loss gate and
  its multi-metric promotion rule are satisfied.
- Do not optimize or narrate RQ1 around PCC alone.
- Do not revive or scale Route C assembly or the learned delay-warp design.
- Do not treat an archived roadmap, matrix, ADR, or change log as current
  authorization. The master plan is the sole roadmap.
- Do not describe Shape-200 as long-term continuous ECG evidence.
