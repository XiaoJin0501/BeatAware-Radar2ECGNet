# Current Project State

Last updated: 2026-08-10
Active branch: exp/rem-shape-guided-continuous-decoder
Active GitHub issue: none

## Current Research Objective

Serve RQ1: a new representation-aware radar-to-ECG model with strong **beat
morphology** reconstruction on MMECG and CR_RVS under subject-independent
protocols, favorable vs published/reproduced baselines, plus a mechanistic
**discovery** about the continuous-ECG bottleneck. Main line = morphology +
external validation + fair comparison (claims M1–M4). Anchor-guided continuous
assembly (route C) is a demoted discovery/mechanism section (D1–D2).

## Verified Evidence (all pilot, publishable=false)

- MMECG Shape-200 morphology: REM-AlignNet-MTL PCC ≈0.89 (vs radarODE adapted
  0.73, BeatAware 0.67). Main-line strength.
- CR_RVS Shape-200: CW-STFT+REM PCC ≈0.887 (> VMD-CBNeT reported 0.856);
  lag-aware refine 5-fold full-window PCC 0.500 vs TCN 0.438.
- Discovery (route C): oracle-cycle OLA 0.93–0.98 with morphology ≈0.96 ⇒
  continuous gap is beat-anchor localization, not morphology (D1).
- Route C robustness FAILED under 3-fold LOSO: fold_01 fusion F1@50 0.82 / OLA
  0.51 was a non-representative easy fold; 3-fold mean F1@50 0.56 / OLA 0.17,
  folds 02/03 OLA ~0. Diagnosis: recall collapse was mostly a threshold artifact,
  but threshold-tuned re-eval did NOT recover OLA ⇒ anchor **precision** does not
  generalize across subjects (fold_02/03 MAE 40–55 ms vs fold_01 16–20 ms) (D2).
- Falsified earlier: electromechanical-delay warp (latent/output) does not close
  the MMECG continuous gap (ALIGNMENT_DESIGN §7b).

## Current Blockers

- Final morphology/external tables not yet locked (tiers, aggregation).
- Reported-literature ledger tables not finalized.
- radarODE long-assembly not faithfully reproducible → compare in Shape-200 space,
  labeled adapted.
- Robust sub-50 ms anchor localization under LOSO is unsolved (route C future work).

## Next Approved Tasks (main line)

1. Build the final morphology table (T1) with fair tier labels.
2. Build CR_RVS external table (T3) aligned to VMD-CBNeT metrics.
3. Build reported-literature comparison table (Table D).
4. Write the route-C discovery/mechanism section from D1–D2 evidence.

## Do Not Do

- Do not promote pilots.
- Do not present route C as the Q1 main continuous model (demoted).
- Do not scale route C to 11-fold or Stage 2 unless anchor precision generalization
  is solved first.
- Do not describe Shape-200 as long-term continuous ECG evidence.
