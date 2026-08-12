# Baseline Reproduction Policy

This file defines how to handle published radar-to-ECG papers whose original
protocol is not strict LOSO or not identical to our protocol. The goal is to gain
comparison breadth without contaminating the fair evidence chain.

## Two-Lane Reproduction

Every external method can have up to two reproductions:

| Lane | Purpose | Protocol | Where it appears |
|---|---|---|---|
| Original-protocol reproduction | Check whether we can faithfully reproduce the paper's own reported result. | As close as possible to the paper: sample split, window length, preprocessing, metrics, epochs. | Table D / appendix only, labeled `reproduced-original-protocol`. |
| Common-protocol reproduction | Fair comparison against REM-Net. | Our locked subject-independent split and metric contract: MMECG 11-fold LOSO or CR_RVS 5-fold subject-independent; fold-macro aggregation. | Main comparison table, labeled `adapted` or `reproduced-common-protocol`. |

The original-protocol lane is useful, but it is not a fair head-to-head unless
the protocol already matches ours.

## Claim Rules

- Allowed: "We reproduced Method X under its reported sample-level protocol and
  obtained comparable results."
- Allowed: "When adapted to our subject-independent protocol, Method X achieved
  ..."
- Forbidden: "REM-Net outperforms Method X" using REM-Net common-protocol numbers
  against Method X original-protocol numbers.
- Forbidden: promoting a sample-level or subject-specific result into the fair
  strict-LOSO main table.
- Required: every reproduced baseline must state whether it is `original-protocol`
  or `common-protocol`.

## Candidate-Specific Decisions

### Attention-DCGAN / Jin 2025 on MMECG

- Original-protocol lane: allowed if we implement their 2000-train / 100-test
  sample protocol, but this lives only in Table D or appendix.
- Common-protocol lane: valuable only if adapted to MMECG 11-fold LOSO. This is
  the only version that can enter T1.
- Risk: the reported PCC 0.9853 likely reflects an easier sample-level protocol;
  do not use it as a main-table competitor.

### MultiResLinkNet / Chowdhury 2024 on CR_RVS

- Original-protocol lane: feasible and useful because they report 5-fold
  subject-wise CR_RVS/Schellenberger results over Resting/Valsalva/Apnea.
- Common-protocol lane: high-value and probably the first baseline to reproduce
  under our CR_RVS 5-fold subject-independent split.
- Risk: their temporal/spectral correlation metrics are not identical to our PCC;
  report both their metrics and ours when possible.

### LifWavNet / Kundu 2025 on CR_RVS

- Original-protocol lane: useful if code or a faithful reimplementation is
  available; their reported train/test chunk split is not our split.
- Common-protocol lane: high-value but more expensive than MultiResLinkNet.
- Risk: architecture details and preprocessing may require careful engineering.

### Radar2ECG / Li 2025 on CR_RVS Source Data

- Original-protocol lane: possible only after matching their radar heart-sound
  extraction from CR_RVS resting data.
- Common-protocol lane: lower priority because the task differs from our CW-STFT
  radar-to-morphology setup.
- Risk: not an apples-to-apples radar representation.

## Minimal Baseline Plan for the 3-Day Draft

1. Keep reported-only Table D broad and well-caveated.
2. Add one common-protocol reproduced baseline if time permits:
   **MultiResLinkNet on CR_RVS 5-fold**.
3. Leave Attention-DCGAN and LifWavNet as reported-only/candidate-to-reproduce
   unless there is enough time for a clean spec, implementation, and audit.
