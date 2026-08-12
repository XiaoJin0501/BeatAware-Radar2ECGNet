# Table D — Reported-Literature Comparison — DESIGN / AUDIT

Purpose: isolate ALL **reported-only** literature numbers (not reproduced under our
protocol) into one clearly-caveated table, so they never contaminate the
same-protocol main evidence (T1 MMECG morphology, T3 CR_RVS). Design/audit first;
no numbers are cited unless verified. Source ledger:
`docs/LITERATURE_COMPARISON_LEDGER.csv`; PDFs under `reference_papers/`.

## Hygiene rule (locked)

- Reported-only numbers are **descriptive context**, never a same-protocol
  head-to-head with ours.
- **Different/private-dataset** numbers must NEVER share a comparison cell with our
  results.
- Only two things may appear next to our numbers: (a) baselines **reproduced under
  our protocol** (T1 "adapted" rows); (b) **same-public-dataset reported-only**
  results with an explicit protocol caveat (T3 Block B).
- Every Table D row states: dataset, same-dataset?, protocol, comparability tier.

## Comparability tiers

- **R1 — reproduced under our protocol** (lives in T1/T3 as "adapted"; Table D just
  cites the original paper).
- **R2 — same public dataset, reported-only, different protocol** (citable with
  caveat; = T3 Block B).
- **R3 — different/private dataset, reported-only** (context only; NEVER numerically
  compared to ours).
- **R4 — pending audit** (dataset/protocol/numbers not verified → do NOT cite
  numbers).

## Verified entries

| Paper | Method | Dataset | Same? | Protocol | Reported numbers (verified) | Tier | Placement in our paper |
|---|---|---|---|---|---|---|---|
| Zhang 2025 radarODE-MTL | radarODE-MTL | MMECG/RCG2ECG (FMCW) | yes | ECG_shape/PPI/Anchor; long-assembly not locally reproduced | no single clean PCC in ledger; ECG_shape/robustness metrics | R1 | T1 as **adapted** (0.728 Shape-200); no long-assembly claim |
| VMD-CBNeT 2025 | VMD-CBNeT | Nature/CR_RVS (CW) | yes | leave-one-out | PCC 0.856, RMSE 0.112, RR 5.625, QT 5.625, PR 4.911, QSR 9.643 (PDF Table I) | R2 | T3 Block B (reported-only caveat) |
| VMD-CBNeT 2025 | CNN+LSTM | Nature/CR_RVS (CW) | yes | leave-one-out | PCC 0.647, RMSE 0.127, RR 48.534, QT 48.534, PR 7.414, QSR 29.052 (PDF Table I) | R2 | T3 Block B |
| Jin 2025 Attention-DCGAN | Attention-DCGAN | MMECG public | yes | reported 2000 train samples / 100 test samples; no strict LOSO found | PCC 0.9853, RMSE 0.0329 mV | R2 | Table D only; MMECG reported-only caveat |
| Chowdhury 2024 MultiResLinkNet | MultiResLinkNet | Schellenberger/CR_RVS (CW) | yes | 5-fold subject-wise; Resting/Valsalva/Apnea; 1024-sample segments | Resting temporal CC 66.095±19.33; Valsalva 60.136±21.92; Apnea 55.331±18.95 | R2 | Table D; metric-space caveat |
| Kundu 2025 LifWavNet | LifWavNet | CR_RVS + Med-Radar (CW) | yes for CR_RVS | train/test chunks; not our 5-fold split | CR_RVS rho 0.926, MRE 0.823, MAEHR 0.63 bpm, MAERMSSD 13.67 ms | R2 | Table D; candidate-to-reproduce |
| Li 2025 Radar2ECG | Radar2ECG | Scidata Phase / CR_RVS resting subset | yes source dataset | 8:2 sample split, resting only, radar heart-sound features | RMSE 0.0237, MAE 0.0134, COS 0.987, R2 0.879 | R2 | Table D; strong caveat |
| G&Sharma 2025 BiLSTM-CW | BiLSTM | CR_RVS resting subset | yes source dataset | subject-specific and pooled/global; 12 resting subjects | subject-specific Pearson r 96.61%; pooled/global Pearson r 96.17%, F1 93.24% | R2 | Table D; not main comparison |
| Wu 2024 RSSRNet | RSSRNet | custom mmWave | no | their split | median ECG PCC 0.964, NRMSE 0.049 (their data) | R3 | T1 has our **adapted** RSSRNet (0.262); do NOT compare 0.964 to ours |
| Zhao 2024 AirECG | AirECG | custom arrhythmia mmWave | no | user-dep/indep | PCC 0.955 normal / 0.860 abnormal (their data) | R3 | T1 has our **adapted** AirECG (0.180); do NOT compare |
| Cui 2025 SS-S2SA | SS-S2SA | custom FMCW | no | 10 healthy subjects train / 5 evaluated subjects | average correlation about 0.92; P/R/T timing MAE 13.9/8.1/11.1 ms | R3 | related work only |
| Liu 2024 Cross-Modal AE | Cross-modal autoencoder | custom 60GHz FMCW | no | 16 participants train / 4 test participants | XCorr 0.914, RMSE 55.4e-3, timing errors below 31 ms | R3 | related work only |
| Feghoul 2024 ECGDiff | ECGDiff | Doppler (unaudited) | unknown | not audited | not audited | R4 | not cited until audited |

`QT == RR` in the VMD-CBNeT rows is reproduced exactly from PDF Table I (verified,
commit c80f11c); quote "as reported" only, never as independent QT evidence.

## Pending audit (R4) — related radar-ECG papers, numbers NOT cited yet

From `reference_papers/` (dataset/protocol not yet audited; treat as related work,
not comparators, until each is checked): Kong 2026 (SSM/FMCW), Zhang 2025 radarODE,
Zhang 2026 conditional diffusion, Xu 2025 WaveGRU-net, Li 2025 CUHeart,
Liu 2025 Finding-Order-in-Chaos, Liu mmJEPA-ECG, Chen 2024, Cao 2024 fiducial,
Cao 2026, Li 2025 non-intrusive, Wang 2022 UWB, phonocardiogram(2024).

Rule: promote an R4 paper to R2/R3 only after auditing its dataset + protocol +
exact reported numbers from its PDF; otherwise cite as qualitative related work.

## Locked claim rules for Table D

- ALLOWED: cite reported numbers with dataset + protocol + tier, as context.
- ALLOWED (R2, same dataset): the T3-locked phrasing — "exceeds reported VMD-CBNeT
  PCC 0.856 under our morphology protocol, protocol caveat".
- FORBIDDEN: comparing R3 (different-dataset) numbers to ours in any ranking; any
  "SOTA/outperforms" claim built on reported-only numbers under different protocols;
  citing R4 numbers before audit.

## Required actions before Table D is paper-ready

1. Decide which R4 papers to audit into R2/R3 (only those likely to be expected by
   reviewers, e.g., radarODE variants, Radar2ECG, SS-S2SA, LifWavNet, WaveGRU).
2. For each audited paper, record dataset/protocol/numbers in
   `LITERATURE_COMPARISON_LEDGER.csv` with a source (PDF + table/page).
3. Confirm QSR definition (QRS duration vs Q-S-R composite) across cited papers so
   the column is consistent.
4. Highest-value reproduction candidates, if time permits: (a) LifWavNet on our
   CR_RVS 5-fold split because it reports CR_RVS and continuous 1024-point ECG;
   (b) MultiResLinkNet on CR_RVS because it is simple 1D segmentation; (c)
   Attention-DCGAN only if the MMECG sample protocol can be made strict-LOSO.
