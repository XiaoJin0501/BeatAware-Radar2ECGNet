# Results — Skeleton (Q1 manuscript)

Draft scaffold for the Results section. Each subsection lists: what is reported,
the locked numbers, the figure/table, the allowed claim language, and what must NOT
be written. Sources: `T1_MMECG_MORPHOLOGY_LOCKED.md`, `T3_CRRVS_EXTERNAL_AUDIT.md`,
`ASSEMBLY_DESIGN.md`, `PAPER_CLAIM_LEDGER.md`. Numbers are multi-seed (G1) where
available.

## 5.1 MMECG Beat-Morphology Reconstruction

Reports: Table T1 (Block A strict-LOSO) + Fig F1 (method comparison) + Fig F5
(transparent high/median/hard morphology overlays) + Fig F3-left (multi-seed
stability) + the morphology fiducial booster
(`docs/MORPHOLOGY_FIDUCIAL_BOOSTER.md`).

Key numbers (Shape-200, 11-fold strict LOSO, fold-macro):
- **REM-Net (proposed): PCC 0.885 ± 0.002** (multi-seed, 3 seeds; across-fold std ≈0.05).
- radarODE-MTL (adapted, ECG_shape space): 0.728 ± 0.133.
- REM-Event 0.279; WSST-TCN 0.273; RSSRNet(adapted) 0.262; AirECG(adapted) 0.180.

Narrative beats:
- REM-Net reconstructs beat morphology with the highest fidelity among all
  strict-LOSO methods, a large margin over the strongest reproduced baseline
  (radarODE-MTL, adapted, ECG_shape).
- Region-wise morphology analysis shows the gain is not only global PCC:
  REM-Net preserves QRS-edge morphology (0.962 PCC) and ST/T morphology
  (0.787 PCC), while radarODE-MTL is competitive at QRS-edge (0.961) but weak in
  ST/T (0.157) and diastolic morphology (0.034).
- Qualitative examples must use the deterministic 90th/median/10th percentile
  selection rule, not hand-picked overlays. Use the signed-error panels to show
  both successful QRS/ST-T recovery and hard-case amplitude mismatch.
- The ablation chain (WSST-TCN → +event head → +morphology MTL decoder) shows the
  morphology multi-task decoder is the key component; the alignment module was
  ablated with no effect and removed (G5).
- Multi-seed std 0.002 → result is seed-robust.

Allowed: "state-of-the-art beat-morphology reconstruction on MMECG under strict
11-fold LOSO." Forbidden: continuous/long-term ECG SOTA; comparing to
different-dataset reported numbers (RSSRNet 0.964 / AirECG 0.955); presenting the
LOSO+calibration BeatAware number in the fair column (Block B only, caveated).

## 5.2 External Validation on CR_RVS (CW radar)

Reports: Table T3 (Block A local 5-fold + Block B reported-only) + Fig F2 +
Fig F3-right (multi-seed stability).

Key numbers (Shape-200, 5-fold subject-independent, verified no subject overlap):
- **CW-STFT + REM-Net (proposed): PCC 0.886 ± 0.001** (multi-seed).
- CW-STFT + TCN (local baseline): 0.550 ± 0.075.
- VMD-CBNeT (reported-only, different protocol): PCC 0.856; CNN+LSTM 0.647.
- Region-wise booster: REM-Net vs CW-STFT+TCN reaches QRS-edge PCC 0.941 vs
  0.464 and ST/T PCC 0.793 vs 0.744; diastolic remains weak (0.380 vs 0.358).

Narrative beats:
- REM-Net transfers to a different radar modality (24-GHz CW) under
  subject-independent 5-fold CV, again strongly above the local representation
  baseline, and seed-robust (std 0.001).
- Against published Nature/CR_RVS results, our morphology PCC (0.886) exceeds the
  reported VMD-CBNeT PCC (0.856) — **as reported, under our morphology protocol**.

Allowed (locked): "under our subject-independent 5-fold CR_RVS morphology protocol,
REM-Net reaches PCC 0.886 ± 0.001, exceeding the reported VMD-CBNeT PCC 0.856
(reported-only, different protocol; not a same-protocol comparison)."
Forbidden: "strictly outperforms VMD-CBNeT under identical protocol"; any PR/QRS/QT
interval superiority (our full-window interval errors are worse); using the reported
RR==QT value as independent QT evidence.

## 5.3 Mechanistic Analysis of the Continuous Gap (discovery)

Reports: Fig F4 (oracle-vs-predicted anchor bound; 3-fold generalization) + the
route-C evidence chain.

Key numbers (fold-level):
- Morphology ≈0.96; **oracle-anchor cycle assembly 0.93–0.98**; predicted-anchor
  assembly 0.22–0.51 (fold-01); dense full-window floor ≈0.39.
- 3-fold: predicted-anchor OLA collapses on unseen subjects (fold_02/03 ≈0.02);
  recall recovers with threshold tuning but OLA does not (anchor **precision** MAE
  40–55 ms vs fold_01 16–20 ms).

Narrative beats:
- With oracle anchors, the learned morphology assembles into near-oracle continuous
  ECG → the continuous barrier is **beat-anchor localization, not morphology** (D1).
- Robust sub-50 ms anchor localization does not generalize across subjects under
  LOSO (D2): detection recall transfers, precision does not. This is presented as a
  characterized open problem + future direction, not a solved contribution.
- (Optional side-note D3) a per-beat electromechanical offset is observable but
  subject-dependent (not globally correctable).

Allowed: "we identify beat-anchor localization precision as the governing barrier to
continuous radar-to-ECG, evidenced by a high oracle upper bound." Forbidden:
presenting anchor-guided assembly as a working continuous model; claiming the delay
mechanism closes the gap (falsified).

## Limitations (to draft)

Continuous/full-window fidelity remains limited; clinical interval errors exceed
reported VMD-CBNeT under different protocols; radarODE long-term assembly not
reproduced (compared only in Shape-200 space); adapted-baseline preprocessing partly
unrecorded; Shape-200 is beat-normalized morphology (not continuous ECG);
low-amplitude diastolic morphology remains visibly and quantitatively weaker than
QRS/ST-T morphology.
