# Q1 Manuscript Blueprint

How this paper becomes a submission. Grounded in the locked evidence
(`docs/T1_MMECG_MORPHOLOGY_LOCKED.md`, `docs/T3_CRRVS_EXTERNAL_AUDIT.md`,
`docs/TABLE_D_REPORTED_LITERATURE_AUDIT.md`, `docs/PAPER_CLAIM_LEDGER.md`). The
guiding principle is **claim–result alignment**: every headline maps to verified,
protocol-clean evidence; nothing is claimed that the experiments cannot support.

## 1. Positioning & model naming (redefined)

**Positioning: morphology-first, beat-structured radar-to-ECG reconstruction.** The
model reconstructs ECG **beat morphology** with high fidelity across two radar
modalities under subject-independent protocols. Continuous full-window ECG and
anchor-guided assembly are NOT the main claim; the continuous story is a
**mechanism analysis / discovery** (why continuous is hard) + future direction.

**Rename (drop "Align" — it referenced the falsified delay thesis):**
- Option A (recommended, keeps lineage): **REM-Net — Representation–Event–Morphology
  Network** (morphology-first; "Align" dropped).
- Option B: **BeatMorphNet** (beat-structured morphology).
- Option C: **CardioREM**.
The current code class `REMAlignMTLNet` stays as the implementation; the paper name
is a labeling decision (user to confirm). Whichever is chosen, describe the model as
representation encoder (WSST) → event structure head → **morphology multi-task
decoder** (the component that carries the Shape-200 strength).

## 2. Central claim

> A representation-aware, beat-structured radar-to-ECG network reconstructs ECG beat
> morphology with state-of-the-art fidelity across FMCW (MMECG) and CW (CR_RVS)
> radar under subject-independent protocols; and we show the remaining barrier to
> continuous ECG is **beat-anchor localization precision**, not morphology.

## 3. Contributions (each mapped to evidence)

- **C1 (M1):** SOTA MMECG beat morphology — Shape-200 PCC 0.882 ± 0.061
  (seed-42 final-test; 3-seed stability 0.885 ± 0.002) vs radarODE(adapted)
  0.728, under 11-fold strict LOSO, fold-macro. [T1 Block A]
- **C2 (M2/M4):** Cross-modality external validation on CW CR_RVS — morphology PCC
  0.881 ± 0.013 (3-seed stability 0.886 ± 0.001), exceeding reported VMD-CBNeT
  0.856 *under our morphology protocol* (caveat). [T3 Block A + Block B]
- **C3 (D1/D2):** Mechanistic discovery — the continuous-ECG gap is governed by
  beat-anchor localization: with oracle anchors, learned morphology assembles to
  0.93–0.98, but predicted anchor **precision** does not generalize under LOSO.
  [route-C evidence chain]
- **C4:** Output-formulation-aware, fairness-audited evaluation (strict-LOSO fair
  block vs calibrated/reported-only blocks; documented negative results).

## 4. Figure / table list

Main:
- **T1** MMECG morphology (Block A fair; BeatAware calib as Block B footnote).
- **T3** CR_RVS external (Block A local 5-fold; Block B reported-only, caveated).
- **T4** morphology ablation: WSST-TCN → REM-Event → +morphology head (from T1 rows;
  see gap G5 re: alignment-module ablation).
- **Table D** reported-literature (context, tiered).
- **Fig-1** method overview (encoder → event head → morphology decoder).
- **Fig-2** qualitative morphology reconstructions (have report figs).
- **Fig-3** discovery: oracle-vs-predicted anchor OLA bound + precision-generalization
  across folds (from route-C evidence).
Secondary/appendix:
- **T2** MMECG continuous diagnostic (clearly secondary).
- **T5** efficiency (params/FLOPs/latency) — GAP.
- statewise / PCC-CDF figures (have report figs).

## 5. Section plan (what each says)

- **Intro:** contactless ECG motivation; morphology vs continuous; gap statement.
- **Related work:** radar-to-ECG (radarODE, VMD-CBNeT, AirECG, RSSRNet, …) via
  Table D tiers; position ours as morphology-first + the discovery.
- **Method:** representation encoder, event head, morphology multi-task decoder;
  (alignment module described honestly as ablated/optional).
- **Datasets & protocol:** MMECG 11-fold strict LOSO; CR_RVS 5-fold
  subject-independent (verified); Shape-200 definition; aggregation (fold-macro).
- **Results:** T1, T3, T4, qualitative figs.
- **Discovery/mechanism:** D1/D2 (oracle bound, localization precision, LOSO
  generalization); route C as analysis + future direction.
- **Discussion/limitations:** continuous weakness; interval errors worse than
  reported VMD-CBNeT; single-seed→multi-seed; adapted-baseline caveats.
- **Conclusion.**

## 6. Claims: verified / forbidden

Verified (writable): M1, M2 (caveated), M4, D1, D2, C4. See PAPER_CLAIM_LEDGER.
Forbidden (do NOT write): continuous-ECG SOTA; "strictly outperforms VMD-CBNeT under
identical protocol"; PR/QRS/QT interval superiority; Shape-200 as long-term
continuous evidence; route C as the main continuous model; any different-dataset
(R3) head-to-head (e.g., vs RSSRNet 0.964 / AirECG 0.955).

## 7. Remaining gaps (must-have vs optional)

Must-have for a credible Q1 submission:
- **G1 Multi-seed (≥3) — DONE (2026-08-11):** final no-alignment REM-Net, seeds
  42/123/2024. MMECG Shape-200 **0.885 ± 0.002**, CR_RVS **0.886 ± 0.001**
  (across-seed). Seed-robust; confirms M1/M2 and the G5 alignment drop. The
  main table uses the locked seed-42 final-test run: MMECG **0.882 ± 0.061**,
  CR_RVS **0.881 ± 0.013**.
- **G2 Publication-quality T1/T3 figures — DONE as report figures; needs visual
  polishing only.**
- **G3 Naming/positioning cleanup — DONE:** paper model name is REM-Net; code
  class may remain `REMAlignMTLNet`.

Should-have:
- **G4 Efficiency table (T5)** — params/FLOPs/latency (model vs preprocessing).
- **G5 Alignment-module ablation — RESOLVED (2026-08-10):** align vs no-align gives
  Δ Shape-200 PCC +0.005 (fold01) / −0.004 (fold05) = within noise. **Alignment
  module DROPPED.** Final REM-Net = encoder → event head → morphology MTL decoder
  (use_alignment=False). G1 multi-seed runs on this final model.

Optional / decide:
- **G6 strict-LOSO BeatAware** re-run to get a fair prior-conf row in T1 Block A
  (currently only calibrated Block B). New experiment — decide if worth it.
- **G7 CR_RVS fairer continuous comparison** to reduce the PCC-space mismatch with
  VMD-CBNeT (our full-window vs their reconstruction space).
- **G8 CR_RVS +event ablation** (CW-STFT: TCN → +event → +morphology) to mirror T4.
- **G9 Audit selected R4 papers** into Table D (radarODE variants, Radar2ECG,
  SS-S2SA, LifWavNet, WaveGRU) if reviewers would expect them.

## 8. Honest current position (one paragraph)

Two usable main results (MMECG morphology; CR_RVS morphology/external), one valuable
but negative continuous mechanism analysis (route C), and now a locked evidence chain
(T1/T3/Table D) with fairness audits. The supervised final baseline is fixed as
REM-Net with alignment disabled. The remaining work for a three-day draft is not
large-scale model redesign; it is paper assembly: final tables, publication-quality
qualitative figures, Results prose, and optional efficiency reporting.

## 9. Target journals (order)

IEEE J-BHI or IEEE Sensors (fit: biomedical signal + sensing, morphology + external
validation + honest mechanism study); Biomedical Signal Processing & Control as
strong alternative. radarODE-MTL (TBME/JBHI-tier) is the key comparator to position
against — at the morphology level, with the continuous discovery as the novel angle.
