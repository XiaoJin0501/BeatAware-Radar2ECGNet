# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

## Central thesis (revised 2026-08-10)

Radar-to-ECG continuous fidelity is gated by accurate **beat localization**
(rhythm/anchor prediction), not by beat morphology. We reconstruct continuous ECG
by (1) a strong beat-localization head predicting precise anchor positions + local
rhythm from radar, and (2) assembling the learned morphology (~0.96) at those
positions. See `docs/ASSEMBLY_DESIGN.md` and `docs/Q1_EXPERIMENT_MATRIX.md`.

> Superseded thesis: "time-varying electromechanical delay `δ(t)` closes the gap"
> was FALSIFIED on MMECG fold_01 (delay warp in latent/output space did not move
> continuous PCC beyond ~0.40; see `docs/ALIGNMENT_DESIGN.md` §7b). The δ analysis
> is retained only as an interpretability side-figure.

## Thesis claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| C1 | A radar beat-localization head predicts accurate beat anchors/rhythm, enabling structured continuous ECG assembly. | T2, T4 (L1/L3), localization error | planned | Needs the localization head (route C). Diagnostic ceiling: oracle-cycle 0.93–0.96 vs predicted-anchor 0.23–0.37. |
| C2 | Dual-track evaluation shows morphology is easy but localization/timing is hard; the localization head closes the continuous gap. | T2, Fig-A, T4 | planned | Requires L0→L3 gap-closing evidence. |
| C3 | Beat morphology is reconstructed with high fidelity and, given correct anchors, assembles into near-oracle continuous ECG. | T1, oracle-cycle OLA | partial | Shape-head PCC ≈0.96; oracle-cycle OLA ≈0.93–0.96 on fold_01. |
| C4 | The framework generalizes across FMCW (MMECG) and CW (CR_RVS) under subject-independent protocols. | T3, CR_RVS | partial | CR_RVS lag-aware refine 5-fold pilot exists (PCC 0.500 vs TCN 0.438); route-C localization not yet run on CR_RVS. |
| C5 (side) | A residual electromechanical delay (predicted anchor vs GT R) is observable per beat. | delay side-analysis | optional | Demoted interpretability analysis, not a main claim. |

## Supporting / boundary claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| S1 | Strong beat morphology reconstruction on MMECG and CR_RVS. | T1, T3 | pilot | Shape-200 PCC ≈0.890 (MMECG), ≈0.887 (CR_RVS); morphology evidence only. |
| S2 | Compares favorably with published/reproduced baselines under fair protocols. | T1/T2/T3 tiers | partial | Requires fair-tier labels; sample-level reported results separated from subject-independent main comparison. |
| S3 | CR_RVS external morphology exceeds VMD-CBNeT reported PCC. | T3 | pilot | 0.887 vs reported 0.856; protocol definitions must be stated. |

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
- Promotion from pilot→final only via the result-promotion workflow.
