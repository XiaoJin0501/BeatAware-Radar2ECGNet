# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

## Central thesis

Radar↔ECG reconstruction is limited by a time-varying **electromechanical delay**
`δ(t)` (≈ pre-ejection period). Explicit event structure plus a self-emergently
learned alignment closes the morphology↔continuous gap. See
`docs/ALIGNMENT_DESIGN.md` and `docs/Q1_EXPERIMENT_MATRIX.md`.

## Thesis claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| C1 | Explicit event structure + learned time-varying electromechanical alignment enables continuous ECG reconstruction on radar. | T1, T2, T4 (B1/B4) | planned | Needs the fold_01 minimal loop, then 11-fold LOSO. |
| C2 | Dual-track evaluation shows morphology is easy but timing is hard, and the alignment closes the gap. | T2, Fig-A, T4 | planned | Requires B0→B4 gap-closing evidence. |
| C3 | The learned delay `δ` is interpretable and varies with physiological state (HR, Valsalva/apnea). | Fig-B (pseudo-GT lag), Fig-C (HR), Fig-D (state) | planned | Pseudo-GT delay is validation-only, never in training/inference. |
| C4 | The framework generalizes across FMCW (MMECG) and CW (CR_RVS) under subject-independent protocols. | T3, CR_RVS ablation | partial | CR_RVS lag-aware refine 5-fold pilot exists (PCC 0.500 vs TCN 0.438), but not under the new hybrid-δ design yet. |

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
