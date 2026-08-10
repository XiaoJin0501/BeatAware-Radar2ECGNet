# Paper Claim Ledger

This ledger maps manuscript claims to verified experimental evidence. Do not
write paper claims that cannot be traced here.

## Main thesis (revised 2026-08-10, after route-C demotion)

A new representation-aware radar-to-ECG model achieves strong **beat-morphology**
reconstruction on two radar modalities (FMCW MMECG, CW CR_RVS) under
subject-independent protocols, compares favorably with published/reproduced
baselines, and — as a mechanistic **discovery** — shows that the remaining
continuous-ECG gap is governed by **beat-anchor localization precision**, not by
morphology. See `docs/Q1_EXPERIMENT_MATRIX.md`, `docs/ASSEMBLY_DESIGN.md`.

> Demoted (2026-08-10): "anchor-guided assembly as the main continuous model" is
> NOT the main claim. 3-fold LOSO showed anchor detection recall generalizes but
> anchor PRECISION does not (fold_02/03 MAE 40–55 ms vs fold_01 16–20 ms;
> continuous OLA ~0 on 2/3 folds even after recall is restored). Route C is kept
> as a discovery + mechanism/future-direction section, not the Q1 main model.
> The earlier "electromechanical delay `δ(t)`" thesis was falsified before this
> (see `docs/ALIGNMENT_DESIGN.md` §7b).

## Main claims

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| M1 | Strong beat-morphology reconstruction on MMECG (Shape-200). | T1 | pilot | Shape-200 PCC ≈0.89 vs radarODE(adapted) 0.73, BeatAware 0.67. |
| M2 | External validation on CR_RVS (CW radar) matches/exceeds published results. | T3 | pilot | CW-STFT+REM Shape-200 PCC ≈0.887 vs VMD-CBNeT reported 0.856; protocol caveats stated. |
| M3 | Compares favorably with published/reproduced baselines under fair, output-formulation-aware protocols. | T1/T2/T3 tiers | partial | Needs final tier-labeled tables; sample-level results kept separate from subject-independent. |
| M4 | Generalizes across FMCW (MMECG) and CW (CR_RVS) under subject-independent protocols. | T1, T3 | partial | Morphology track; CR_RVS lag-aware refine 5-fold pilot (PCC 0.500 vs TCN 0.438). |

## Discovery / mechanism claims (route C, demoted from main model)

| Claim ID | Manuscript Claim | Supporting Evidence | Status | Notes |
|---|---|---|---|---|
| D1 | The radar→ECG continuous gap is governed by beat-anchor localization, not morphology: with oracle anchors, learned morphology assembles into near-oracle continuous ECG. | oracle-cycle OLA 0.93–0.98; morphology ≈0.96 | supported (fold-level) | Strong upper-bound evidence. |
| D2 | Robust sub-50 ms anchor localization under LOSO is the open bottleneck: detection recall generalizes across subjects but precision does not. | 3-fold anchor validation + failure diagnosis + threshold re-eval | supported | fold_02/03 MAE 40–55 ms; OLA ~0; recovering recall does not recover OLA. Frame as finding + future direction. |
| D3 (side) | A per-beat electromechanical offset (predicted anchor vs GT R) is observable but subject-dependent. | anchor signed-bias analysis | optional | Interpretability side-note; not globally correctable (LOSO val→test offset does not transfer). |

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
