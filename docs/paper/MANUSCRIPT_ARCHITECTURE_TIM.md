# Manuscript Architecture — IEEE TIM Submission

**Title:** Contactless Radar-to-ECG Reconstruction: A Rigorous Leak-Free Benchmark
and Factorized Deep Architecture

**Venues:** IEEE Transactions on Instrumentation and Measurement (primary); IEEE
JBHI; IEEE IoT-J. Positioning locked 2026-09-19 in `docs/Q1_JOURNAL_MASTER_PLAN.md`.

This blueprint replaces the Shape-200-headline structure of `docs/paper/RQ1.tex`.
Every quantitative sentence must map to a claim in `docs/PAPER_CLAIM_LEDGER.md`
("Benchmark paper claims"), and every claim to a registered experiment.

## Framing for a measurement journal

Treat radar-to-ECG reconstruction as a **measurement system** whose output is an
estimate of the ECG, and the evaluation protocol as the **measurement procedure**
that must be traceable, leak-free and uncertainty-aware. The paper's contribution is
to show that the procedure the field uses mis-states the measurand in three ways --
split leakage, a saturated beat-level metric, and conflation of timing with
morphology -- and to supply a corrected procedure together with the true state of
the art under it. Uncertainty is carried by fold-level dispersion, paired tests,
three seeds, and radar-free reference floors.

## Section plan

| § | Content | Evidence | Status |
|---|---|---|---|
| I Introduction | Contactless ECG promise; reported beat-level PCC of 0.8-0.98; three protocol faults that inflate it; contributions C1-C4 | Ledger C1-C4 | Draft after W2 |
| II Related work | Radar cardiac sensing; learning-based radar-to-ECG (the five families); evaluation practice (sample-wise vs subject-wise splits, beat-normalised metrics) | `docs/TABLE_D_REPORTED_LITERATURE_AUDIT.md` | Reusable from RQ1.tex §II with a new evaluation-practice subsection |
| III Measurement procedure | Measurand (continuous 1 x 1600 ECG, stored units); split construction and the repaired `loso_v2`; train-fold scaler; validation-only selection; the test-access state machine; metric definitions, aggregation labels, coverage gate, heavy-tailed ratios; **template floor** as a radar-free reference | METRIC_CONTRACT; MMECG split audit; C1, C2 | Ready to write |
| IV Systems under test | REM-Net as a representative factorized architecture (representation / event / morphology); the five matched families and every adaptation (D-lists); identical budget and checkpoint rule | `docs/MATCHED_BASELINE_PROTOCOL.md`; `ADAPTATIONS` | Ready to write |
| V Datasets | MMECG (77 GHz FMCW, 11 subjects, 11-fold LOSO); CR_RVS (24 GHz CW, 5-fold subject-independent, per-record z-scored target) | DATA_PROTOCOL; W3 specs | Ready (W3 numbers pending) |
| VI-A Protocol fault 1: split leakage | Two validation subjects across 11 folds; cross-fold selection bias; the repair | `docs/MMECG_SPLIT_PROVENANCE_AUDIT.md` | Ready |
| VI-B Protocol fault 2: saturated beat metric | Template floor 0.879-0.887 on both datasets; prior Tier-1 numbers at the floor; radarODE-MTL below it | C2; `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md` | Ready |
| VI-C Matched continuous benchmark | Table 1: six models, eleven folds, all metrics, paired tests; R² ≈ 0 throughout | C3; W2 for seeds | Seed 42 ready; three-seed pending W2 |
| VI-D Mechanism: timing, not morphology | Template at GT peaks R² 0.73 vs models ≈ 0; template at radar peaks R² < 0; amplitude shrinkage as loss-optimal hedge; Tier-3 tolerance curve | C4; `docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md` | Ready (MMECG); CR_RVS pending W3 |
| VI-E External validity (CR_RVS) | Same benchmark and diagnostics on CW radar | W3 | Pending W3 |
| VI-F Engineering | Parameters, latency; radarODE-MTL EGA cost | `docs/EFFICIENCY_AUDIT.md`; W2 timing | Update with baseline params |
| VII Discussion | What the field's numbers measure; what a credible continuous claim requires (timing); recommended reporting checklist | C1-C4 | After W2/W3 |
| VIII Limitations | Two datasets, small subject counts, single CR_RVS validation subject per fold, adapted reproductions, template is a single fixed shape | — | Draft now |
| IX Conclusion | — | — | Last |

## Tables and figures

| ID | Content | Source | Status |
|---|---|---|---|
| Table 1 | Matched continuous benchmark, MMECG: PCC, R², NRMSE, RMSE, MAE, waveform F1@50/100/150, QRS / ST-T MAE, variance ratio, >10 Hz median, gated RR/HR, params; three-seed mean ± std; paired tests vs REM-Net | W2 summaries | Seed 42 done |
| Table 2 | Template floor versus every Tier-1 number (old split, `loso_v2`, CR_RVS), with per-fold paired tests | Tier-1 audit | Done |
| Table 3 | Template-timing diagnostics T0-T3 | Timing audit | Done (MMECG) |
| Table 4 | CR_RVS benchmark (continuous + template floor) | W3 | Pending |
| Table 5 | Adaptation register for the five families | `ADAPTATIONS` | Done |
| Fig. 1 | Measurement pipeline and where each fault enters (split, metric, timing) | new diagram | To draw |
| Fig. 2 | Per-fold PCC of six models with the template-at-GT upper bound as a reference line | W2 + timing audit | To draw |
| Fig. 3 | Waveform examples: GT, REM-Net, Chen-TCN, template@GT, template@radar peaks, at quantitatively chosen best / median / worst folds | frozen predictions | To draw |
| Fig. 4 | Template floor vs model Shape-200 PCC per fold (both datasets) | Tier-1 audit | To draw |
| Fig. 5 | Tier-3 tolerance curve with the template-at-GT bound | temporal-tolerance audit | To draw |

## Material to retire from RQ1.tex

- §VI-A "MMECG Beat-Morphology Reconstruction" as a headline result (0.882 is BLOCKED and at the template floor).
- §VI "External Validation on CR_RVS" framed as exceeding VMD-CBNeT (claim S3 suspended).
- §III "Morphology–Anchor–Assembly Decomposition" and "Removed Alignment Module" as method content; the closed routes move to a short negative-results paragraph in Discussion.
- Any sentence stating that REM-Net outperforms, resolves collapse, or recovers morphology from radar.

## Material to keep

- Robustness to input noise and efficiency sections, re-labelled as engineering characterisation of REM-Net, not superiority.
- Reported-literature context, now explicitly separated from matched results.
- The physiological-state breakdown if re-run on the continuous benchmark (optional).

## Open items before drafting results

1. W2: seeds 123 and 2024 for all six models (queued; about 32 GPU hours).
2. W3: CR_RVS continuous benchmark for all six models (queued after W2).
3. Template-floor and template-timing diagnostics on CR_RVS (analysis only, after W3).
4. Three-seed aggregation script with paired tests pooled appropriately (seed-averaged fold values).
