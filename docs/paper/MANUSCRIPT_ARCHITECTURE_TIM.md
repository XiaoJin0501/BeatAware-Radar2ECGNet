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
| III Measurement procedure (**drafted**: `sections/03_MEASUREMENT_PROCEDURE.md`) | Measurand (continuous 1 x 1600 ECG, stored units); split construction and the repaired `loso_v2`; train-fold scaler; validation-only selection; the test-access state machine; metric definitions, aggregation labels, coverage gate, heavy-tailed ratios; **template floor** as a radar-free reference | METRIC_CONTRACT; MMECG split audit; C1, C2 | Ready to write |
| IV Systems under test | REM-Net as a representative factorized architecture (representation / event / morphology); the five matched families and every adaptation (D-lists); identical budget and checkpoint rule | `docs/MATCHED_BASELINE_PROTOCOL.md`; `ADAPTATIONS` | Ready to write |
| V Datasets | MMECG (77 GHz FMCW, 11 subjects, 11-fold LOSO); CR_RVS (24 GHz CW, 5-fold subject-independent, per-record z-scored target) | DATA_PROTOCOL; W3 specs | Ready (W3 numbers pending) |
| VI-A Protocol fault 1: split leakage (**drafted**: `sections/06_PROTOCOL_FAULTS_AND_LIMITS.md`) | Two validation subjects across 11 folds; cross-fold selection bias; the repair | `docs/MMECG_SPLIT_PROVENANCE_AUDIT.md` | Ready |
| VI-B Protocol fault 2: saturated beat metric (**drafted**) | Template floor 0.879-0.887 on both datasets; prior Tier-1 numbers at the floor; radarODE-MTL below it | C2; `docs/audits/Q1_TIER1_TEMPLATE_FLOOR_AUDIT.md` | Ready |
| VI-C Matched continuous benchmark | Table 1: six models, eleven folds, all metrics, paired tests; R² ≈ 0 throughout | C3; W2 for seeds | Seed 42 ready; three-seed pending W2 |
| VI-D Mechanism: timing, not morphology (**drafted**) | Template at GT peaks R² 0.73 vs models ≈ 0; template at radar peaks R² < 0; amplitude shrinkage as loss-optimal hedge; Tier-3 tolerance curve | C4; `docs/audits/Q1_TEMPLATE_TIMING_AUDIT.md` | Ready (MMECG); CR_RVS pending W3 |
| VI-E External validity (CR_RVS) | REM-Net, Chen-TCN, WaveGRU-Net on 5 subject-independent folds, overall and Resting-only; template floor and template-timing diagnostics on CW radar | W3 | Pending W3 |
| VI-F Reported literature on CR_RVS | Protocol-annotated table of Wu 2024, Xiao 2026, Li 2025, Chowdhury 2024; RSSRnet's own 0.959 → 0.39-0.90 contrast as the field's evidence of split-driven inflation | `docs/audits/Q1_CRRVS_LITERATURE_PROTOCOL_AUDIT.md` | Ready |
| VI-G Engineering | Parameters, latency; radarODE-MTL EGA cost | `docs/EFFICIENCY_AUDIT.md`; W2 timing | Update with baseline params |
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
| Table 6 | Reported CR_RVS literature with protocol columns (scenario, split unit, window overlap, target normalisation, metric) | CR_RVS literature audit | Done |
| Fig. 1 | Measurement pipeline and where each fault enters (split, metric, timing) | new diagram | To draw |
| Fig. 2 | Per-fold PCC of six models with the template-at-GT upper bound as a reference line | W2 + timing audit | To draw |
| Fig. 3 | Waveform examples: GT, REM-Net, Chen-TCN, template@GT, template@radar peaks, at quantitatively chosen best / median / worst folds | frozen predictions | To draw |
| Fig. 4 | Template floor vs model Shape-200 PCC per fold (both datasets) | Tier-1 audit | To draw |
| Fig. 5 | Tier-3 tolerance curve with the template-at-GT bound | temporal-tolerance audit | To draw |
| Fig. 6 | Objective schematic: blunt versus sharp QRS and the high-frequency band they produce (synthetic; caption must say so) | `scripts/figures/plot_mechanism_stft.py` | Drawn |
| Fig. 6b | Bare version of Fig. 6 for external annotation: waveforms and spectrograms only, no text | `scripts/figures/plot_mechanism_base.py` | Drawn |
| Fig. 7 | Measured spectral deficit: reference ECG versus the MR-STFT estimate on a median held-out window, with the above-10 Hz share (4 % here, benchmark median 0.122) | `scripts/figures/plot_spectral_deficit_measured.py` | Drawn |

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

1. ~~W2: seeds 123 and 2024 for all six models~~ — complete (198 sealed folds, `docs/audits/Q1_MULTISEED_BENCHMARK_AUDIT.md`).
2. ~~W3: CR_RVS continuous benchmark~~ — complete, with scenario, template-floor and timing diagnostics (`docs/audits/Q1_CRRVS_DIAGNOSTICS_AUDIT.md`, `docs/audits/Q1_SCENARIO_BREAKDOWN_AUDIT.md`).
3. Remaining to draft: Sections I, II, IV, V, VI-C, VI-E, VI-F, VI-G, VII-IX and the five figures.
4. ~~Three-seed aggregation~~ — `scripts/aggregate_multiseed_benchmark.py`, Table 1 frozen.
