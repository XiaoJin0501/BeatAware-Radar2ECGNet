# Shared Research Context

This is the stable knowledge layer shared by Claude Code and Codex. It should
change slowly. Do not store daily progress, transient metrics, or temporary
experiment status here.

## Research Scope

This project serves one Q1 journal paper objective: propose a new radar-to-ECG
reconstruction model and demonstrate strong performance on both MMECG and
CR_RVS, outperforming most relevant published and reproduced baselines under
fair protocols.

The project should not drift into a broad multi-paper platform, arrhythmia
classification project, general foundation-model project, or open-ended agent
governance project. Governance files exist only to protect this RQ1 objective.

The Q1 journal extension should not be framed as merely adding modules to the
ICONIP BeatAware model. The intended framing is:

```text
radar-to-ECG reconstruction requires cardiac-informative representation,
cardiac event structure, morphology-aware supervision, and clinically meaningful
continuous ECG evaluation.
```

## Single Research Question

| ID | Question |
|---|---|
| RQ1 | Can a new representation-aware, event/morphology-structured radar-to-ECG model achieve state-of-the-art or near-state-of-the-art ECG reconstruction on both MMECG and CR_RVS under fair evaluation against published and reproduced baselines? |

All other analyses are evidence dimensions for RQ1, not independent research
questions:

- representation analysis explains why the model works;
- event/morphology supervision supports the model mechanism;
- strict blind continuous reconstruction and temporal-tolerance diagnostics
  quantify the remaining waveform/timing gap without promoting an assembly route;
- CR_RVS validation demonstrates cross-radar-modality robustness.

## Dataset Roles

| Dataset | Role |
|---|---|
| MMECG public | Main FMCW radar benchmark with strict 11-fold LOSO protocol |
| CR_RVS / Nature | External 24-GHz CW radar validation with subject-independent 5-fold CV |

The local MMECG public dataset is not the unrelated dataset also called MMECG
in some external papers.

## Model Lineage

| Model | Role |
|---|---|
| BeatAware | ICONIP baseline with raw/FMCW physiology-aware modules |
| radarODE-MTL | Strong adapted morphology baseline; its official long-term assembly is not locally reproducible and must not be claimed |
| REM-Net (code: REMAlignMTLNet; formerly "REM-AlignNet-MTL") | Q1 main model: morphology-first, beat-structured multi-task reconstruction |

Canonical paper model name (locked 2026-08-10): **REM-Net**
(Representation–Event–Morphology Network), positioned as **morphology-first,
beat-structured** radar-to-ECG reconstruction. The former working name
"REM-AlignNet-MTL" and the code class `REMAlignMTLNet` refer to the same model;
"Align" was dropped because the electromechanical-alignment/continuous thesis was
closed (see `docs/PAPER_CLAIM_LEDGER.md` and `docs/Q1_JOURNAL_MASTER_PLAN.md`).
The delay-warp and anchor-guided Route C assembly designs are closed historical
routes. Their retained audits document negative evidence; they are not candidate
models, next experiments, or manuscript performance claims. Do not introduce
other names without a separate naming decision.

## Non-Negotiable Caveat

Shape-200 results are beat morphology evidence only. They must not be described
as final long-term continuous ECG reconstruction results.

The current locked RQ1 main evidence uses Shape-200 beat morphology as the
primary supervised output formulation: MMECG seed-42 final-test PCC 0.882 ±
0.061 with 3-seed stability 0.885 ± 0.002, and CR_RVS seed-42 final-test PCC
0.881 ± 0.013 with 3-seed stability 0.886 ± 0.001. Current continuous ECG work
uses the three-tier benchmark in `docs/Q1_JOURNAL_MASTER_PLAN.md`: Shape-200
morphology, strict blind full-window reconstruction, and GT-assisted temporal
tolerance diagnostics. Max-Lag NCC is diagnostic only.
