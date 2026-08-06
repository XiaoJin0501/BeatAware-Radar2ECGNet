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
cardiac event structure, dynamic electromechanical alignment, and clinically
meaningful continuous ECG evaluation.
```

## Single Research Question

| ID | Question |
|---|---|
| RQ1 | Can a new representation-aware, event/morphology-structured radar-to-ECG model achieve state-of-the-art or near-state-of-the-art ECG reconstruction on both MMECG and CR_RVS under fair evaluation against published and reproduced baselines? |

All other analyses are evidence dimensions for RQ1, not independent research
questions:

- representation analysis explains why the model works;
- event/morphology supervision supports the model mechanism;
- long-term ECG assembly makes comparison with radarODE-MTL fair;
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
| radarODE-MTL | Strong pipeline baseline; final paper comparison requires its long-term assembly protocol |
| REM-AlignNet-MTL | Current Q1 candidate model for rhythm-event-morphology multi-task reconstruction |

Use `REM-AlignNet-MTL` as the working model name unless the user explicitly
renames the method. Avoid introducing new names such as PhysioStruct unless a
separate naming decision is made.

## Non-Negotiable Caveat

Shape-200 results are beat morphology evidence only. They must not be described
as final long-term continuous ECG reconstruction results.

Current high Shape-200 numbers are pilot evidence for RQ1. They become
manuscript support only after the paper defines exactly whether the final table
uses Shape-200 morphology, 8-second continuous ECG, long-term assembled ECG, or
multiple complementary protocols.
