# Shared Research Context

This is the stable knowledge layer shared by Claude Code and Codex. It should
change slowly. Do not store daily progress, transient metrics, or temporary
experiment status here.

## Research Scope

This project studies non-contact radar-to-ECG reconstruction for reliable
cardiac monitoring. The Q1 journal extension should not be framed as merely
adding modules to the ICONIP BeatAware model. The intended framing is:

```text
radar-to-ECG reconstruction requires cardiac-informative representation,
cardiac event structure, dynamic electromechanical alignment, and clinically
meaningful continuous ECG evaluation.
```

## Research Questions

| ID | Question |
|---|---|
| RQ1 | Which radar representations expose cardiac mechanical information useful for ECG reconstruction? |
| RQ2 | Can explicit rhythm/event/morphology structure improve ECG morphology and beat timing? |
| RQ3 | Can a fair long-term ECG assembly protocol show reliable continuous reconstruction beyond beat-normalized Shape-200 evidence? |

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
| PhysioStruct-style framework | Future manuscript framing if long-term evidence supports it |

## Non-Negotiable Caveat

Shape-200 results are beat morphology evidence only. They must not be described
as final long-term continuous ECG reconstruction results.

Current high Shape-200 numbers are pilot evidence until a locked long-term ECG
assembly protocol, manifest, metric audit, and fair baseline comparison exist.
