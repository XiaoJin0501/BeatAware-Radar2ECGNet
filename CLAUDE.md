# CLAUDE.md - BeatAware-Radar2ECGNet Q1 Workspace

This file is the handoff entry point for Claude Code and other coding agents.
The repository is now organized around the Q1 journal extension, not the older
diffusion/samplewise notes.

## Read First

Use these files as the source of truth:

```text
docs/PROJECT_STRUCTURE_Q1.md
docs/NEXT_EXPERIMENTS_Q1.md
experiments_mmecg/EXPERIMENT_REGISTRY.csv
experiments_mmecg/README.md
scripts/README.md
src/README.md
docs/DELETED_EXPERIMENTS_LOG.md
reference_papers/README.md
```

Before launching or citing any experiment, check
`experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Current Paper Direction

The Q1 paper should not be framed as "adding modules to BeatAware." The current
position is:

```text
Radar-to-ECG reconstruction should be evaluated as a representation-aware,
event-structured, and morphology-guided continuous ECG reconstruction problem.
```

The current strongest local evidence is Shape-200 beat morphology. It is not
yet a final long-term continuous ECG claim.

## Current Main Model

`REM-AlignNet-MTL` means:

```text
Rhythm-Event-Morphology Alignment Network with Multi-Task Learning
```

The model uses a WSST/radar representation frontend, rhythm/event/morphology
heads, and multi-task training to improve beat-shape reconstruction. The next
formal milestone is to evaluate it under a radarODE-style long-term ECG
assembly protocol.

## Experiment Categories

Every experiment belongs to exactly one category:

| Category | Meaning | Publishable |
|---|---|---|
| `smoke` | Code-running check | false |
| `probe` | Quick trend check | false |
| `pilot` | Complete-ish, but protocol/model claim not locked | false by default |
| `final` | Protocol-locked formal experiment | true |
| `invalid` | Wrong, unfair, broken, or untraceable | false |

No result is paper-ready unless the registry says:

```text
category=final
publishable=true
```

## Retained Result Directories

The retained top-level experiment directories are:

```text
q1_remalign_mtl_shapefirst_loso
q1_remalign_wsst_tcn_loso
q1_remalign_event_lowpeak_loso
q1_crrvs_remalign_mtl_5fold
q1_crrvs_wsst_tcn_5fold
mmecg_radarode_mtl_ega_strict_loso
mmecg_beataware_fewshot40v10
mmecg_airecg_strict_loso_v4
mmecg_rssrnet_strict_loso_v2
mmecg_cnn_strict_loso_baseline
mmecg_lstm_strict_loso_baseline
mmecg_bilstm_strict_loso_baseline
mmecg_conformer_strict_loso_baseline
```

Generated report packages currently retained:

```text
q1_paper_style_report
q1_crrvs_external_report_5fold
q1_supervisor_shape200_snapshot
q1_teacher_report_remalign_mtl_shapefirst
```

Deleted probe/smoke/obsolete directories are recorded in
`docs/DELETED_EXPERIMENTS_LOG.md`.

## Dataset Protocols

| Dataset | Protocol | Current Status |
|---|---|---|
| MMECG public | 11-fold strict LOSO | Main local benchmark |
| CR_RVS / Nature | 5-fold subject-independent CV | External validation |

The local MMECG is the public MMECG dataset. It is not the unrelated dataset
called MMECG in some external papers.

## Important Result Caveat

Current high-performing results are Shape-200 beat morphology:

```text
MMECG REM-AlignNet-MTL Shape-200 PCC ~= 0.8898
CR_RVS REM-AlignNet-MTL Shape-200 PCC ~= 0.8865
```

These should be reported as progress/pilot evidence only. For a final Q1
comparison with radarODE-MTL, implement and run the same long-term ECG assembly
protocol for both radarODE-MTL and REM-AlignNet-MTL.

## Next Formal Task

Follow `docs/NEXT_EXPERIMENTS_Q1.md`:

```text
1. Implement radarODE-style long-term ECG assembly evaluation.
2. Evaluate radarODE-MTL and REM-AlignNet-MTL under the same protocol.
3. Promote only completed, fair, traceable runs to category=final.
4. Regenerate paper-style tables and visualizations from final runs only.
```

## Working Rules

- Do not create unregistered experiment directories.
- Do not cite smoke/probe/pilot results as final paper results.
- Add every new experiment to `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- Keep temporary debugging outputs outside paper-facing directories or delete
  them after use.
- Use `scripts/README.md` before running training/evaluation scripts.
