# BeatAware-Radar2ECGNet Q1 Workspace

This repository is being reorganized for a Q1 journal extension of the ICONIP
BeatAware-Radar2ECGNet work.

## Source of Truth

Read these files first:

```text
CLAUDE.md
docs/PROJECT_STRUCTURE_Q1.md
docs/NEXT_EXPERIMENTS_Q1.md
experiments_mmecg/EXPERIMENT_REGISTRY.csv
experiments_mmecg/README.md
scripts/README.md
src/README.md
docs/DELETED_EXPERIMENTS_LOG.md
reference_papers/README.md
```

## Current Claim Status

The current strongest local result is Shape-200 beat morphology reconstruction,
not final long-term continuous ECG reconstruction.

```text
MMECG REM-AlignNet-MTL Shape-200 PCC ~= 0.8898
CR_RVS REM-AlignNet-MTL Shape-200 PCC ~= 0.8865
```

All current results are treated as `pilot` unless explicitly promoted in
`experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Main Model

`REM-AlignNet-MTL`:

```text
Rhythm-Event-Morphology Alignment Network with Multi-Task Learning
```

The intended Q1 framing is representation-aware, event-structured, and
morphology-guided radar-to-ECG reconstruction.

## Current Protocols

| Dataset | Protocol | Use |
|---|---|---|
| MMECG public | 11-fold strict LOSO | Main benchmark |
| CR_RVS / Nature | 5-fold subject-independent CV | External validation |

## Next Formal Milestone

The next paper-critical experiment is a fair radarODE-style long-term ECG
assembly evaluation:

```text
radarODE-MTL original-style long ECG protocol
REM-AlignNet-MTL under the same long ECG protocol
MMECG strict LOSO
CR_RVS 5-fold subject-independent validation
```

Only after this should results be promoted from `pilot` to `final`.

## Hygiene Rules

- Do not create unregistered experiment directories.
- Do not cite `smoke`, `probe`, or `pilot` as final manuscript results.
- Add each experiment to `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- Keep temporary outputs out of paper-facing directories.
- Use `scripts/README.md` before running scripts.
