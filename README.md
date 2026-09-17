# BeatAware-Radar2ECGNet Q1 Workspace

This repository supports one Q1 journal objective: propose a new radar-to-ECG
reconstruction model and demonstrate strong performance on both MMECG and
CR_RVS, outperforming or comparing favorably with most relevant published and
reproduced baselines under fair protocols.

## Source of Truth

Read these files first:

```text
AGENTS.md
CLAUDE.md
docs/agent/SHARED_CONTEXT.md
docs/Q1_JOURNAL_MASTER_PLAN.md
docs/PROJECT_STATE.md
docs/RESEARCH_CONTRACT.md
docs/DATA_PROTOCOL.md
docs/METRIC_CONTRACT.md
docs/EXPERIMENT_POLICY.md
docs/AGENT_DEVELOPMENT_GUIDE.md
docs/PAPER_CLAIM_LEDGER.md
docs/GITHUB_WORKFLOW.md
experiments_mmecg/EXPERIMENT_REGISTRY.csv
experiments_mmecg/README.md
experiments/README.md
scripts/README.md
src/README.md
docs/DELETED_EXPERIMENTS_LOG.md
reference_papers/README.md
```

## Current Claim Status

The project uses a three-tier benchmark: Shape-200 morphology, strict blind
full-window continuous ECG, and GT-assisted temporal-tolerance diagnostics.
These endpoints must never be conflated.

```text
MMECG REM-Net Shape-200 PCC ~= 0.885 (multi-seed stability result)
CR_RVS REM-Net Shape-200 PCC ~= 0.886 (multi-seed stability result)
MMECG repaired-split continuous PCC@0 = 0.223 (folds 1/5/9 diagnostic baseline)
```

All current results are treated as `pilot` unless explicitly promoted in
`experiments_mmecg/EXPERIMENT_REGISTRY.csv`.

## Main Model

`REM-Net`:

```text
Representation–Event–Morphology Network
```

The intended Q1 framing is a new representation-aware, event/morphology-guided
radar-to-ECG model validated on two datasets.

## Current Protocols

| Dataset | Protocol | Use |
|---|---|---|
| MMECG public | 11-fold strict LOSO | Main benchmark |
| CR_RVS / Nature | 5-fold subject-independent CV | External validation |

## Next Formal Milestone

The only active experiment is the registered `q1_loss_gate`, currently waiting
for GPU release. It compares three single-term waveform objectives on repaired
MMECG folds 1/5/9:

```text
L1: PCC loss
L2: first-derivative loss
L3: multi-resolution STFT loss
```

Only a term satisfying the preregistered multi-metric rule may advance. The
former delay-warp and Route C assembly plans are archived, not pending work.

## Hygiene Rules

- Do not create unregistered experiment directories.
- Do not cite `smoke`, `probe`, or `pilot` as final manuscript results.
- Add each experiment to `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- Keep temporary outputs out of paper-facing directories.
- Use `scripts/README.md` before running scripts.
- Use `agent_skills/` as the single source for shared Claude/Codex skills.
- Run `python3 scripts/agent/sync_skills.py --check` before PRs that touch
  agent skills.
