# GitHub Workflow

This repository should be managed as a research codebase with reproducible
experiment bookkeeping.

## Branch Policy

| Branch Type | Pattern | Purpose |
|---|---|---|
| stable | `main` | Clean, reviewed project state only |
| Q1 project work | `q1/<topic>` | Main journal-extension development |
| experiment implementation | `exp/<dataset>-<model>-<protocol>` | Training/evaluation code for a specific registered experiment |
| report artifacts | `report/<topic>` | Tables, figures, and supervisor/manuscript reports |
| hotfix | `fix/<topic>` | Small bug fixes against current active work |

Current cleanup branch:

```text
q1/project-cleanup-and-registry
```

Recommended next branches:

```text
q1/radarode-long-ecg-evaluator
exp/mmecg-remalign-mtl-radarode-long
exp/crrvs-remalign-mtl-5fold-long
report/q1-long-ecg-comparison
```

## Rules

- `main` must remain the current clean baseline: runnable, stable, and not
  polluted by failed experiments.
- Do not commit directly to `main` unless the change is tiny and already
  reviewed.
- One branch should have one clear objective.
- Every experiment branch must add or update
  `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- New formal experiments should also add a spec in `experiments/specs/` and a
  run manifest in `experiments/manifests/` after execution.
- Experiment outputs, checkpoints, logs, and generated fold directories stay
  out of Git.
- Commit before train: any non-smoke experiment must start from a committed
  code state, and the registry must record the commit and branch.
- Push completed stages to GitHub so the remote repository is the durable
  backup.
- Commit source code, configs, runbooks, registry rows, and reproducible report
  scripts.
- PDF literature files are local-only under `reference_papers/`; cite papers
  through BibTeX/DOI or manuscript references instead of committing large PDFs.

## Commit Style

Use concise conventional prefixes:

```text
chore: reorganize q1 experiment workspace
feat: add radarode-style long ecg evaluator
fix: correct crrvs subject split leakage
docs: update q1 experiment registry policy
report: add supervisor shape200 comparison
```

## Pull Request Checklist

Before merging into `main`:

```text
[ ] Branch name follows the policy.
[ ] New or changed experiments are registered with commit ID and branch.
[ ] No checkpoints, logs, fold outputs, or local dataset files are committed.
[ ] Metric/preprocessing/hyperparameter/protocol changes update docs in the same PR.
[ ] Relevant scripts pass syntax/import checks.
[ ] Agent skills are synchronized if `agent_skills/` changed.
[ ] README/CLAUDE handoff remains accurate.
[ ] Results are not marked publishable unless category=final.
```

## Suggested Review Flow

1. Push the branch to GitHub.
2. Open a pull request into `main`.
3. Use the PR description to state:
   - objective,
   - changed source files,
   - experiment registry changes,
   - validation performed,
   - remaining risks.
4. Merge only after the branch has a coherent project state.
