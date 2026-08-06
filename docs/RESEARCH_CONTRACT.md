# Research Contract

This is the scientific rule layer for the project. It applies to humans,
Claude Code, Codex, and any future agent.

## Data Integrity

- No data leakage is allowed.
- Test subjects must not influence normalization, range-bin selection,
  hyperparameter selection, checkpoint selection, or model selection.
- Subject-independent protocols must be audited for subject overlap.
- Oracle experiments must be explicitly labeled as upper bounds.
- Ground-truth ECG events may be used for training supervision and oracle
  analysis only, never for final inference.

## Experiment Integrity

- Probe or smoke results must not be used as manuscript evidence.
- Do not tune metrics to improve numbers.
- Do not select checkpoints from best test performance.
- Do not change evaluation code after seeing unfavorable results without
  documenting the change and re-running affected baselines.
- Every quantitative conclusion must trace to an experiment ID and manifest.

## Baseline Integrity

- Adapted baselines must be described as adapted implementations, not official
  original implementations.
- Strong pipeline baselines such as radarODE-MTL must be compared under matched
  protocols before claiming superiority.
- Do not compare Shape-200 outputs against long continuous ECG outputs as if
  they were the same task.

## Claim Integrity

- Do not claim causality without a controlled ablation.
- Do not claim external generalization without subject-independent external
  validation.
- Do not claim clinical reliability without locked fiducial/interval metrics
  and failure handling.
- Manuscript claims must appear in `docs/PAPER_CLAIM_LEDGER.md`.
