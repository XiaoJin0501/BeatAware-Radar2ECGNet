---
name: experiment-preflight
description: Check that an experiment is registered, traceable, and protocol-safe before training starts.
---

# Experiment Preflight

Use before any training or formal evaluation run.

Check:

- experiment spec exists;
- registry has no duplicate ID;
- branch follows `docs/GITHUB_WORKFLOW.md`;
- git status is understood;
- dataset split is subject-independent when required;
- normalization does not use test-subject statistics;
- checkpoint rule uses validation only;
- output path is unique;
- smoke test passed for new code paths.

If any item fails, do not launch the formal run.
