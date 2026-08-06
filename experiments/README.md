# Experiment Governance

This directory stores machine-readable experiment specifications and run
manifests. Large experiment outputs remain under ignored local result
directories and should not be committed.

```text
experiments/specs/       planned experiment specifications
experiments/manifests/   completed run manifests
```

There is exactly one experiment registry:

```text
experiments_mmecg/EXPERIMENT_REGISTRY.csv
```

New formal Q1 experiments should use `experiments/specs/` and
`experiments/manifests/` in addition to the single registry. Do not create a
second registry.
