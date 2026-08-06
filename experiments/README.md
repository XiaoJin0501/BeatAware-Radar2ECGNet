# Experiment Governance

This directory stores machine-readable experiment specifications and run
manifests. Large experiment outputs remain under ignored local result
directories and should not be committed.

```text
experiments/specs/       planned experiment specifications
experiments/manifests/   completed run manifests
experiments/registry.csv lightweight registry for future formal experiments
```

The historical retained result registry is still:

```text
experiments_mmecg/EXPERIMENT_REGISTRY.csv
```

New formal Q1 experiments should use `experiments/specs/` and
`experiments/manifests/` in addition to the registry.
