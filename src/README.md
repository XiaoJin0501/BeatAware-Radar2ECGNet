# Source Module Registry

This directory contains the code that is still relevant after the Q1 cleanup.

## Current Q1 Main Path

| Path | Status | Role |
|---|---|---|
| `src/models/remalign/` | active | REM-AlignNet / REM-AlignNet-MTL candidate architecture |
| `src/models/model_factory.py` | active | Central model construction entry point for training/testing |
| `src/data/mmecg_dataset.py` | active | MMECG and compatible H5 dataset loading, LOSO/samplewise split support |
| `src/data/wu_adapter.py` | active | Adapter utilities for external radar-to-ECG datasets |
| `src/losses/losses.py` | active | Waveform, spectral, event, and shape-related losses |
| `src/utils/metrics.py` | active | Evaluation metrics used by reports and test scripts |

## Retained Baselines

| Path | Status | Role |
|---|---|---|
| `src/models/BeatAwareNet/` | frozen baseline | ICONIP BeatAware model and backward-compatible architecture |
| `src/models/baselines/` | frozen baseline | Generic CNN/TCN/RNN/Conformer/RSSRNet-style baselines |
| `src/models/modules/fmcw_encoder.py` | shared | KI/FMCW range encoder used by BeatAware and raw/FMCW baselines |
| `src/models/modules/peak_module.py` | shared | Event/peak auxiliary module reference |
| `src/models/modules/tfilm.py` | shared | Temporal FiLM utilities |

## Legacy Compatibility

| Path | Status | Why It Remains |
|---|---|---|
| `src/models/modules/diffusion_decoder.py` | legacy-compatible | Still referenced by `BeatAwareNet` when `use_diffusion=True`; not part of the current Q1 main claim |

## Rules

- New architectures should enter through `src/models/model_factory.py`.
- Do not add a new top-level model package unless it supports a registered
  experiment in `experiments_mmecg/EXPERIMENT_REGISTRY.csv`.
- Keep published baselines frozen unless a protocol bug is being fixed.
- Treat diffusion/samplewise code as compatibility support, not the Q1 main
  direction.
