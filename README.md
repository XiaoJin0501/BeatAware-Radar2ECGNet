# REM-Net

Radar-to-ECG research project. The current paper model is **REM-Net, `phase_pd`**.
Formal results use MMECG 11-fold LOSO and CR_RVS 5-fold subject-independent evaluation.
The `seed42` label means base seed42; actual initialization is42+fold.

## Start here

- [Paper facts and source citations](docs/paper/PAPER_FACTS.md)
- [Latest numeric results manifest](docs/paper/RESULTS_MANIFEST.md)
- [Paper materials](docs/paper/)
- [Current scripts](scripts/README.md)
- [2026-11-04 cleanup / recovery map](docs/CLEANUP_20261104.md)

## Directory map

| Path | Purpose |
|---|---|
| `experiments/phase_resampling_20261027/` | Current architecture, formal ablation, selected weights and protocol |
| `experiments/development/structured_decoder_v0_20261017_ab/` | Still-required frozen inference, raw-reference and preprocessing dependency |
| `experiments_mmecg/` | Retained original dual-head/baseline weights used by current results |
| `scripts/figures/` | Current figure/table scripts and their shared helper chain |
| `docs/paper/source_data/` | Required baseline predictions, references and frozen code snapshot |
| `docs/paper/sections/results_verified_20261101/` | Verified main-comparison table sources |
| `docs/paper/hrv_agreement_20261103/` | Final60s HR/HRV paired exports and provenance |
| `src/`, `tests/`, `configs/` | Shared implementation, active tests and configuration |
| `reference_papers/`, `ICONIP2026/` | Reference literature and separate conference-paper material |

Historical experiments/scripts/specifications were moved **outside the working tree**,
not mixed into the current model. Recovery details are in the cleanup report.
Paper source data and selected weights are not disposable caches.

## Runtime dependencies

Existing evaluators pin `/tmp/rem_dual_head_formal`, the local datasets under
`/home/qhh2237/Datasets/`, and the official radarODE source where applicable.
Those external dependencies were not removed. The legacy WSST implementation
still needed by inference now resides beside that inference helper, byte-for-byte unchanged.

See `AGENTS.md` before development. Cleanup authorizes no new training or architecture route.
