# radarODE-MTL On MMECG Runbook

This document records the steps for running the radarODE-MTL/EGA baseline used by the ICONIP2026 BeatAware-Radar2ECGNet paper.

The implementation in this repository is an adapter around the external project:

```text
/home/qhh2237/Projects/radarODE-MTL
```

The retained local scripts are:

```text
scripts/precompute_wsst4s_mmecg.py
scripts/train_radarode_mtl_ega_mmecg.py
scripts/run_radarode_ega_strict_loso_all.sh
scripts/evaluate_mmecg_shape200.py
```

## Purpose

radarODE-MTL is used as a reproduced external baseline for Table 1 of the ICONIP2026 paper. It is not the BeatAware model.

The paper protocol is strict 11-fold LOSO on MMECG:

- No target-subject calibration.
- One held-out subject per fold.
- 4-second center crop for radarODE-style WSST input.
- Radar input converted to WSST tensors with shape `(N, 50, 71, 120)`.
- radarODE outputs ECG shape predictions in the 200-point single-cycle space.

## Required Inputs

MMECG split files must already exist:

```text
/home/qhh2237/Datasets/MMECG/processed/loso/fold_01/train.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_01/val.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_01/test.h5
...
/home/qhh2237/Datasets/MMECG/processed/loso/fold_11/
```

Each split H5 must contain `rcg`. The precompute script crops the center 4 seconds from each 8-second segment:

```text
samples 400:1200 at 200 Hz
```

## Environment

Use the same environment used by the project scripts:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet
export PYTHONNOUSERSITE=1
```

Python dependencies needed by this path include:

```text
torch
h5py
numpy
scipy
tqdm
ssqueezepy
LibMTL from /home/qhh2237/Projects/radarODE-MTL
radarODE_plus modules from /home/qhh2237/Projects/radarODE-MTL/Projects/radarODE_plus
```

If WSST precompute fails with a missing dependency, install `ssqueezepy` in `cyberbrain`.

## Step 1: Precompute WSST4s

For a single fold:

```bash
python scripts/precompute_wsst4s_mmecg.py \
  --mode loso \
  --fold_idx 1 \
  --batch_size 4
```

For all folds:

```bash
python scripts/precompute_wsst4s_mmecg.py \
  --mode loso \
  --batch_size 4
```

Generated files are written next to each original split:

```text
train.h5 -> train_wsst4s.h5
val.h5   -> val_wsst4s.h5
test.h5  -> test_wsst4s.h5
```

The script skips existing WSST files unless `--overwrite` is passed.

## Step 2: Train One Strict LOSO Fold

Example for fold 01:

```bash
python scripts/train_radarode_mtl_ega_mmecg.py \
  --exp_tag mmecg_radarode_mtl_ega_strict_loso \
  --protocol strict_loso \
  --fold_idx 1 \
  --epochs 200 \
  --batch_size 22 \
  --num_workers 4
```

Default optimizer settings match the local reproduction script:

```text
optimizer: SGD
lr: 5e-3
momentum: 0.937
weight_decay: 5e-4
scheduler: cosine
EGA_temp: 1.0
device: cuda:0
```

Outputs for fold 01:

```text
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_01/config.json
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_01/best_radarode_mtl_ega_fold01.pt
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_01/best_radarode_mtl_ega_fold01_best_result.npy
```

The model tasks are:

```text
ECG_shape: (B, 1, 200)
PPI:       (B, 1, 260)
Anchor:    (B, 1, 800)
```

## Step 3: Train All Strict LOSO Folds

Use the wrapper:

```bash
bash scripts/run_radarode_ega_strict_loso_all.sh
```

The wrapper does this for folds 1 through 11:

1. Precompute `*_wsst4s.h5` for the fold.
2. Train `train_radarode_mtl_ega_mmecg.py` with `--protocol strict_loso`.
3. Write a fold log under `experiments_mmecg/`.
4. Skip a fold if its best result file exists and the fold log indicates completion.

Log paths:

```text
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso_all.log
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso_fold01.log
...
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso_fold11.log
```

These log files can be deleted after results are confirmed; they are not checkpoints.

## Step 4: Evaluate Against BeatAware Outputs

For the paper-style shape-200 comparison, use:

```bash
python scripts/evaluate_mmecg_shape200.py \
  --model_type radarode \
  --exp_tag mmecg_radarode_mtl_ega_strict_loso \
  --fold_idx -1
```

`evaluate_mmecg_shape200.py` also imports BeatAware, AirECG, and RSSRNet helpers so the four-model comparison can share one evaluation path.

## Resume And Recovery

If one fold fails:

1. Confirm WSST cache exists:

```text
/home/qhh2237/Datasets/MMECG/processed/loso/fold_XX/train_wsst4s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_XX/val_wsst4s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_XX/test_wsst4s.h5
```

2. Rerun only that fold:

```bash
python scripts/precompute_wsst4s_mmecg.py --mode loso --fold_idx XX --batch_size 4

python scripts/train_radarode_mtl_ega_mmecg.py \
  --exp_tag mmecg_radarode_mtl_ega_strict_loso \
  --protocol strict_loso \
  --fold_idx XX \
  --epochs 200 \
  --batch_size 22 \
  --num_workers 4
```

3. Then rerun the all-fold wrapper if needed. Completed folds should be skipped.

## Common Failure Modes

Missing external radarODE-MTL project:

```text
ModuleNotFoundError: LibMTL
ModuleNotFoundError: Projects.radarODE_plus...
```

Check that `/home/qhh2237/Projects/radarODE-MTL` exists and contains the expected `LibMTL` and `Projects/radarODE_plus` modules.

Missing WSST cache:

```text
FileNotFoundError: Missing WSST cache
```

Run `scripts/precompute_wsst4s_mmecg.py` for the fold.

Wrong WSST shape:

```text
wrong WSST shape
```

Delete the affected `*_wsst4s.h5` file and regenerate with `--overwrite`.

CUDA memory pressure:

Reduce `--batch_size`, for example:

```bash
--batch_size 12
```

Long-running wrapper interrupted:

Rerun:

```bash
bash scripts/run_radarode_ega_strict_loso_all.sh
```

The wrapper is designed to skip completed folds.

## Cleanup Policy

Safe to delete:

```text
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso*.log
experiments_mmecg/*.pid
```

Do not delete unless intentionally rerunning:

```text
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_*/config.json
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_*/*.pt
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_*/*best_result.npy
/home/qhh2237/Datasets/MMECG/processed/loso/fold_*/*_wsst4s.h5
```

## Current Paper Alignment

For ICONIP2026, radarODE-MTL should be described as a reproduced strict-LOSO external baseline with ODE-constrained multi-task training. It should not be described as the main BeatAware method and should not use the old few-shot/calibration protocol for the reported paper result.
