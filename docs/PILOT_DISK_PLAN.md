# Disk Plan — q1_continuous_ecg_b0_b2_pilot

Measured 2026-09-09. Filesystem `/home` (1.8 T), **26 GB free, 99 % used**.

## Current footprint, MMECG `processed/loso`

| set | size | files | status |
|---|---|---|---|
| `_wsst4s.h5` (legacy 4 s) | **24.7 GB** | 33 | superseded; no *config* selects it (all 54 use `_wsst.h5`), but several scripts still name the suffix as an option |
| `_wsst.h5` (legacy 4 s) | 17.6 GB | 33 | **in use** by the frozen incumbent (`wsst_suffix="_wsst.h5"`) |
| `_wsst8s.h5` (new, full support) | 10.0 GB | 6 | required by this pilot |
| base `.h5` (`rcg` + `ecg`) | 9.4 GB | 33 | required |
| total | 61.6 GB | | |

`_wsst.h5` and `_wsst4s.h5` are **not** byte-identical (different content, same shape),
so they are two legacy generations, not a mirror pair.

## The pilot fits with room to spare — no action needed now

15 runs (5 arms x folds 1/5/9). The only new data is checkpoints and logs:

| item | size |
|---|---|
| `_wsst8s.h5` for folds 1/5/9 train+val | 10.0 GB — **already written** |
| checkpoints, 15 x ~5.7 MiB | ~86 MiB |
| history/config JSON | < 5 MiB |
| **new space required** | **< 100 MiB against 26 GB free** |

## The constraint is the *next* stage, not this one

An 11-fold final-test run needs `_wsst8s.h5` for all 11 folds including test splits:
roughly `11/3 x 10 GB ≈ 37 GB`. **That does not fit in 26 GB.** A decision will be
required before final test — but not before this pilot.

Options, in order of preference, to be decided later and **not executed now**:

1. Delete `_wsst4s.h5` (**24.7 GB**). No experiment config selects it — all 54
   `config.json` files carry `wsst_suffix="_wsst.h5"` — so no recorded run depends on
   it. Caveat: `train_mmecg.py`, `mmecg_dataset.py`, `train_radarode_mtl_ega_mmecg.py`
   and `evaluate_mmecg_long_assembly.py` still name `_wsst4s` as a selectable suffix,
   so deleting the data would make that option fail at runtime. Reproducing any older
   run that used it would require regenerating. This alone makes the 11-fold set fit.
2. Generate `_wsst8s.h5` per fold, run, then delete that fold's copy before the next.
   Peak usage stays near current levels at the cost of recomputation.
3. Move legacy WSST to slower storage if any is available.

**No deletion is proposed as part of this pilot**, consistent with the standing
instruction not to modify or remove existing experiments and artifacts.

## Guard

The pilot writes only under
`experiments_mmecg/q1_continuous_ecg_b0_b2_pilot{,_smoke}/`. The WSST generator
refuses the test split without `--allow_test`, so no further large file can be created
by accident during this stage.
