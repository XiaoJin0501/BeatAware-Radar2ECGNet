# Deleted Experiments Log

This log records experiments and scripts intentionally removed during project
cleanup. They should not be restored unless a specific reproducibility question
requires it.

## Deleted Experiment Result Directories

Deleted because they were smoke/probe/invalid, superseded, or historical
few-shot artifacts that do not serve the current Q1 protocol.

```text
experiments_mmecg/_archive_probe_runs
experiments_mmecg/q1_journal_summary
experiments_mmecg/iconip2026_figures
experiments_mmecg/mmecg_beataware_ablation_no_calib_strict_loso
experiments_mmecg/mmecg_beataware_ablation_no_cp_fewshot40v10
experiments_mmecg/mmecg_beataware_ablation_no_pa_fewshot40v10
experiments_mmecg/mmecg_bilstm_paper_fewshot40v10_baseline
experiments_mmecg/mmecg_conformer_fewshot40v10_baseline
experiments_mmecg/mmecg_gru_fewshot40v10_baseline
experiments_mmecg/mmecg_guided_tcn_fewshot40v10
experiments_mmecg/mmecg_ki3_tcn_fewshot40v10
experiments_mmecg/mmecg_ki8_tcn_fewshot40v10
experiments_mmecg/mmecg_ki16_tcn_fewshot40v10
experiments_mmecg/mmecg_ki16_tcn_cp_fewshot40v10
experiments_mmecg/mmecg_kitcn_cp_pa_fewshot40v10
experiments_mmecg/mmecg_tcn_fewshot40v10_baseline
experiments_mmecg/mmecg_transformer_fewshot40v10_baseline
experiments_mmecg/mmecg_unet1d_fewshot40v10_baseline
experiments_mmecg/mmecg_wavegru_fewshot40v10_baseline
```

### 2026-09-10 method-development cleanup

The following generated outputs were removed before starting the next REM-Net
method-development stage. They were smoke checks or explicitly quarantined,
truncated runs and were never eligible for scientific comparison. Their specs,
registry records, audit notes, and source code remain tracked.

```text
experiments_mmecg/q1_cgd_smoke
experiments_mmecg/q1_continuous_ecg_b0_b2_pilot/ABORTED_INVALID_B0_fold01_11epochs_budget30
experiments_mmecg/q1_continuous_ecg_b0_b2_pilot/ABORTED_INVALID_B0_fold01_4epochs
experiments_mmecg/q1_continuous_ecg_b0_b2_pilot_smoke
experiments_mmecg/q1_morphology_consistency_overfit_smoke_20260908
experiments_mmecg/q1_morphology_consistency_smoke_20260908
experiments_mmecg/q1_timing_identifiability_gate_smoke
```

The six old-protocol full-support WSST caches below were also removed. They
were generated from the superseded `processed/loso` manifest whose repeated
validation assignment makes cross-fold model selection protocol-invalid. The
source H5 files, generator, experiment records, and corrected `loso_v2` caches
are retained, so these derived files can be regenerated if an audit requires
them.

```text
/home/qhh2237/Datasets/MMECG/processed/loso/fold_01/train_wsst8s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_01/val_wsst8s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_05/train_wsst8s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_05/val_wsst8s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_09/train_wsst8s.h5
/home/qhh2237/Datasets/MMECG/processed/loso/fold_09/val_wsst8s.h5
```

Recorded size before deletion: 10,691,865,349 bytes for the six WSST caches
and approximately 40 MiB for the seven result directories.

## Deleted Script Directories

```text
scripts/archive_probes
scripts/run_iconip_ablation_mmecg.sh
```

The deleted scripts were historical probe/ablation runners. Formal entry
points are now listed in `scripts/README.md`.

## Literature Directory Cleanup

The old tracked `Papers/` directory was removed as an active project entry.
Reference PDFs are now consolidated locally under:

```text
reference_papers/
```

PDF files in this directory are ignored by Git. This keeps literature separate
from executable code, scripts, and experiment outputs while avoiding large PDF
blobs in the GitHub repository.

## Retained Core Results

```text
experiments_mmecg/q1_remalign_mtl_shapefirst_loso
experiments_mmecg/q1_remalign_wsst_tcn_loso
experiments_mmecg/q1_remalign_event_lowpeak_loso
experiments_mmecg/q1_crrvs_remalign_mtl_5fold
experiments_mmecg/q1_crrvs_wsst_tcn_5fold
experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso
experiments_mmecg/mmecg_beataware_fewshot40v10
experiments_mmecg/mmecg_airecg_strict_loso_v4
experiments_mmecg/mmecg_rssrnet_strict_loso_v2
```
