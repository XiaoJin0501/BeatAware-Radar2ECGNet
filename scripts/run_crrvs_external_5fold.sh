#!/usr/bin/env bash
set -euo pipefail

# Full CR_RVS external validation under the existing subject-level 5-fold split.
#
# This runs:
#   1) CR_RVS raw I/Q -> CW-STFT H5 preprocessing for folds 1-5
#   2) CW-STFT + TCN baseline
#   3) CW-STFT + REM-AlignNet-MTL shape-first model
#   4) Shape-200 evaluation for both models
#   5) Multi-fold external report
#
# Usage:
#   bash scripts/run_crrvs_external_5fold.sh
#   FOLDS="2 3 4 5" EPOCHS=50 bash scripts/run_crrvs_external_5fold.sh

FOLDS="${FOLDS:-1 2 3 4 5}"
EPOCHS="${EPOCHS:-50}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python || command -v python3)"
fi

CR_H5_DIR="${CR_H5_DIR:-/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold}"
MAX_WINDOWS_PER_RECORD="${MAX_WINDOWS_PER_RECORD:-80}"

EXP_REM="${EXP_REM:-q1_crrvs_remalign_mtl_5fold}"
EXP_TCN="${EXP_TCN:-q1_crrvs_wsst_tcn_5fold}"

echo "[CR_RVS] Building CW-STFT H5 folds: ${FOLDS}"
"${PYTHON}" scripts/build_crrvs_wsst_shape_h5.py \
  --folds ${FOLDS} \
  --max_windows_per_record "${MAX_WINDOWS_PER_RECORD}" \
  --output_dir "${CR_H5_DIR}"

for fold in ${FOLDS}; do
  printf -v fold2 "%02d" "${fold}"

  echo
  echo "[CR_RVS] Fold ${fold}: CW-STFT + TCN"
  if [[ ! -f "experiments_mmecg/${EXP_TCN}/fold_${fold2}/results_shape200/global_summary_shape200.json" ]]; then
    "${PYTHON}" scripts/train_mmecg.py \
      --exp_tag "${EXP_TCN}" \
      --protocol loso \
      --fold_idx "${fold}" \
      --loso_h5_dir "${CR_H5_DIR}" \
      --n_range_bins 1 \
      --model_arch wsst_tcn \
      --epochs "${EPOCHS}" \
      --early_stop_patience 100 \
      --narrow_bandpass false \
      --balance_by subject \
      --beta_peak 0.0 \
      --use_shape200_loss true \
      --lambda_shape200 0.8 \
      --shape200_pcc_weight 0.5

    "${PYTHON}" scripts/evaluate_mmecg_shape200.py \
      --model_type continuous \
      --exp_tag "${EXP_TCN}" \
      --fold_idx "${fold}" \
      --device "${DEVICE}"
  else
    echo "[CR_RVS] Fold ${fold}: TCN result exists, skipping."
  fi

  echo
  echo "[CR_RVS] Fold ${fold}: REM-AlignNet-MTL"
  if [[ ! -f "experiments_mmecg/${EXP_REM}/fold_${fold2}/results_shape_head/global_summary_shape_head.json" ]]; then
    "${PYTHON}" scripts/train_mmecg.py \
      --exp_tag "${EXP_REM}" \
      --protocol loso \
      --fold_idx "${fold}" \
      --loso_h5_dir "${CR_H5_DIR}" \
      --n_range_bins 1 \
      --model_arch rem_align_mtl \
      --epochs "${EPOCHS}" \
      --early_stop_patience 100 \
      --narrow_bandpass false \
      --balance_by subject \
      --beta_peak 0.3 \
      --use_shape200_loss true \
      --lambda_shape200 0.8 \
      --shape200_pcc_weight 0.5

    "${PYTHON}" scripts/evaluate_remalign_shape_head.py \
      --exp_tag "${EXP_REM}" \
      --fold_idx "${fold}" \
      --checkpoint best_shape200.pt \
      --device "${DEVICE}"
  else
    echo "[CR_RVS] Fold ${fold}: REM result exists, skipping."
  fi
done

"${PYTHON}" scripts/make_crrvs_external_report.py \
  --rem_tag "${EXP_REM}" \
  --tcn_tag "${EXP_TCN}" \
  --folds ${FOLDS} \
  --output_dir "experiments_mmecg/q1_crrvs_external_report_5fold"

echo
echo "[CR_RVS] Done. Report: experiments_mmecg/q1_crrvs_external_report_5fold"
