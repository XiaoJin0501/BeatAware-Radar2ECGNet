#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cyberbrain
cd /home/qhh2237/Projects/BeatAware-Radar2ECGNet
export PYTHONNOUSERSITE=1

MASTER_LOG="experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso_all.log"
mkdir -p experiments_mmecg

for fold in 1 2 3 4 5 6 7 8 9 10 11; do
  fold2=$(printf "%02d" "${fold}")
  exp_dir="experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso/fold_${fold2}"
  fold_log="experiments_mmecg/mmecg_radarode_mtl_ega_strict_loso_fold${fold2}.log"
  best_result="${exp_dir}/best_radarode_mtl_ega_fold${fold2}_best_result.npy"

  echo "=== radarODE-MTL EGA strict LOSO fold_${fold2} ===" | tee -a "${MASTER_LOG}"

  if [[ -f "${best_result}" ]] && grep -q "Best Result" "${fold_log}" 2>/dev/null; then
    echo "[skip] fold_${fold2} already completed: ${best_result}" | tee -a "${MASTER_LOG}"
    continue
  fi

  mkdir -p "${exp_dir}/logs"
  ln -sf "/home/qhh2237/Projects/BeatAware-Radar2ECGNet/${fold_log}" "${exp_dir}/logs/train.log"

  echo "=== precompute WSST4s fold_${fold2} ===" | tee "${fold_log}" | tee -a "${MASTER_LOG}"
  python scripts/precompute_wsst4s_mmecg.py --mode loso --fold_idx "${fold}" --batch_size 4 2>&1 | tee -a "${fold_log}" | tee -a "${MASTER_LOG}"

  echo "=== train radarODE-MTL EGA strict LOSO fold_${fold2} ===" | tee -a "${fold_log}" | tee -a "${MASTER_LOG}"
  python scripts/train_radarode_mtl_ega_mmecg.py \
    --exp_tag mmecg_radarode_mtl_ega_strict_loso \
    --protocol strict_loso \
    --fold_idx "${fold}" \
    --epochs 200 \
    --batch_size 22 \
    --num_workers 4 \
    2>&1 | tee -a "${fold_log}" | tee -a "${MASTER_LOG}"
done

echo "=== radarODE-MTL EGA strict LOSO all folds done ===" | tee -a "${MASTER_LOG}"
