#!/usr/bin/env bash
# run_model_d.sh — 完整模型（Model D）全部 5 个 fold 训练
#
# 用法：
#   bash scripts/run_model_d.sh
#   bash scripts/run_model_d.sh --epochs 10   # 快速调试
#
# Model D = KI + PA + CP（use_pam=true, use_emd=true）
# fold_idx 默认 -1 = 自动依次运行 fold 0 → fold 4

set -e

echo "========================================"
echo " Model D (Full) — 5-Fold 训练"
echo " $(date)"
echo "========================================"

python scripts/train.py \
    --exp_tag     ModelD_full \
    --input_type  phase \
    --use_pam     true \
    --use_emd     true \
    --fold_idx    -1 \
    --epochs      150 \
    --batch_size  32 \
    --lr          1e-4 \
    --seed        42 \
    "$@"

echo "========================================"
echo " 训练完成！Results: experiments/ModelD_full/"
echo " $(date)"
echo "========================================"
