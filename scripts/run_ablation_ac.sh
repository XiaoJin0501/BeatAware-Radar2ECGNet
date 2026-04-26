#!/usr/bin/env bash
# run_ablation_ac.sh — 消融实验：Model A + Model C（Model D 已完成）
#
# 消融框架：
#   Model A (Baseline)  : KI only (use_pam=false, use_emd=false)
#   Model C (+KI+PA)    : KI + EMD 物理对齐层 (use_pam=false, use_emd=true)
#   Model D (Full)      : KI + PA + CP，已完成
#
# 注：Model B 与 Model A 参数相同（KI 对 phase 输入始终激活），跳过。

set -e

COMMON="--epochs 150 --batch_size 32 --lr 1e-4 --seed 42 --input_type phase $@"
TRAIN="python scripts/train.py"
TEST="python scripts/test.py"

echo "============================================================"
echo " BeatAware-Radar2ECGNet — Ablation Study (Model A + C)"
echo " $(date)"
echo "============================================================"

# ── Model A: Baseline（KI only，无 PAM/EMD）──────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] ===== Model A: Baseline (no PAM, no EMD) ====="
$TRAIN $COMMON --exp_tag ModelA_baseline --use_pam false --use_emd false
echo "[$(date +%H:%M:%S)] Model A training done. Running test..."
$TEST --exp_tag ModelA_baseline --use_pam false --use_emd false
echo "[$(date +%H:%M:%S)] Model A complete."

# ── Model C: +KI+PA（加 EMD 物理对齐层，无多头 PAM）────────────────────
echo ""
echo "[$(date +%H:%M:%S)] ===== Model C: +PA (no PAM, with EMD) ====="
$TRAIN $COMMON --exp_tag ModelC_ki_pa --use_pam false --use_emd true
echo "[$(date +%H:%M:%S)] Model C training done. Running test..."
$TEST --exp_tag ModelC_ki_pa --use_pam false --use_emd true
echo "[$(date +%H:%M:%S)] Model C complete."

echo ""
echo "============================================================"
echo " Model A + C done. Generating comparison with Model D..."
echo " $(date)"
echo "============================================================"

# ── 汇总对比 ──────────────────────────────────────────────────────────────
python scripts/summarize_ablation.py

echo ""
echo "============================================================"
echo " All done. Results: experiments/"
echo "============================================================"
