#!/usr/bin/env bash
# run_ablation.sh — V2 消融实验批量运行脚本
#
# 用法：
#   bash scripts/run_ablation.sh               # 全部实验，全部 folds
#   bash scripts/run_ablation.sh --fold_idx 0  # 只跑 fold 0（调试用）
#
# V2 消融实验框架（三维 KI / PA / CP）：
#   Model A — Baseline：use_pam=false, use_emd=false（无 KI/PA/CP）
#   Model B — +KI：导数感知编码器（3通道 diff 输入），use_pam=false, use_emd=false
#   Model C — +KI+PA：在 Model B 基础上加 EMD 物理对齐层，use_pam=false, use_emd=true
#   Model D — Full（+KI+PA+CP）：完整模型，use_pam=true, use_emd=true
#
# 其他消融：
#   D4 — 去除 ConformerFusionBlock（待模型支持 --use_conformer 参数）
#   D5 — 仅 resting 训练，全场景测试（跨场景泛化）

set -e

# ── 公共参数 ──────────────────────────────────────────────────────────────
COMMON="--epochs 150 --batch_size 32 --lr 1e-4 --seed 42 --input_type phase $@"
TRAIN="python scripts/train.py"
TEST="python scripts/test.py"

echo "============================================================"
echo " BeatAware-Radar2ECGNet V2 Ablation Study"
echo "============================================================"

# ── Model A: Baseline（无 PAM/TFiLM/EMD）────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Model A (baseline, no PAM/EMD)..."
$TRAIN $COMMON --exp_tag ModelA_baseline --use_pam false --use_emd false
$TEST         --exp_tag ModelA_baseline --use_pam false --use_emd false

# ── Model B: +KI（导数感知 Encoder，仍无 PAM/EMD）───────────────────────
# 注：KI (torch.diff 三通道输入) 在 radar_phase 输入时始终激活
# Model A vs B 的差异体现在有无完整 PAM 多头监督；
# 若需严格隔离 KI 效果，可为 Model A 额外加 --use_ki false 参数（待实现）
echo "[$(date +%H:%M:%S)] Starting Model B (+KI, no PAM/EMD)..."
$TRAIN $COMMON --exp_tag ModelB_ki      --use_pam false --use_emd false
$TEST         --exp_tag ModelB_ki      --use_pam false --use_emd false

# ── Model C: +KI+PA（加 EMD 物理对齐层，仍无多头 PAM）───────────────────
echo "[$(date +%H:%M:%S)] Starting Model C (+KI+PA, no full PAM)..."
$TRAIN $COMMON --exp_tag ModelC_ki_pa   --use_pam false --use_emd true
$TEST         --exp_tag ModelC_ki_pa   --use_pam false --use_emd true

# ── Model D: Full（+KI+PA+CP，完整 V2 架构）──────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Model D (full model)..."
$TRAIN $COMMON --exp_tag ModelD_full    --use_pam true  --use_emd true
$TEST         --exp_tag ModelD_full    --use_pam true  --use_emd true

# ── D5: 仅 resting 训练 → 全场景测试（跨场景泛化）────────────────────────
echo "[$(date +%H:%M:%S)] Starting D5 (resting-only train, all-scenario test)..."
$TRAIN $COMMON --exp_tag D5_generalization --use_pam true --use_emd true \
               --scenarios resting
$TEST         --exp_tag D5_generalization --use_pam true --use_emd true
# test 不传 --scenarios，默认使用全部三种场景，观察跨场景泛化

echo "============================================================"
echo " All experiments done."
echo " Results: experiments/"
echo "============================================================"
