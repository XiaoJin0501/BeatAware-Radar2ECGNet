#!/usr/bin/env bash
# run_ablation.sh — 消融实验批量运行脚本
#
# 用法：
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --fold_idx 0   # 只跑单个 fold（调试用）
#
# 实验列表：
#   Exp A  — 基线：无 PAM/TFiLM，输入 radar_phase
#   Exp B1 — 完整模型，输入 radar_raw
#   Exp B2 — 完整模型，输入 radar_phase（主实验）
#   Exp B3 — 完整模型，输入 radar_spec
#   D1     — PAM 二值标签 vs 高斯软标签（需替换数据，暂用同配置占位）
#   D2     — 去除 ConformerFusionBlock（待模型支持 --no_conformer 参数）
#   D3     — PAM 单核 k=15 vs 多尺度（待模型支持）
#   D4     — 仅 resting 场景训练

set -e

# ── 公共参数 ──────────────────────────────────────────────────────────────
COMMON="--epochs 150 --batch_size 32 --lr 1e-4 --seed 42 $@"
TRAIN="python scripts/train.py"
TEST="python scripts/test.py"

echo "============================================================"
echo " BeatAware-Radar2ECGNet Ablation Study"
echo "============================================================"

# ── Exp A: 基线（无 PAM/TFiLM）──────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp A (baseline, no PAM/TFiLM)..."
$TRAIN $COMMON --exp_tag ExpA_baseline  --input_type phase --use_pam false
$TEST  $COMMON --exp_tag ExpA_baseline  --input_type phase --use_pam false

# ── Exp B1: 完整模型 × radar_raw ─────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B1 (full model, radar_raw)..."
$TRAIN $COMMON --exp_tag ExpB1_raw    --input_type raw
$TEST  $COMMON --exp_tag ExpB1_raw    --input_type raw

# ── Exp B2: 完整模型 × radar_phase（主实验）──────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B2 (full model, radar_phase)..."
$TRAIN $COMMON --exp_tag ExpB2_phase  --input_type phase
$TEST  $COMMON --exp_tag ExpB2_phase  --input_type phase

# ── Exp B3: 完整模型 × radar_spec ────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B3 (full model, radar_spec)..."
$TRAIN $COMMON --exp_tag ExpB3_spec   --input_type spec
$TEST  $COMMON --exp_tag ExpB3_spec   --input_type spec

# ── D4: 仅 resting 场景 ──────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting D4 (resting only, generalization test)..."
$TRAIN $COMMON --exp_tag D4_resting_only --input_type phase \
               --scenarios resting
$TEST  $COMMON --exp_tag D4_resting_only --input_type phase \
               --scenarios resting

echo "============================================================"
echo " All experiments done."
echo " Results: experiments/"
echo "============================================================"
