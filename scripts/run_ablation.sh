#!/usr/bin/env bash
# run_ablation.sh — 消融实验批量运行脚本
#
# 用法：
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --fold_idx 0   # 只跑单个 fold（调试用）
#
# 实验列表（与 CLAUDE.md / docs/ARCHITECTURE.md 编号一致）：
#   Exp A  — 基线：无 PAM/TFiLM，输入 radar_phase
#   Exp B1 — 完整模型，输入 radar_raw
#   Exp B2 — 完整模型，输入 radar_phase（主实验）
#   Exp B3 — 完整模型，输入 radar_spec
#   D1     — PAM 二值标签 vs 高斯软标签（待数据支持）
#   D2     — PAM→TFiLM 串联 vs 平行（待模型支持 --tfilm_parallel 参数）
#   D3     — PAM 多尺度 vs 单卷积核 k=15（待模型支持 --pam_single_scale 参数）
#   D4     — 去除 ConformerFusionBlock（待模型支持 --use_conformer 参数）
#   D5     — 仅 resting 训练，全场景测试（跨场景泛化）

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
$TEST         --exp_tag ExpA_baseline  --input_type phase --use_pam false

# ── Exp B1: 完整模型 × radar_raw ─────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B1 (full model, radar_raw)..."
$TRAIN $COMMON --exp_tag ExpB1_raw    --input_type raw
$TEST         --exp_tag ExpB1_raw    --input_type raw

# ── Exp B2: 完整模型 × radar_phase（主实验）──────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B2 (full model, radar_phase)..."
$TRAIN $COMMON --exp_tag ExpB2_phase  --input_type phase
$TEST         --exp_tag ExpB2_phase  --input_type phase

# ── Exp B3: 完整模型 × radar_spec ────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Starting Exp B3 (full model, radar_spec)..."
$TRAIN $COMMON --exp_tag ExpB3_spec   --input_type spec
$TEST         --exp_tag ExpB3_spec   --input_type spec

# ── D5: 仅 resting 训练 → 全场景测试（跨场景泛化）────────────────────────
# 训练：只用 resting 数据
# 测试：用全部场景（resting + valsalva + apnea），才能观察泛化能力
echo "[$(date +%H:%M:%S)] Starting D5 (resting-only train, all-scenario test)..."
$TRAIN $COMMON --exp_tag D5_generalization --input_type phase \
               --scenarios resting
$TEST         --exp_tag D5_generalization --input_type phase
# 注意：test 不传 --scenarios，默认使用全部三种场景

echo "============================================================"
echo " All experiments done. D1/D2/D3/D4 pending model support."
echo " Results: experiments/"
echo " Next: python scripts/summarize_ablation.py"
echo "============================================================"
