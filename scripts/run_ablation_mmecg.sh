#!/usr/bin/env bash
# run_ablation_mmecg.sh — MMECG 消融实验批量脚本
#
# 消融维度（MMECG / FMCW 输入）：
#   PAM  — 峰值辅助模块 + TFiLM 节律调制
#   EMD  — 物理对齐层（可学习 FIR 延迟补偿）
#
#   Model A  use_pam=false  use_emd=false   纯 Backbone（基线）
#   Model B  use_pam=false  use_emd=true    +EMD
#   Model C  use_pam=true   use_emd=false   +PAM
#   Model D  use_pam=true   use_emd=true    Full（已由 mmecg_v1 覆盖）
#
# 用法：
#   bash scripts/run_ablation_mmecg.sh                    # LOSO 全部 11 折
#   bash scripts/run_ablation_mmecg.sh --fold_idx 1       # 单折调试
#   bash scripts/run_ablation_mmecg.sh --protocol samplewise
#
# 注：Model D（完整扩散模型）的结果直接复用 exp_tag=mmecg_diff_v2，不重复训练。
#     若需重跑 Model D，取消下方注释并修改 exp_tag。

set -e

# 扩散消融版（v2）：T=1000, hidden=256, n_blocks=8，与 mmecg_diff_v2 一致
TRAIN="python scripts/train_mmecg.py --epochs 300 --use_diffusion true --balance_by subject --narrow_bandpass true"
TEST="python scripts/test_mmecg.py"
EXTRA_ARGS="$@"   # 透传：--fold_idx / --protocol / --batch_size 等

run_model() {
    local tag="$1" pam="$2" emd="$3"
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── $tag  (use_pam=$pam  use_emd=$emd) ──"
    $TRAIN --exp_tag "$tag" --use_pam "$pam" --use_emd "$emd" $EXTRA_ARGS
    $TEST  --exp_tag "$tag" $EXTRA_ARGS
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ── $tag done ──"
}

echo "========================================================"
echo " BeatAware MMECG Ablation (Diffusion v2: T=1000)"
echo " Extra args: $EXTRA_ARGS"
echo "========================================================"

run_model  mmecg_diff_A   false  false
run_model  mmecg_diff_B   false  true
run_model  mmecg_diff_C   true   false
# run_model  mmecg_diff_D  true   true   # 已由 mmecg_diff_v2 覆盖，如需重跑取消注释

echo ""
echo "========================================================"
echo " All ablation runs done."
echo " Results: experiments_mmecg/"
echo "========================================================"
