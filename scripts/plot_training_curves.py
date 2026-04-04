"""
plot_training_curves.py — 从 train_history.json 生成训练曲线图

用法：
  # 单个 fold
  python scripts/plot_training_curves.py --exp_tag smoke_test --fold_idx 0

  # 所有已存在的 fold（自动发现）
  python scripts/plot_training_curves.py --exp_tag smoke_test

输出：
  experiments/<EXP_TAG>/fold_N/results/training_curves.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# =============================================================================
# 绘图
# =============================================================================

def plot_one_fold(history: list[dict], save_path: Path, fold: int) -> None:
    """
    生成两行四列共 7 个子图的训练曲线图：
      Row 1: Loss 分量（Total / Time / Freq / Peak）
      Row 2: Val 指标（MAE / PCC / PRD / R-peak F1）
    """
    epochs = [h["epoch"] for h in history]

    # ── 提取各序列（有些 epoch 可能没有 f1，用 nan 填充）────────────────
    def get(key):
        return [h.get(key, float("nan")) for h in history]

    train_total = get("total")
    train_time  = get("time")
    train_freq  = get("freq")
    train_peak  = get("peak")

    val_mae  = get("val_mae")
    val_pcc  = get("val_pcc")
    val_prd  = get("val_prd")
    val_f1   = get("val_rpeak_f1")
    val_loss = get("val_loss")

    has_f1 = any(not np.isnan(v) for v in val_f1)

    # ── 布局：2 行，上行 4 列，下行 4 列 ─────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        f"Training Curves — {save_path.parent.parent.parent.name} / Fold {fold}",
        fontsize=13, y=1.01,
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # 颜色方案
    C_TOTAL = "#2C3E50"
    C_TIME  = "#2980B9"
    C_FREQ  = "#27AE60"
    C_PEAK  = "#E67E22"
    C_VAL   = "#8E44AD"

    # ── Row 1：Loss 分量 ──────────────────────────────────────────────────

    # 1-1  Total Loss（train + val）
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_total, color=C_TOTAL, lw=1.5, label="train total")
    ax.plot(epochs, val_loss,    color=C_TOTAL, lw=1.5, ls="--", label="val loss")
    ax.set_title("Total Loss", fontsize=10)
    ax.legend(fontsize=7)
    _style(ax)

    # 1-2  L_time（MAE）
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, train_time, color=C_TIME, lw=1.5)
    ax.set_title("L_time  (MAE, train)", fontsize=10)
    _style(ax)

    # 1-3  L_freq（STFT）
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, train_freq, color=C_FREQ, lw=1.5)
    ax.set_title("L_freq  (STFT, train)", fontsize=10)
    _style(ax)

    # 1-4  L_peak（BCE）
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(epochs, train_peak, color=C_PEAK, lw=1.5)
    ax.set_title("L_peak  (BCE, train)", fontsize=10)
    _style(ax)

    # ── Row 2：Val 指标 ───────────────────────────────────────────────────

    # 2-1  Val MAE
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, val_mae, color=C_VAL, lw=1.5, marker="o", ms=3)
    best_epoch = epochs[int(np.nanargmin(val_mae))]
    best_val   = float(np.nanmin(val_mae))
    ax.axvline(best_epoch, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Val MAE  (best={best_val:.4f} @ ep{best_epoch})", fontsize=9)
    _style(ax, ylabel="MAE")

    # 2-2  Val PCC
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, val_pcc, color=C_VAL, lw=1.5, marker="o", ms=3)
    best_epoch = epochs[int(np.nanargmax(val_pcc))]
    best_val   = float(np.nanmax(val_pcc))
    ax.axvline(best_epoch, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Val PCC  (best={best_val:.4f} @ ep{best_epoch})", fontsize=9)
    _style(ax, ylabel="PCC")

    # 2-3  Val PRD
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(epochs, val_prd, color=C_VAL, lw=1.5, marker="o", ms=3)
    best_epoch = epochs[int(np.nanargmin(val_prd))]
    best_val   = float(np.nanmin(val_prd))
    ax.axvline(best_epoch, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Val PRD  (best={best_val:.2f}% @ ep{best_epoch})", fontsize=9)
    _style(ax, ylabel="PRD (%)")

    # 2-4  Val R-peak F1（若有数据则画，否则显示 N/A）
    ax = fig.add_subplot(gs[1, 3])
    if has_f1:
        ax.plot(epochs, val_f1, color=C_VAL, lw=1.5, marker="o", ms=3)
        valid_f1 = [(e, v) for e, v in zip(epochs, val_f1) if not np.isnan(v)]
        best_epoch = max(valid_f1, key=lambda x: x[1])[0]
        best_val   = max(v for _, v in valid_f1)
        ax.axvline(best_epoch, color="gray", ls=":", lw=0.8)
        ax.set_title(
            f"Val R-peak F1  (best={best_val:.4f} @ ep{best_epoch})", fontsize=9
        )
    else:
        ax.text(0.5, 0.5, "N/A\n(f1_every not reached)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        ax.set_title("Val R-peak F1", fontsize=9)
    _style(ax, ylabel="F1")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {save_path}")


def _style(ax, ylabel: str = "") -> None:
    """统一子图样式。"""
    ax.set_xlabel("Epoch", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 train_history.json 生成训练曲线图"
    )
    parser.add_argument("--exp_tag",  required=True, help="实验名称")
    parser.add_argument(
        "--fold_idx", type=int, default=-1,
        help="fold 索引（默认 -1 = 自动发现所有已存在的 fold）",
    )
    parser.add_argument(
        "--exp_dir", type=str, default="experiments",
        help="实验根目录（默认 experiments/）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    exp_root     = project_root / args.exp_dir / args.exp_tag

    if not exp_root.exists():
        print(f"[ERROR] 实验目录不存在: {exp_root}")
        sys.exit(1)

    # 发现所有 fold 目录
    if args.fold_idx == -1:
        fold_dirs = sorted(exp_root.glob("fold_*"))
    else:
        fold_dirs = [exp_root / f"fold_{args.fold_idx}"]

    if not fold_dirs:
        print(f"[ERROR] 未找到任何 fold 目录: {exp_root}")
        sys.exit(1)

    for fold_dir in fold_dirs:
        fold_idx  = int(fold_dir.name.split("_")[1])
        hist_path = fold_dir / "results" / "train_history.json"

        if not hist_path.exists():
            print(f"[SKIP] Fold {fold_idx}: train_history.json 不存在")
            continue

        with open(hist_path, encoding="utf-8") as f:
            history = json.load(f)

        if not history:
            print(f"[SKIP] Fold {fold_idx}: train_history.json 为空")
            continue

        result_dir = fold_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        save_path = result_dir / "training_curves.png"

        print(f"Fold {fold_idx}: {len(history)} epochs")
        plot_one_fold(history, save_path, fold_idx)


if __name__ == "__main__":
    main()
