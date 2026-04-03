"""
test.py — BeatAware-Radar2ECGNet 推理与评估脚本

用法：
  # 评估单个 fold
  python scripts/test.py --exp_tag ExpB_phase --fold_idx 0

  # 评估全部 5 folds 并汇总
  python scripts/test.py --exp_tag ExpB_phase
"""

import csv
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import Config, get_config
from src.data.dataset import RadarECGDataset
from src.losses.losses import TotalLoss
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.metrics import compute_all_metrics
from src.utils.seeding import set_seed


# =============================================================================
# 单 Fold 测试
# =============================================================================

def test_one_fold(cfg: Config, fold: int) -> dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.ckpt_dir(fold) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please run train.py first."
        )

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[Fold {fold}] Loaded checkpoint (epoch={ckpt['epoch']}, "
          f"val_mae={ckpt['val_mae']:.4f})")

    model = BeatAwareRadar2ECGNet(
        input_type=cfg.input_type,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=0.0,        # test 时关闭 dropout
        use_pam=cfg.use_pam,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 数据 ──────────────────────────────────────────────────────────────
    val_ds = RadarECGDataset(
        cfg.dataset_dir, fold_idx=fold, split="val",
        input_type=cfg.input_type, scenarios=cfg.scenarios,
    )
    loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    print(f"[Fold {fold}] Val samples: {len(val_ds)}")

    criterion = TotalLoss(alpha=cfg.alpha, beta=cfg.beta)

    # ── 推理 ──────────────────────────────────────────────────────────────
    all_pred, all_gt, total_loss = [], [], 0.0
    sample_preds, sample_gts = [], []     # 保存前 N 个样本用于可视化

    with torch.no_grad():
        for batch in loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)

            ecg_pred, peak_pred = model(radar)
            losses = criterion(ecg_pred, ecg_gt, peak_pred, rpeak_gt)
            total_loss += losses["total"].item()

            all_pred.append(ecg_pred.cpu())
            all_gt.append(ecg_gt.cpu())

            if len(sample_preds) < 8:
                sample_preds.append(ecg_pred.cpu())
                sample_gts.append(ecg_gt.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)

    # ── 指标计算（含 R 峰 F1）────────────────────────────────────────────
    metrics = compute_all_metrics(all_pred, all_gt, compute_f1=True)
    metrics["loss"] = total_loss / len(loader)

    print(f"[Fold {fold}] MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
          f"PCC={metrics['pcc']:.4f} | PRD={metrics['prd']:.2f}% | "
          f"F1={metrics.get('rpeak_f1', float('nan')):.4f}")

    # ── 保存 CSV ──────────────────────────────────────────────────────────
    result_dir = cfg.result_dir(fold)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + list(metrics.keys()))
        writer.writeheader()
        writer.writerow({"fold": fold, **metrics})
    print(f"[Fold {fold}] Metrics saved: {csv_path}")

    # ── 可视化（前 8 个样本对比图）───────────────────────────────────────
    _save_comparison_figure(
        sample_preds, sample_gts,
        save_path=result_dir / "sample_predictions.png",
        fold=fold,
    )

    return metrics


def _save_comparison_figure(
    preds: list, gts: list,
    save_path: Path,
    fold: int,
    n_show: int = 8,
) -> None:
    pred_cat = torch.cat(preds, dim=0)[:n_show].numpy().squeeze(1)  # (N, 1600)
    gt_cat   = torch.cat(gts,   dim=0)[:n_show].numpy().squeeze(1)

    n = len(pred_cat)
    t = np.arange(pred_cat.shape[1]) / 200.0

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Fold {fold} — GT (blue) vs Pred (red), first {n} samples",
                 fontsize=12)

    for i, ax in enumerate(axes):
        ax.plot(t, gt_cat[i],   color="#2980B9", linewidth=0.9, label="GT",   alpha=0.8)
        ax.plot(t, pred_cat[i], color="#E74C3C", linewidth=0.9, label="Pred", alpha=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fold {fold}] Prediction plot saved: {save_path}")


# =============================================================================
# Main — 汇总所有 fold
# =============================================================================

def main() -> None:
    cfg = get_config()
    folds = list(range(cfg.n_folds)) if cfg.fold_idx == -1 else [cfg.fold_idx]

    all_metrics = []
    for fold in folds:
        try:
            m = test_one_fold(cfg, fold)
            m["fold"] = fold
            all_metrics.append(m)
        except FileNotFoundError as e:
            print(f"[Fold {fold}] SKIP: {e}")

    if len(all_metrics) == 0:
        print("No checkpoints found. Run train.py first.")
        return

    # 汇总 CSV
    summary_path = cfg.exp_root / "test_summary.csv"
    keys = [k for k in all_metrics[0] if k != "fold"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + keys)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

        # 均值行
        mean_row = {"fold": "mean"}
        for k in keys:
            vals = [m[k] for m in all_metrics if not (isinstance(m[k], float) and m[k] != m[k])]
            mean_row[k] = sum(vals) / len(vals) if vals else float("nan")
        writer.writerow(mean_row)

    print(f"\nSummary saved: {summary_path}")
    print(f"\n{'Metric':<12} {'Mean':>10}")
    print("-" * 25)
    for k in ["mae", "rmse", "pcc", "prd", "rpeak_f1"]:
        if k in mean_row:
            print(f"{k:<12} {mean_row[k]:>10.4f}")

    # 汇总 JSON
    with open(cfg.exp_root / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({"folds": all_metrics, "mean": mean_row}, f, indent=2)


if __name__ == "__main__":
    main()
