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
from src.utils.metrics import (
    compute_all_metrics,
    compute_advanced_metrics,
    compute_waveform_metrics,
    compute_peak_metrics,
    detect_rpeaks,
)
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
          f"val_pcc={ckpt.get('val_pcc', float('nan')):.4f}, "
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
    all_pred, all_gt, all_scenarios, all_subjects, total_loss = [], [], [], [], 0.0
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
            all_scenarios.extend(batch["scenario"])   # list[str]
            all_subjects.extend(batch["subject"])     # list[str]

            if len(sample_preds) < 8:
                sample_preds.append(ecg_pred.cpu())
                sample_gts.append(ecg_gt.cpu())

    all_pred      = torch.cat(all_pred, dim=0)   # (N, 1, L)
    all_gt        = torch.cat(all_gt,   dim=0)
    all_scenarios = np.array(all_scenarios)       # (N,) str
    all_subjects  = np.array(all_subjects)        # (N,) str

    # ── 全局指标（MAE/RMSE/PCC/PRD/F1）──────────────────────────────────
    metrics = compute_all_metrics(all_pred, all_gt, compute_f1=True)
    metrics["loss"] = total_loss / len(loader)

    # ── 高级指标（DTW / RR interval / QRS width）─────────────────────────
    print(f"[Fold {fold}] Computing advanced metrics (DTW / RR / QRS)...")
    metrics.update(compute_advanced_metrics(all_pred, all_gt, fs=200))

    print(
        f"[Fold {fold}] MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | "
        f"PCC={metrics['pcc']:.4f} | PRD={metrics['prd']:.2f}% | "
        f"F1={metrics.get('rpeak_f1', float('nan')):.4f} | "
        f"DTW={metrics.get('dtw', float('nan')):.4f} | "
        f"RR_MAE={metrics.get('rr_interval_mae', float('nan')):.2f}ms | "
        f"QRS_MAE={metrics.get('qrs_width_mae', float('nan')):.2f}ms"
    )

    result_dir = cfg.result_dir(fold)
    result_dir.mkdir(parents=True, exist_ok=True)

    # ── 全局 CSV ──────────────────────────────────────────────────────────
    csv_path = result_dir / "test_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + list(metrics.keys()))
        writer.writeheader()
        writer.writerow({"fold": fold, **metrics})
    print(f"[Fold {fold}] Metrics saved: {csv_path}")

    # ── Per-scenario 指标 ─────────────────────────────────────────────────
    _save_scenario_metrics(
        all_pred, all_gt, all_scenarios,
        save_path=result_dir / "test_metrics_by_scenario.csv",
        fold=fold,
    )

    # ── Per-subject 指标（按 subject × scenario 双维度）─────────────────
    _save_subject_metrics(
        all_pred, all_gt, all_subjects, all_scenarios,
        save_path=result_dir / "test_metrics_by_subject.csv",
        fold=fold,
    )

    # ── 可视化（前 8 个样本对比图）───────────────────────────────────────
    _save_comparison_figure(
        sample_preds, sample_gts,
        save_path=result_dir / "sample_predictions.png",
        fold=fold,
    )

    return metrics


def _save_scenario_metrics(
    pred:      torch.Tensor,    # (N, 1, L)
    gt:        torch.Tensor,    # (N, 1, L)
    scenarios: np.ndarray,      # (N,) str
    save_path: Path,
    fold:      int,
) -> None:
    """
    按场景分组计算 MAE/RMSE/PCC/PRD/F1，保存为 CSV。

    用于 D5 泛化实验：训练用 resting，测试观察各场景的指标差异。
    """
    scenario_list = sorted(set(scenarios))
    rows = []

    for scenario in scenario_list:
        mask      = scenarios == scenario
        s_pred    = pred[mask]
        s_gt      = gt[mask]
        n_samples = int(mask.sum())

        if n_samples == 0:
            continue

        s_metrics = compute_all_metrics(s_pred, s_gt, compute_f1=True)
        row = {"fold": fold, "scenario": scenario, "n_samples": n_samples}
        row.update(s_metrics)
        rows.append(row)

        print(
            f"  [{scenario}] n={n_samples:4d} | "
            f"MAE={s_metrics['mae']:.4f} | PCC={s_metrics['pcc']:.4f} | "
            f"F1={s_metrics.get('rpeak_f1', float('nan')):.4f}"
        )

    if not rows:
        return

    fieldnames = ["fold", "scenario", "n_samples"] + [
        k for k in rows[0] if k not in ("fold", "scenario", "n_samples")
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Fold {fold}] Per-scenario metrics saved: {save_path}")


def _save_subject_metrics(
    pred:      torch.Tensor,    # (N, 1, L)
    gt:        torch.Tensor,    # (N, 1, L)
    subjects:  np.ndarray,      # (N,) str
    scenarios: np.ndarray,      # (N,) str
    save_path: Path,
    fold:      int,
) -> None:
    """
    按受试者 × 场景计算 MAE/RMSE/PCC/PRD/F1，保存为 CSV。

    用于：
      - D5 跨场景泛化实验：每个受试者在 3 个场景下的指标
      - 30-subject 柱状图：将所有 fold 的 CSV 合并即可覆盖全部受试者
    """
    rows = []

    for subject in sorted(set(subjects)):
        for scenario in sorted(set(scenarios[subjects == subject])):
            mask = (subjects == subject) & (scenarios == scenario)
            s_pred = pred[mask]
            s_gt   = gt[mask]
            n_samples = int(mask.sum())

            if n_samples == 0:
                continue

            s_metrics = compute_all_metrics(s_pred, s_gt, compute_f1=True)
            row = {
                "fold":     fold,
                "subject":  subject,
                "scenario": scenario,
                "n_samples": n_samples,
            }
            row.update(s_metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = ["fold", "subject", "scenario", "n_samples"] + [
        k for k in rows[0] if k not in ("fold", "subject", "scenario", "n_samples")
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Fold {fold}] Per-subject metrics saved: {save_path}")


def _save_comparison_figure(
    preds: list, gts: list,
    save_path: Path,
    fold: int,
    n_show: int = 8,
    fs: int = 200,
) -> None:
    """
    每个样本生成两列子图：
      左列：时域波形对比（GT 蓝色 / Pred 红色）+ R 峰标注
      右列：功率谱对比（GT 蓝色 / Pred 红色，log 纵轴）
    """
    pred_np = torch.cat(preds, dim=0)[:n_show].numpy().squeeze(1)  # (N, L)
    gt_np   = torch.cat(gts,   dim=0)[:n_show].numpy().squeeze(1)
    n = len(pred_np)
    L = pred_np.shape[1]
    t = np.arange(L) / fs

    # 频率轴（单边）
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)

    fig, axes = plt.subplots(n, 2, figsize=(18, 2.6 * n),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n == 1:
        axes = axes[np.newaxis, :]   # (1, 2)

    fig.suptitle(
        f"Fold {fold} — GT (blue) vs Pred (red), first {n} samples\n"
        f"Left: time-domain + R-peaks   Right: power spectrum",
        fontsize=11, y=1.01,
    )

    for i in range(n):
        gt_sig   = gt_np[i]
        pred_sig = pred_np[i]

        # ── 左列：时域 + R 峰 ──────────────────────────────────────────
        ax_t = axes[i, 0]
        ax_t.plot(t, gt_sig,   color="#2980B9", lw=0.9, label="GT",   alpha=0.85)
        ax_t.plot(t, pred_sig, color="#E74C3C", lw=0.9, label="Pred", alpha=0.85)

        # R 峰标注
        gt_peaks   = detect_rpeaks(gt_sig,   fs=fs)
        pred_peaks = detect_rpeaks(pred_sig, fs=fs)
        if len(gt_peaks) > 0:
            ax_t.scatter(gt_peaks / fs, gt_sig[gt_peaks],
                         color="#1A5276", s=18, zorder=5,
                         label=f"GT peaks ({len(gt_peaks)})")
        if len(pred_peaks) > 0:
            ax_t.scatter(pred_peaks / fs, pred_sig[pred_peaks],
                         color="#922B21", s=18, marker="x", zorder=5,
                         label=f"Pred peaks ({len(pred_peaks)})")

        ax_t.set_ylim(-0.05, 1.15)
        ax_t.set_xlim(t[0], t[-1])
        ax_t.grid(True, alpha=0.25)
        ax_t.tick_params(labelsize=7)
        if i == 0:
            ax_t.legend(loc="upper right", fontsize=7, ncol=2)
        if i == n - 1:
            ax_t.set_xlabel("Time (s)", fontsize=9)

        # ── 右列：功率谱（0–40 Hz，心电信号有效带宽）─────────────────
        ax_f = axes[i, 1]
        psd_gt   = np.abs(np.fft.rfft(gt_sig)) ** 2
        psd_pred = np.abs(np.fft.rfft(pred_sig)) ** 2

        # 只显示 0.5–40 Hz 心电有效频段
        mask = (freqs >= 0.5) & (freqs <= 40.0)
        ax_f.semilogy(freqs[mask], psd_gt[mask],
                      color="#2980B9", lw=0.9, alpha=0.85)
        ax_f.semilogy(freqs[mask], psd_pred[mask],
                      color="#E74C3C", lw=0.9, alpha=0.85)

        ax_f.set_xlim(0.5, 40.0)
        ax_f.grid(True, alpha=0.25, which="both")
        ax_f.tick_params(labelsize=7)
        if i == 0:
            ax_f.set_title("Power Spectrum\n(0.5–40 Hz)", fontsize=8)
        if i == n - 1:
            ax_f.set_xlabel("Freq (Hz)", fontsize=9)
        ax_f.set_ylabel("Power", fontsize=7)

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
