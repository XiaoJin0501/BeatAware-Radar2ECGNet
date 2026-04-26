"""
train_mmecg.py — MMECG LOSO 训练脚本

用法：
  python scripts/train_mmecg.py --exp_tag mmecg_D --fold_idx 0 --epochs 150
  python scripts/train_mmecg.py --exp_tag mmecg_D --fold_idx -1   # 全部 11 折

LOSO: fold_idx=i 表示留出第 i 号受试者（0-based 索引）作测试集。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders
from src.losses.losses import TotalLoss
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics
from src.utils.seeding import set_seed


# =============================================================================
# 工具函数
# =============================================================================

def _build_peak_gts_mmecg(batch: dict, rpeak_gt: torch.Tensor,
                            device: torch.device) -> dict:
    """
    MMECG 数据集只有 QRS 标注（无 P/T 波），
    返回 peak_gts 时把 p/t 置为全零 + valid=False，
    让 TotalLoss 中的 masked BCE 项自动忽略。
    """
    B, _, L = rpeak_gt.shape
    gts = {"qrs": rpeak_gt}
    zeros = torch.zeros_like(rpeak_gt)
    gts["p"]       = zeros
    gts["t"]       = zeros
    gts["p_valid"] = torch.zeros(B, dtype=torch.bool, device=device)
    gts["t_valid"] = torch.zeros(B, dtype=torch.bool, device=device)
    return gts


def _exp_dir(cfg: MMECGConfig, exp_tag: str, fold: int) -> Path:
    return Path(cfg.exp_dir) / exp_tag / f"fold_{fold}"


# =============================================================================
# 单 Fold 训练
# =============================================================================

def train_one_fold(cfg: MMECGConfig, exp_tag: str, fold: int,
                   args: argparse.Namespace) -> None:
    set_seed(42 + fold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dir = _exp_dir(cfg, exp_tag, fold)
    ckpt_dir   = fold_dir / "checkpoints"
    log_dir    = fold_dir / "logs"
    result_dir = fold_dir / "results"
    for d in [ckpt_dir, log_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_dir, name=f"fold{fold}")
    logger.info(f"{'='*60}")
    logger.info(f"MMECG LOSO | fold={fold} | test_subject=fold_{fold}")
    logger.info(f"device={device} | use_pam={cfg.use_pam} | use_emd={cfg.use_emd}")
    logger.info(f"{'='*60}")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_loso_loaders(
        dataset_dir=cfg.dataset_dir,
        fold_idx=fold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=cfg.balanced_sampling,
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples | "
                f"Test: {len(val_loader.dataset)} samples")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        use_pam=cfg.use_pam,
        use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    criterion = TotalLoss(alpha_stft=0.1, beta_peak=1.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2,
    )

    # ── 保存配置 ──────────────────────────────────────────────────────────────
    cfg_dict = {k: v for k, v in cfg.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}
    cfg_dict["exp_tag"] = exp_tag
    cfg_dict["fold_idx"] = fold
    with open(fold_dir / "config.json", "w") as jf:
        json.dump(cfg_dict, jf, indent=2, default=str)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_pcc        = -float("inf")
    best_val_loss       = float("inf")   # 早停用 val_loss（更平滑）
    early_stop_cnt      = 0
    early_stop_min_epoch = 30            # 前 30 epoch 不触发早停（给 FMCWEncoder 收敛）
    global_step         = 0
    history             = []
    val_every           = 5
    f1_every            = 10

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "time": 0.0,
                        "freq": 0.0, "peak": 0.0}
        t0 = time.time()

        for batch in train_loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)
            peak_gts = _build_peak_gts_mmecg(batch, rpeak_gt, device)

            optimizer.zero_grad()
            ecg_pred, peak_preds = model(radar)

            if not torch.isfinite(ecg_pred).all():
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        m.running_mean.nan_to_num_(nan=0.0)
                        m.running_var.nan_to_num_(nan=1.0, posinf=1.0).clamp_(min=0.0)
                continue

            losses = criterion(ecg_pred, ecg_gt, peak_preds, peak_gts, epoch=epoch)
            if not torch.isfinite(losses["total"]):
                continue

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            global_step += 1

        scheduler.step()

        n_batches = max(len(train_loader), 1)
        avg_train = {k: v / n_batches for k, v in epoch_losses.items()}
        logger.log_dict(avg_train, step=epoch, prefix=f"fold{fold}/train_epoch")

        hist_row: dict = {"epoch": epoch, **avg_train}

        if epoch % val_every == 0:
            val_metrics = _evaluate(
                model, val_loader, criterion, device, epoch, f1_every,
            )
            logger.log_dict(val_metrics, step=epoch, prefix=f"fold{fold}/val")

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train={avg_train['total']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_pcc={val_metrics['pcc']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )
            hist_row.update({f"val_{k}": v for k, v in val_metrics.items()})

            # ── 最佳 checkpoint：以 val_pcc 为准（临床意义最强）──────────────
            if val_metrics["pcc"] > best_val_pcc:
                best_val_pcc = val_metrics["pcc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_pcc": best_val_pcc,
                }, ckpt_dir / "best.pt")
                logger.info(f"  -> Best ckpt saved (val_pcc={best_val_pcc:.4f})")

            # ── 早停：以 val_loss 为准（平滑，不受 warmup 切换扰动）───────────
            # 同时要求已过最小 epoch 保护期
            val_loss = val_metrics.get("loss", float("inf"))
            if epoch >= early_stop_min_epoch:
                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                    if early_stop_cnt >= cfg.early_stop_patience:
                        logger.info(
                            f"  -> Early stop at epoch {epoch} "
                            f"(val_loss no improvement for "
                            f"{cfg.early_stop_patience} checks)"
                        )
                        history.append(hist_row)
                        break

        history.append(hist_row)

    with open(result_dir / "train_history.json", "w") as jf:
        json.dump(history, jf, indent=2)
    logger.info(f"Best val PCC: {best_val_pcc:.4f}")
    logger.close()


# =============================================================================
# Evaluation
# =============================================================================

def _evaluate(model, loader, criterion, device, epoch, f1_every):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    compute_f1 = (epoch % f1_every == 0)

    with torch.no_grad():
        for batch in loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)
            peak_gts = _build_peak_gts_mmecg(batch, rpeak_gt, device)

            ecg_pred, peak_preds = model(radar)
            if not torch.isfinite(ecg_pred).all():
                continue

            losses = criterion(ecg_pred, ecg_gt, peak_preds, peak_gts, epoch=epoch)
            if torch.isfinite(losses["total"]):
                total_loss += losses["total"].item()

            all_pred.append(ecg_pred.cpu())
            all_gt.append(ecg_gt.cpu())

    if not all_pred:
        return {"mae": float("nan"), "rmse": float("nan"),
                "pcc": float("nan"), "prd": float("nan"), "loss": float("nan")}

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)
    metrics  = compute_all_metrics(all_pred, all_gt, compute_f1=compute_f1)
    metrics["loss"] = total_loss / max(len(all_pred) // loader.batch_size, 1)
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag",  type=str, default="mmecg_D")
    parser.add_argument("--fold_idx", type=int, default=-1,
                        help="-1 = all 11 folds; 0-10 = single fold")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--use_pam",  type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--use_emd",  type=lambda x: x.lower() != "false", default=None)
    args = parser.parse_args()

    cfg = MMECGConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.use_pam is not None:
        cfg.use_pam = args.use_pam
    if args.use_emd is not None:
        cfg.use_emd = args.use_emd

    folds = list(range(cfg.n_folds)) if args.fold_idx == -1 else [args.fold_idx]
    print(f"MMECG LOSO training | exp_tag={args.exp_tag} | folds={folds}")
    print(f"use_pam={cfg.use_pam} | use_emd={cfg.use_emd}")

    for fold in folds:
        train_one_fold(cfg, args.exp_tag, fold, args)

    print(f"\nAll folds done. Results: {cfg.exp_dir}/{args.exp_tag}/")


if __name__ == "__main__":
    main()
