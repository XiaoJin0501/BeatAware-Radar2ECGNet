"""
train.py — BeatAware-Radar2ECGNet 训练脚本

用法：
  python scripts/train.py --exp_tag ExpB_phase --input_type phase --epochs 150
  python scripts/train.py --exp_tag ExpA_base  --input_type phase --use_pam false

5-Fold Cross-Validation：
  默认运行全部 5 folds（fold_idx=-1）。
  指定 --fold_idx N 只运行单个 fold（用于调试或并行训练）。
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import Config, get_config
from src.data.dataset import RadarECGDataset
from src.losses.losses import TotalLoss
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics
from src.utils.seeding import set_seed


# =============================================================================
# 单 Fold 训练
# =============================================================================

def train_one_fold(cfg: Config, fold: int) -> None:
    set_seed(cfg.seed + fold)   # 每个 fold 独立种子，保证可复现但不完全相同

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ── 目录 ──────────────────────────────────────────────────────────────
    cfg.ckpt_dir(fold).mkdir(parents=True, exist_ok=True)
    cfg.log_dir(fold).mkdir(parents=True, exist_ok=True)
    cfg.result_dir(fold).mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(cfg.log_dir(fold), name=f"fold{fold}")
    logger.info(f"{'='*60}")
    logger.info(f"Fold {fold}/{cfg.n_folds-1} | exp_tag={cfg.exp_tag}")
    logger.info(f"device={device} | input_type={cfg.input_type} | use_pam={cfg.use_pam}")
    logger.info(f"{'='*60}")

    # ── 数据集 ────────────────────────────────────────────────────────────
    train_ds = RadarECGDataset(
        cfg.dataset_dir, fold_idx=fold, split="train",
        input_type=cfg.input_type, scenarios=cfg.scenarios,
    )
    val_ds = RadarECGDataset(
        cfg.dataset_dir, fold_idx=fold, split="val",
        input_type=cfg.input_type, scenarios=cfg.scenarios,
    )
    logger.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # ── 模型 ──────────────────────────────────────────────────────────────
    model = BeatAwareRadar2ECGNet(
        input_type=cfg.input_type,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        use_pam=cfg.use_pam,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────
    criterion = TotalLoss(alpha=cfg.alpha, beta=cfg.beta)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    global_step  = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "time": 0.0, "freq": 0.0, "peak": 0.0}
        t0 = time.time()

        for batch in train_loader:
            radar   = batch["radar"].to(device)
            ecg_gt  = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)

            optimizer.zero_grad()
            ecg_pred, peak_pred = model(radar)
            losses = criterion(ecg_pred, ecg_gt, peak_pred, rpeak_gt)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if global_step % cfg.log_every == 0:
                logger.log_dict(
                    {k: losses[k].item() for k in losses},
                    step=global_step, prefix=f"fold{fold}/train_step"
                )
            global_step += 1

        scheduler.step()

        # epoch 平均 train loss
        n_batches = len(train_loader)
        avg_train = {k: v / n_batches for k, v in epoch_losses.items()}
        logger.log_dict(avg_train, step=epoch, prefix=f"fold{fold}/train_epoch")

        # ── Validation ────────────────────────────────────────────────────
        hist_row: dict = {"epoch": epoch, **avg_train}

        if epoch % cfg.val_every == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, cfg, epoch)
            logger.log_dict(val_metrics, step=epoch, prefix=f"fold{fold}/val")

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train_loss={avg_train['total']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_pcc={val_metrics['pcc']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

            # 保存最佳 checkpoint
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                ckpt_path = cfg.ckpt_dir(fold) / "best.pt"
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae":    best_val_mae,
                    "config":     cfg.__dict__,
                }, ckpt_path)
                logger.info(f"  -> Best checkpoint saved (val_mae={best_val_mae:.4f})")

            hist_row.update({f"val_{k}": v for k, v in val_metrics.items()})

        history.append(hist_row)

    # ── 保存训练历史 ──────────────────────────────────────────────────────
    hist_path = cfg.result_dir(fold) / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {hist_path}")
    logger.info(f"Best val MAE: {best_val_mae:.4f}")
    logger.close()


# =============================================================================
# Validation Loop
# =============================================================================

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  TotalLoss,
    device:     torch.device,
    cfg:        Config,
    epoch:      int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    compute_f1 = (epoch % cfg.f1_every == 0)

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

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)

    metrics = compute_all_metrics(all_pred, all_gt, compute_f1=compute_f1)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    cfg = get_config()

    # 保存配置
    cfg.exp_root.mkdir(parents=True, exist_ok=True)
    with open(cfg.exp_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    folds = list(range(cfg.n_folds)) if cfg.fold_idx == -1 else [cfg.fold_idx]
    print(f"\nRunning folds: {folds}")

    for fold in folds:
        train_one_fold(cfg, fold)

    print(f"\nAll folds done. Results: {cfg.exp_root}")


if __name__ == "__main__":
    main()
