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
# 辅助：构建 peak_gts dict
# =============================================================================

def _build_peak_gts(batch: dict, rpeak_gt: torch.Tensor, device: torch.device) -> dict:
    """
    从 batch 中整理多头 PAM 所需的 GT 字典。

    batch 中 'pwave'/'twave'/'pwave_valid'/'twave_valid' 在
    dataset.py 里始终存在（step2b 未运行时为零 tensor + valid=False）。
    """
    gts = {"qrs": rpeak_gt}

    pwave_gt    = batch["pwave"].to(device)      # (B,1,1600)
    twave_gt    = batch["twave"].to(device)      # (B,1,1600)
    pwave_valid = batch["pwave_valid"].to(device) # (B,) bool
    twave_valid = batch["twave_valid"].to(device) # (B,) bool

    gts["p"]       = pwave_gt
    gts["t"]       = twave_gt
    gts["p_valid"] = pwave_valid
    gts["t_valid"] = twave_valid

    return gts


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
        use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────
    criterion = TotalLoss(
        alpha_stft=cfg.alpha_stft,
        warmup_epochs=cfg.warmup_epochs,
    ).to(device)
    # log_vars 是可学习参数，须纳入 optimizer（与模型参数共享 lr 和 weight_decay）
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────
    best_val_pcc     = -float("inf")
    best_val_mae     = float("inf")   # 仅用于日志记录
    early_stop_count = 0
    global_step      = 0
    history          = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0, "recon": 0.0, "time": 0.0,
            "freq": 0.0, "peak": 0.0, "der": 0.0, "interval": 0.0,
        }
        t0 = time.time()

        for batch in train_loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)

            # P/T 波 GT（V2 多头 PAM，需 step2b 预处理）
            peak_gts = _build_peak_gts(batch, rpeak_gt, device)

            optimizer.zero_grad()
            ecg_pred, peak_preds = model(radar)
            losses = criterion(ecg_pred, ecg_gt, peak_preds, peak_gts, epoch=epoch)

            # NaN/Inf 保护：SSM 偶发数值爆炸时跳过该 batch，不让训练崩溃
            if not torch.isfinite(losses["total"]):
                continue

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

        # log_vars 权重轨迹（adaptive loss weights，4 tasks）
        lv = criterion.log_vars.detach().cpu()
        precision = (0.5 * torch.exp(-lv)).tolist()
        logger.log_dict(
            {
                "recon": precision[0], "peak": precision[1],
                "der":   precision[2], "interval": precision[3],
            },
            step=epoch, prefix=f"fold{fold}/loss_weight",
        )

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

            # val 指标先并入 hist_row（保证 early stop 和正常路径都有完整记录）
            hist_row.update({f"val_{k}": v for k, v in val_metrics.items()})

            # 保存最佳 checkpoint（以 val_pcc 为准，PCC 比 MAE 更稳定）
            if val_metrics["pcc"] > best_val_pcc:
                best_val_pcc = val_metrics["pcc"]
                best_val_mae = val_metrics["mae"]
                early_stop_count = 0
                ckpt_path = cfg.ckpt_dir(fold) / "best.pt"
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_pcc":    best_val_pcc,
                    "val_mae":    best_val_mae,
                    "config":     cfg.__dict__,
                }, ckpt_path)
                logger.info(f"  -> Best checkpoint saved (val_pcc={best_val_pcc:.4f}, val_mae={best_val_mae:.4f})")
            else:
                early_stop_count += 1
                if early_stop_count >= cfg.early_stop_patience:
                    logger.info(
                        f"  -> Early stopping at epoch {epoch} "
                        f"(no val_pcc improvement for {cfg.early_stop_patience} epochs)"
                    )
                    history.append(hist_row)
                    break

        history.append(hist_row)

    # ── 保存训练历史 ──────────────────────────────────────────────────────
    hist_path = cfg.result_dir(fold) / "train_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved: {hist_path}")
    logger.info(f"Best val PCC: {best_val_pcc:.4f} | Best val MAE: {best_val_mae:.4f}")
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
            peak_gts = _build_peak_gts(batch, rpeak_gt, device)

            ecg_pred, peak_preds = model(radar)
            losses = criterion(ecg_pred, ecg_gt, peak_preds, peak_gts, epoch=epoch)
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
