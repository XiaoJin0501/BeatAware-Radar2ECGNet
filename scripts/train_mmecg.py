"""
train_mmecg.py — MMECG 训练脚本（LOSO / Samplewise）

用法：
  # LOSO：单折
  python scripts/train_mmecg.py --exp_tag mmecg_v1 --fold_idx 1 --epochs 150

  # LOSO：全部 11 折
  python scripts/train_mmecg.py --exp_tag mmecg_v1 --fold_idx -1

  # Samplewise
  python scripts/train_mmecg.py --exp_tag sw_v1 --protocol samplewise --epochs 150
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders_h5, build_samplewise_loaders_h5
from src.losses.losses import DiffusionLoss, TotalLoss
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
    zeros = torch.zeros_like(rpeak_gt)
    return {
        "qrs":     rpeak_gt,
        "p":       zeros,
        "t":       zeros,
        "p_valid": torch.zeros(B, dtype=torch.bool, device=device),
        "t_valid": torch.zeros(B, dtype=torch.bool, device=device),
    }


def _exp_dir(cfg: MMECGConfig, exp_tag: str, fold_label: str) -> Path:
    return Path(cfg.exp_dir) / exp_tag / fold_label


# =============================================================================
# 单次训练（LOSO 单折 or Samplewise 一次）
# =============================================================================

def train_one_run(
    cfg: MMECGConfig,
    exp_tag: str,
    train_loader,
    val_loader,
    run_label: str,
    args: argparse.Namespace,
) -> None:
    """
    通用训练函数。run_label 用于目录命名和日志前缀。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir    = _exp_dir(cfg, exp_tag, run_label)
    ckpt_dir   = run_dir / "checkpoints"
    log_dir    = run_dir / "logs"
    result_dir = run_dir / "results"
    for d in [ckpt_dir, log_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(log_dir, name=run_label)
    logger.info(f"{'='*60}")
    logger.info(f"MMECG training | run={run_label} | protocol={args.protocol}")
    logger.info(f"device={device} | use_pam={cfg.use_pam} | use_emd={cfg.use_emd}")
    logger.info(f"Train: {len(train_loader.dataset)} samples | "
                f"Val: {len(val_loader.dataset)} samples")
    logger.info(f"{'='*60}")

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
        use_diffusion=cfg.use_diffusion,
        diff_T=cfg.diff_T,
        diff_ddim_steps=cfg.diff_ddim_steps,
        diff_hidden=cfg.diff_hidden,
        diff_n_blocks=cfg.diff_n_blocks,
        use_output_lag_align=cfg.use_output_lag_align,
        output_lag_max_samples=int(round(cfg.output_lag_max_ms / 1000.0 * cfg.fs)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")
    logger.info(f"use_diffusion={cfg.use_diffusion} | "
                f"balance_by={cfg.balance_by} | narrow_bandpass={cfg.narrow_bandpass}")
    logger.info(f"use_lag_aware_loss={cfg.use_lag_aware_loss} | "
                f"lag_max_ms={cfg.lag_max_ms} | "
                f"lambda_lag_pcc={cfg.lambda_lag_pcc} | "
                f"lambda_lag_l1={cfg.lambda_lag_l1} | "
                f"lambda_zero_pcc={cfg.lambda_zero_pcc} | "
                f"lambda_lag_penalty={cfg.lambda_lag_penalty} | "
                f"lag_softmax_tau={cfg.lag_softmax_tau}")
    logger.info(f"use_output_lag_align={cfg.use_output_lag_align} | "
                f"output_lag_max_ms={cfg.output_lag_max_ms} | "
                f"lambda_output_lag_l1={cfg.lambda_output_lag_l1}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    if cfg.use_diffusion:
        criterion = DiffusionLoss(beta_peak=1.0).to(device)
    else:
        lag_max_samples = int(round(cfg.lag_max_ms / 1000.0 * cfg.fs))
        criterion = TotalLoss(
            alpha_stft=0.1,
            beta_peak=1.0,
            use_lag_aware=cfg.use_lag_aware_loss,
            lag_max_samples=lag_max_samples,
            lambda_lag_pcc=cfg.lambda_lag_pcc,
            lambda_lag_l1=cfg.lambda_lag_l1,
            lambda_zero_pcc=cfg.lambda_zero_pcc,
            lambda_lag_penalty=cfg.lambda_lag_penalty,
            lag_softmax_tau=cfg.lag_softmax_tau,
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2,
    )

    # ── 保存配置 ──────────────────────────────────────────────────────────────
    # 读取类级别默认值，再用实例属性覆盖（防止 __class__.__dict__ 漏掉 runtime 修改）
    cfg_dict = {k: v for k, v in cfg.__class__.__dict__.items()
                if not k.startswith("_") and not callable(v)}
    cfg_dict.update({k: v for k, v in cfg.__dict__.items()
                     if not k.startswith("_")})
    cfg_dict["exp_tag"]  = exp_tag
    cfg_dict["run_label"] = run_label
    cfg_dict["protocol"] = args.protocol
    with open(run_dir / "config.json", "w") as jf:
        json.dump(cfg_dict, jf, indent=2, default=str)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_pcc         = -float("inf")
    best_val_loss        = float("inf")
    early_stop_cnt       = 0
    early_stop_min_epoch = 30
    global_step          = 0
    history              = []
    val_every            = 10 if cfg.use_diffusion else 5   # 扩散 DDIM 采样慢，降低 val 频率
    f1_every             = 20 if cfg.use_diffusion else 10

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "recon": 0.0, "time": 0.0,
                        "freq": 0.0, "peak": 0.0,
                        "lag_pcc": 0.0, "lag_l1": 0.0,
                        "zero_pcc": 0.0, "lag_penalty": 0.0,
                        "output_lag_l1": 0.0}
        t0 = time.time()

        for batch in train_loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)
            peak_gts = _build_peak_gts_mmecg(batch, rpeak_gt, device)

            optimizer.zero_grad()

            if cfg.use_diffusion:
                model_out, peak_preds = model(radar, ecg_gt=ecg_gt)
                # model_out = (eps_pred, eps_true)
                check_tensor = model_out[0]
            else:
                model_out, peak_preds = model(radar)
                check_tensor = model_out

            if not torch.isfinite(check_tensor).all():
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        m.running_mean.nan_to_num_(nan=0.0)
                        m.running_var.nan_to_num_(nan=1.0, posinf=1.0).clamp_(min=0.0)
                continue

            losses = criterion(model_out, ecg_gt, peak_preds, peak_gts, epoch=epoch)
            output_lag_penalty = torch.tensor(0.0, device=device)
            if (
                cfg.use_output_lag_align
                and cfg.lambda_output_lag_l1 > 0
                and getattr(model, "last_output_lag_samples", None) is not None
            ):
                max_lag_samples = max(
                    cfg.output_lag_max_ms / 1000.0 * cfg.fs,
                    1.0,
                )
                output_lag_penalty = (
                    model.last_output_lag_samples.abs().mean() / max_lag_samples
                )
                losses["total"] = (
                    losses["total"]
                    + cfg.lambda_output_lag_l1 * output_lag_penalty
                )
            losses["output_lag_l1"] = output_lag_penalty.detach()
            if not torch.isfinite(losses["total"]):
                continue

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            global_step += 1

        scheduler.step()

        n_batches = max(len(train_loader), 1)
        avg_train = {k: v / n_batches for k, v in epoch_losses.items()}
        logger.log_dict(avg_train, step=epoch, prefix=f"{run_label}/train_epoch")

        hist_row: dict = {"epoch": epoch, **avg_train}

        if epoch % val_every == 0:
            val_metrics = _evaluate(
                model, val_loader, criterion, device, epoch, f1_every,
            )
            logger.log_dict(val_metrics, step=epoch, prefix=f"{run_label}/val")

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

            pcc_improved = val_metrics["pcc"] > best_val_pcc
            if pcc_improved:
                best_val_pcc = val_metrics["pcc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_pcc": best_val_pcc,
                }, ckpt_dir / "best.pt")
                logger.info(f"  -> Best ckpt saved (val_pcc={best_val_pcc:.4f})")

            # 扩散模式 val_loss 恒为 0（DDIM 无 eps），改用 val_pcc 做 early stopping
            if cfg.use_diffusion:
                es_improved = pcc_improved
            else:
                val_loss = val_metrics.get("loss", float("inf"))
                es_improved = val_loss < best_val_loss
                if es_improved:
                    best_val_loss = val_loss

            if epoch >= early_stop_min_epoch:
                if es_improved:
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                    if early_stop_cnt >= cfg.early_stop_patience:
                        metric_name = "val_pcc" if cfg.use_diffusion else "val_loss"
                        logger.info(
                            f"  -> Early stop at epoch {epoch} "
                            f"({metric_name} no improvement for "
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
# Evaluation（val loop）
# =============================================================================

def _evaluate(model, loader, criterion, device, epoch, f1_every):
    model.eval()
    total_loss = 0.0
    all_pred, all_gt = [], []
    compute_f1 = (epoch % f1_every == 0)
    use_diff = getattr(model, "use_diffusion", False)

    with torch.no_grad():
        for batch in loader:
            radar    = batch["radar"].to(device)
            ecg_gt   = batch["ecg"].to(device)
            rpeak_gt = batch["rpeak"].to(device)
            peak_gts = _build_peak_gts_mmecg(batch, rpeak_gt, device)

            # In eval mode, diffusion model auto-triggers DDIM → ecg_pred
            ecg_pred, peak_preds = model(radar)
            if not torch.isfinite(ecg_pred).all():
                continue

            # For val loss use a dummy model_out compatible with both losses
            if use_diff:
                # Approximate diffusion loss as zero during val (DDIM has no eps)
                pass
            else:
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
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag",  type=str, default="mmecg_v1")
    parser.add_argument("--fold_idx", type=int, default=-1,
                        help="LOSO: 1-based fold (1~11); -1 = all folds")
    parser.add_argument("--protocol", type=str, default="loso",
                        choices=["loso", "samplewise"])
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--use_pam",  type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--use_emd",  type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--use_diffusion",
                        type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--balance_by", type=str, default=None,
                        choices=["subject", "class"],
                        help="Sampler weight strategy (default: config value)")
    parser.add_argument("--narrow_bandpass",
                        type=lambda x: x.lower() != "false", default=None,
                        help="Apply 0.8-3.5 Hz heartbeat bandpass to RCG")
    parser.add_argument("--diff_T",          type=int, default=None,
                        help="Diffusion steps T (default: config value, currently 1000)")
    parser.add_argument("--diff_ddim_steps", type=int, default=None,
                        help="DDIM inference steps (default: config value, currently 50)")
    parser.add_argument("--diff_hidden",     type=int, default=None,
                        help="Diffusion ResBlock hidden channels (default: config value)")
    parser.add_argument("--diff_n_blocks",   type=int, default=None,
                        help="Number of diffusion ResBlocks (default: config value)")
    parser.add_argument("--use_lag_aware_loss",
                        type=lambda x: x.lower() != "false", default=None,
                        help="Use small-window shift-invariant PCC/L1 waveform loss")
    parser.add_argument("--lag_max_ms", type=float, default=None,
                        help="Max lag window for lag-aware loss in milliseconds")
    parser.add_argument("--lambda_lag_pcc", type=float, default=None,
                        help="Weight for lag-aware PCC loss")
    parser.add_argument("--lambda_lag_l1", type=float, default=None,
                        help="Weight for lag-aware L1 loss")
    parser.add_argument("--lambda_zero_pcc", type=float, default=None,
                        help="Weight for zero-lag PCC anchor loss")
    parser.add_argument("--lambda_lag_penalty", type=float, default=None,
                        help="Weight for soft absolute-lag penalty")
    parser.add_argument("--lag_softmax_tau", type=float, default=None,
                        help="Temperature for differentiable lag penalty")
    parser.add_argument("--use_output_lag_align",
                        type=lambda x: x.lower() != "false", default=None,
                        help="Predict and apply per-segment scalar output lag")
    parser.add_argument("--output_lag_max_ms", type=float, default=None,
                        help="Max absolute predicted output lag in milliseconds")
    parser.add_argument("--lambda_output_lag_l1", type=float, default=None,
                        help="L1 regularization weight for normalized output lag")
    args = parser.parse_args()

    cfg = MMECGConfig()
    if args.epochs          is not None: cfg.epochs          = args.epochs
    if args.use_pam         is not None: cfg.use_pam         = args.use_pam
    if args.use_emd         is not None: cfg.use_emd         = args.use_emd
    if args.use_diffusion   is not None: cfg.use_diffusion   = args.use_diffusion
    if args.balance_by      is not None: cfg.balance_by      = args.balance_by
    if args.narrow_bandpass is not None: cfg.narrow_bandpass = args.narrow_bandpass
    if args.diff_T          is not None: cfg.diff_T          = args.diff_T
    if args.diff_ddim_steps is not None: cfg.diff_ddim_steps = args.diff_ddim_steps
    if args.diff_hidden     is not None: cfg.diff_hidden     = args.diff_hidden
    if args.diff_n_blocks   is not None: cfg.diff_n_blocks   = args.diff_n_blocks
    if args.use_lag_aware_loss is not None: cfg.use_lag_aware_loss = args.use_lag_aware_loss
    if args.lag_max_ms      is not None: cfg.lag_max_ms      = args.lag_max_ms
    if args.lambda_lag_pcc  is not None: cfg.lambda_lag_pcc  = args.lambda_lag_pcc
    if args.lambda_lag_l1   is not None: cfg.lambda_lag_l1   = args.lambda_lag_l1
    if args.lambda_zero_pcc is not None: cfg.lambda_zero_pcc = args.lambda_zero_pcc
    if args.lambda_lag_penalty is not None: cfg.lambda_lag_penalty = args.lambda_lag_penalty
    if args.lag_softmax_tau is not None: cfg.lag_softmax_tau = args.lag_softmax_tau
    if args.use_output_lag_align is not None: cfg.use_output_lag_align = args.use_output_lag_align
    if args.output_lag_max_ms is not None: cfg.output_lag_max_ms = args.output_lag_max_ms
    if args.lambda_output_lag_l1 is not None: cfg.lambda_output_lag_l1 = args.lambda_output_lag_l1

    print(f"MMECG training | exp_tag={args.exp_tag} | protocol={args.protocol}")
    print(f"use_pam={cfg.use_pam} | use_emd={cfg.use_emd} | epochs={cfg.epochs}")
    print(f"use_lag_aware_loss={cfg.use_lag_aware_loss} | lag_max_ms={cfg.lag_max_ms} | "
          f"lambda_zero_pcc={cfg.lambda_zero_pcc} | "
          f"lambda_lag_penalty={cfg.lambda_lag_penalty}")
    print(f"use_output_lag_align={cfg.use_output_lag_align} | "
          f"output_lag_max_ms={cfg.output_lag_max_ms} | "
          f"lambda_output_lag_l1={cfg.lambda_output_lag_l1}")

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=cfg.balanced_sampling,
        balance_by=cfg.balance_by,
        narrow_bandpass=cfg.narrow_bandpass,
    )

    if args.protocol == "samplewise":
        train_loader, val_loader, _ = build_samplewise_loaders_h5(
            sw_dir=cfg.samplewise_h5_dir, **loader_kwargs,
        )
        train_one_run(cfg, args.exp_tag, train_loader, val_loader,
                      run_label="samplewise", args=args)
    else:
        # LOSO: fold_idx is 1-based (1~11)
        folds = list(range(1, cfg.n_folds + 1)) if args.fold_idx == -1 else [args.fold_idx]
        print(f"LOSO folds: {folds}")

        for fold in folds:
            set_seed(42 + fold)
            train_loader, val_loader, _ = build_loso_loaders_h5(
                fold_idx=fold,
                loso_dir=cfg.loso_h5_dir,
                **loader_kwargs,
            )
            train_one_run(cfg, args.exp_tag, train_loader, val_loader,
                          run_label=f"fold_{fold:02d}", args=args)

    print(f"\nAll runs done. Results: {cfg.exp_dir}/{args.exp_tag}/")


if __name__ == "__main__":
    main()
