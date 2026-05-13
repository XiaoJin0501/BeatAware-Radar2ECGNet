"""
AirECG model on MMECG, using AirECG's source training hyperparameters.

Kept from AirECG:
  - AirECG DiT model implementation from /home/qhh2237/Projects/AirECG
  - AirECG Gaussian diffusion training objective (MSE noise prediction)
  - AirECG 32x32 representation, so MMECG 1600-point windows are resampled to
    1024 for training/inference, then predictions are resampled back to 1600.

Changed for this project:
  - MMECG LOSO or fixed-shot target calibration split
  - MMECG four-level metrics from scripts/test_mmecg.py

Kept from AirECG source training where possible:
  - AdamW lr=1e-4, weight_decay=0
  - constant LR (no scheduler)
  - no gradient clipping
  - EMA model
  - diffusion MSE training loss

Important: AirECG's paper uses historical ECG as calibration guidance. This
script supports both source-only strict LOSO guidance and a fair few-shot
protocol where the same target calibration windows used by our model are also
used as AirECG's historical ECG guidance pool. Test ECG is never used as
guidance.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
AIR_ECG_ROOT = Path("/home/qhh2237/Projects/AirECG")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AIR_ECG_ROOT))

from configs.mmecg_config import MMECGConfig
from scripts.test_mmecg import _evaluate_segment, _json_default, _loso_summary
from src.data.mmecg_dataset import MMECGWindowedH5Dataset, _split_calibration_indices
from src.utils.metrics import compute_all_metrics
from src.utils.metrics import summarize_global_metrics, summarize_subject_metrics

from diffusion import create_diffusion
from models import AirECG_model


AIR_LEN = 1024
MMECG_LEN = 1600


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


class AirECGMMECGDataset(Dataset):
    def __init__(
        self,
        base: MMECGWindowedH5Dataset,
        ref_base: MMECGWindowedH5Dataset,
        include_peak_indices: bool,
    ):
        self.base = base
        self.ref_base = ref_base
        self.include_peak_indices = include_peak_indices

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _resize_1d(x: torch.Tensor, length: int) -> torch.Tensor:
        return F.interpolate(
            x.unsqueeze(0), size=length, mode="linear", align_corners=False
        ).squeeze(0)

    @staticmethod
    def _to_img(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], 32, 32)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        ref_item = self.ref_base[idx % len(self.ref_base)]
        radar_1024 = self._resize_1d(item["radar"].float(), AIR_LEN)
        ecg_1024 = self._resize_1d(item["ecg"].float(), AIR_LEN) * 2 - 1
        ref_1024 = self._resize_1d(ref_item["ecg"].float(), AIR_LEN) * 2 - 1
        out = {
            "mmwave": self._to_img(radar_1024).float(),
            "ecg_img": self._to_img(ecg_1024).float(),
            "ref_img": self._to_img(ref_1024).float(),
            "ecg_1600": item["ecg"].float(),
            "subject": item["subject"],
            "state": item["state"],
        }
        if self.include_peak_indices:
            for key in ("r_idx", "q_idx", "s_idx", "t_idx", "delin_valid"):
                out[key] = item[key]
        return out


def collate_air(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ("mmwave", "ecg_img", "ref_img", "ecg_1600"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("subject", "state"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    for key in ("r_idx", "q_idx", "s_idx", "t_idx"):
        if key in batch[0]:
            out[key] = [b[key] for b in batch]
    if "delin_valid" in batch[0]:
        out["delin_valid"] = torch.tensor(
            [bool(b["delin_valid"]) for b in batch], dtype=torch.bool
        )
    return out


def subject_sampler(base: MMECGWindowedH5Dataset) -> WeightedRandomSampler:
    subjects = base._subj.astype(np.int64)
    unique, counts = np.unique(subjects, return_counts=True)
    inv = {int(s): 1.0 / float(c) for s, c in zip(unique, counts)}
    weights = torch.tensor([inv[int(s)] for s in subjects], dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_loaders(
    cfg: MMECGConfig,
    fold: int,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool,
    protocol: str = "loso",
    calib_n_train: int | None = None,
    calib_n_val: int | None = None,
    calib_seed: int = 42,
):
    fold_dir = Path(cfg.loso_h5_dir) / f"fold_{fold:02d}"
    ds_kwargs = dict(
        narrow_bandpass=False,
        target_norm="minmax",
        topk_bins=None,
        topk_method="energy",
    )
    train_base = MMECGWindowedH5Dataset(fold_dir / "train.h5", include_peak_indices=False, **ds_kwargs)

    if protocol == "loso":
        val_base = MMECGWindowedH5Dataset(fold_dir / "val.h5", include_peak_indices=False, **ds_kwargs)
        test_base = MMECGWindowedH5Dataset(fold_dir / "test.h5", include_peak_indices=True, **ds_kwargs)
        train_data = train_base
        ref_data = train_base
        val_data = val_base
        test_data = test_base
    elif protocol == "loso_calib":
        # Same fixed-count personalization split as our few-shot experiment.
        # The target-subject calibration windows are used both as labeled
        # training samples and as AirECG historical ECG guidance.
        test_plain = MMECGWindowedH5Dataset(fold_dir / "test.h5", include_peak_indices=False, **ds_kwargs)
        test_full = MMECGWindowedH5Dataset(fold_dir / "test.h5", include_peak_indices=True, **ds_kwargs)
        calib_idx, calib_val_idx, eval_idx = _split_calibration_indices(
            test_plain._state,
            calib_ratio=0.4,
            calib_val_ratio=0.1,
            calib_n_train=calib_n_train,
            calib_n_val=calib_n_val,
            seed=calib_seed + fold,
        )
        ref_data = Subset(test_plain, calib_idx.tolist())
        train_data = ConcatDataset([train_base, ref_data])
        val_data = Subset(test_plain, calib_val_idx.tolist())
        test_data = Subset(test_full, eval_idx.tolist())
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    train_ds = AirECGMMECGDataset(train_data, ref_data, include_peak_indices=False)
    val_ds = AirECGMMECGDataset(val_data, ref_data, include_peak_indices=False)
    test_ds = AirECGMMECGDataset(test_data, ref_data, include_peak_indices=True)

    sampler = subject_sampler(train_base) if (balanced_sampling and protocol == "loso") else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_air,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_air,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_air,
    )
    return train_loader, val_loader, test_loader


def air_loss(model, diffusion, batch: dict, device: torch.device) -> torch.Tensor:
    ecg = batch["ecg_img"].to(device)
    mmwave = batch["mmwave"].to(device)
    ref = batch["ref_img"].to(device)
    t = torch.randint(0, diffusion.num_timesteps, (ecg.shape[0],), device=device)
    return diffusion.training_losses(model, ecg, t, {"y1": mmwave, "y2": ref})["loss"].mean()


@torch.no_grad()
def evaluate_val_loss(model, diffusion, loader, device: torch.device, max_batches: int) -> float:
    model.eval()
    vals = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        vals.append(float(air_loss(model, diffusion, batch, device).item()))
    model.train()
    return float(np.mean(vals)) if vals else float("inf")


@torch.no_grad()
def sample_ecg(model, diffusion, batch: dict, device: torch.device) -> np.ndarray:
    n = batch["ecg_img"].shape[0]
    z = torch.randn(n, 1, 32, 32, device=device)
    samples = diffusion.p_sample_loop(
        model.forward,
        z.shape,
        z,
        clip_denoised=True,
        model_kwargs={
            "y1": batch["mmwave"].to(device),
            "y2": batch["ref_img"].to(device),
        },
        progress=False,
        device=device,
    )
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1], align with ecg_1600 metrics range
    samples = samples.reshape(n, 1, AIR_LEN)
    return F.interpolate(samples, size=MMECG_LEN, mode="linear", align_corners=False).cpu().numpy()[:, 0]


@torch.no_grad()
def evaluate_val_metrics(
    model,
    train_diffusion,
    sample_diffusion,
    loader,
    device: torch.device,
    epoch: int,
    max_batches: int,
    f1_every: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    preds = []
    gts = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(float(air_loss(model, train_diffusion, batch, device).item()))
        pred_np = sample_ecg(model, sample_diffusion, batch, device)
        preds.append(torch.from_numpy(pred_np[:, None, :]))
        gts.append(batch["ecg_1600"])
    model.train()

    if not preds:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "pcc": float("nan"),
            "prd": float("nan"),
            "rpeak_f1": float("nan"),
            "loss": float("nan"),
        }

    pred = torch.cat(preds, dim=0)
    gt = torch.cat(gts, dim=0)
    metrics = compute_all_metrics(
        pred,
        gt,
        compute_f1=(epoch % f1_every == 0),
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float("inf")
    return metrics


def train_fold(args, cfg: MMECGConfig, fold: int) -> None:
    set_seed(args.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.exp_dir) / args.exp_tag / f"fold_{fold:02d}"
    ckpt_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_loaders(
        cfg,
        fold,
        args.batch_size,
        args.num_workers,
        balanced_sampling=args.balanced_sampling,
        protocol=args.protocol,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
    )
    model = AirECG_model(input_size=32, mm_channels=50).to(device)
    ema = deepcopy(model).to(device).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    diffusion = create_diffusion(timestep_respacing="")
    sample_diffusion = create_diffusion(str(args.num_sampling_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg_dict = vars(args).copy()
    reference_guidance = (
        "source-train-ECG-only, no target-subject ECG"
        if args.protocol == "loso"
        else (
            f"target calibration ECG guidance only "
            f"({args.calib_n_train} train + {args.calib_n_val} val split; no test ECG)"
        )
    )
    cfg_dict.update(
        fold=fold,
        protocol=args.protocol,
        model_source=str(AIR_ECG_ROOT),
        model_name="AirECG_model",
        loss="AirECG GaussianDiffusion MSE noise-prediction loss",
        optimizer="AdamW",
        scheduler="constant",
        balanced_sampling=args.balanced_sampling,
        balance_by=("subject" if args.balanced_sampling else "none"),
        target_norm="minmax",
        mm_channels=50,
        air_len=AIR_LEN,
        metric_len=MMECG_LEN,
        reference_guidance=reference_guidance,
        num_params=sum(p.numel() for p in model.parameters()),
    )
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))

    history = []
    best_val_pcc = -float("inf")
    best_val_loss = float("inf")
    best_val_score = -float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    update_ema(ema, model, decay=0.0)

    with open(log_dir / "train.log", "w") as lf:
        def log(msg: str) -> None:
            print(msg, flush=True)
            lf.write(msg + "\n")
            lf.flush()

        log(f"AirECG-on-MMECG {args.protocol} fold_{fold:02d}")
        log(f"Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} Params={cfg_dict['num_params']:,}")
        log(f"Reference guidance: {reference_guidance}")
        log(
            f"epochs={args.epochs} patience={args.early_stop_patience} "
            f"lr={args.lr} wd={args.weight_decay} "
            f"balanced_sampling={args.balanced_sampling} grad_clip={args.grad_clip}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            losses = []
            for batch in train_loader:
                loss = air_loss(model, diffusion, batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                update_ema(ema, model, decay=args.ema_decay)
                losses.append(float(loss.item()))

            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            if epoch % args.val_every == 0:
                val_metrics = evaluate_val_metrics(
                    ema,
                    diffusion,
                    sample_diffusion,
                    val_loader,
                    device,
                    epoch=epoch,
                    max_batches=args.val_batches,
                    f1_every=args.f1_every,
                )
                val_f1 = val_metrics.get("rpeak_f1", float("nan"))
                score_f1 = 0.0 if not np.isfinite(val_f1) else val_f1
                val_score = (
                    val_metrics.get("pcc", 0.0)
                    - val_metrics.get("rmse", 0.0)
                    + 0.10 * score_f1
                )
                val_metrics["composite_score"] = float(val_score)
                row.update({f"val_{k}": v for k, v in val_metrics.items()})

                ckpt_payload = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "val_pcc": val_metrics.get("pcc"),
                    "val_rmse": val_metrics.get("rmse"),
                    "val_mae": val_metrics.get("mae"),
                    "val_rpeak_f1": val_metrics.get("rpeak_f1"),
                    "val_loss": val_metrics.get("loss"),
                    "val_composite_score": val_score,
                    "config": cfg_dict,
                }

                pcc_improved = val_metrics["pcc"] > best_val_pcc
                if pcc_improved:
                    best_val_pcc = val_metrics["pcc"]
                    best_epoch = epoch
                    no_improve_epochs = 0
                    torch.save(ckpt_payload, ckpt_dir / "best.pt")
                    torch.save(ckpt_payload, ckpt_dir / "best_pcc.pt")
                else:
                    if epoch >= args.min_epochs:
                        no_improve_epochs += args.val_every

                val_loss = val_metrics.get("loss", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ckpt_payload, ckpt_dir / "best_loss.pt")
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(ckpt_payload, ckpt_dir / "best_composite.pt")

                log(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"train={row['train_loss']:.4f} | "
                    f"val_mae={val_metrics['mae']:.4f} | "
                    f"val_rmse={val_metrics['rmse']:.4f} | "
                    f"val_pcc={val_metrics['pcc']:.4f} | "
                    f"val_prd={val_metrics.get('prd', float('nan')):.2f} | "
                    f"val_f1={val_f1:.4f} | "
                    f"val_loss={val_metrics.get('loss', float('nan')):.4f} | "
                    f"val_score={val_score:.4f} | "
                    f"lr={args.lr:.2e} | "
                    f"{time.time()-t0:.1f}s"
                )
                if epoch >= args.min_epochs and no_improve_epochs >= args.early_stop_patience:
                    log(
                        f"  -> Early stop at epoch {epoch} "
                        f"(val_pcc no improvement for {args.early_stop_patience} epochs)"
                    )
                    history.append(row)
                    break
            else:
                log(
                    f"Epoch {epoch:03d}/{args.epochs} train_loss={row['train_loss']:.4f} "
                    f"lr={args.lr:.2e} time={time.time()-t0:.1f}s"
                )
            history.append(row)

    (result_dir / "train_history.json").write_text(json.dumps(history, indent=2))


def test_fold(args, cfg: MMECGConfig, fold: int) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.exp_dir) / args.exp_tag / f"fold_{fold:02d}"
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    _, _, test_loader = build_loaders(
        cfg,
        fold,
        args.eval_batch_size,
        args.num_workers,
        balanced_sampling=False,
        protocol=args.protocol,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
    )

    model = AirECG_model(input_size=32, mm_channels=50).to(device).eval()
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    diffusion = create_diffusion(str(args.num_sampling_steps))

    rows = []
    beat_rows = []
    seg_id = 0
    print(f"\nTesting AirECG {args.protocol} fold_{fold:02d} N={len(test_loader.dataset)} ckpt_epoch={ckpt.get('epoch')}")
    for batch in test_loader:
        pred = sample_ecg(model, diffusion, batch, device)
        gt = batch["ecg_1600"].numpy()[:, 0]
        for i in range(pred.shape[0]):
            row, beats = _evaluate_segment(
                seg_id=seg_id,
                pred_1d=pred[i],
                gt_1d=gt[i],
                gt_r=batch["r_idx"][i],
                gt_q=batch["q_idx"][i],
                gt_s=batch["s_idx"][i],
                gt_t=batch["t_idx"][i],
                subject_id=int(batch["subject"][i].item()),
                state_code=int(batch["state"][i].item()),
                delin_valid=bool(batch["delin_valid"][i].item()),
            )
            rows.append(row)
            beat_rows.extend(beats)
            seg_id += 1
        if seg_id % 200 < args.eval_batch_size:
            print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    seg_df = pd.DataFrame(rows)
    beat_df = pd.DataFrame(beat_rows)
    subj_df = summarize_subject_metrics(rows)
    global_dict = summarize_global_metrics(rows)
    seg_df.to_csv(result_dir / "segment_metrics.csv", index=False)
    beat_df.to_csv(result_dir / "beat_metrics.csv", index=False)
    subj_df.to_csv(result_dir / "subject_summary.csv", index=False)
    with open(result_dir / "global_summary.json", "w") as jf:
        json.dump(global_dict, jf, indent=2, default=_json_default)
    print(
        f"  PCC={global_dict.get('pcc_raw_mean', float('nan')):.4f} "
        f"RMSE={global_dict.get('rmse_norm_mean', float('nan')):.4f} "
        f"R2={global_dict.get('r2_mean', float('nan')):.4f} "
        f"QMR={global_dict.get('qualified_monitoring_rate', float('nan')):.1f}%"
    )
    return seg_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", default="mmecg_airecg_calib40v10_source")
    parser.add_argument("--protocol", choices=["loso", "loso_calib"], default="loso_calib")
    parser.add_argument("--fold_idx", type=int, default=1)
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Optional project-side sampler. Off by default to respect AirECG source code.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--f1_every", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--calib_n_train", type=int, default=40)
    parser.add_argument("--calib_n_val", type=int, default=10)
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = MMECGConfig()
    folds = list(range(1, cfg.n_folds + 1)) if args.fold_idx == -1 else [args.fold_idx]
    if args.mode in ("train", "train_test"):
        for fold in folds:
            train_fold(args, cfg, fold)
    if args.mode in ("test", "train_test"):
        all_seg = [test_fold(args, cfg, fold) for fold in folds]
        if len(all_seg) > 1:
            _loso_summary(all_seg, Path(cfg.exp_dir) / args.exp_tag / "loso_summary")


if __name__ == "__main__":
    main()
