"""
Probe the output lag head for a trained MMECG checkpoint.

Prints predicted lag distribution and quick PCC/MAE on the first N validation
and test segments. This is intentionally lightweight and does not write files.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_samplewise_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet


def _pcc_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred.flatten(1)
    gt = gt.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    gt = gt - gt.mean(dim=1, keepdim=True)
    den = torch.sqrt((pred.square().sum(dim=1) * gt.square().sum(dim=1)).clamp_min(1e-12))
    return (pred * gt).sum(dim=1) / den


def _load_cfg(run_dir: Path) -> MMECGConfig:
    cfg = MMECGConfig()
    with open(run_dir / "config.json") as f:
        saved = json.load(f)
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _load_model(run_dir: Path, cfg: MMECGConfig, device: torch.device):
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
        use_output_lag_align=getattr(cfg, "use_output_lag_align", False),
        output_lag_max_samples=int(round(
            getattr(cfg, "output_lag_max_ms", 200.0) / 1000.0 * getattr(cfg, "fs", 200)
        )),
    ).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


@torch.no_grad()
def _probe(name: str, loader, model, device, max_segments: int, fs: float):
    lags, pccs, maes = [], [], []
    seen = 0
    for batch in loader:
        radar = batch["radar"].to(device)
        gt = batch["ecg"].to(device)
        pred, _ = model(radar)
        if model.last_output_lag_samples is not None:
            lags.append(model.last_output_lag_samples.detach().cpu().numpy())
        pccs.append(_pcc_batch(pred, gt).detach().cpu().numpy())
        maes.append((pred - gt).abs().mean(dim=(1, 2)).detach().cpu().numpy())
        seen += radar.shape[0]
        if seen >= max_segments:
            break

    pccs_np = np.concatenate(pccs)
    maes_np = np.concatenate(maes)
    print(f"\n{name}: n={len(pccs_np)}")
    print(f"  pcc mean/median: {pccs_np.mean():.4f} / {np.median(pccs_np):.4f}")
    print(f"  mae mean/median: {maes_np.mean():.4f} / {np.median(maes_np):.4f}")
    if lags:
        lags_np = np.concatenate(lags)
        lag_ms = lags_np / fs * 1000.0
        print(
            "  lag samples mean/median/std/min/max: "
            f"{lags_np.mean():.2f} / {np.median(lags_np):.2f} / "
            f"{lags_np.std():.2f} / {lags_np.min():.2f} / {lags_np.max():.2f}"
        )
        print(
            "  lag ms mean/median/mean_abs: "
            f"{lag_ms.mean():.1f} / {np.median(lag_ms):.1f} / {np.abs(lag_ms).mean():.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--max_segments", type=int, default=256)
    args = parser.parse_args()

    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / "samplewise"
    cfg = _load_cfg(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(run_dir, cfg, device)
    print(f"checkpoint epoch={ckpt.get('epoch')} val_pcc={ckpt.get('val_pcc')}")
    print(f"use_output_lag_align={getattr(cfg, 'use_output_lag_align', False)}")

    _, val_loader, test_loader = build_samplewise_loaders_h5(
        sw_dir=cfg.samplewise_h5_dir,
        batch_size=32,
        num_workers=0,
        balanced_sampling=False,
        narrow_bandpass=cfg.narrow_bandpass,
    )
    _probe("val", val_loader, model, device, args.max_segments, cfg.fs)
    _probe("test", test_loader, model, device, args.max_segments, cfg.fs)


if __name__ == "__main__":
    main()
