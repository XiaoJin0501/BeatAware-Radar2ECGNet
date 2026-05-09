"""
plot_loso_predictions.py — LOSO test prediction visualization

每个 fold 一张图，3 行 subplot 各为 best/median/worst PCC 的 sample，
ECG GT (黑实线) vs Pred (红虚线) overlay。

用法：
    python scripts/plot_loso_predictions.py --exp_tag mmecg_reg_loso --fold_idx 1
    python scripts/plot_loso_predictions.py --exp_tag mmecg_reg_loso --fold_idx 2
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet


FS = 200
WIN_LEN = 1600
T_AXIS = np.arange(WIN_LEN) / FS  # 0..8s


def load_model(run_dir: Path, device: torch.device):
    cfg = MMECGConfig()
    cfg_path = run_dir / "config.json"
    saved = json.loads(cfg_path.read_text())
    for k in ("C", "d_state", "dropout", "use_pam", "use_emd", "emd_max_delay",
              "n_range_bins", "use_diffusion", "diff_T", "diff_ddim_steps",
              "diff_hidden", "diff_n_blocks", "narrow_bandpass"):
        if k in saved:
            setattr(cfg, k, saved[k])

    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C, d_state=cfg.d_state, dropout=cfg.dropout,
        use_pam=cfg.use_pam, use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
        use_diffusion=cfg.use_diffusion,
        diff_T=cfg.diff_T, diff_ddim_steps=cfg.diff_ddim_steps,
        diff_hidden=cfg.diff_hidden, diff_n_blocks=cfg.diff_n_blocks,
    ).to(device)

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt.get("epoch", "?"), ckpt.get("val_pcc", float("nan"))


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns (preds, gts, subjects) as numpy arrays / lists."""
    preds, gts, subs = [], [], []
    for batch in loader:
        radar = batch["radar"].to(device)
        ecg_gt = batch["ecg"]
        sub = batch["subject"].tolist()
        ecg_pred, _ = model(radar)
        preds.append(ecg_pred.cpu().numpy()[:, 0])
        gts.append(ecg_gt.numpy()[:, 0])
        subs.extend(sub)
    return np.concatenate(preds), np.concatenate(gts), subs


def per_sample_pcc(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """PCC for each sample (N,)."""
    pc = preds - preds.mean(axis=1, keepdims=True)
    gc = gts - gts.mean(axis=1, keepdims=True)
    num = (pc * gc).sum(axis=1)
    den = np.sqrt((pc ** 2).sum(axis=1) * (gc ** 2).sum(axis=1)) + 1e-8
    return num / den


def per_sample_rmse(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    return np.sqrt(((preds - gts) ** 2).mean(axis=1))


def plot_fold(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MMECGConfig()
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    if not run_dir.exists():
        sys.exit(f"Run dir not found: {run_dir}")

    print(f"[fold_{args.fold_idx:02d}] loading model + test data...")
    model, mcfg, epoch, val_pcc = load_model(run_dir, device)

    _, _, test_loader = build_loso_loaders_h5(
        fold_idx=args.fold_idx,
        loso_dir=cfg.loso_h5_dir,
        batch_size=16,
        num_workers=2,
        balanced_sampling=False,
        narrow_bandpass=mcfg.narrow_bandpass,
    )

    print(f"[fold_{args.fold_idx:02d}] running inference...")
    preds, gts, subs = collect_predictions(model, test_loader, device)
    pccs = per_sample_pcc(preds, gts)
    rmses = per_sample_rmse(preds, gts)
    print(f"  collected {len(preds)} segments, PCC range "
          f"[{pccs.min():.3f}, {pccs.max():.3f}], mean={pccs.mean():.3f}")

    # Pick representative samples
    order = np.argsort(pccs)
    worst_i = int(order[0])
    median_i = int(order[len(order) // 2])
    best_i = int(order[-1])
    picks = [
        ("Best",   best_i,   pccs[best_i],   rmses[best_i]),
        ("Median", median_i, pccs[median_i], rmses[median_i]),
        ("Worst",  worst_i,  pccs[worst_i],  rmses[worst_i]),
    ]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 7),
                             gridspec_kw={"hspace": 0.55})
    sub_id = subs[best_i]  # all same in LOSO test (one subject)
    val_pcc_str = f"{val_pcc:.4f}" if isinstance(val_pcc, (int, float)) else "n/a"
    fig.suptitle(
        f"LOSO fold_{args.fold_idx:02d}  |  test subject = {sub_id}  |  "
        f"best ckpt epoch {epoch}, val PCC {val_pcc_str}  |  "
        f"test mean PCC = {pccs.mean():.3f} (n={len(preds)})",
        fontsize=11, y=0.995,
    )

    for ax, (label, idx, pcc, rmse) in zip(axes, picks):
        ax.plot(T_AXIS, gts[idx],   color="black", lw=1.2, alpha=0.9, label="GT")
        ax.plot(T_AXIS, preds[idx], color="#E74C3C", lw=1.2, alpha=0.85,
                linestyle="--", label="Pred")
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Normalized ECG", fontsize=9)
        ax.set_title(
            f"{label}  —  segment {idx}   PCC = {pcc:.3f}   RMSE_norm = {rmse:.3f}",
            fontsize=10, loc="left",
        )
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig_dir = ROOT / "experiments_mmecg" / args.exp_tag / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"fold_{args.fold_idx:02d}_pred_vs_gt.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}  ({out.stat().st_size / 1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_tag", default="mmecg_reg_loso")
    p.add_argument("--fold_idx", type=int, required=True)
    plot_fold(p.parse_args())


if __name__ == "__main__":
    main()
