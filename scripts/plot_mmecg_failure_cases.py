"""
plot_mmecg_failure_cases.py — MMECG prediction failure analysis.

Creates:
  1. Representative GT-vs-pred ECG overlays for selected subject/scene groups.
  2. Subject-wise PCC bar plot.
  3. Subject x scene PCC heatmap.

Examples:
  python scripts/plot_mmecg_failure_cases.py \
    --exp_tag mmecg_reg_samplewise_subject --protocol samplewise

  python scripts/plot_mmecg_failure_cases.py \
    --exp_tag mmecg_reg_loso_subject --protocol loso --fold_idx 1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders_h5, build_samplewise_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.metrics import detect_rpeaks


FS = 200
WIN_LEN = 1600
T_AXIS = np.arange(WIN_LEN) / FS


def _run_label(protocol: str, fold_idx: int) -> str:
    return "samplewise" if protocol == "samplewise" else f"fold_{fold_idx:02d}"


def _load_saved_cfg(run_dir: Path) -> MMECGConfig:
    cfg = MMECGConfig()
    saved = json.loads((run_dir / "config.json").read_text())
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
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def _build_test_loader(protocol: str, fold_idx: int, cfg: MMECGConfig):
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
        narrow_bandpass=cfg.narrow_bandpass,
    )
    if protocol == "samplewise":
        _, _, loader = build_samplewise_loaders_h5(
            sw_dir=cfg.samplewise_h5_dir, **kwargs,
        )
    else:
        _, _, loader = build_loso_loaders_h5(
            fold_idx=fold_idx,
            loso_dir=cfg.loso_h5_dir,
            **kwargs,
        )
    return loader


@torch.no_grad()
def _collect_predictions(model, loader, device):
    preds, gts = [], []
    subjects, states, gt_rpeaks = [], [], []
    for batch in loader:
        radar = batch["radar"].to(device)
        pred, _ = model(radar)
        preds.append(pred.cpu().numpy()[:, 0])
        gts.append(batch["ecg"].numpy()[:, 0])
        subjects.extend(batch["subject"].tolist())
        states.extend(batch["state"].tolist())
        gt_rpeaks.extend(batch["r_idx"])
    return {
        "pred": np.concatenate(preds, axis=0),
        "gt": np.concatenate(gts, axis=0),
        "subject": np.asarray(subjects, dtype=int),
        "state": np.asarray(states, dtype=int),
        "gt_rpeaks": gt_rpeaks,
    }


def _pick_indices(seg_df: pd.DataFrame, max_groups: int = 6) -> list[tuple[str, int]]:
    """Pick representative best/median/worst samples for the most informative groups."""
    picks: list[tuple[str, int]] = []

    group_stats = (
        seg_df.groupby(["subject_id", "scene"])
        .agg(n=("pcc_raw", "size"), pcc=("pcc_raw", "mean"))
        .reset_index()
    )
    group_stats = group_stats[group_stats["n"] >= 10].copy()
    if group_stats.empty:
        order = seg_df["pcc_raw"].sort_values()
        return [
            ("Worst overall", int(order.index[0])),
            ("Median overall", int(order.index[len(order) // 2])),
            ("Best overall", int(order.index[-1])),
        ]

    # Take the three best and three worst subject-scene groups.
    selected = pd.concat([
        group_stats.sort_values("pcc", ascending=True).head(max_groups // 2),
        group_stats.sort_values("pcc", ascending=False).head(max_groups // 2),
    ]).drop_duplicates(["subject_id", "scene"])

    for _, row in selected.iterrows():
        mask = (
            (seg_df["subject_id"] == row["subject_id"])
            & (seg_df["scene"] == row["scene"])
        )
        sub = seg_df[mask].copy()
        sub = sub.sort_values("pcc_raw")
        median_row = sub.iloc[len(sub) // 2]
        label = (
            f"S{int(row['subject_id'])} {row['scene']} "
            f"group mean PCC={row['pcc']:.3f}"
        )
        picks.append((label, int(median_row["segment_id"])))

    return picks


def _plot_waveform_cases(picks, data, seg_df, out_path: Path, title: str):
    n = len(picks)
    fig_h = max(2.6 * n, 5)
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=12, y=0.995)
    for ax, (label, idx) in zip(axes, picks):
        pred = data["pred"][idx]
        gt = data["gt"][idx]
        row = seg_df[seg_df["segment_id"] == idx].iloc[0]
        pred_r = detect_rpeaks(pred, fs=FS)
        gt_r = np.asarray(data["gt_rpeaks"][idx], dtype=int)
        gt_r = gt_r[(gt_r >= 0) & (gt_r < WIN_LEN)]

        ax.plot(T_AXIS, gt, color="black", lw=1.15, label="GT ECG")
        ax.plot(T_AXIS, pred, color="#D43F3A", lw=1.15, ls="--", label="Pred ECG")
        if len(gt_r):
            ax.vlines(gt_r / FS, ymin=-0.05, ymax=1.05, color="black",
                      alpha=0.18, lw=0.8, label="GT R")
        if len(pred_r):
            ax.vlines(pred_r / FS, ymin=-0.05, ymax=1.05, color="#D43F3A",
                      alpha=0.18, lw=0.8, label="Pred R")

        ax.set_xlim(0, WIN_LEN / FS)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_ylabel("ECG")
        ax.set_title(
            f"{label} | seg={idx} | PCC={row.pcc_raw:.3f} | "
            f"RMSE={row.rmse_norm:.3f} | Rerr={row.r_peak_error_ms_mean:.1f}ms | "
            f"QMR={int(row.qualified_flag)}",
            fontsize=9,
            loc="left",
        )
        ax.legend(loc="upper right", ncol=4, fontsize=7, frameon=False)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_bar(seg_df: pd.DataFrame, out_path: Path, title: str):
    sub = (
        seg_df.groupby("subject_id")
        .agg(pcc=("pcc_raw", "mean"), qmr=("qualified_flag", "mean"), n=("pcc_raw", "size"))
        .reset_index()
        .sort_values("subject_id")
    )
    fig, ax1 = plt.subplots(figsize=(max(8, len(sub) * 0.75), 4.5))
    x = np.arange(len(sub))
    bars = ax1.bar(x, sub["pcc"], color="#3B75AF", edgecolor="white", linewidth=0.6)
    worst = int(sub["pcc"].argmin())
    best = int(sub["pcc"].argmax())
    bars[worst].set_color("#C0392B")
    bars[best].set_color("#2E8B57")
    ax1.axhline(sub["pcc"].mean(), color="#777", ls="--", lw=1,
                label=f"Mean PCC={sub['pcc'].mean():.3f}")
    ax1.set_ylabel("PCC")
    ax1.set_ylim(min(-0.1, sub["pcc"].min() - 0.05), min(1.0, sub["pcc"].max() + 0.12))
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{int(s)}\n(n={int(n)})" for s, n in zip(sub["subject_id"], sub["n"])])
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper left", frameon=False)

    ax2 = ax1.twinx()
    ax2.plot(x, sub["qmr"] * 100, color="#F39C12", marker="o", lw=1.4, label="QMR")
    ax2.set_ylabel("QMR (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", frameon=False)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(seg_df: pd.DataFrame, out_path: Path, title: str):
    pivot = seg_df.pivot_table(
        index="subject_id", columns="scene", values="pcc_raw", aggfunc="mean"
    )
    scene_order = [s for s in ["NB", "IB", "SP", "PE"] if s in pivot.columns]
    pivot = pivot[scene_order]

    fig, ax = plt.subplots(figsize=(max(6, len(scene_order) * 1.4), max(4, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"S{int(s)}" for s in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            text = "" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title)
    ax.set_xlabel("Scene")
    ax.set_ylabel("Subject")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Mean PCC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--protocol", choices=["samplewise", "loso"], default="samplewise")
    parser.add_argument("--fold_idx", type=int, default=1)
    args = parser.parse_args()

    run_label = _run_label(args.protocol, args.fold_idx)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / run_label
    result_dir = run_dir / "results"
    seg_path = result_dir / "segment_metrics.csv"
    if not seg_path.exists():
        raise FileNotFoundError(f"Missing segment metrics: {seg_path}. Run test_mmecg.py first.")

    out_dir = ROOT / "experiments_mmecg" / args.exp_tag / "figures" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_saved_cfg(run_dir)
    model, ckpt = _load_model(run_dir, cfg, device)
    loader = _build_test_loader(args.protocol, args.fold_idx, cfg)
    data = _collect_predictions(model, loader, device)
    seg_df = pd.read_csv(seg_path)

    title_base = (
        f"{args.exp_tag}/{run_label} | best epoch={ckpt.get('epoch', '?')} "
        f"val PCC={ckpt.get('val_pcc', float('nan')):.3f} | "
        f"test mean PCC={seg_df['pcc_raw'].mean():.3f}"
    )

    picks = _pick_indices(seg_df)
    _plot_waveform_cases(
        picks, data, seg_df,
        out_dir / "representative_waveforms.png",
        f"Representative prediction cases | {title_base}",
    )
    _plot_subject_bar(
        seg_df, out_dir / "subject_pcc_qmr.png",
        f"Subject-wise performance | {title_base}",
    )
    _plot_heatmap(
        seg_df, out_dir / "subject_scene_pcc_heatmap.png",
        f"Subject x scene PCC | {title_base}",
    )

    print(f"Saved figures to: {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}: {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
