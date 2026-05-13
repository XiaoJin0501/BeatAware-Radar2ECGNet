"""
Visualize strict LOSO vs supervised subject calibration on MMECG.

This script compares two trained fold checkpoints on the same calibration
evaluation subset. It also plots the headline test summaries already produced by
scripts/test_mmecg.py.

Example:
  python scripts/plot_mmecg_calibration_comparison.py \
    --strict_exp mmecg_reg_loso_slim \
    --calib_exp mmecg_reg_loso_calib40v10_slim_fold01 \
    --fold_idx 1 \
    --calib_ratio 0.4 \
    --calib_val_ratio 0.1
"""

from __future__ import annotations

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
from src.data.mmecg_dataset import build_loso_calibration_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.metrics import detect_rpeaks


FS = 200
WIN_LEN = 1600
STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _saved_cfg(run_dir: Path) -> MMECGConfig:
    cfg = MMECGConfig()
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        saved = _read_json(cfg_path)
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def _loader_overrides(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    saved = _read_json(cfg_path)
    topk = saved.get("topk_bins", 0)
    return {
        "narrow_bandpass": bool(saved.get("narrow_bandpass", False)),
        "target_norm": saved.get("target_norm", "minmax"),
        "topk_method": saved.get("topk_method", "energy"),
        "topk_bins": topk if (topk and topk > 0) else None,
    }


def _load_model(run_dir: Path, device: torch.device) -> BeatAwareRadar2ECGNet:
    cfg = _saved_cfg(run_dir)
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
            getattr(cfg, "output_lag_max_ms", 200.0) / 1000.0 * getattr(cfg, "fs", FS)
        )),
        fmcw_selector=getattr(cfg, "fmcw_selector", "se"),
        fmcw_topk=getattr(cfg, "fmcw_topk", 10),
        fmcw_tau=getattr(cfg, "fmcw_tau_final", 0.1),
    ).to(device)

    ckpt_path = run_dir / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"[{run_dir.name}] dropped {len(unexpected)} unexpected keys")
    if missing:
        print(f"[{run_dir.name}] missing {len(missing)} keys")
    print(
        f"Loaded {run_dir}: epoch={ckpt.get('epoch', '?')} "
        f"val_pcc={ckpt.get('val_pcc', float('nan')):.4f}"
    )
    model.eval()
    return model


@torch.no_grad()
def _collect(model: BeatAwareRadar2ECGNet, loader, device: torch.device) -> dict:
    pred, gt, subject, state, gt_r = [], [], [], [], []
    for batch in loader:
        radar = batch["radar"].to(device)
        y, _ = model(radar)
        pred.append(y.cpu().numpy()[:, 0])
        gt.append(batch["ecg"].numpy()[:, 0])
        subject.extend(batch["subject"].tolist())
        state.extend(batch["state"].tolist())
        gt_r.extend(batch["r_idx"])
    return {
        "pred": np.concatenate(pred, axis=0),
        "gt": np.concatenate(gt, axis=0),
        "subject": np.asarray(subject, dtype=int),
        "state": np.asarray(state, dtype=int),
        "gt_r": gt_r,
    }


def _wave_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pcc, rmse, mae, r2 = [], [], [], []
    for p, g in zip(pred, gt):
        p = np.asarray(p, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        if np.std(p) < 1e-8 or np.std(g) < 1e-8:
            corr = np.nan
        else:
            corr = float(np.corrcoef(p, g)[0, 1])
        err = p - g
        mse = float(np.mean(err ** 2))
        denom = float(np.sum((g - np.mean(g)) ** 2))
        pcc.append(corr)
        rmse.append(float(np.sqrt(mse)))
        mae.append(float(np.mean(np.abs(err))))
        r2.append(float(1.0 - np.sum(err ** 2) / denom) if denom > 1e-12 else np.nan)
    return {
        "pcc": np.asarray(pcc, dtype=float),
        "rmse": np.asarray(rmse, dtype=float),
        "mae": np.asarray(mae, dtype=float),
        "r2": np.asarray(r2, dtype=float),
    }


def _mean(x: np.ndarray) -> float:
    return float(np.nanmean(x))


def _plot_headline_summary(strict_summary: dict, calib_summary: dict, out_path: Path) -> None:
    metrics = [
        ("PCC", "pcc_raw_mean", "higher"),
        ("R2", "r2_mean", "higher"),
        ("F1@150ms", "average_f1_150ms_mean", "higher"),
        ("QMR (%)", "qualified_monitoring_rate", "higher"),
        ("RMSE", "rmse_norm_mean", "lower"),
        ("R error (ms)", "r_peak_error_ms_mean_mean", "lower"),
        ("RR error (ms)", "rr_interval_error_ms_mean_mean", "lower"),
    ]
    labels = [m[0] for m in metrics]
    strict = [strict_summary.get(k, np.nan) for _, k, _ in metrics]
    calib = [calib_summary.get(k, np.nan) for _, k, _ in metrics]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.2))
    for ax, label, s, c, (_, _, direction) in zip(axes, labels, strict, calib, metrics):
        colors = ["#7A8793", "#2176AE"] if direction == "higher" else ["#7A8793", "#D1495B"]
        ax.bar([0, 1], [s, c], color=colors, width=0.68)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Strict\nLOSO", "40/10/50\nCalib"], fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.22)
        top = np.nanmax([s, c])
        bottom = np.nanmin([s, c])
        if np.isfinite(top):
            pad = max(abs(top - bottom) * 0.18, abs(top) * 0.08, 0.05)
            ax.set_ylim(bottom - pad if bottom < 0 else 0, top + pad)
        for x, y in enumerate([s, c]):
            if np.isfinite(y):
                ax.text(x, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Fold 01 Test Summary: Strict Zero-Shot LOSO vs Supervised Subject Calibration",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_same_subset_distributions(strict_m: dict, calib_m: dict, out_path: Path) -> None:
    panels = [
        ("PCC", "pcc", "higher"),
        ("RMSE", "rmse", "lower"),
        ("MAE", "mae", "lower"),
        ("R2", "r2", "higher"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, key, _) in zip(axes, panels):
        vals = [strict_m[key], calib_m[key]]
        bp = ax.boxplot(
            vals,
            tick_labels=["Strict\nLOSO", "40/10/50\nCalib"],
            patch_artist=True,
            showfliers=False,
            widths=0.55,
        )
        for patch, color in zip(bp["boxes"], ["#9AA6B2", "#61A5C2"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        for i, arr in enumerate(vals, start=1):
            jitter = np.linspace(-0.08, 0.08, len(arr))
            ax.scatter(np.full(len(arr), i) + jitter, arr, s=10, alpha=0.35, color="#333333")
            ax.text(i, np.nanmean(arr), f"mean={np.nanmean(arr):.3f}", ha="center",
                    va="bottom", fontsize=8, color="#111111")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Same Calibration Eval Subset: Segment-Level Distribution", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_scene_pcc(strict_m: dict, calib_m: dict, state: np.ndarray, out_path: Path) -> None:
    scenes = [s for s in ["NB", "IB", "SP", "PE"] if np.any([STATE_NAMES.get(x) == s for x in state])]
    x = np.arange(len(scenes))
    strict_vals, calib_vals = [], []
    for scene in scenes:
        mask = np.asarray([STATE_NAMES.get(x) == scene for x in state], dtype=bool)
        strict_vals.append(_mean(strict_m["pcc"][mask]))
        calib_vals.append(_mean(calib_m["pcc"][mask]))

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.36
    ax.bar(x - w / 2, strict_vals, width=w, color="#9AA6B2", label="Strict LOSO")
    ax.bar(x + w / 2, calib_vals, width=w, color="#2176AE", label="40/10/50 Calib")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("PCC")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    ax.set_title("Same Eval Subset: PCC by MMECG Scene")
    for i, (s, c) in enumerate(zip(strict_vals, calib_vals)):
        ax.text(i - w / 2, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pick_waveform_cases(strict_m: dict, calib_m: dict) -> list[int]:
    pcc_s = strict_m["pcc"]
    pcc_c = calib_m["pcc"]
    gain = pcc_c - pcc_s
    candidates = [
        int(np.nanargmax(gain)),
        int(np.nanargmin(pcc_s)),
        int(np.nanargmin(np.abs(pcc_c - np.nanmedian(pcc_c)))),
        int(np.nanargmax(pcc_c)),
    ]
    unique = []
    for idx in candidates:
        if idx not in unique:
            unique.append(idx)
    return unique[:4]


def _plot_waveforms(
    strict_data: dict,
    calib_data: dict,
    strict_m: dict,
    calib_m: dict,
    out_path: Path,
) -> None:
    idxs = _pick_waveform_cases(strict_m, calib_m)
    t = np.arange(WIN_LEN) / FS
    fig, axes = plt.subplots(len(idxs), 1, figsize=(14, 2.8 * len(idxs)), sharex=True)
    if len(idxs) == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        gt = calib_data["gt"][idx]
        strict_pred = strict_data["pred"][idx]
        calib_pred = calib_data["pred"][idx]
        scene = STATE_NAMES.get(int(calib_data["state"][idx]), "UNK")
        subj = int(calib_data["subject"][idx])
        gt_r = np.asarray(calib_data["gt_r"][idx], dtype=int)
        gt_r = gt_r[(gt_r >= 0) & (gt_r < WIN_LEN)]
        strict_r = detect_rpeaks(strict_pred, fs=FS)
        calib_r = detect_rpeaks(calib_pred, fs=FS)

        ax.plot(t, gt, color="#111111", lw=1.35, label="GT ECG")
        ax.plot(t, strict_pred, color="#9AA6B2", lw=1.1, ls="--", label="Strict LOSO")
        ax.plot(t, calib_pred, color="#D1495B", lw=1.1, label="40/10/50 Calib")
        if len(gt_r):
            ax.vlines(gt_r / FS, ymin=-0.06, ymax=1.06, color="#111111", alpha=0.16, lw=0.8)
        if len(strict_r):
            ax.scatter(strict_r / FS, strict_pred[strict_r], s=16, color="#9AA6B2", alpha=0.75)
        if len(calib_r):
            ax.scatter(calib_r / FS, calib_pred[calib_r], s=16, color="#D1495B", alpha=0.75)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlim(0, WIN_LEN / FS)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_ylabel("ECG")
        ax.set_title(
            f"S{subj} {scene} eval idx={idx} | "
            f"PCC strict={strict_m['pcc'][idx]:.3f}, calib={calib_m['pcc'][idx]:.3f} | "
            f"RMSE strict={strict_m['rmse'][idx]:.3f}, calib={calib_m['rmse'][idx]:.3f}",
            fontsize=9,
            loc="left",
        )
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Representative Waveforms on the Same Calibration Eval Subset", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    out_path: Path,
    strict_summary: dict,
    calib_summary: dict,
    strict_m: dict,
    calib_m: dict,
    figures: list[Path],
) -> None:
    lines = [
        "# MMECG Fold 01 Calibration Visualization Summary",
        "",
        "## Headline Test Results",
        "",
        "| Protocol | Test subset | PCC | RMSE | R2 | QMR | R error | RR error | F1@150ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Strict LOSO | {strict_summary.get('num_segments', 'NA')} | "
            f"{strict_summary.get('pcc_raw_mean', float('nan')):.4f} | "
            f"{strict_summary.get('rmse_norm_mean', float('nan')):.4f} | "
            f"{strict_summary.get('r2_mean', float('nan')):.4f} | "
            f"{strict_summary.get('qualified_monitoring_rate', float('nan')):.1f}% | "
            f"{strict_summary.get('r_peak_error_ms_mean_mean', float('nan')):.1f} ms | "
            f"{strict_summary.get('rr_interval_error_ms_mean_mean', float('nan')):.1f} ms | "
            f"{strict_summary.get('average_f1_150ms_mean', float('nan')):.4f} |"
        ),
        (
            f"| LOSO + 40/10/50 supervised calibration | {calib_summary.get('num_segments', 'NA')} | "
            f"{calib_summary.get('pcc_raw_mean', float('nan')):.4f} | "
            f"{calib_summary.get('rmse_norm_mean', float('nan')):.4f} | "
            f"{calib_summary.get('r2_mean', float('nan')):.4f} | "
            f"{calib_summary.get('qualified_monitoring_rate', float('nan')):.1f}% | "
            f"{calib_summary.get('r_peak_error_ms_mean_mean', float('nan')):.1f} ms | "
            f"{calib_summary.get('rr_interval_error_ms_mean_mean', float('nan')):.1f} ms | "
            f"{calib_summary.get('average_f1_150ms_mean', float('nan')):.4f} |"
        ),
        "",
        "## Same 50% Evaluation Subset",
        "",
        "| Model checkpoint | PCC | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Strict LOSO checkpoint | {_mean(strict_m['pcc']):.4f} | "
            f"{_mean(strict_m['rmse']):.4f} | {_mean(strict_m['mae']):.4f} | "
            f"{_mean(strict_m['r2']):.4f} |"
        ),
        (
            f"| Calibration checkpoint | {_mean(calib_m['pcc']):.4f} | "
            f"{_mean(calib_m['rmse']):.4f} | {_mean(calib_m['mae']):.4f} | "
            f"{_mean(calib_m['r2']):.4f} |"
        ),
        "",
        "Figures:",
    ]
    for fig in figures:
        lines.append(f"- `{fig.name}`")
    lines.append("")
    lines.append(
        "Note: the calibration result is supervised subject calibration, not strict "
        "zero-shot LOSO. It should be reported as a personalization/adaptation setting."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict_exp", default="mmecg_reg_loso_slim")
    parser.add_argument("--calib_exp", default="mmecg_reg_loso_calib40v10_slim_fold01")
    parser.add_argument("--fold_idx", type=int, default=1)
    parser.add_argument("--calib_ratio", type=float, default=0.4)
    parser.add_argument("--calib_val_ratio", type=float, default=0.1)
    parser.add_argument("--calib_n_train", type=int, default=None)
    parser.add_argument("--calib_n_val", type=int, default=None)
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_arrays", action="store_true")
    args = parser.parse_args()

    cfg = MMECGConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    strict_run = Path(cfg.exp_dir) / args.strict_exp / f"fold_{args.fold_idx:02d}"
    calib_run = Path(cfg.exp_dir) / args.calib_exp / f"fold_{args.fold_idx:02d}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.exp_dir) / "calibration_figures" / "fold_01"
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_summary = _read_json(strict_run / "results" / "global_summary.json")
    calib_summary = _read_json(calib_run / "results" / "global_summary.json")

    # Use calibration loader overrides because this defines the exact 50% eval subset.
    loader_kwargs = _loader_overrides(calib_run)
    _, _, test_loader = build_loso_calibration_loaders_h5(
        fold_idx=args.fold_idx,
        loso_dir=cfg.loso_h5_dir,
        calib_ratio=args.calib_ratio,
        calib_val_ratio=args.calib_val_ratio,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
        **loader_kwargs,
    )

    device = torch.device(args.device)
    strict_model = _load_model(strict_run, device)
    calib_model = _load_model(calib_run, device)
    strict_data = _collect(strict_model, test_loader, device)

    # Rebuild loader because iterators can be exhausted and workers hold state.
    _, _, test_loader = build_loso_calibration_loaders_h5(
        fold_idx=args.fold_idx,
        loso_dir=cfg.loso_h5_dir,
        calib_ratio=args.calib_ratio,
        calib_val_ratio=args.calib_val_ratio,
        calib_n_train=args.calib_n_train,
        calib_n_val=args.calib_n_val,
        calib_seed=args.calib_seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
        **loader_kwargs,
    )
    calib_data = _collect(calib_model, test_loader, device)

    strict_m = _wave_metrics(strict_data["pred"], strict_data["gt"])
    calib_m = _wave_metrics(calib_data["pred"], calib_data["gt"])

    figures = [
        out_dir / "headline_test_summary.png",
        out_dir / "same_subset_metric_distributions.png",
        out_dir / "same_subset_pcc_by_scene.png",
        out_dir / "representative_waveforms_same_subset.png",
    ]
    _plot_headline_summary(strict_summary, calib_summary, figures[0])
    _plot_same_subset_distributions(strict_m, calib_m, figures[1])
    _plot_scene_pcc(strict_m, calib_m, calib_data["state"], figures[2])
    _plot_waveforms(strict_data, calib_data, strict_m, calib_m, figures[3])

    if args.save_arrays:
        np.savez_compressed(
            out_dir / "same_subset_predictions.npz",
            strict_pred=strict_data["pred"],
            calib_pred=calib_data["pred"],
            gt=calib_data["gt"],
            subject=calib_data["subject"],
            state=calib_data["state"],
            strict_pcc=strict_m["pcc"],
            calib_pcc=calib_m["pcc"],
            strict_rmse=strict_m["rmse"],
            calib_rmse=calib_m["rmse"],
        )

    _write_markdown(
        out_dir / "README.md",
        strict_summary,
        calib_summary,
        strict_m,
        calib_m,
        figures,
    )

    print("\nSaved figures:")
    for fig in figures:
        print(f"  {fig}")
    print(f"  {out_dir / 'README.md'}")
    print(
        "\nSame subset mean PCC: "
        f"strict={_mean(strict_m['pcc']):.4f}, calib={_mean(calib_m['pcc']):.4f}"
    )


if __name__ == "__main__":
    main()
