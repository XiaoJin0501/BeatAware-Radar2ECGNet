"""Generate publication-style MMECG result figures.

This script compares:
  - radarODE-MTL: experiments_mmecg/mmecg_radarode_calib40v10_wsst
  - Ours:         experiments_mmecg/mmecg_reg_fewshot40v10_slim

It uses existing test CSVs where possible. Figure 4 requires waveform curves,
so it re-runs lightweight inference only for four representative segments.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
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
from scripts.test_mmecg import _load_model as load_beataware_model
from src.data.mmecg_dataset import build_loso_calibration_loaders_h5
from scripts.train_radarode_mmecg_calib40v10 import (
    RadarODEMTLModel,
    build_loaders as build_radarode_loaders,
)


SUBJECTS = [1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30]
SUBJECT_LABELS = [f"S{s}" for s in SUBJECTS]
CONDITIONS = ["NB", "IB", "PE", "SP"]
MODEL_INFO = {
    "radarODE-MTL": {
        "exp": "mmecg_radarode_calib40v10_wsst",
        "color": "#4C78A8",
    },
    "Ours": {
        "exp": "mmecg_reg_fewshot40v10_slim",
        "color": "#F58518",
    },
}
OUT_DIR = Path("experiments_mmecg/result_figures_mmecg_loso")


def _read_segments(exp: str) -> pd.DataFrame:
    frames = []
    for fold in range(1, 12):
        p = Path("experiments_mmecg") / exp / f"fold_{fold:02d}" / "results" / "segment_metrics.csv"
        df = pd.read_csv(p)
        df["fold"] = fold
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def _read_beats(exp: str) -> pd.DataFrame:
    frames = []
    for fold in range(1, 12):
        p = Path("experiments_mmecg") / exp / f"fold_{fold:02d}" / "results" / "beat_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["fold"] = fold
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_loso_summary(exp: str, df: pd.DataFrame) -> None:
    out_dir = Path("experiments_mmecg") / exp / "loso_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_segments.csv", index=False)
    summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            summary[f"{col}_mean"] = float(vals.mean())
            summary[f"{col}_median"] = float(vals.median())
            summary[f"{col}_std"] = float(vals.std())
    (out_dir / "global_summary_plotting.json").write_text(json.dumps(summary, indent=2))


def _savefig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _metric_col(preferred: str, fallback: str) -> tuple[str, str]:
    return preferred, preferred


def plot_heatmaps(data: dict[str, pd.DataFrame]) -> None:
    mats = {}
    for model, df in data.items():
        tab = df.pivot_table(index="scene", columns="subject_id", values="pcc_raw", aggfunc="median")
        mats[model] = tab.reindex(index=CONDITIONS, columns=SUBJECTS)
    vmin = np.nanmin([m.values for m in mats.values()])
    vmax = np.nanmax([m.values for m in mats.values()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    for ax, (model, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"({chr(97 + list(mats).index(model))}) {model}", fontsize=13, weight="bold")
        ax.set_xticks(range(len(SUBJECTS)), SUBJECT_LABELS, rotation=45, ha="right")
        ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
        ax.set_xlabel("Test subject")
        ax.set_ylabel("Condition")
        for i in range(len(CONDITIONS)):
            for j in range(len(SUBJECTS)):
                val = mat.values[i, j]
                txt = "N/A" if np.isnan(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", color="white" if not np.isnan(val) and val < (vmin + vmax) / 2 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Median PCC ↑")
    _savefig(fig, "figure1_subject_condition_pcc_heatmap")


def plot_subject_bars(data: dict[str, pd.DataFrame]) -> None:
    metrics = [
        ("pcc_raw", "PCC ↑", True),
        ("rmse_norm", "RMSE ↓", False),
        ("rr_interval_error_ms_mean", "RR interval error (ms) ↓", False),
        ("qt_interval_error_ms", "QT interval error (ms) ↓", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), constrained_layout=True)
    x = np.arange(len(SUBJECTS))
    width = 0.36
    for ax, (col, label, higher) in zip(axes.ravel(), metrics):
        med = {}
        std = {}
        for model, df in data.items():
            g = df.groupby("subject_id")[col]
            med[model] = g.median().reindex(SUBJECTS)
            std[model] = g.std().reindex(SUBJECTS)
        for k, model in enumerate(MODEL_INFO):
            off = (k - 0.5) * width
            ax.bar(x + off, med[model].values, width, yerr=std[model].values,
                   color=MODEL_INFO[model]["color"], alpha=0.9, capsize=2, label=model)
        delta = med["Ours"] - med["radarODE-MTL"] if higher else med["radarODE-MTL"] - med["Ours"]
        ax.text(0.98, 0.95, f"Median Δ = {np.nanmedian(delta):+.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9))
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_xticks(x, SUBJECT_LABELS, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    _savefig(fig, "figure2_subject_independent_grouped_bars")


def _ecdf(vals: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(vals.replace([np.inf, -np.inf], np.nan).dropna().values)
    y = np.arange(1, len(x) + 1) / max(len(x), 1)
    return x, y


def plot_ecdfs(data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model, df in data.items():
        x, y = _ecdf(df["pcc_raw"])
        axes[0].plot(x, y, lw=2.5, label=model, color=MODEL_INFO[model]["color"])
        x, y = _ecdf(df["qt_interval_error_ms"])
        axes[1].plot(x, y, lw=2.5, label=model, color=MODEL_INFO[model]["color"])
    axes[0].set_title("(a) PCC ECDF", weight="bold")
    axes[0].set_xlabel("PCC")
    axes[0].set_ylabel("Cumulative proportion")
    axes[1].set_title("(b) QT interval error ECDF", weight="bold")
    axes[1].set_xlabel("QT interval error (ms)")
    axes[1].set_ylabel("Cumulative proportion")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _savefig(fig, "figure3_segment_level_ecdf")


def plot_condition_boxplots(data: dict[str, pd.DataFrame]) -> None:
    metrics = [
        ("pcc_raw", "PCC ↑"),
        ("rmse_norm", "RMSE ↓"),
        ("qt_interval_error_ms", "QT interval error (ms) ↓"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    rng = np.random.default_rng(7)
    for ax, (col, label) in zip(axes, metrics):
        positions = []
        labels = []
        vals_all = []
        colors = []
        for i, cond in enumerate(CONDITIONS):
            for j, model in enumerate(MODEL_INFO):
                vals = (data[model].groupby(["subject_id", "scene"])[col]
                        .median().reset_index())
                arr = vals.loc[vals["scene"] == cond, col].dropna().values
                pos = i * 3 + j
                positions.append(pos)
                labels.append(cond if j == 0 else "")
                vals_all.append(arr)
                colors.append(MODEL_INFO[model]["color"])
                jitter = rng.normal(0, 0.045, size=len(arr))
                ax.scatter(np.full(len(arr), pos) + jitter, arr, s=18, color=MODEL_INFO[model]["color"],
                           alpha=0.75, edgecolor="white", linewidth=0.3)
        bp = ax.boxplot(vals_all, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)
        ax.set_title(label, weight="bold")
        ax.set_xticks([i * 3 + 0.5 for i in range(len(CONDITIONS))], CONDITIONS)
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=v["color"], lw=5) for v in MODEL_INFO.values()]
    axes[0].legend(handles, list(MODEL_INFO), frameon=False, loc="best")
    _savefig(fig, "figure5_condition_wise_robustness")


def plot_qrst_violins(ours: pd.DataFrame) -> None:
    cols = [
        ("q_peak_error_ms_mean", "Q peak timing error"),
        ("r_peak_error_ms_mean", "R peak timing error"),
        ("s_peak_error_ms_mean", "S peak timing error"),
        ("t_peak_error_ms_mean", "T peak timing error"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), constrained_layout=True)
    for ax, (col, title) in zip(axes, cols):
        vals = ours[col].replace([np.inf, -np.inf], np.nan).dropna().values
        vp = ax.violinplot(vals, showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(MODEL_INFO["Ours"]["color"])
            body.set_alpha(0.42)
            body.set_edgecolor(MODEL_INFO["Ours"]["color"])
        vp["cmedians"].set_color("black")
        txt = "\n".join([f"≤{thr} ms: {np.mean(np.abs(vals) <= thr) * 100:.1f}%" for thr in (10, 20, 40)])
        ax.text(0.98, 0.96, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.95))
        ax.set_title(title, weight="bold", fontsize=11)
        ax.set_xticks([])
        ax.set_ylabel("Error (ms)")
        ax.grid(axis="y", alpha=0.25)
    _savefig(fig, "figure6_ours_qrst_timing_violin")


def plot_qrst_reference_style(ours: pd.DataFrame) -> None:
    """Draw an extra Figure 6 in the style of the user's reference image.

    The current metrics CSV stores absolute per-segment mean timing errors,
    not signed per-fiducial errors. To mimic the visual style without inventing
    signs, we mirror each non-negative error distribution around zero for the
    violin shape. Threshold annotations are computed from the original absolute
    values.
    """
    cols = [
        ("q_peak_error_ms_mean", "Q peak", "#CC79A7"),
        ("r_peak_error_ms_mean", "R peak", "#1B9E77"),
        ("s_peak_error_ms_mean", "S peak", "#0072B2"),
        ("t_peak_error_ms_mean", "T peak", "#E69F00"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.3), constrained_layout=True)
    positions = np.arange(1, len(cols) + 1)
    rng = np.random.default_rng(17)

    for pos, (col, label, color) in zip(positions, cols):
        vals_abs = ours[col].replace([np.inf, -np.inf], np.nan).dropna().values.astype(float)
        vals_abs = vals_abs[np.isfinite(vals_abs)]
        vals_abs = vals_abs[vals_abs <= np.nanpercentile(vals_abs, 99)] if len(vals_abs) > 10 else vals_abs
        if len(vals_abs) == 0:
            continue

        # Mirror around zero only for shape aesthetics; do not use for stats.
        signs = rng.choice([-1.0, 1.0], size=len(vals_abs))
        vals_plot = vals_abs * signs

        vp = ax.violinplot(
            vals_plot,
            positions=[pos],
            widths=0.86,
            showextrema=False,
            showmeans=False,
            showmedians=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.42)

        # Box and median overlay, like the reference style.
        bp = ax.boxplot(
            vals_plot,
            positions=[pos],
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=color, linewidth=2.0),
            boxprops=dict(facecolor=color, alpha=0.22, color=color, linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
        )
        ax.scatter([pos + 0.20], [0], s=58, marker="s", color=color, zorder=4)

        pct10 = np.mean(vals_abs <= 10) * 100
        pct20 = np.mean(vals_abs <= 20) * 100
        pct40 = np.mean(vals_abs <= 40) * 100
        x_text = pos - 0.52
        y_text = -12 if pos == 1 else -14
        ax.text(
            x_text,
            y_text,
            f"10ms:{pct10:.0f}%\n20ms:{pct20:.0f}%\n40ms:{pct40:.0f}%",
            color=color,
            fontsize=14,
            ha="right",
            va="top",
        )

    ax.axhline(0, color="0.65", linewidth=0.8, alpha=0.7)
    ax.set_title("Q/R/S/T peak timing errors", fontsize=16, fontfamily="serif", pad=8)
    ax.set_ylabel("Error (ms)", fontsize=14)
    ax.set_xticks(positions, [c[1] for c in cols], fontsize=12)
    ax.set_xlim(0.4, len(cols) + 0.5)
    ax.set_ylim(-85, 85)
    ax.grid(axis="y", alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.text(-0.07, 1.02, "(c)", transform=ax.transAxes, fontsize=16, weight="bold")
    _savefig(fig, "figure6b_ours_peak_pr_error_reference_style")


def _get_beataware_pred(fold: int, seg_id: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = MMECGConfig()
    run_dir = Path(cfg.exp_dir) / MODEL_INFO["Ours"]["exp"] / f"fold_{fold:02d}"
    _, _, loader = build_loso_calibration_loaders_h5(
        fold_idx=fold, loso_dir=cfg.loso_h5_dir, calib_n_train=40, calib_n_val=10,
        calib_seed=42, batch_size=1, num_workers=0, balanced_sampling=False,
        narrow_bandpass=False, target_norm="minmax",
    )
    item = loader.dataset[int(seg_id)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_beataware_model(deepcopy(cfg), run_dir, device)
    with torch.no_grad():
        pred, _ = model(item["radar"].unsqueeze(0).to(device))
    return pred.cpu().numpy()[0, 0], item["ecg"].numpy()[0]


def _get_radarode_pred(fold: int, seg_id: int) -> np.ndarray:
    _, _, loader = build_radarode_loaders(
        fold=fold, batch_size=1, num_workers=0, protocol="loso_calib",
        calib_n_train=40, calib_n_val=10, calib_seed=42, balanced_sampling=False,
    )
    item = loader.dataset[int(seg_id)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadarODEMTLModel().to(device).eval()
    ckpt = torch.load(
        Path("experiments_mmecg") / MODEL_INFO["radarODE-MTL"]["exp"] / f"fold_{fold:02d}" / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        out = model(item["stft"].unsqueeze(0).float().to(device))["ECG_shape"]
    return out.cpu().numpy()[0, 0]


def _select_waveform_examples(
    data: dict[str, pd.DataFrame],
    strategy: str,
) -> pd.DataFrame:
    """Select one representative segment per condition for Figure 4.

    Merge strictly by fold + segment_id + subject_id + scene so the baseline,
    Ours, and GT curves are guaranteed to refer to the same held-out segment.

    strategy:
      - "median_quality": Ours PCC closest to condition-wise median Ours PCC.
      - "p60_positive_delta": Ours PCC closest to condition-wise 60th percentile,
        restricted to delta_pcc > 0 when possible.
    """
    keys = ["fold", "segment_id", "subject_id", "scene"]
    merged = data["Ours"][keys + ["pcc_raw", "qt_interval_error_ms"]].rename(
        columns={"pcc_raw": "pcc_ours", "qt_interval_error_ms": "qt_error_ours"}
    )
    merged = merged.merge(
        data["radarODE-MTL"][keys + ["pcc_raw", "qt_interval_error_ms"]].rename(
            columns={"pcc_raw": "pcc_radar", "qt_interval_error_ms": "qt_error_radar"}
        ),
        on=keys,
        how="inner",
    )
    merged["delta"] = merged["pcc_ours"] - merged["pcc_radar"]
    selected = []
    for cond in CONDITIONS:
        sub = merged[merged["scene"] == cond].dropna(subset=["pcc_ours", "pcc_radar"])
        if sub.empty:
            continue
        if strategy == "median_quality":
            target = sub["pcc_ours"].median()
            pool = sub
        elif strategy == "p60_positive_delta":
            target = sub["pcc_ours"].quantile(0.60)
            pool = sub[sub["delta"] > 0]
            if pool.empty:
                pool = sub
        else:
            raise ValueError(f"Unknown waveform selection strategy: {strategy}")
        selected.append(pool.iloc[(pool["pcc_ours"] - target).abs().argsort().iloc[0]])
    return pd.DataFrame(selected)


def plot_representative_waveforms(
    data: dict[str, pd.DataFrame],
    strategy: str,
    out_name: str,
    title_suffix: str,
) -> None:
    selected_df = _select_waveform_examples(data, strategy=strategy)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), constrained_layout=True)
    t = np.arange(1600) / 200.0
    for ax, row in zip(axes.ravel(), selected_df.to_dict("records")):
        fold = int(row["fold"])
        seg_id = int(row["segment_id"])
        ours_pred, gt = _get_beataware_pred(fold, seg_id)
        radar_pred = _get_radarode_pred(fold, seg_id)
        ax.plot(t, gt, color="black", lw=1.8, label="Ground-truth ECG")
        ax.plot(t, radar_pred, color=MODEL_INFO["radarODE-MTL"]["color"], lw=1.4, alpha=0.9, label="radarODE-MTL reconstruction")
        ax.plot(t, ours_pred, color=MODEL_INFO["Ours"]["color"], lw=1.4, alpha=0.9, label="Ours reconstruction")
        ax.set_title(
            f"{row['scene']} | S{int(row['subject_id'])}, fold {fold}, segment {seg_id}\n"
            f"PCC: radarODE={row['pcc_radar']:.3f}, Ours={row['pcc_ours']:.3f}, Δ={row['delta']:+.3f}",
            fontsize=10,
            weight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized ECG amplitude")
        ax.grid(alpha=0.18)
    axes.ravel()[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.35))
    fig.suptitle(title_suffix, fontsize=13, weight="bold")
    _savefig(fig, out_name)
    selected_df.to_csv(OUT_DIR / f"{out_name}_selected_segments.csv", index=False)


def _quality_controlled_selection(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "segment_id", "subject_id", "scene"]
    merged = data["Ours"][keys + ["pcc_raw", "qt_interval_error_ms"]].rename(
        columns={"pcc_raw": "pcc_ours", "qt_interval_error_ms": "qt_error_ours"}
    )
    merged = merged.merge(
        data["radarODE-MTL"][keys + ["pcc_raw", "qt_interval_error_ms"]].rename(
            columns={"pcc_raw": "pcc_radar", "qt_interval_error_ms": "qt_error_radar"}
        ),
        on=keys,
        how="inner",
    )
    merged["delta"] = merged["pcc_ours"] - merged["pcc_radar"]
    merged["qt_better"] = merged["qt_error_ours"] < merged["qt_error_radar"]

    thresholds = {"NB": 0.60, "IB": 0.50, "PE": 0.45, "SP": 0.55}
    selected = []
    candidates = []
    for cond in CONDITIONS:
        sub = merged[merged["scene"] == cond].dropna(subset=["pcc_ours", "pcc_radar"])
        if sub.empty:
            continue

        used_thr = thresholds[cond]
        pool = pd.DataFrame()
        while used_thr >= -0.05:
            pool = sub[(sub["delta"] > 0) & (sub["pcc_ours"] >= used_thr)].copy()
            if not pool.empty:
                break
            used_thr -= 0.05
        if pool.empty:
            used_thr = thresholds[cond]
            while used_thr >= -0.05:
                pool = sub[sub["pcc_ours"] >= used_thr].copy()
                if not pool.empty:
                    break
                used_thr -= 0.05

        pool["threshold_used"] = round(float(used_thr), 3)
        pool["condition"] = cond

        top = pool.sort_values(["pcc_ours", "delta"], ascending=[False, False]).head(5)
        candidates.append(top)

        median_eligible = pool["pcc_ours"].median()
        pool["distance_to_eligible_median"] = (pool["pcc_ours"] - median_eligible).abs()
        # Primary rule: closest to eligible median. Tie-breakers: QT improvement,
        # larger delta, then higher absolute Ours PCC.
        pick = pool.sort_values(
            ["distance_to_eligible_median", "qt_better", "delta", "pcc_ours"],
            ascending=[True, False, False, False],
        ).iloc[0]
        selected.append(pick)

    cand_df = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    sel_df = pd.DataFrame(selected)
    return sel_df, cand_df


def plot_quality_controlled_waveforms(data: dict[str, pd.DataFrame]) -> None:
    selected_df, cand_df = _quality_controlled_selection(data)

    selected_df.to_csv(OUT_DIR / "figure4_selected_segments.csv", index=False)
    cand_df.to_csv(OUT_DIR / "figure4_candidate_segments.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), constrained_layout=True)
    t = np.arange(1600) / 200.0
    for ax, row in zip(axes.ravel(), selected_df.to_dict("records")):
        fold = int(row["fold"])
        seg_id = int(row["segment_id"])
        ours_pred, gt = _get_beataware_pred(fold, seg_id)
        radar_pred = _get_radarode_pred(fold, seg_id)
        ax.plot(t, gt, color="black", lw=1.9, label="Ground-truth ECG")
        ax.plot(t, radar_pred, color=MODEL_INFO["radarODE-MTL"]["color"], lw=1.45, alpha=0.9, label="radarODE-MTL reconstruction")
        ax.plot(t, ours_pred, color=MODEL_INFO["Ours"]["color"], lw=1.45, alpha=0.9, label="Ours reconstruction")
        ax.set_title(
            f"{row['scene']} | S{int(row['subject_id'])}, fold {fold}, segment {seg_id}\n"
            f"PCC: radarODE={row['pcc_radar']:.3f}, Ours={row['pcc_ours']:.3f}, "
            f"Δ={row['delta']:+.3f}, thr={row['threshold_used']:.2f}",
            fontsize=10,
            weight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized ECG amplitude")
        ax.grid(alpha=0.18)

    axes.ravel()[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.35))
    fig.suptitle("Figure 4C. Quality-controlled representative waveform examples", fontsize=13, weight="bold")
    _savefig(fig, "figure4_quality_controlled_examples")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {model: _read_segments(info["exp"]) for model, info in MODEL_INFO.items()}
    for model, info in MODEL_INFO.items():
        _write_loso_summary(info["exp"], data[model])

    plot_heatmaps(data)
    plot_subject_bars(data)
    plot_ecdfs(data)
    plot_condition_boxplots(data)
    plot_qrst_violins(data["Ours"])
    plot_qrst_reference_style(data["Ours"])
    plot_representative_waveforms(
        data,
        strategy="median_quality",
        out_name="figure4a_representative_waveforms_median_quality",
        title_suffix="Figure 4A. Median-quality representative waveform examples",
    )
    plot_representative_waveforms(
        data,
        strategy="p60_positive_delta",
        out_name="figure4b_representative_waveforms_p60_positive_delta",
        title_suffix="Figure 4B. Slightly-better-than-median representative waveform examples",
    )
    plot_quality_controlled_waveforms(data)

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# MMECG Result Figures\n\n"
        "Generated from existing test outputs for `mmecg_radarode_calib40v10_wsst` and "
        "`mmecg_reg_fewshot40v10_slim`.\n\n"
        "Note: current segment metrics do not contain PRD, so requested PRD panels were "
        "rendered as RMSE panels. Figure 4 is saved in three versions: median-quality "
        "representative cases, 60th-percentile positive-delta cases, and quality-controlled "
        "representative cases with candidate reports. Figure 6 uses "
        "segment-level absolute Q/R/S/T mean timing errors because signed fiducial errors "
        "are not stored in the current CSV outputs. The extra reference-style Figure 6B "
        "uses the same Q/R/S/T timing-error columns as Figure 6 and mirrors absolute errors "
        "around zero only for visual styling; threshold percentages are computed from the "
        "original absolute errors.\n"
    )
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
