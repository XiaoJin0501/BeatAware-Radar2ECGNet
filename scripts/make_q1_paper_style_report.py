#!/usr/bin/env python3
"""Create paper-style MMECG comparison figures for Q1 reporting.

This script extends the supervisor report by adding reproduced literature
baselines already present in `experiments_mmecg`, plus figures closer to a
paper result section: main comparison tables, method ranking, paired fold
comparison, segment CDFs, state-wise bars, and waveform panels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments_mmecg"


METHODS = [
    {"name": "AirECG", "tag": "mmecg_airecg_strict_loso_v4", "kind": "published baseline"},
    {"name": "RSSRNet", "tag": "mmecg_rssrnet_strict_loso_v2", "kind": "published baseline"},
    {"name": "CNN", "tag": "mmecg_cnn_strict_loso_baseline", "kind": "generic baseline"},
    {"name": "LSTM", "tag": "mmecg_lstm_strict_loso_baseline", "kind": "generic baseline"},
    {"name": "BiLSTM", "tag": "mmecg_bilstm_strict_loso_baseline", "kind": "generic baseline"},
    {"name": "Conformer", "tag": "mmecg_conformer_strict_loso_baseline", "kind": "generic baseline"},
    {"name": "ICONIP BeatAware", "tag": "mmecg_beataware_fewshot40v10", "kind": "our ICONIP"},
    {"name": "WSST-TCN", "tag": "q1_remalign_wsst_tcn_loso", "kind": "representation baseline"},
    {"name": "REM-Event", "tag": "q1_remalign_event_lowpeak_loso", "kind": "ablation"},
    {"name": "radarODE-MTL", "tag": "mmecg_radarode_mtl_ega_strict_loso", "kind": "strong pipeline"},
    {"name": "REM-AlignNet-MTL", "tag": "q1_remalign_mtl_shapefirst_loso", "kind": "proposed"},
]


COLORS = {
    "AirECG": "#8DA0CB",
    "RSSRNet": "#66C2A5",
    "CNN": "#B3B3B3",
    "LSTM": "#B3B3B3",
    "BiLSTM": "#B3B3B3",
    "Conformer": "#B3B3B3",
    "ICONIP BeatAware": "#FC8D62",
    "WSST-TCN": "#A6CEE3",
    "REM-Event": "#FDBF6F",
    "radarODE-MTL": "#5F6C72",
    "REM-AlignNet-MTL": "#2F80ED",
}


def _json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def _mean_std(x: pd.Series | np.ndarray) -> str:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "NA"
    return f"{arr.mean():.4f} ± {arr.std(ddof=0):.4f}"


def _summary_path(tag: str, fold: int, metric_space: str) -> Path:
    fold_dir = EXP / tag / f"fold_{fold:02d}"
    if tag == "q1_remalign_mtl_shapefirst_loso" and metric_space == "shape":
        return fold_dir / "results_shape_head" / "global_summary_shape_head.json"
    if metric_space == "shape":
        return fold_dir / "results_shape200" / "global_summary_shape200.json"
    return fold_dir / "results" / "global_summary.json"


def _read_fold_metric(tag: str, fold: int, metric_space: str) -> dict[str, float] | None:
    path = _summary_path(tag, fold, metric_space)
    if not path.exists():
        return None
    d = _json(path)
    if tag == "q1_remalign_mtl_shapefirst_loso" and metric_space == "shape":
        return {
            "pcc": _float(d.get("pcc_mean")),
            "rmse": _float(d.get("rmse_mean")),
            "mae": _float(d.get("mae_mean")),
            "r2": _float(d.get("r2_mean")),
            "qmr": float("nan"),
            "r_f1": float("nan"),
            "n": _float(d.get("num_segments")),
        }
    return {
        "pcc": _float(d.get("pcc_raw_mean")),
        "rmse": _float(d.get("rmse_norm_mean")),
        "mae": _float(d.get("mae_norm_mean")),
        "r2": _float(d.get("r2_mean")),
        "qmr": _float(d.get("qualified_monitoring_rate")),
        "r_f1": _float(d.get("rpeak_f1_150ms_mean", d.get("average_f1_150ms_mean"))),
        "n": _float(d.get("num_segments")),
    }


def collect_fold_table() -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for space in ("full", "shape"):
            for fold in range(1, 12):
                metrics = _read_fold_metric(method["tag"], fold, space)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "method": method["name"],
                        "exp_tag": method["tag"],
                        "kind": method["kind"],
                        "output_space": "Shape-200" if space == "shape" else "8-s continuous",
                        "fold": fold,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, space), sub in df.groupby(["method", "output_space"]):
        rows.append(
            {
                "method": method,
                "output_space": space,
                "available_folds": int(sub["fold"].nunique()),
                "total_segments": int(np.nansum(sub["n"])),
                "pcc_mean": float(sub["pcc"].mean()),
                "pcc_std": float(sub["pcc"].std(ddof=0)),
                "pcc_median": float(sub["pcc"].median()),
                "rmse_mean": float(sub["rmse"].mean()),
                "mae_mean": float(sub["mae"].mean()),
                "r2_mean": float(sub["r2"].mean()),
                "qmr_mean": float(sub["qmr"].mean()),
                "r_f1_mean": float(sub["r_f1"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["output_space", "pcc_mean"], ascending=[True, False])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )


def plot_shape_ranking(summary: pd.DataFrame, out: Path) -> None:
    sub = summary[(summary["output_space"] == "Shape-200") & (summary["available_folds"] >= 3)].copy()
    sub = sub.sort_values("pcc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y = np.arange(len(sub))
    ax.barh(y, sub["pcc_mean"], xerr=sub["pcc_std"], color=[COLORS.get(m, "#AAAAAA") for m in sub["method"]])
    ax.set_yticks(y)
    ax.set_yticklabels(sub["method"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Shape-200 PCC")
    ax.set_title("Main shape-200 comparison across reproduced methods")
    for yi, row in zip(y, sub.itertuples(index=False)):
        ax.text(row.pcc_mean + 0.02, yi, f"{row.pcc_mean:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_main_shape200_method_ranking.png")
    fig.savefig(out / "fig_main_shape200_method_ranking.pdf")
    plt.close(fig)


def plot_metric_radarode_style(summary: pd.DataFrame, out: Path) -> None:
    methods = ["AirECG", "RSSRNet", "ICONIP BeatAware", "radarODE-MTL", "REM-AlignNet-MTL"]
    metrics = [("pcc_mean", "PCC", 0, 1), ("rmse_mean", "RMSE", 0, 0.32), ("mae_mean", "MAE", 0, 0.25), ("r2_mean", "R2", -2.5, 1)]
    shape = summary[(summary["output_space"] == "Shape-200")].set_index("method")
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0))
    for ax, (key, title, ymin, ymax) in zip(axes, metrics):
        vals = [shape.loc[m, key] if m in shape.index else np.nan for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, vals, color=[COLORS.get(m, "#AAAAAA") for m in methods], width=0.72)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylim(ymin, ymax)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + (ymax - ymin) * 0.025, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("Shape-200 morphology metrics", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_shape200_four_metric_panel.png", bbox_inches="tight")
    fig.savefig(out / "fig_shape200_four_metric_panel.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fold_heatmap(df: pd.DataFrame, out: Path) -> None:
    sub = df[(df["output_space"] == "Shape-200")].copy()
    keep = ["AirECG", "RSSRNet", "ICONIP BeatAware", "WSST-TCN", "REM-Event", "radarODE-MTL", "REM-AlignNet-MTL"]
    pivot = sub.pivot_table(index="method", columns="fold", values="pcc", aggfunc="mean").reindex(keep)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels([str(i) for i in range(1, 12)])
    ax.set_xlabel("LOSO fold")
    ax.set_title("Shape-200 PCC heatmap")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.65 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PCC")
    fig.tight_layout()
    fig.savefig(out / "fig_shape200_fold_heatmap.png")
    fig.savefig(out / "fig_shape200_fold_heatmap.pdf")
    plt.close(fig)


def _segment_csv(method: str, tag: str, fold: int) -> Path:
    if tag == "q1_remalign_mtl_shapefirst_loso":
        return EXP / tag / f"fold_{fold:02d}" / "results_shape_head" / "segment_metrics_shape_head.csv"
    return EXP / tag / f"fold_{fold:02d}" / "results_shape200" / "segment_metrics_shape200.csv"


def collect_segments() -> pd.DataFrame:
    methods = [
        ("AirECG", "mmecg_airecg_strict_loso_v4"),
        ("RSSRNet", "mmecg_rssrnet_strict_loso_v2"),
        ("ICONIP BeatAware", "mmecg_beataware_fewshot40v10"),
        ("radarODE-MTL", "mmecg_radarode_mtl_ega_strict_loso"),
        ("REM-AlignNet-MTL", "q1_remalign_mtl_shapefirst_loso"),
    ]
    rows = []
    for name, tag in methods:
        for fold in range(1, 12):
            path = _segment_csv(name, tag, fold)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            pcc_col = "pcc" if "pcc" in df.columns else "pcc_raw"
            rmse_col = "rmse" if "rmse" in df.columns else "rmse_norm"
            for i, row in df.iterrows():
                rows.append(
                    {
                        "method": name,
                        "fold": fold,
                        "segment": i,
                        "pcc": float(row[pcc_col]),
                        "rmse": float(row[rmse_col]),
                        "state": row.get("state", np.nan),
                        "scene": row.get("scene", ""),
                    }
                )
    return pd.DataFrame(rows)


def plot_cdf(seg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in ["AirECG", "RSSRNet", "ICONIP BeatAware", "radarODE-MTL", "REM-AlignNet-MTL"]:
        vals = np.asarray(seg.loc[seg["method"] == method, "pcc"], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, linewidth=2.0, label=method, color=COLORS.get(method))
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segment-level shape-200 PCC")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF of segment-level reconstruction quality")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_segment_pcc_cdf.png")
    fig.savefig(out / "fig_segment_pcc_cdf.pdf")
    plt.close(fig)


def plot_state_panel(seg: pd.DataFrame, out: Path) -> pd.DataFrame:
    show = ["radarODE-MTL", "REM-AlignNet-MTL", "ICONIP BeatAware", "RSSRNet", "AirECG"]
    state = (
        seg[seg["method"].isin(show)]
        .groupby(["method", "state", "scene"], dropna=False)
        .agg(n=("pcc", "size"), pcc=("pcc", "mean"), rmse=("rmse", "mean"))
        .reset_index()
    )
    scenes = ["NB", "IB", "SP", "PE"]
    if not set(scenes).intersection(set(state["scene"].astype(str))):
        scenes = sorted(state["scene"].dropna().astype(str).unique().tolist())
    x = np.arange(len(scenes))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for i, method in enumerate(show):
        vals = []
        for scene in scenes:
            sub = state[(state["method"] == method) & (state["scene"].astype(str) == scene)]
            vals.append(float(sub["pcc"].mean()) if len(sub) else np.nan)
        ax.bar(x + (i - 2) * width, vals, width=width, color=COLORS.get(method), label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylim(0, 1)
    ax.set_xlabel("MMECG state")
    ax.set_ylabel("Shape-200 PCC")
    ax.set_title("State-wise generalization within MMECG")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_statewise_multimodel_pcc.png")
    fig.savefig(out / "fig_statewise_multimodel_pcc.pdf")
    plt.close(fig)
    return state


def plot_wave_panel(out: Path) -> None:
    fold = 9
    specs = [
        ("AirECG", "mmecg_airecg_strict_loso_v4", "results_shape200/predictions_shape200.npz"),
        ("RSSRNet", "mmecg_rssrnet_strict_loso_v2", "results_shape200/predictions_shape200.npz"),
        ("ICONIP BeatAware", "mmecg_beataware_fewshot40v10", "results_shape200/predictions_shape200.npz"),
        ("radarODE-MTL", "mmecg_radarode_mtl_ega_strict_loso", "results_shape200/predictions_shape200.npz"),
        ("REM-AlignNet-MTL", "q1_remalign_mtl_shapefirst_loso", "results_shape_head/predictions_shape_head.npz"),
    ]
    loaded = {}
    n = None
    for name, tag, rel in specs:
        path = EXP / tag / f"fold_{fold:02d}" / rel
        if not path.exists():
            continue
        z = np.load(path)
        pred_key = "pred"
        gt_key = "target" if "target" in z.files else "gt"
        loaded[name] = (z[pred_key], z[gt_key])
        n = len(z[pred_key]) if n is None else min(n, len(z[pred_key]))
    if "REM-AlignNet-MTL" not in loaded or "radarODE-MTL" not in loaded:
        return
    rem_pred, rem_gt = loaded["REM-AlignNet-MTL"]
    radar_pred, radar_gt = loaded["radarODE-MTL"]
    gains = np.asarray([_pcc(rem_pred[i], rem_gt[i]) - _pcc(radar_pred[i], radar_gt[i]) for i in range(n)])
    rem_scores = np.asarray([_pcc(rem_pred[i], rem_gt[i]) for i in range(n)])
    picks = [
        int(np.argmax(gains)),
        int(np.argsort(np.abs(rem_scores - np.median(rem_scores)))[0]),
        int(np.argmin(rem_scores)),
    ]
    titles = ["largest REM gain", "median REM case", "hard REM case"]
    t = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True)
    for ax, idx, title in zip(axes, picks, titles):
        ax.plot(t, rem_gt[idx], color="#111111", linewidth=2.0, label="GT")
        for name in ["AirECG", "RSSRNet", "ICONIP BeatAware", "radarODE-MTL", "REM-AlignNet-MTL"]:
            if name not in loaded:
                continue
            pred, gt = loaded[name]
            ax.plot(t, pred[idx], linewidth=1.35 if name != "REM-AlignNet-MTL" else 1.8, color=COLORS.get(name), label=f"{name} ({_pcc(pred[idx], gt[idx]):.2f})")
        ax.set_title(f"Fold {fold}, segment {idx}: {title}", fontsize=10)
        ax.set_ylabel("Normalized ECG")
        ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Normalized cardiac cycle")
    fig.suptitle("Representative morphology reconstructions across methods", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_multimodel_shape_waveforms.png", bbox_inches="tight")
    fig.savefig(out / "fig_multimodel_shape_waveforms.pdf", bbox_inches="tight")
    plt.close(fig)


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def write_markdown(out: Path, summary: pd.DataFrame, state: pd.DataFrame) -> None:
    shape = summary[summary["output_space"] == "Shape-200"].sort_values("pcc_mean", ascending=False)
    full = summary[summary["output_space"] == "8-s continuous"].sort_values("pcc_mean", ascending=False)
    lines = ["# Q1 Paper-Style MMECG Result Package\n"]
    lines.append("## Shape-200 Main Comparison\n")
    lines.append("| Method | Folds | PCC | RMSE | MAE | R2 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in shape.itertuples(index=False):
        lines.append(f"| {row.method} | {row.available_folds} | {row.pcc_mean:.4f} ± {row.pcc_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} | {row.r2_mean:.4f} |\n")
    lines.append("\n## Continuous ECG Audit\n")
    lines.append("| Method | Folds | PCC | RMSE | QMR | R-F1 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in full.itertuples(index=False):
        lines.append(f"| {row.method} | {row.available_folds} | {row.pcc_mean:.4f} ± {row.pcc_std:.4f} | {row.rmse_mean:.4f} | {row.qmr_mean:.2f} | {row.r_f1_mean:.4f} |\n")
    lines.append("\n## Interpretation\n")
    lines.append(
        "REM-AlignNet-MTL is currently strongest in the radarODE-comparable shape-200 morphology space. "
        "The paper claim should therefore emphasize morphology reconstruction and event-structured representation learning, "
        "while treating 8-s continuous ECG as an unresolved branch that needs a morphology-guided renderer.\n"
    )
    lines.append("\n## Figures\n")
    for name in [
        "fig_main_shape200_method_ranking.png",
        "fig_shape200_four_metric_panel.png",
        "fig_shape200_fold_heatmap.png",
        "fig_segment_pcc_cdf.png",
        "fig_statewise_multimodel_pcc.png",
        "fig_multimodel_shape_waveforms.png",
    ]:
        lines.append(f"- `{name}`\n")
    (out / "Q1_PAPER_STYLE_REPORT.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(EXP / "q1_paper_style_report"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    _style()

    folds = collect_fold_table()
    summary = aggregate(folds)
    segments = collect_segments()
    folds.to_csv(out / "fold_level_all_methods.csv", index=False)
    summary.to_csv(out / "summary_all_methods.csv", index=False)
    segments.to_csv(out / "segment_level_selected_methods.csv", index=False)

    plot_shape_ranking(summary, out)
    plot_metric_radarode_style(summary, out)
    plot_fold_heatmap(folds, out)
    plot_cdf(segments, out)
    state = plot_state_panel(segments, out)
    state.to_csv(out / "state_level_selected_methods.csv", index=False)
    plot_wave_panel(out)
    write_markdown(out, summary, state)
    print(summary.to_string(index=False))
    print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
