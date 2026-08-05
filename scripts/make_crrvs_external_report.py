#!/usr/bin/env python3
"""Create a compact CR_RVS external-validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments_mmecg"
SCENE_NAMES = {0: "Resting", 1: "Valsalva", 2: "Apnea", 3: "PE"}


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12))


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_summary(rem_tag: str, tcn_tag: str, fold: int) -> pd.DataFrame:
    rem = _json(EXP / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "global_summary_shape_head.json")
    tcn = _json(EXP / tcn_tag / f"fold_{fold:02d}" / "results_shape200" / "global_summary_shape200.json")
    return pd.DataFrame(
        [
            {
                "fold": fold,
                "method": "CW-STFT + TCN",
                "pcc": tcn["pcc_raw_mean"],
                "median_pcc": tcn["pcc_raw_median"],
                "rmse": tcn["rmse_norm_mean"],
                "mae": tcn["mae_norm_mean"],
                "r2": tcn["r2_mean"],
                "n": tcn["num_segments"],
            },
            {
                "fold": fold,
                "method": "CW-STFT + REM-AlignNet-MTL",
                "pcc": rem["pcc_mean"],
                "median_pcc": rem["pcc_median"],
                "rmse": rem["rmse_mean"],
                "mae": rem["mae_mean"],
                "r2": rem["r2_mean"],
                "n": rem["num_segments"],
            },
        ]
    )


def _load_segments(rem_tag: str, tcn_tag: str, fold: int) -> pd.DataFrame:
    rem = pd.read_csv(EXP / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "segment_metrics_shape_head.csv")
    tcn = pd.read_csv(EXP / tcn_tag / f"fold_{fold:02d}" / "results_shape200" / "segment_metrics_shape200.csv")
    n = min(len(rem), len(tcn))
    rows = []
    for i in range(n):
        rows.append(
            {
                "segment": i,
                "fold": fold,
                "subject_id": int(rem.iloc[i]["subject_id"]),
                "state": int(rem.iloc[i]["state"]),
                "scene": SCENE_NAMES.get(int(rem.iloc[i]["state"]), str(rem.iloc[i]["state"])),
                "rem_pcc": float(rem.iloc[i]["pcc"]),
                "rem_rmse": float(rem.iloc[i]["rmse"]),
                "tcn_pcc": float(tcn.iloc[i]["pcc_raw"]),
                "tcn_rmse": float(tcn.iloc[i]["rmse_norm"]),
            }
        )
    df = pd.DataFrame(rows)
    df["delta_pcc"] = df["rem_pcc"] - df["tcn_pcc"]
    return df


def _load_all_summaries(rem_tag: str, tcn_tag: str, folds: list[int]) -> pd.DataFrame:
    parts = []
    for fold in folds:
        try:
            parts.append(_load_summary(rem_tag, tcn_tag, fold))
        except FileNotFoundError as exc:
            print(f"[WARN] missing fold {fold}: {exc}")
    if not parts:
        raise FileNotFoundError("No fold summaries found.")
    return pd.concat(parts, ignore_index=True)


def _load_all_segments(rem_tag: str, tcn_tag: str, folds: list[int]) -> pd.DataFrame:
    parts = []
    for fold in folds:
        try:
            parts.append(_load_segments(rem_tag, tcn_tag, fold))
        except FileNotFoundError as exc:
            print(f"[WARN] missing fold {fold}: {exc}")
    if not parts:
        raise FileNotFoundError("No fold segment metrics found.")
    return pd.concat(parts, ignore_index=True)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("method")
        .agg(
            folds=("fold", "nunique"),
            n=("n", "sum"),
            pcc=("pcc", "mean"),
            pcc_std=("pcc", "std"),
            median_pcc=("median_pcc", "mean"),
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
    )


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


def plot_main(summary: pd.DataFrame, out: Path) -> None:
    if "pcc_std" not in summary.columns:
        summary = summary.copy()
        summary["pcc_std"] = 0.0
        summary["rmse_std"] = 0.0
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    colors = ["#8795A1", "#2F80ED"]
    for ax, key, err_key, title, ymax, lower_better in [
        (axes[0], "pcc", "pcc_std", "Shape-200 PCC", 1.0, False),
        (axes[1], "rmse", "rmse_std", "Shape-200 RMSE", 0.22, True),
        (axes[2], "mae", None, "Shape-200 MAE", 0.16, True),
    ]:
        yerr = summary[err_key] if err_key and err_key in summary.columns else None
        ax.bar(np.arange(len(summary)), summary[key], yerr=yerr, color=colors, width=0.65, capsize=4)
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary["method"], rotation=25, ha="right")
        ax.set_ylim(0, ymax)
        ax.set_title(title)
        for i, v in enumerate(summary[key]):
            ax.text(i, v + ymax * 0.035, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("CR_RVS external validation", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_crrvs_external_main_metrics.png", bbox_inches="tight")
    fig.savefig(out / "fig_crrvs_external_main_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_foldwise(summary: pd.DataFrame, out: Path) -> None:
    pivot = summary.pivot_table(index="fold", columns="method", values="pcc", aggfunc="mean")
    methods = ["CW-STFT + TCN", "CW-STFT + REM-AlignNet-MTL"]
    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for offset, method, color in [(-width / 2, methods[0], "#8795A1"), (width / 2, methods[1], "#2F80ED")]:
        vals = pivot[method].to_numpy(dtype=float) if method in pivot.columns else np.full(len(x), np.nan)
        ax.bar(x + offset, vals, width=width, color=color, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(f)) for f in pivot.index])
    ax.set_ylim(0, 1)
    ax.set_xlabel("CR_RVS subject-level fold")
    ax.set_ylabel("Shape-200 PCC")
    ax.set_title("Fold-wise external validation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_crrvs_external_foldwise_pcc.png")
    fig.savefig(out / "fig_crrvs_external_foldwise_pcc.pdf")
    plt.close(fig)


def plot_cdf(seg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, color in [("tcn_pcc", "CW-STFT + TCN", "#8795A1"), ("rem_pcc", "CW-STFT + REM-AlignNet-MTL", "#2F80ED")]:
        xs = np.sort(seg[key].to_numpy(dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=label, color=color, linewidth=2.1)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segment-level shape-200 PCC")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CR_RVS segment-level CDF")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_crrvs_external_pcc_cdf.png")
    fig.savefig(out / "fig_crrvs_external_pcc_cdf.pdf")
    plt.close(fig)


def plot_state(seg: pd.DataFrame, out: Path) -> pd.DataFrame:
    state = (
        seg.groupby(["state", "scene"])
        .agg(
            n=("rem_pcc", "size"),
            rem_pcc=("rem_pcc", "mean"),
            tcn_pcc=("tcn_pcc", "mean"),
            rem_rmse=("rem_rmse", "mean"),
            tcn_rmse=("tcn_rmse", "mean"),
            rem_win_rate=("delta_pcc", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    state["delta_pcc"] = state["rem_pcc"] - state["tcn_pcc"]
    x = np.arange(len(state))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, state["tcn_pcc"], width, color="#8795A1", label="CW-STFT + TCN")
    ax.bar(x + width / 2, state["rem_pcc"], width, color="#2F80ED", label="REM-AlignNet-MTL")
    ax.set_xticks(x)
    ax.set_xticklabels(state["scene"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("CR_RVS state")
    ax.set_ylabel("Shape-200 PCC")
    ax.set_title("State-wise external performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_crrvs_external_statewise.png")
    fig.savefig(out / "fig_crrvs_external_statewise.pdf")
    plt.close(fig)
    return state


def plot_waves(rem_tag: str, tcn_tag: str, fold: int, out: Path) -> pd.DataFrame:
    rem_z = np.load(EXP / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "predictions_shape_head.npz")
    tcn_z = np.load(EXP / tcn_tag / f"fold_{fold:02d}" / "results_shape200" / "predictions_shape200.npz")
    rem_pred, rem_gt = rem_z["pred"], rem_z["target"]
    tcn_pred, tcn_gt = tcn_z["pred"], tcn_z["gt"]
    n = min(len(rem_pred), len(tcn_pred))
    rem_scores = np.asarray([_pcc(rem_pred[i], rem_gt[i]) for i in range(n)])
    tcn_scores = np.asarray([_pcc(tcn_pred[i], tcn_gt[i]) for i in range(n)])
    gain = rem_scores - tcn_scores
    picks = [
        int(np.argmax(gain)),
        int(np.argsort(np.abs(rem_scores - np.median(rem_scores)))[0]),
        int(np.argmin(rem_scores)),
    ]
    labels = ["largest REM gain", "median REM", "hard REM case"]
    t = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
    rows = []
    for ax, idx, label in zip(axes, picks, labels):
        ax.plot(t, rem_gt[idx], color="#111111", linewidth=2.0, label="GT")
        ax.plot(t, tcn_pred[idx], color="#8795A1", linewidth=1.5, label=f"TCN {tcn_scores[idx]:.2f}")
        ax.plot(t, rem_pred[idx], color="#2F80ED", linewidth=1.7, label=f"REM {rem_scores[idx]:.2f}")
        ax.set_title(f"Segment {idx}: {label}, gain={gain[idx]:+.2f}", fontsize=10)
        ax.set_ylabel("Normalized ECG")
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        rows.append({"segment": idx, "case": label, "rem_pcc": rem_scores[idx], "tcn_pcc": tcn_scores[idx], "gain": gain[idx]})
    axes[-1].set_xlabel("Normalized cardiac cycle")
    fig.suptitle("CR_RVS representative shape reconstructions", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_crrvs_external_wave_examples.png", bbox_inches="tight")
    fig.savefig(out / "fig_crrvs_external_wave_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def write_markdown(out: Path, summary: pd.DataFrame, state: pd.DataFrame, seg: pd.DataFrame) -> None:
    tcn = summary[summary["method"] == "CW-STFT + TCN"].iloc[0]
    rem = summary[summary["method"] == "CW-STFT + REM-AlignNet-MTL"].iloc[0]
    lines = ["# CR_RVS External Validation Report\n"]
    lines.append("## Main Result\n")
    lines.append("| Method | Folds | N | PCC | Median PCC | RMSE | MAE | R2 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in summary.itertuples(index=False):
        pcc_text = f"{row.pcc:.4f}"
        rmse_text = f"{row.rmse:.4f}"
        if hasattr(row, "pcc_std") and pd.notna(row.pcc_std):
            pcc_text = f"{row.pcc:.4f} ± {row.pcc_std:.4f}"
        if hasattr(row, "rmse_std") and pd.notna(row.rmse_std):
            rmse_text = f"{row.rmse:.4f} ± {row.rmse_std:.4f}"
        folds = int(getattr(row, "folds", 1))
        lines.append(f"| {row.method} | {folds} | {int(row.n)} | {pcc_text} | {row.median_pcc:.4f} | {rmse_text} | {row.mae:.4f} | {row.r2:.4f} |\n")
    lines.append("\n## Key Takeaway\n")
    lines.append(
        f"On CR_RVS subject-level external validation, REM-AlignNet-MTL improves shape-200 PCC by "
        f"{rem.pcc - tcn.pcc:+.4f} over the CW-STFT + TCN baseline. Segment-level REM wins "
        f"{(seg['delta_pcc'] > 0).mean() * 100:.1f}% of windows.\n"
    )
    lines.append("\n## State-Level Result\n")
    lines.append("| State | N | REM PCC | TCN PCC | Gain | REM win rate |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in state.itertuples(index=False):
        lines.append(f"| {row.scene} | {int(row.n)} | {row.rem_pcc:.4f} | {row.tcn_pcc:.4f} | {row.delta_pcc:.4f} | {row.rem_win_rate * 100:.1f}% |\n")
    lines.append("\n## Figures\n")
    for name in [
        "fig_crrvs_external_main_metrics.png",
        "fig_crrvs_external_foldwise_pcc.png",
        "fig_crrvs_external_pcc_cdf.png",
        "fig_crrvs_external_statewise.png",
        "fig_crrvs_external_wave_examples.png",
    ]:
        lines.append(f"- `{name}`\n")
    (out / "CR_RVS_EXTERNAL_REPORT.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rem_tag", default="q1_crrvs_remalign_mtl_fold1")
    parser.add_argument("--tcn_tag", default="q1_crrvs_wsst_tcn_fold1")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--output_dir", default=str(EXP / "q1_crrvs_external_report_fold1"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    _style()
    folds = args.folds if args.folds else [args.fold]
    fold_summary = _load_all_summaries(args.rem_tag, args.tcn_tag, folds)
    summary = _aggregate(fold_summary) if len(folds) > 1 else fold_summary.assign(folds=1, pcc_std=0.0, rmse_std=0.0)
    seg = _load_all_segments(args.rem_tag, args.tcn_tag, folds)
    fold_summary.to_csv(out / "fold_summary_crrvs_external.csv", index=False)
    summary.to_csv(out / "summary_crrvs_external.csv", index=False)
    seg.to_csv(out / "segment_crrvs_external.csv", index=False)
    plot_main(summary, out)
    plot_foldwise(fold_summary, out)
    plot_cdf(seg, out)
    state = plot_state(seg, out)
    state.to_csv(out / "state_crrvs_external.csv", index=False)
    examples = plot_waves(args.rem_tag, args.tcn_tag, folds[0], out)
    examples.to_csv(out / "examples_crrvs_external.csv", index=False)
    write_markdown(out, summary, state, seg)
    print(summary.to_string(index=False))
    print(f"\nReport written to: {out}")


if __name__ == "__main__":
    main()
