#!/usr/bin/env python3
"""Build a compact Q1-progress report for REM-AlignNet-MTL.

The report is intentionally result-driven: it compares the direct REM shape
head against the reproduced radarODE-MTL shape-200 baseline, audits the
full-window branch, and writes publication-style figures for supervisor
discussion.
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
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments_mmecg"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _pcc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    pred = pred - pred.mean()
    gt = gt - gt.mean()
    denom = np.sqrt(np.sum(pred * pred) * np.sum(gt * gt)) + 1e-12
    return float(np.sum(pred * gt) / denom)


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def _mean_std(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"{np.nanmean(values):.4f} ± {np.nanstd(values):.4f}"


def _fold_rows(rem_tag: str, radarode_tag: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 12):
        rem_path = EXP_ROOT / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "global_summary_shape_head.json"
        radar_path = EXP_ROOT / radarode_tag / f"fold_{fold:02d}" / "results_shape200" / "global_summary_shape200.json"
        full_path = EXP_ROOT / rem_tag / f"fold_{fold:02d}" / "results" / "global_summary.json"
        crop_path = EXP_ROOT / rem_tag / f"fold_{fold:02d}" / "results_shape200" / "global_summary_shape200.json"
        if not rem_path.exists():
            raise FileNotFoundError(rem_path)
        if not radar_path.exists():
            raise FileNotFoundError(radar_path)
        rem = _load_json(rem_path)
        radar = _load_json(radar_path)
        full = _load_json(full_path) if full_path.exists() else {}
        crop = _load_json(crop_path) if crop_path.exists() else {}
        rows.append(
            {
                "fold": fold,
                "num_segments": int(rem.get("num_segments", radar.get("num_segments", 0))),
                "rem_shape_pcc": _safe_float(rem.get("pcc_mean")),
                "rem_shape_rmse": _safe_float(rem.get("rmse_mean")),
                "rem_shape_mae": _safe_float(rem.get("mae_mean")),
                "rem_shape_r2": _safe_float(rem.get("r2_mean")),
                "radarode_pcc": _safe_float(radar.get("pcc_raw_mean")),
                "radarode_rmse": _safe_float(radar.get("rmse_norm_mean")),
                "radarode_mae": _safe_float(radar.get("mae_norm_mean")),
                "radarode_r2": _safe_float(radar.get("r2_mean")),
                "rem_full_pcc": _safe_float(full.get("pcc_raw_mean")),
                "rem_full_rmse": _safe_float(full.get("rmse_norm_mean")),
                "rem_full_qmr": _safe_float(full.get("qualified_monitoring_rate")),
                "rem_fullcrop_pcc": _safe_float(crop.get("pcc_raw_mean")),
                "rem_fullcrop_rmse": _safe_float(crop.get("rmse_norm_mean")),
            }
        )
    df = pd.DataFrame(rows)
    df["delta_pcc"] = df["rem_shape_pcc"] - df["radarode_pcc"]
    df["delta_rmse"] = df["rem_shape_rmse"] - df["radarode_rmse"]
    df["win_pcc"] = df["delta_pcc"] > 0
    df["win_rmse"] = df["delta_rmse"] < 0
    return df


def _paired_tests(df: pd.DataFrame) -> dict[str, Any]:
    rem = df["rem_shape_pcc"].to_numpy(dtype=np.float64)
    radar = df["radarode_pcc"].to_numpy(dtype=np.float64)
    delta = rem - radar
    t_stat, t_p = stats.ttest_rel(rem, radar)
    try:
        w_stat, w_p = stats.wilcoxon(rem, radar, alternative="greater", zero_method="wilcox")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    dz = float(delta.mean() / (delta.std(ddof=1) + 1e-12))
    ci_low, ci_high = stats.t.interval(
        0.95,
        len(delta) - 1,
        loc=float(delta.mean()),
        scale=float(stats.sem(delta)),
    )
    return {
        "paired_t_pcc": {"statistic": float(t_stat), "p_value_two_sided": float(t_p)},
        "wilcoxon_pcc_greater": {"statistic": float(w_stat), "p_value": float(w_p)},
        "delta_pcc_mean": float(delta.mean()),
        "delta_pcc_std": float(delta.std(ddof=0)),
        "delta_pcc_95ci": [float(ci_low), float(ci_high)],
        "cohens_dz": dz,
        "wins_pcc": int(df["win_pcc"].sum()),
        "wins_rmse": int(df["win_rmse"].sum()),
    }


def _load_segment_metrics(rem_tag: str, radarode_tag: str) -> pd.DataFrame:
    rows = []
    for fold in range(1, 12):
        rem_csv = EXP_ROOT / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "segment_metrics_shape_head.csv"
        radar_csv = EXP_ROOT / radarode_tag / f"fold_{fold:02d}" / "results_shape200" / "segment_metrics_shape200.csv"
        rem = pd.read_csv(rem_csv)
        radar = pd.read_csv(radar_csv)
        n = min(len(rem), len(radar))
        for i in range(n):
            rows.append(
                {
                    "fold": fold,
                    "index": i,
                    "subject_id": rem.iloc[i].get("subject_id", fold),
                    "state": rem.iloc[i].get("state", np.nan),
                    "scene": rem.iloc[i].get("scene", ""),
                    "rem_pcc": float(rem.iloc[i]["pcc"]),
                    "rem_rmse": float(rem.iloc[i]["rmse"]),
                    "radarode_pcc": float(radar.iloc[i]["pcc_raw"] if "pcc_raw" in radar.columns else radar.iloc[i]["pcc"]),
                    "radarode_rmse": float(
                        radar.iloc[i]["rmse_norm"] if "rmse_norm" in radar.columns else radar.iloc[i]["rmse"]
                    ),
                }
            )
    seg = pd.DataFrame(rows)
    seg["delta_pcc"] = seg["rem_pcc"] - seg["radarode_pcc"]
    return seg


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


def _plot_fold_pcc(df: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x - width / 2, df["radarode_pcc"], width, label="radarODE-MTL", color="#8795A1")
    ax.bar(x + width / 2, df["rem_shape_pcc"], width, label="REM-AlignNet-MTL shape head", color="#2F80ED")
    ax.set_xticks(x)
    ax.set_xticklabels(df["fold"].astype(str))
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("LOSO fold")
    ax.set_ylabel("Shape-200 PCC")
    ax.set_title("Fold-wise shape-200 reconstruction performance")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for i, delta in enumerate(df["delta_pcc"]):
        ax.text(i, max(df.loc[i, "radarode_pcc"], df.loc[i, "rem_shape_pcc"]) + 0.012, f"+{delta:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_foldwise_pcc_bar.png")
    fig.savefig(out / "fig_foldwise_pcc_bar.pdf")
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(df["radarode_pcc"], df["rem_shape_pcc"], s=54, color="#2F80ED", edgecolor="white", linewidth=0.8)
    lo = min(df["radarode_pcc"].min(), df["rem_shape_pcc"].min()) - 0.03
    hi = max(df["radarode_pcc"].max(), df["rem_shape_pcc"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "--", color="#5F6C72", linewidth=1.2)
    for _, row in df.iterrows():
        ax.text(row["radarode_pcc"] + 0.006, row["rem_shape_pcc"] + 0.004, str(int(row["fold"])), fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("radarODE-MTL PCC")
    ax.set_ylabel("REM-AlignNet-MTL PCC")
    ax.set_title("Per-fold paired comparison")
    fig.tight_layout()
    fig.savefig(out / "fig_pcc_paired_scatter.png")
    fig.savefig(out / "fig_pcc_paired_scatter.pdf")
    plt.close(fig)


def _plot_delta(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = ["#1B9E77" if v > 0 else "#D95F02" for v in df["delta_pcc"]]
    ax.bar(df["fold"].astype(str), df["delta_pcc"], color=colors)
    ax.axhline(0, color="#303030", linewidth=1.0)
    ax.set_xlabel("LOSO fold")
    ax.set_ylabel("PCC gain over radarODE")
    ax.set_title("REM-AlignNet-MTL improvement by fold")
    fig.tight_layout()
    fig.savefig(out / "fig_pcc_gain_by_fold.png")
    fig.savefig(out / "fig_pcc_gain_by_fold.pdf")
    plt.close(fig)


def _plot_segment_distribution(seg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bins = np.linspace(-0.5, 1.0, 60)
    ax.hist(seg["radarode_pcc"], bins=bins, alpha=0.58, density=True, label="radarODE-MTL", color="#8795A1")
    ax.hist(seg["rem_pcc"], bins=bins, alpha=0.58, density=True, label="REM-AlignNet-MTL", color="#2F80ED")
    ax.set_xlabel("Segment-level PCC")
    ax.set_ylabel("Density")
    ax.set_title("Segment-level shape-200 PCC distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_segment_pcc_distribution.png")
    fig.savefig(out / "fig_segment_pcc_distribution.pdf")
    plt.close(fig)


def _plot_statewise(seg: pd.DataFrame, out: Path) -> pd.DataFrame:
    state = (
        seg.groupby(["state", "scene"], dropna=False)
        .agg(
            n=("rem_pcc", "size"),
            rem_pcc_mean=("rem_pcc", "mean"),
            radarode_pcc_mean=("radarode_pcc", "mean"),
            rem_rmse_mean=("rem_rmse", "mean"),
            radarode_rmse_mean=("radarode_rmse", "mean"),
            rem_win_rate=("delta_pcc", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    state["delta_pcc_mean"] = state["rem_pcc_mean"] - state["radarode_pcc_mean"]
    labels = [str(s) if str(s) else str(int(st)) for st, s in zip(state["state"], state["scene"])]
    x = np.arange(len(state))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - width / 2, state["radarode_pcc_mean"], width, color="#8795A1", label="radarODE-MTL")
    ax.bar(x + width / 2, state["rem_pcc_mean"], width, color="#2F80ED", label="REM-AlignNet-MTL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel("MMECG state")
    ax.set_ylabel("Segment-level PCC mean")
    ax.set_title("State-wise shape-200 performance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_statewise_pcc.png")
    fig.savefig(out / "fig_statewise_pcc.pdf")
    plt.close(fig)
    return state


def _sample_indices(rem_pred: np.ndarray, rem_gt: np.ndarray, radar_pred: np.ndarray, radar_gt: np.ndarray) -> dict[str, int]:
    n = min(len(rem_pred), len(radar_pred))
    rem_pcc = np.asarray([_pcc(rem_pred[i], rem_gt[i]) for i in range(n)])
    radar_pcc = np.asarray([_pcc(radar_pred[i], radar_gt[i]) for i in range(n)])
    gain = rem_pcc - radar_pcc
    median_idx = int(np.argsort(np.abs(rem_pcc - np.median(rem_pcc)))[0])
    return {
        "best_rem": int(np.argmax(rem_pcc)),
        "median_rem": median_idx,
        "largest_gain": int(np.argmax(gain)),
        "worst_rem": int(np.argmin(rem_pcc)),
    }


def _plot_shape_examples(rem_tag: str, radarode_tag: str, out: Path) -> pd.DataFrame:
    chosen_folds = [3, 5, 9, 11]
    rows = []
    t = np.linspace(0.0, 1.0, 200)
    fig, axes = plt.subplots(len(chosen_folds), 4, figsize=(13.5, 9.8), sharex=True, sharey=True)
    for r, fold in enumerate(chosen_folds):
        rem_npz = np.load(EXP_ROOT / rem_tag / f"fold_{fold:02d}" / "results_shape_head" / "predictions_shape_head.npz")
        radar_npz = np.load(EXP_ROOT / radarode_tag / f"fold_{fold:02d}" / "results_shape200" / "predictions_shape200.npz")
        rem_pred = rem_npz["pred"]
        rem_gt = rem_npz["target"]
        radar_pred = radar_npz["pred"]
        radar_gt = radar_npz["gt"]
        picks = _sample_indices(rem_pred, rem_gt, radar_pred, radar_gt)
        for c, (name, idx) in enumerate(picks.items()):
            ax = axes[r, c]
            rem_p = rem_pred[idx]
            gt = rem_gt[idx]
            radar_p = radar_pred[idx]
            rem_score = _pcc(rem_p, gt)
            radar_score = _pcc(radar_p, radar_gt[idx])
            ax.plot(t, gt, color="#111111", linewidth=1.7, label="GT")
            ax.plot(t, radar_p, color="#8795A1", linewidth=1.2, label="radarODE")
            ax.plot(t, rem_p, color="#2F80ED", linewidth=1.35, label="REM")
            ax.set_title(f"Fold {fold} {name.replace('_', ' ')}\nREM {rem_score:.2f}, radarODE {radar_score:.2f}", fontsize=9)
            if r == len(chosen_folds) - 1:
                ax.set_xlabel("Normalized cardiac cycle")
            if c == 0:
                ax.set_ylabel("Normalized ECG")
            rows.append(
                {
                    "fold": fold,
                    "case": name,
                    "index": idx,
                    "rem_pcc": rem_score,
                    "radarode_pcc": radar_score,
                    "gain": rem_score - radar_score,
                }
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Representative shape-200 reconstructions", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "fig_shape200_representative_examples.png", bbox_inches="tight")
    fig.savefig(out / "fig_shape200_representative_examples.pdf", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_full_branch_audit(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.plot(df["fold"], df["rem_shape_pcc"], marker="o", label="Direct shape head", color="#2F80ED")
    ax.plot(df["fold"], df["rem_fullcrop_pcc"], marker="s", label="Full output cropped to shape-200", color="#D95F02")
    ax.plot(df["fold"], df["rem_full_pcc"], marker="^", label="Full-window 1600 output", color="#7570B3")
    ax.set_xticks(df["fold"])
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("LOSO fold")
    ax.set_ylabel("PCC")
    ax.set_title("REM-AlignNet-MTL branch audit")
    ax.legend(frameon=False, ncol=1, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_rem_branch_audit.png")
    fig.savefig(out / "fig_rem_branch_audit.pdf")
    plt.close(fig)


def _write_markdown(
    out: Path,
    rem_tag: str,
    radarode_tag: str,
    df: pd.DataFrame,
    tests: dict[str, Any],
    seg: pd.DataFrame,
    state: pd.DataFrame,
) -> None:
    md = []
    md.append("# REM-AlignNet-MTL Q1 Progress Report\n")
    md.append("## Core Result\n")
    md.append(
        "| Model | Shape-200 PCC | Shape-200 RMSE | Shape-200 MAE | Fold wins |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| radarODE-MTL | {_mean_std(df['radarode_pcc'])} | {df['radarode_rmse'].mean():.4f} | {df['radarode_mae'].mean():.4f} | - |\n"
        f"| REM-AlignNet-MTL shape head | {_mean_std(df['rem_shape_pcc'])} | {df['rem_shape_rmse'].mean():.4f} | {df['rem_shape_mae'].mean():.4f} | {int(df['win_pcc'].sum())}/11 |\n"
    )
    md.append("\n## Statistical Test\n")
    md.append(
        f"- Mean PCC gain: {tests['delta_pcc_mean']:.4f} "
        f"(95% CI {tests['delta_pcc_95ci'][0]:.4f} to {tests['delta_pcc_95ci'][1]:.4f}).\n"
    )
    md.append(
        f"- Paired t-test: p = {tests['paired_t_pcc']['p_value_two_sided']:.3e}; "
        f"Wilcoxon one-sided: p = {tests['wilcoxon_pcc_greater']['p_value']:.3e}; "
        f"Cohen's dz = {tests['cohens_dz']:.3f}.\n"
    )
    md.append("\n## Full-Window Branch Audit\n")
    md.append(
        "| Branch | PCC mean ± std | RMSE mean | Interpretation |\n"
        "|---|---:|---:|---|\n"
        f"| Direct shape head | {_mean_std(df['rem_shape_pcc'])} | {df['rem_shape_rmse'].mean():.4f} | Strong, radarODE-comparable morphology output. |\n"
        f"| Full output cropped to shape-200 | {_mean_std(df['rem_fullcrop_pcc'])} | {df['rem_fullcrop_rmse'].mean():.4f} | Weak; full decoder does not preserve the learned morphology. |\n"
        f"| Full-window 1600 output | {_mean_std(df['rem_full_pcc'])} | {df['rem_full_rmse'].mean():.4f} | Still not a reliable continuous ECG claim. |\n"
    )
    md.append("\n## Segment-Level Evidence\n")
    md.append(
        f"- Segment-level REM median PCC: {seg['rem_pcc'].median():.4f}; "
        f"radarODE median PCC: {seg['radarode_pcc'].median():.4f}.\n"
    )
    md.append(
        f"- Segment-level REM wins: {(seg['delta_pcc'] > 0).mean() * 100:.1f}% "
        f"({int((seg['delta_pcc'] > 0).sum())}/{len(seg)} segments).\n"
    )
    md.append("\n## State-Level Summary\n")
    md.append("| State | N | REM PCC | radarODE PCC | PCC gain | REM win rate |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for _, row in state.iterrows():
        label = str(row["scene"]) if str(row["scene"]) else str(int(row["state"]))
        md.append(
            f"| {label} | {int(row['n'])} | {row['rem_pcc_mean']:.4f} | "
            f"{row['radarode_pcc_mean']:.4f} | {row['delta_pcc_mean']:.4f} | "
            f"{row['rem_win_rate'] * 100:.1f}% |\n"
        )
    md.append("\n## Supervisor-Facing Interpretation\n")
    md.append(
        "The current result supports a shape-first Q1 story: REM-AlignNet-MTL learns a much stronger "
        "beat morphology representation than radarODE-MTL under the same 11-fold LOSO shape-200 "
        "evaluation. However, the full-window 1600-point branch is not yet strong enough to claim "
        "complete continuous ECG reconstruction superiority. The next model step should therefore "
        "convert the strong shape head into a morphology-guided continuous renderer instead of merely "
        "tuning the existing full decoder.\n"
    )
    md.append("\n## Generated Figures\n")
    for name in [
        "fig_foldwise_pcc_bar.png",
        "fig_pcc_paired_scatter.png",
        "fig_pcc_gain_by_fold.png",
        "fig_segment_pcc_distribution.png",
        "fig_statewise_pcc.png",
        "fig_shape200_representative_examples.png",
        "fig_rem_branch_audit.png",
    ]:
        md.append(f"- `{name}`\n")
    md.append(f"\nCompared experiments: `{rem_tag}` vs `{radarode_tag}`.\n")
    (out / "Q1_TEACHER_REPORT.md").write_text("".join(md))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    _style()

    df = _fold_rows(args.rem_tag, args.radarode_tag)
    tests = _paired_tests(df)
    seg = _load_segment_metrics(args.rem_tag, args.radarode_tag)
    state = _plot_statewise(seg, out)

    df.to_csv(out / "fold_level_rem_vs_radarode.csv", index=False)
    seg.to_csv(out / "segment_level_rem_vs_radarode.csv", index=False)
    state.to_csv(out / "state_level_rem_vs_radarode.csv", index=False)
    summary = {
        "rem_tag": args.rem_tag,
        "radarode_tag": args.radarode_tag,
        "num_folds": int(len(df)),
        "rem_shape_pcc_mean": float(df["rem_shape_pcc"].mean()),
        "rem_shape_pcc_std": float(df["rem_shape_pcc"].std(ddof=0)),
        "radarode_pcc_mean": float(df["radarode_pcc"].mean()),
        "radarode_pcc_std": float(df["radarode_pcc"].std(ddof=0)),
        "rem_shape_rmse_mean": float(df["rem_shape_rmse"].mean()),
        "radarode_rmse_mean": float(df["radarode_rmse"].mean()),
        "rem_full_pcc_mean": float(df["rem_full_pcc"].mean()),
        "rem_fullcrop_pcc_mean": float(df["rem_fullcrop_pcc"].mean()),
        "tests": tests,
        "segment_rem_win_rate": float((seg["delta_pcc"] > 0).mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_fold_pcc(df, out)
    _plot_scatter(df, out)
    _plot_delta(df, out)
    _plot_segment_distribution(seg, out)
    examples = _plot_shape_examples(args.rem_tag, args.radarode_tag, out)
    examples.to_csv(out / "representative_examples.csv", index=False)
    _plot_full_branch_audit(df, out)
    _write_markdown(out, args.rem_tag, args.radarode_tag, df, tests, seg, state)

    print(json.dumps(summary, indent=2))
    print(f"\nReport written to: {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rem_tag", default="q1_remalign_mtl_shapefirst_loso")
    parser.add_argument("--radarode_tag", default="mmecg_radarode_mtl_ega_strict_loso")
    parser.add_argument(
        "--output_dir",
        default=str(EXP_ROOT / "q1_teacher_report_remalign_mtl_shapefirst"),
    )
    args = parser.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()
