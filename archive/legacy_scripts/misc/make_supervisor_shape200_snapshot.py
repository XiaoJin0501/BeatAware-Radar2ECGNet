#!/usr/bin/env python3
"""Create an English supervisor-facing Shape-200 snapshot report.

The report intentionally separates reproduced/local Shape-200 results from
external literature values. It is meant for fast progress reporting, not as the
final radarODE-style long-term ECG protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments_mmecg"
OUT = EXP / "q1_supervisor_shape200_snapshot"


SOURCE_INFO = {
    "REM-AlignNet-MTL": {
        "source": "Proposed model, this work",
        "status": "local reproduction",
        "note": "Shape head output evaluated against ECG-derived center-cycle Shape-200 target.",
    },
    "radarODE-MTL": {
        "source": "Zhang et al., IEEE TIM 2025, DOI:10.1109/TIM.2025.3562975",
        "status": "adapted local reproduction",
        "note": "ECG_shape=(1,200) subtask; not the original paper's long-term ECG assembly result.",
    },
    "ICONIP BeatAware": {
        "source": "Our ICONIP2026 BeatAware-Radar2ECGNet version",
        "status": "local reproduction",
        "note": "Continuous 8-s output cropped/resampled to center-cycle Shape-200 for this snapshot.",
    },
    "WSST-TCN": {
        "source": "TCN representation baseline, this work",
        "status": "local reproduction",
        "note": "WSST/STFT representation plus lightweight TCN decoder.",
    },
    "REM-Event": {
        "source": "Event-structured ablation, this work",
        "status": "local reproduction",
        "note": "REM event layer without the final MTL morphology-first configuration.",
    },
    "AirECG": {
        "source": "Zhao et al., IMWUT 2024, DOI:10.1145/3678550",
        "status": "adapted local reproduction",
        "note": "External published architecture adapted to the local MMECG protocol.",
    },
    "RSSRNet": {
        "source": "Wu et al., IEEE Sensors Journal 2024, DOI:10.1109/JSEN.2023.3333025",
        "status": "adapted local reproduction",
        "note": "External published architecture adapted to the local MMECG protocol.",
    },
    "CNN": {
        "source": "Generic neural baseline",
        "status": "local baseline",
        "note": "Not a radar-to-ECG paper contribution; included as a sanity baseline.",
    },
    "LSTM": {
        "source": "Generic recurrent baseline",
        "status": "local baseline",
        "note": "Not a radar-to-ECG paper contribution; included as a sanity baseline.",
    },
    "BiLSTM": {
        "source": "Generic recurrent baseline",
        "status": "local baseline",
        "note": "Not a radar-to-ECG paper contribution; included as a sanity baseline.",
    },
    "GRU": {
        "source": "Generic recurrent baseline",
        "status": "local baseline",
        "note": "Not a radar-to-ECG paper contribution; included as a sanity baseline.",
    },
    "Conformer": {
        "source": "Gulati et al., Interspeech 2020 / arXiv:2005.08100",
        "status": "local baseline",
        "note": "Generic sequence backbone adapted as a baseline.",
    },
    "CW-STFT + TCN": {
        "source": "CR_RVS representation baseline, this work",
        "status": "local reproduction",
        "note": "24-GHz CW I/Q converted to phase/displacement-like signal and STFT.",
    },
    "CW-STFT + REM-AlignNet-MTL": {
        "source": "Proposed model, this work",
        "status": "local reproduction",
        "note": "CR_RVS external validation of the morphology-first REM-AlignNet-MTL shape head.",
    },
    "VMD-CBNeT": {
        "source": "Xu et al., IGARSS 2025, DOI:10.1109/IGARSS55030.2025.11243555",
        "status": "external reported result",
        "note": "Nature/CR_RVS value from the table provided by the user; not reproduced locally.",
    },
    "CNN+LSTM (VMD-CBNeT paper)": {
        "source": "Xu et al., IGARSS 2025, DOI:10.1109/IGARSS55030.2025.11243555",
        "status": "external reported result",
        "note": "Nature/CR_RVS value from the VMD-CBNeT paper table; not reproduced locally.",
    },
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def _source(method: str, field: str) -> str:
    return SOURCE_INFO.get(method, {}).get(field, "")


def load_mmecg_shape() -> pd.DataFrame:
    src = EXP / "q1_paper_style_report" / "summary_all_methods.csv"
    df = pd.read_csv(src)
    df = df[df["output_space"] == "Shape-200"].copy()
    df = df.rename(
        columns={
            "available_folds": "folds",
            "total_segments": "n",
            "pcc_mean": "pcc",
            "pcc_std": "pcc_std",
            "rmse_mean": "rmse",
            "mae_mean": "mae",
            "r2_mean": "r2",
        }
    )
    cols = ["method", "folds", "n", "pcc", "pcc_std", "pcc_median", "rmse", "mae", "r2"]
    df = df[cols].copy()
    df["dataset"] = "MMECG public"
    df["protocol"] = "11-fold strict LOSO; Shape-200 center-cycle morphology"
    df["status"] = df["method"].map(lambda x: _source(x, "status"))
    df["paper_source"] = df["method"].map(lambda x: _source(x, "source"))
    df["note"] = df["method"].map(lambda x: _source(x, "note"))
    return df.sort_values("pcc", ascending=False)


def load_crrvs_shape() -> pd.DataFrame:
    src = EXP / "q1_crrvs_external_report_5fold" / "summary_crrvs_external.csv"
    df = pd.read_csv(src)
    df = df.rename(columns={"folds": "folds", "n": "n", "pcc": "pcc", "rmse": "rmse"})
    df["dataset"] = "CR_RVS / Nature 24-GHz CW"
    df["protocol"] = "5-fold subject-independent CV; Shape-200 center-cycle morphology"
    df["status"] = df["method"].map(lambda x: _source(x, "status"))
    df["paper_source"] = df["method"].map(lambda x: _source(x, "source"))
    df["note"] = df["method"].map(lambda x: _source(x, "note"))
    keep = ["method", "folds", "n", "pcc", "pcc_std", "median_pcc", "rmse", "rmse_std", "mae", "r2", "dataset", "protocol", "status", "paper_source", "note"]
    df = df[keep].copy()

    # External values from the user-provided VMD-CBNeT table.
    external = pd.DataFrame(
        [
            {
                "method": "VMD-CBNeT",
                "folds": np.nan,
                "n": np.nan,
                "pcc": 0.856,
                "pcc_std": np.nan,
                "median_pcc": np.nan,
                "rmse": 0.112,
                "rmse_std": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "dataset": "CR_RVS / Nature 24-GHz CW",
                "protocol": "External paper protocol; likely not identical to our Shape-200 protocol",
                "status": _source("VMD-CBNeT", "status"),
                "paper_source": _source("VMD-CBNeT", "source"),
                "note": _source("VMD-CBNeT", "note"),
            },
            {
                "method": "CNN+LSTM (VMD-CBNeT paper)",
                "folds": np.nan,
                "n": np.nan,
                "pcc": 0.647,
                "pcc_std": np.nan,
                "median_pcc": np.nan,
                "rmse": 0.127,
                "rmse_std": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "dataset": "CR_RVS / Nature 24-GHz CW",
                "protocol": "External paper protocol; likely not identical to our Shape-200 protocol",
                "status": _source("CNN+LSTM (VMD-CBNeT paper)", "status"),
                "paper_source": _source("CNN+LSTM (VMD-CBNeT paper)", "source"),
                "note": _source("CNN+LSTM (VMD-CBNeT paper)", "note"),
            },
        ]
    )
    return pd.concat([df, external], ignore_index=True).sort_values("pcc", ascending=False)


def load_crrvs_state() -> pd.DataFrame:
    src = EXP / "q1_crrvs_external_report_5fold" / "state_crrvs_external.csv"
    return pd.read_csv(src)


def save_source_table(methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "paper_or_origin": _source(method, "source"),
                "result_status": _source(method, "status"),
                "important_note": _source(method, "note"),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates("method")
    out.to_csv(OUT / "model_sources.csv", index=False)
    return out


def plot_combined_pcc(mm: pd.DataFrame, cr: pd.DataFrame) -> None:
    mm_show = mm.copy().sort_values("pcc", ascending=True)
    cr_show = cr.copy().sort_values("pcc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1.55, 1.0]})
    for ax, data, title in [
        (axes[0], mm_show, "MMECG public: 11-fold LOSO"),
        (axes[1], cr_show, "CR_RVS/Nature: 5-fold subject-independent CV"),
    ]:
        y = np.arange(len(data))
        colors = ["#2F80ED" if "REM" in m else "#5F6C72" if "radarODE" in m or "VMD" in m else "#9AA5B1" for m in data["method"]]
        xerr = data["pcc_std"].to_numpy(dtype=float) if "pcc_std" in data else None
        if xerr is not None:
            xerr = np.where(np.isfinite(xerr), xerr, 0.0)
        ax.barh(y, data["pcc"], xerr=xerr, color=colors, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(data["method"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("PCC")
        ax.set_title(title)
        for yi, row in zip(y, data.itertuples(index=False)):
            ax.text(row.pcc + 0.015, yi, f"{row.pcc:.3f}", va="center", fontsize=8)
    fig.suptitle("Current Beat-Morphology / Shape-200 Snapshot", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_snapshot_pcc_two_datasets.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_snapshot_pcc_two_datasets.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_mmecg_metrics(mm: pd.DataFrame) -> None:
    methods = ["AirECG", "RSSRNet", "ICONIP BeatAware", "radarODE-MTL", "REM-AlignNet-MTL"]
    sub = mm.set_index("method")
    keys = [("pcc", "PCC", 0, 1), ("rmse", "RMSE", 0, 0.32), ("mae", "MAE", 0, 0.25), ("r2", "R2", -2.4, 0.9)]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2))
    for ax, (key, title, ymin, ymax) in zip(axes, keys):
        vals = [float(sub.loc[m, key]) if m in sub.index else np.nan for m in methods]
        colors = ["#8DA0CB", "#66C2A5", "#FC8D62", "#5F6C72", "#2F80ED"]
        ax.bar(np.arange(len(methods)), vals, color=colors)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=38, ha="right")
        ax.set_ylim(ymin, ymax)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + (ymax - ymin) * 0.025, f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("MMECG Shape-200 Metrics for Main Published/Proposed Methods", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mmecg_shape200_main_metrics.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_mmecg_shape200_main_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_crrvs_state(state: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(len(state))
    width = 0.36
    ax.bar(x - width / 2, state["tcn_pcc"], width=width, color="#9AA5B1", label="CW-STFT + TCN")
    ax.bar(x + width / 2, state["rem_pcc"], width=width, color="#2F80ED", label="CW-STFT + REM-AlignNet-MTL")
    ax.set_xticks(x)
    ax.set_xticklabels(state["scene"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Shape-200 PCC")
    ax.set_title("CR_RVS External Generalization by Physiological State")
    for i, row in enumerate(state.itertuples(index=False)):
        ax.text(i + width / 2, row.rem_pcc + 0.025, f"{row.rem_pcc:.3f}", ha="center", fontsize=8)
        ax.text(i - width / 2, row.tcn_pcc + 0.025, f"{row.tcn_pcc:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_crrvs_statewise_pcc_clean.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_crrvs_statewise_pcc_clean.pdf", bbox_inches="tight")
    plt.close(fig)


def _fmt_mean_std(mean: float, std: float | None) -> str:
    if std is None or not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def write_report(mm: pd.DataFrame, cr: pd.DataFrame, state: pd.DataFrame, sources: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Preliminary Shape-200 / Beat-Morphology Result Snapshot\n\n")
    lines.append("This report is prepared for a fast supervisor update. It summarizes the currently available beat-normalized Shape-200 results on MMECG and CR_RVS. It should not be presented as the final radarODE-style long-term ECG comparison yet.\n\n")
    lines.append("## Key Message\n\n")
    lines.append("- REM-AlignNet-MTL is currently the strongest local model in the beat-morphology / Shape-200 space on both datasets.\n")
    lines.append("- On MMECG, REM-AlignNet-MTL improves over the adapted radarODE-MTL ECG_shape reproduction by +{:.4f} PCC.\n".format(float(mm.loc[mm.method == "REM-AlignNet-MTL", "pcc"].iloc[0] - mm.loc[mm.method == "radarODE-MTL", "pcc"].iloc[0])))
    lines.append("- On CR_RVS, REM-AlignNet-MTL improves over CW-STFT + TCN by +{:.4f} PCC and is also above the VMD-CBNeT reported PCC from the user-provided table.\n".format(float(cr.loc[cr.method == "CW-STFT + REM-AlignNet-MTL", "pcc"].iloc[0] - cr.loc[cr.method == "CW-STFT + TCN", "pcc"].iloc[0])))
    lines.append("- Important limitation: Shape-200 evaluates normalized single-cycle morphology, not complete 8-s or trial-level continuous ECG reconstruction.\n\n")

    lines.append("## MMECG Public Dataset: Shape-200 Results\n\n")
    lines.append("Protocol: 11-fold strict LOSO. The target is the center cardiac cycle resampled to 200 points.\n\n")
    lines.append("| Method | Source / Origin | Status | Folds | N | PCC | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for row in mm.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.paper_source} | {row.status} | {int(row.folds)} | {int(row.n)} | "
            f"{_fmt_mean_std(row.pcc, row.pcc_std)} | {row.rmse:.4f} | {row.mae:.4f} | {row.r2:.4f} |\n"
        )

    lines.append("\n## CR_RVS / Nature Dataset: Shape-200 and Reported Literature Results\n\n")
    lines.append("Protocol for our local runs: 5-fold subject-independent cross-validation. External VMD-CBNeT values are reported values, not local reproductions, and may not use the identical Shape-200 protocol.\n\n")
    lines.append("| Method | Source / Origin | Status | Folds | N | PCC | RMSE | MAE | R2 |\n")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for row in cr.itertuples(index=False):
        folds = "" if not np.isfinite(row.folds) else str(int(row.folds))
        n = "" if not np.isfinite(row.n) else str(int(row.n))
        mae = "" if not np.isfinite(row.mae) else f"{row.mae:.4f}"
        r2 = "" if not np.isfinite(row.r2) else f"{row.r2:.4f}"
        lines.append(
            f"| {row.method} | {row.paper_source} | {row.status} | {folds} | {n} | "
            f"{_fmt_mean_std(row.pcc, row.pcc_std)} | {row.rmse:.4f} | {mae} | {r2} |\n"
        )

    lines.append("\n## CR_RVS State-Level External Generalization\n\n")
    lines.append("| State | N | REM PCC | TCN PCC | PCC Gain | REM Win Rate |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in state.itertuples(index=False):
        lines.append(f"| {row.scene} | {int(row.n)} | {row.rem_pcc:.4f} | {row.tcn_pcc:.4f} | {row.delta_pcc:.4f} | {row.rem_win_rate * 100:.1f}% |\n")

    lines.append("\n## Interpretation for Supervisor Meeting\n\n")
    lines.append("The current evidence supports the claim that REM-AlignNet-MTL learns a strong event-structured morphology representation. It is consistently strong on the MMECG public FMCW dataset and on the external CR_RVS/Nature 24-GHz CW dataset. However, the final journal claim should wait for a radarODE-style long-term ECG assembly protocol, because the original radarODE-MTL paper reports final long-term ECG reconstruction after combining ECG_shape, Anchor, and PPI.\n\n")
    lines.append("## Generated Figures\n\n")
    for name in [
        "fig_snapshot_pcc_two_datasets.png",
        "fig_mmecg_shape200_main_metrics.png",
        "fig_crrvs_statewise_pcc_clean.png",
    ]:
        lines.append(f"- `{name}`\n")
    lines.append("\n## Model Source Notes\n\n")
    lines.append("| Method | Paper / Origin | Status | Note |\n")
    lines.append("|---|---|---|---|\n")
    for row in sources.itertuples(index=False):
        lines.append(f"| {row.method} | {row.paper_or_origin} | {row.result_status} | {row.important_note} |\n")
    (OUT / "SUPERVISOR_SHAPE200_SNAPSHOT.md").write_text("".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    mm = load_mmecg_shape()
    cr = load_crrvs_shape()
    state = load_crrvs_state()
    all_methods = mm["method"].tolist() + cr["method"].tolist()
    sources = save_source_table(all_methods)
    mm.to_csv(OUT / "mmecg_shape200_results_with_sources.csv", index=False)
    cr.to_csv(OUT / "crrvs_shape200_results_with_sources.csv", index=False)
    state.to_csv(OUT / "crrvs_state_level_results.csv", index=False)
    plot_combined_pcc(mm, cr)
    plot_mmecg_metrics(mm)
    plot_crrvs_state(state)
    write_report(mm, cr, state, sources)
    print(f"Wrote supervisor snapshot to: {OUT}")
    print(mm[["method", "pcc", "pcc_std", "rmse", "r2"]].to_string(index=False))
    print()
    print(cr[["method", "pcc", "pcc_std", "rmse", "r2", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
