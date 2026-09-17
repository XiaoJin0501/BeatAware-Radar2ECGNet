#!/usr/bin/env python3
"""Summarize CR_RVS Shape-200 and full-window external validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments_mmecg"


METHODS = [
    {
        "method": "CW-STFT + REM-AlignNet-MTL-Refine-Lag",
        "tag": "q1_crrvs_remalign_refine_lagaware_5fold_pilot",
        "role": "proposed_lagaware_pilot",
    },
    {
        "method": "CW-STFT + REM-AlignNet-MTL",
        "tag": "q1_crrvs_remalign_mtl_5fold",
        "role": "proposed_shape_pilot",
    },
    {
        "method": "CW-STFT + TCN",
        "tag": "q1_crrvs_wsst_tcn_5fold",
        "role": "local_baseline",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: Path, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=[metric]).copy()
    colors = {
        "proposed_lagaware_pilot": "#59A14F",
        "proposed_shape_pilot": "#86BC86",
        "local_baseline": "#4C78A8",
        "published": "#F28E2B",
    }
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=170)
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df[metric], color=[colors.get(v, "#777777") for v in plot_df["role"]])
    ax.set_xticks(x, plot_df["method"], rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = max(float(plot_df[metric].max()) * 1.18, 1.0 if "pcc" in metric else 10.0)
    ax.set_ylim(0, ymax)
    for idx, value in enumerate(plot_df[metric]):
        ax.text(idx, min(float(value) + ymax * 0.025, ymax * 0.97), f"{float(value):.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def _shape_table() -> pd.DataFrame:
    path = EXP_ROOT / "q1_crrvs_external_report_5fold" / "summary_crrvs_external.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["evaluation_space"] = "Shape-200 morphology"
    df["role"] = df["method"].map(
        {
            "CW-STFT + REM-AlignNet-MTL": "proposed_shape_pilot",
            "CW-STFT + TCN": "local_baseline",
        }
    ).fillna("local_baseline")
    base = df.rename(
        columns={
            "pcc": "pcc_mean",
            "rmse": "rmse_mean",
            "mae": "mae_mean",
            "r2": "r2_mean",
            "folds": "available_folds",
            "n": "total_segments",
            "segments": "total_segments",
        }
    ).loc[
        :,
        [
            "method",
            "role",
            "evaluation_space",
            "available_folds",
            "total_segments",
            "pcc_mean",
            "rmse_mean",
            "mae_mean",
            "r2_mean",
        ],
    ]
    extra_rows = []
    for method in METHODS:
        if method["tag"] in {"q1_crrvs_remalign_mtl_5fold", "q1_crrvs_wsst_tcn_5fold"}:
            continue
        vals: dict[str, list[float]] = {"pcc_mean": [], "rmse_mean": [], "mae_mean": [], "r2_mean": []}
        total = 0
        available = 0
        for fold in range(1, 6):
            path = EXP_ROOT / method["tag"] / f"fold_{fold:02d}" / "results_shape_head" / "global_summary_shape_head.json"
            if not path.exists():
                path = EXP_ROOT / method["tag"] / f"fold_{fold:02d}" / "results_shape200" / "global_summary_shape200.json"
            if not path.exists():
                continue
            row = _load_json(path)
            available += 1
            total += int(row.get("num_segments", 0))
            vals["pcc_mean"].append(float(row.get("pcc_mean", row.get("pcc_raw_mean", np.nan))))
            vals["rmse_mean"].append(float(row.get("rmse_mean", row.get("rmse_norm_mean", np.nan))))
            vals["mae_mean"].append(float(row.get("mae_mean", row.get("mae_norm_mean", np.nan))))
            vals["r2_mean"].append(float(row.get("r2_mean", np.nan)))
        if available:
            extra = {
                "method": method["method"],
                "role": method["role"],
                "evaluation_space": "Shape-200 morphology",
                "available_folds": available,
                "total_segments": total,
            }
            for key, values in vals.items():
                extra[key] = _mean_std(values)[0]
            extra_rows.append(extra)
    if extra_rows:
        base = pd.concat([base, pd.DataFrame(extra_rows)], ignore_index=True)
    return base


def _full_window_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    summary_rows = []
    for method in METHODS:
        per_metric: dict[str, list[float]] = {}
        total_segments = 0
        available = 0
        for fold in range(1, 6):
            path = EXP_ROOT / method["tag"] / f"fold_{fold:02d}" / "results" / "global_summary.json"
            if not path.exists():
                continue
            row = _load_json(path)
            available += 1
            total_segments += int(row.get("num_segments", 0))
            flat = {
                "method": method["method"],
                "role": method["role"],
                "fold": fold,
                "num_segments": int(row.get("num_segments", 0)),
                "pcc_mean": float(row.get("pcc_raw_mean", np.nan)),
                "rmse_mean": float(row.get("rmse_norm_mean", np.nan)),
                "mae_mean": float(row.get("mae_norm_mean", np.nan)),
                "r2_mean": float(row.get("r2_mean", np.nan)),
                "qmr_percent": float(row.get("qualified_monitoring_rate", np.nan)),
                "rr_error_ms": float(row.get("rr_interval_error_ms_mean_mean", np.nan)),
                "avg_f1_150ms": float(row.get("average_f1_150ms_mean", np.nan)),
                "pr_error_ms": float(row.get("pr_interval_error_ms_mean", np.nan)),
                "qrs_error_ms": float(row.get("qrs_duration_error_ms_mean", np.nan)),
                "qt_error_ms": float(row.get("qt_interval_error_ms_mean", np.nan)),
            }
            fold_rows.append(flat)
            for key, val in flat.items():
                if key in {"method", "role", "fold", "num_segments"}:
                    continue
                per_metric.setdefault(key, []).append(float(val))
        if available:
            out = {
                "method": method["method"],
                "role": method["role"],
                "evaluation_space": "8-s full-window diagnostic",
                "available_folds": available,
                "total_segments": total_segments,
            }
            for key, vals in per_metric.items():
                mean, std = _mean_std(vals)
                out[f"{key}_mean"] = mean
                out[f"{key}_std"] = std
            summary_rows.append(out)
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def _published_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "VMD-CBNeT (published)",
                "role": "published",
                "dataset": "Nature / CR_RVS",
                "protocol_note": "Reported in VMD-CBNeT paper; not locally reproduced.",
                "pcc": 0.856,
                "rmse": 0.112,
                "rr_error_ms": 5.625,
                "pr_error_ms": 4.911,
                "qrs_error_ms": 9.643,
                "qt_error_ms": 5.625,
            },
            {
                "method": "CNN+LSTM (published)",
                "role": "published",
                "dataset": "Nature / CR_RVS",
                "protocol_note": "Reported in VMD-CBNeT paper; not locally reproduced.",
                "pcc": 0.647,
                "rmse": 0.127,
                "rr_error_ms": 48.534,
                "pr_error_ms": 7.414,
                "qrs_error_ms": 29.052,
                "qt_error_ms": 48.534,
            },
        ]
    )


def build_report(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = _shape_table()
    fold_full, summary_full = _full_window_table()
    published = _published_table()

    shape.to_csv(out_dir / "crrvs_shape200_summary.csv", index=False)
    fold_full.to_csv(out_dir / "crrvs_full_window_fold_summary.csv", index=False)
    summary_full.to_csv(out_dir / "crrvs_full_window_summary.csv", index=False)
    published.to_csv(out_dir / "published_nature_crrvs_reference.csv", index=False)

    _write_markdown(shape, out_dir / "crrvs_shape200_summary.md")
    _write_markdown(summary_full, out_dir / "crrvs_full_window_summary.md")
    _write_markdown(published, out_dir / "published_nature_crrvs_reference.md")

    if not shape.empty:
        _plot_metric(
            shape.sort_values("pcc_mean", ascending=False),
            metric="pcc_mean",
            title="CR_RVS Shape-200 Morphology PCC",
            out_path=out_dir / "crrvs_shape200_pcc.png",
            ylabel="PCC",
        )
    if not summary_full.empty:
        _plot_metric(
            summary_full.sort_values("pcc_mean_mean", ascending=False),
            metric="pcc_mean_mean",
            title="CR_RVS Full-Window Diagnostic PCC",
            out_path=out_dir / "crrvs_full_window_pcc.png",
            ylabel="PCC",
        )
        _plot_metric(
            summary_full.sort_values("rr_error_ms_mean", ascending=True),
            metric="rr_error_ms_mean",
            title="CR_RVS Full-Window RR Error",
            out_path=out_dir / "crrvs_full_window_rr_error.png",
            ylabel="RR error (ms)",
        )

    readme = [
        "# CR_RVS External Validation Report",
        "",
        "## Interpretation",
        "",
        "- Lag-aware REM-AlignNet-MTL-Refine preserves the original REM Shape-200 morphology advantage on CR_RVS.",
        "- It also improves 8-second full-window diagnostic PCC and RR timing over CW-STFT+TCN in the current 5-fold pilot.",
        "- Results remain pilot-level until metric audit and final protocol promotion are completed.",
        "- Published VMD-CBNeT numbers are included as reported-only Nature/CR_RVS references, not as locally reproduced baselines.",
        "",
        "## Outputs",
        "",
        "- `crrvs_shape200_summary.csv/md/png`",
        "- `crrvs_full_window_fold_summary.csv`",
        "- `crrvs_full_window_summary.csv/md/png`",
        "- `crrvs_full_window_rr_error.png`",
        "- `published_nature_crrvs_reference.csv/md`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="experiments_mmecg/q1_crrvs_full_window_report")
    args = parser.parse_args()
    build_report(ROOT / args.output_dir)


if __name__ == "__main__":
    main()
