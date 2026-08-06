#!/usr/bin/env python3
"""Build paper-first comparison tables and figures.

The report separates directly comparable reproduced/local results from
published reported-only numbers whose datasets or split protocols differ.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                values.append("")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path, color_col: str | None = None) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(df)), 4.2), dpi=170)
    if color_col:
        palette = {
            "proposed_pilot": "#59A14F",
            "adapted_reproduction": "#E15759",
            "local_baseline": "#4C78A8",
            "reported_same_dataset": "#F28E2B",
            "adapted_code_or_reported_only": "#B07AA1",
            "literature_pending": "#9C755F",
        }
        colors = [palette.get(str(v), "#777777") for v in df[color_col]]
    else:
        colors = "#4C78A8"
    ax.bar(range(len(df)), df[y], color=colors)
    ax.set_xticks(range(len(df)), df[x], rotation=24, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title)
    upper = max(1.0, float(df[y].max()) * 1.15)
    ax.set_ylim(0, upper)
    for i, v in enumerate(df[y]):
        ax.text(i, min(float(v) + upper * 0.025, upper * 0.97), f"{float(v):.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")


def build_report(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = _load_csv(ROOT / "docs" / "LITERATURE_COMPARISON_LEDGER.csv")
    mmecg = _load_csv(ROOT / "experiments_mmecg" / "q1_paper_style_report" / "summary_all_methods.csv")
    crrvs = _load_csv(ROOT / "experiments_mmecg" / "q1_crrvs_external_report_5fold" / "summary_crrvs_external.csv")

    mm_shape = (
        mmecg[mmecg["output_space"].eq("Shape-200")]
        .loc[:, ["method", "available_folds", "total_segments", "pcc_mean", "pcc_std", "pcc_median", "rmse_mean", "mae_mean", "r2_mean"]]
        .sort_values("pcc_mean", ascending=False)
        .reset_index(drop=True)
    )
    mm_shape["comparison_class"] = mm_shape["method"].map(
        {
            "REM-AlignNet-MTL": "proposed_pilot",
            "radarODE-MTL": "adapted_reproduction",
            "ICONIP BeatAware": "local_baseline",
            "WSST-TCN": "local_baseline",
            "REM-Event": "local_baseline",
            "RSSRNet": "adapted_reproduction",
            "AirECG": "adapted_reproduction",
        }
    ).fillna("local_baseline")

    mm_cont = (
        mmecg[mmecg["output_space"].eq("8-s continuous")]
        .loc[:, ["method", "available_folds", "total_segments", "pcc_mean", "pcc_std", "pcc_median", "rmse_mean", "mae_mean", "r2_mean", "qmr_mean", "r_f1_mean"]]
        .sort_values("pcc_mean", ascending=False)
        .reset_index(drop=True)
    )

    vmd_rows = ledger[
        ledger["comparison_status"].eq("reported_same_dataset")
        & ledger["dataset_in_paper"].str.contains("Nature", na=False)
        & ledger["reported_key_numbers"].str.contains("PCC", na=False)
    ].copy()
    published_crrvs = []
    for _, row in vmd_rows.iterrows():
        text = str(row["reported_key_numbers"])
        pcc_match = re.search(r"\bPCC\s+([0-9]*\.?[0-9]+)", text)
        rmse_match = re.search(r"\bRMSE\s+([0-9]*\.?[0-9]+)", text)
        pcc = float(pcc_match.group(1)) if pcc_match else None
        rmse = float(rmse_match.group(1)) if rmse_match else None
        published_crrvs.append(
            {
                "method": str(row["method"]) + " (published)",
                "source": str(row["paper"]),
                "comparison_status": str(row["comparison_status"]),
                "pcc": pcc,
                "rmse": rmse,
                "notes": str(row["notes"]),
            }
        )
    crrvs_ours = crrvs.rename(columns={"folds": "available_folds"}).copy()
    crrvs_ours["source"] = "local 5-fold subject-independent pilot"
    crrvs_ours["comparison_status"] = crrvs_ours["method"].map(
        {
            "CW-STFT + REM-AlignNet-MTL": "proposed_pilot",
            "CW-STFT + TCN": "local_baseline",
        }
    )
    crrvs_table = pd.concat(
        [
            crrvs_ours.loc[:, ["method", "source", "comparison_status", "pcc", "rmse"]],
            pd.DataFrame(published_crrvs).loc[:, ["method", "source", "comparison_status", "pcc", "rmse"]],
        ],
        ignore_index=True,
    ).sort_values("pcc", ascending=False).reset_index(drop=True)

    ledger.to_csv(out_dir / "literature_comparison_ledger.csv", index=False)
    mm_shape.to_csv(out_dir / "mmecg_shape200_comparison.csv", index=False)
    mm_cont.to_csv(out_dir / "mmecg_continuous_comparison_secondary.csv", index=False)
    crrvs_table.to_csv(out_dir / "crrvs_reported_vs_ours_comparison.csv", index=False)

    _write_markdown_table(mm_shape, out_dir / "mmecg_shape200_comparison.md")
    _write_markdown_table(mm_cont, out_dir / "mmecg_continuous_comparison_secondary.md")
    _write_markdown_table(crrvs_table, out_dir / "crrvs_reported_vs_ours_comparison.md")

    _plot_bar(
        mm_shape.head(8),
        x="method",
        y="pcc_mean",
        title="MMECG Shape-200 Morphology Comparison",
        out_path=out_dir / "mmecg_shape200_pcc.png",
        color_col="comparison_class",
    )
    _plot_bar(
        crrvs_table.dropna(subset=["pcc"]),
        x="method",
        y="pcc",
        title="CR_RVS / Nature Reported-vs-Ours PCC",
        out_path=out_dir / "crrvs_reported_vs_ours_pcc.png",
        color_col="comparison_status",
    )

    summary = [
        "# Paper-First Comparison Report",
        "",
        "## Key Findings",
        "",
        f"- MMECG Shape-200: best local method is `{mm_shape.iloc[0]['method']}` with PCC `{mm_shape.iloc[0]['pcc_mean']:.3f}`.",
        f"- CR_RVS/Nature: best listed method is `{crrvs_table.iloc[0]['method']}` with PCC `{crrvs_table.iloc[0]['pcc']:.3f}`.",
        "- Full-window continuous ECG is reported as a secondary capability, not the sole success criterion.",
        "- Sample-level or private-dataset literature results should remain reported-only unless reproduced under our subject-independent protocol.",
        "",
        "## Outputs",
        "",
        "- `mmecg_shape200_comparison.csv/md/png`",
        "- `mmecg_continuous_comparison_secondary.csv/md`",
        "- `crrvs_reported_vs_ours_comparison.csv/md/png`",
        "- `literature_comparison_ledger.csv`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="experiments_mmecg/q1_paper_first_comparison_report",
    )
    args = parser.parse_args()
    build_report(ROOT / args.output_dir)


if __name__ == "__main__":
    main()
