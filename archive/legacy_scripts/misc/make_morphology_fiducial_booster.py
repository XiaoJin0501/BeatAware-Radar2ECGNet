"""Compute phase-region morphology metrics from existing Shape-200 predictions.

This is a no-retraining evidence booster for the Q1 paper. Shape-200 is an
R-to-R, min-max normalized cycle, so the metrics below are deliberately named as
phase-region morphology metrics rather than clinical QT/QRS diagnoses.

Outputs are written under experiments_mmecg/q1_morphology_fiducial_booster/
(ignored by git). The script itself is the reproducible artifact.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments_mmecg"
OUT = EXP / "q1_morphology_fiducial_booster"
OUT.mkdir(parents=True, exist_ok=True)

SHAPE_LEN = 200
SEEDS = [42, 123, 2024]

# Shape-200 cycles start/end near R peaks. These fixed phase regions are coarse
# morphology checks; they are not full clinical delineation.
REGIONS = {
    "qrs_edge": np.r_[0:24, 176:200],
    "st_t": np.r_[24:96],
    "diastolic": np.r_[96:176],
}
T_REGION = np.arange(32, 96)
P_REGION = np.arange(132, 176)


@dataclass(frozen=True)
class MethodSpec:
    dataset: str
    method: str
    tag_template: str
    folds: range
    result_dir: str
    npz_name: str
    seeds: tuple[int, ...] | None = None
    tier: str = "local"


METHODS = [
    MethodSpec(
        dataset="MMECG",
        method="REM-Net",
        tag_template="q1_g1_remnet_mmecg_seed{seed}",
        folds=range(1, 12),
        result_dir="results_shape_head",
        npz_name="predictions_shape_head.npz",
        seeds=tuple(SEEDS),
        tier="proposed_multi_seed",
    ),
    MethodSpec(
        dataset="MMECG",
        method="radarODE-MTL",
        tag_template="mmecg_radarode_mtl_ega_strict_loso",
        folds=range(1, 12),
        result_dir="results_shape200",
        npz_name="predictions_shape200.npz",
        tier="adapted",
    ),
    MethodSpec(
        dataset="MMECG",
        method="REM-Event",
        tag_template="q1_remalign_event_lowpeak_loso",
        folds=range(1, 12),
        result_dir="results_shape200",
        npz_name="predictions_shape200.npz",
        tier="ablation",
    ),
    MethodSpec(
        dataset="MMECG",
        method="WSST-TCN",
        tag_template="q1_remalign_wsst_tcn_loso",
        folds=range(1, 12),
        result_dir="results_shape200",
        npz_name="predictions_shape200.npz",
        tier="representation_baseline",
    ),
    MethodSpec(
        dataset="CR_RVS",
        method="REM-Net",
        tag_template="q1_g1_remnet_crrvs_seed{seed}",
        folds=range(1, 6),
        result_dir="results_shape_head",
        npz_name="predictions_shape_head.npz",
        seeds=tuple(SEEDS),
        tier="proposed_multi_seed",
    ),
    MethodSpec(
        dataset="CR_RVS",
        method="CW-STFT+TCN",
        tag_template="q1_crrvs_wsst_tcn_5fold",
        folds=range(1, 6),
        result_dir="results_shape200",
        npz_name="predictions_shape200.npz",
        tier="local_baseline",
    ),
]


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    pred = z["pred"].astype(np.float64)
    if "target" in z:
        target = z["target"].astype(np.float64)
    elif "gt" in z:
        target = z["gt"].astype(np.float64)
    else:
        raise KeyError(f"{path} has no target/gt key; keys={z.files}")
    if pred.ndim == 3:
        pred = pred[:, 0]
    if target.ndim == 3:
        target = target[:, 0]
    if pred.shape != target.shape or pred.shape[1] != SHAPE_LEN:
        raise ValueError(f"{path} shape mismatch: pred={pred.shape}, target={target.shape}")
    return pred, target


def _safe_pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a0 * a0).sum(axis=1) * (b0 * b0).sum(axis=1))
    return (a0 * b0).sum(axis=1) / np.maximum(den, 1e-12)


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    err = pred - target
    out["whole_pcc"] = float(np.nanmean(_safe_pcc(pred, target)))
    out["whole_mae"] = float(np.nanmean(np.abs(err)))
    out["whole_rmse"] = float(np.sqrt(np.nanmean(err * err)))

    for name, idx in REGIONS.items():
        p = pred[:, idx]
        t = target[:, idx]
        e = p - t
        out[f"{name}_pcc"] = float(np.nanmean(_safe_pcc(p, t)))
        out[f"{name}_mae"] = float(np.nanmean(np.abs(e)))
        out[f"{name}_bias"] = float(np.nanmean(e))
        out[f"{name}_area_mae"] = float(np.nanmean(np.abs(p.mean(axis=1) - t.mean(axis=1))))

    for name, idx in {"t_wave": T_REGION, "p_region": P_REGION}.items():
        p = pred[:, idx]
        t = target[:, idx]
        p_peak = idx[np.argmax(p, axis=1)]
        t_peak = idx[np.argmax(t, axis=1)]
        out[f"{name}_peak_phase_mae_pts"] = float(np.nanmean(np.abs(p_peak - t_peak)))
        out[f"{name}_peak_phase_mae_pct"] = float(np.nanmean(np.abs(p_peak - t_peak)) / SHAPE_LEN)
        p_amp = p[np.arange(len(p)), np.argmax(p, axis=1)]
        t_amp = t[np.arange(len(t)), np.argmax(t, axis=1)]
        out[f"{name}_peak_amp_mae"] = float(np.nanmean(np.abs(p_amp - t_amp)))

    return out


def _iter_method_paths(spec: MethodSpec):
    seeds = spec.seeds or (None,)
    for seed in seeds:
        tag = spec.tag_template.format(seed=seed) if seed is not None else spec.tag_template
        for fold in spec.folds:
            path = EXP / tag / f"fold_{fold:02d}" / spec.result_dir / spec.npz_name
            if path.exists():
                yield seed, fold, tag, path


def _iter_remnet_paths(dataset: str):
    prefix = "mmecg" if dataset == "MMECG" else "crrvs"
    folds = range(1, 12) if dataset == "MMECG" else range(1, 6)
    for seed in SEEDS:
        tag = f"q1_g1_remnet_{prefix}_seed{seed}"
        for fold in folds:
            path = EXP / tag / f"fold_{fold:02d}" / "results_shape_head" / "predictions_shape_head.npz"
            if path.exists():
                yield seed, fold, tag, path


def compute_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in METHODS:
        for seed, fold, tag, path in _iter_method_paths(spec):
            pred, target = _load_npz(path)
            m = _metrics(pred, target)
            row: dict[str, object] = {
                "dataset": spec.dataset,
                "method": spec.method,
                "tier": spec.tier,
                "seed": seed if seed is not None else "",
                "fold": fold,
                "tag": tag,
                "n": int(pred.shape[0]),
            }
            row.update(m)
            rows.append(row)
    return rows


def collect_remnet_predictions(dataset: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    preds = []
    targets = []
    meta: list[dict[str, object]] = []
    for seed, fold, tag, path in _iter_remnet_paths(dataset):
        pred, target = _load_npz(path)
        preds.append(pred)
        targets.append(target)
        for idx in range(pred.shape[0]):
            meta.append({"dataset": dataset, "seed": seed, "fold": fold, "tag": tag, "local_index": idx})
    if not preds:
        raise FileNotFoundError(f"No REM-Net predictions found for {dataset}")
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), meta


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # Average folds within each seed/method, then report across-seed statistics
    # for multi-seed REM-Net. Single-seed baselines use fold-macro mean/std.
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for r in rows:
        grouped.setdefault((str(r["dataset"]), str(r["method"]), str(r["seed"])), []).append(r)

    fold_macro_rows: list[dict[str, object]] = []
    metric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    for (dataset, method, seed), items in grouped.items():
        out: dict[str, object] = {"dataset": dataset, "method": method, "seed": seed, "n_folds": len(items)}
        for k in metric_keys:
            vals = np.array([float(x[k]) for x in items], dtype=float)
            out[k] = float(np.nanmean(vals))
            out[f"{k}_fold_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        fold_macro_rows.append(out)

    by_method: dict[tuple[str, str], list[dict[str, object]]] = {}
    for r in fold_macro_rows:
        by_method.setdefault((str(r["dataset"]), str(r["method"])), []).append(r)

    summary: list[dict[str, object]] = []
    for (dataset, method), items in by_method.items():
        out = {"dataset": dataset, "method": method, "n_seeds": len(items)}
        for k in metric_keys:
            vals = np.array([float(x[k]) for x in items], dtype=float)
            out[f"{k}_mean"] = float(np.nanmean(vals))
            out[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float(items[0][f"{k}_fold_std"])
        summary.append(out)
    return summary


def make_figures(summary: list[dict[str, object]]) -> None:
    def get(dataset: str, method: str, key: str) -> float:
        for r in summary:
            if r["dataset"] == dataset and r["method"] == method:
                return float(r[key])
        return float("nan")

    # Region PCC comparison.
    for dataset, methods in {
        "MMECG": ["REM-Net", "radarODE-MTL", "REM-Event", "WSST-TCN"],
        "CR_RVS": ["REM-Net", "CW-STFT+TCN"],
    }.items():
        labels = ["whole", "qrs_edge", "st_t", "diastolic"]
        x = np.arange(len(labels))
        width = 0.8 / len(methods)
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        for i, method in enumerate(methods):
            vals = [get(dataset, method, f"{lab}_pcc_mean") for lab in labels]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.2, 1.0)
        ax.set_ylabel("PCC")
        ax.set_title(f"{dataset}: region-wise Shape-200 morphology fidelity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"{dataset.lower()}_region_pcc.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / f"{dataset.lower()}_region_pcc.pdf", bbox_inches="tight")
        plt.close(fig)

    # Peak timing/amplitude diagnostics for proposed model.
    rows = [r for r in summary if r["method"] == "REM-Net"]
    labels = [r["dataset"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(rows))
    t_err = [float(r["t_wave_peak_phase_mae_pts_mean"]) for r in rows]
    p_err = [float(r["p_region_peak_phase_mae_pts_mean"]) for r in rows]
    ax.bar(x - 0.18, t_err, 0.36, label="T-region peak phase")
    ax.bar(x + 0.18, p_err, 0.36, label="P/pre-R region peak phase")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (Shape-200 points)")
    ax.set_title("REM-Net intra-cycle peak-location fidelity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "remnet_peak_phase_errors.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "remnet_peak_phase_errors.pdf", bbox_inches="tight")
    plt.close(fig)


def _shade_regions(ax) -> None:
    ax.axvspan(0, 24, color="#d62728", alpha=0.08, lw=0)
    ax.axvspan(176, 199, color="#d62728", alpha=0.08, lw=0)
    ax.axvspan(24, 96, color="#2ca02c", alpha=0.08, lw=0)
    ax.axvspan(96, 176, color="#7f7f7f", alpha=0.08, lw=0)
    ax.text(12, 1.03, "QRS-edge", ha="center", va="bottom", fontsize=7, color="#8c1d18")
    ax.text(60, 1.03, "ST/T", ha="center", va="bottom", fontsize=7, color="#1b6e1b")
    ax.text(136, 1.03, "diastolic", ha="center", va="bottom", fontsize=7, color="#555555")


def _example_indices(pccs: np.ndarray) -> list[tuple[str, int]]:
    order = np.argsort(pccs)
    quantiles = [("high", 0.90), ("median", 0.50), ("hard", 0.10)]
    return [(label, int(order[int(round(q * (len(order) - 1)))])) for label, q in quantiles]


def make_transparent_qualitative_figures() -> list[dict[str, object]]:
    selection_rows: list[dict[str, object]] = []
    x = np.arange(SHAPE_LEN)
    for dataset in ("MMECG", "CR_RVS"):
        pred, target, meta = collect_remnet_predictions(dataset)
        pccs = _safe_pcc(pred, target)
        mae = np.mean(np.abs(pred - target), axis=1)
        picks = _example_indices(pccs)

        fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.8), sharex=True)
        for col, (label, idx) in enumerate(picks):
            ax = axes[0, col]
            _shade_regions(ax)
            ax.plot(x, target[idx], color="black", lw=1.8, label="GT")
            ax.plot(x, pred[idx], color="#1f77b4", lw=1.7, label="REM-Net")
            ax.set_title(f"{label}: PCC={pccs[idx]:.3f}, MAE={mae[idx]:.3f}", fontsize=9)
            ax.set_ylim(-0.05, 1.12)
            if col == 0:
                ax.set_ylabel("normalized ECG")

            err_ax = axes[1, col]
            err_ax.axhline(0, color="black", lw=0.8, alpha=0.5)
            err_ax.fill_between(x, pred[idx] - target[idx], 0, color="#d62728", alpha=0.25)
            err_ax.plot(x, pred[idx] - target[idx], color="#d62728", lw=1.0)
            err_ax.set_ylim(-0.65, 0.65)
            err_ax.set_xlabel("Shape-200 phase point")
            if col == 0:
                err_ax.set_ylabel("prediction error")

            row = dict(meta[idx])
            row.update({"selection": label, "pcc": float(pccs[idx]), "mae": float(mae[idx])})
            selection_rows.append(row)

        fig.suptitle(
            f"{dataset}: transparent REM-Net qualitative examples (90th / median / 10th percentile)",
            fontsize=11,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=True)
        fig.subplots_adjust(bottom=0.11)
        fig.tight_layout()
        fig.savefig(OUT / f"{dataset.lower()}_transparent_qualitative_examples.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / f"{dataset.lower()}_transparent_qualitative_examples.pdf", bbox_inches="tight")
        plt.close(fig)

        # Dataset-level mean and error-band figure.
        p_mean, p_std = pred.mean(axis=0), pred.std(axis=0)
        t_mean, t_std = target.mean(axis=0), target.std(axis=0)
        abs_err = np.abs(pred - target)
        e_mean = abs_err.mean(axis=0)
        e_p75 = np.percentile(abs_err, 75, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)
        _shade_regions(ax1)
        ax1.plot(x, t_mean, color="black", lw=1.8, label="GT mean")
        ax1.fill_between(x, t_mean - t_std, t_mean + t_std, color="black", alpha=0.10, lw=0)
        ax1.plot(x, p_mean, color="#1f77b4", lw=1.8, label="REM-Net mean")
        ax1.fill_between(x, p_mean - p_std, p_mean + p_std, color="#1f77b4", alpha=0.14, lw=0)
        ax1.set_ylim(-0.08, 1.15)
        ax1.set_ylabel("normalized ECG")
        ax1.legend(fontsize=8)
        ax1.set_title(f"{dataset}: mean morphology with 1 SD envelope")

        ax2.plot(x, e_mean, color="#d62728", lw=1.6, label="mean |error|")
        ax2.fill_between(x, e_mean, e_p75, color="#d62728", alpha=0.16, lw=0, label="75th percentile |error|")
        ax2.set_ylabel("|prediction error|")
        ax2.set_xlabel("Shape-200 phase point")
        ax2.set_ylim(0, max(0.45, float(np.nanmax(e_p75)) * 1.05))
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"{dataset.lower()}_mean_error_band.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / f"{dataset.lower()}_mean_error_band.pdf", bbox_inches="tight")
        plt.close(fig)
    return selection_rows


def write_report(summary: list[dict[str, object]]) -> None:
    fields = [
        "dataset",
        "method",
        "whole_pcc_mean",
        "qrs_edge_pcc_mean",
        "st_t_pcc_mean",
        "diastolic_pcc_mean",
        "t_wave_peak_phase_mae_pts_mean",
        "t_wave_peak_amp_mae_mean",
        "p_region_peak_phase_mae_pts_mean",
        "p_region_peak_amp_mae_mean",
    ]
    lines = [
        "# Morphology Fiducial Booster Summary",
        "",
        "No retraining. Metrics are computed from existing Shape-200 prediction NPZ files.",
        "Regions are fixed R-to-R cycle phase regions, not clinical delineation.",
        "",
        "| Dataset | Method | Whole PCC | QRS-edge PCC | ST/T PCC | Diastolic PCC | T peak phase MAE | T amp MAE | P/pre-R phase MAE | P/pre-R amp MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(summary, key=lambda x: (str(x["dataset"]), -float(x["whole_pcc_mean"]))):
        vals = [r.get(f) for f in fields]
        lines.append(
            f"| {vals[0]} | {vals[1]} | {float(vals[2]):.3f} | {float(vals[3]):.3f} | "
            f"{float(vals[4]):.3f} | {float(vals[5]):.3f} | {float(vals[6]):.1f} | "
            f"{float(vals[7]):.3f} | {float(vals[8]):.1f} | {float(vals[9]):.3f} |"
        )
    lines += [
        "",
        "Qualitative panels use a transparent percentile rule: 90th percentile,",
        "median, and 10th percentile by per-sample Shape-200 PCC. See",
        "`transparent_qualitative_selection.csv`.",
        "",
        "Interpretation guardrail: these numbers can support morphology-structure claims,",
        "but they do not support QT/QRS interval superiority or continuous ECG reliability.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = compute_rows()
    _write_csv(OUT / "fold_level_phase_region_metrics.csv", rows)
    summary = aggregate(rows)
    _write_csv(OUT / "summary_phase_region_metrics.csv", summary)
    make_figures(summary)
    selection_rows = make_transparent_qualitative_figures()
    _write_csv(OUT / "transparent_qualitative_selection.csv", selection_rows)
    write_report(summary)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/make_morphology_fiducial_booster.py",
                "note": "No retraining; Shape-200 phase-region morphology metrics.",
                "regions": {k: v.tolist() for k, v in REGIONS.items()},
                "t_region": T_REGION.tolist(),
                "p_region": P_REGION.tolist(),
                "methods": [m.__dict__ for m in METHODS],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
