#!/usr/bin/env python3
"""make_rq1_extra_figures.py -- additional RQ1 manuscript figures.

F7  Per-fold paired comparison            -> reviewer: "fold variation not reported"
F8  Controlled ablation, multi-metric     -> reviewer: component attribution / novelty
F9  Learned delay delta_sys analysis      -> reviewer: "show the delay ... physiological?"
F10 Noise robustness and timing tolerance -> reviewer: generalization across modalities

Analysis-only: reads committed experiment outputs. Writes docs/paper/figures/.
"""
from __future__ import annotations

import csv, glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments_mmecg"
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "axes.axisbelow": True})
C_OURS, C_BASE, C_WARN, C_MUTE = "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def per_fold(tag, sub, keys):
    out = []
    stem = sub.split("results_")[1]
    for p in sorted(EXP.glob(f"{tag}/fold_*/{sub}/global_summary_{stem}.json")):
        d = json.loads(p.read_text())
        for k in keys:
            if d.get(k) is not None:
                out.append(float(d[k])); break
    return np.array(out)


# ---------------------------------------------------------------- F7
def fig_perfold():
    ours = per_fold("q1_final_remnet_incumbent_mmecg_seed42", "results_shape_head", ("pcc_mean",))
    base = per_fold("mmecg_radarode_mtl_ega_strict_loso", "results_shape200", ("pcc_raw_mean", "pcc_mean"))
    crr = per_fold("q1_final_remnet_incumbent_crrvs_seed42", "results_shape_head", ("pcc_mean",))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [2.4, 1]})

    x = np.arange(1, len(ours) + 1)
    for xi, o, r in zip(x, ours, base):
        a.plot([xi, xi], [r, o], color=C_MUTE, lw=1.0, zorder=1)
    a.plot(x, ours, "o", color=C_OURS, ms=6, zorder=3, label=f"REM-Net ({ours.mean():.3f}$\\pm${ours.std(ddof=1):.3f})")
    a.plot(x, base, "s", color=C_BASE, ms=5, zorder=3, label=f"radarODE-MTL ({base.mean():.3f}$\\pm${base.std(ddof=1):.3f})")
    a.axhline(ours.mean(), color=C_OURS, ls="--", lw=0.9, alpha=0.6)
    a.axhline(base.mean(), color=C_BASE, ls="--", lw=0.9, alpha=0.6)
    for f in (5, 9):
        a.annotate("hardest", xy=(f, ours[f - 1]), xytext=(f, ours[f - 1] + 0.11),
                   ha="center", fontsize=7.5, color=C_MUTE,
                   arrowprops=dict(arrowstyle="->", color=C_MUTE, lw=0.8))
    a.set_xticks(x); a.set_xlabel("held-out subject (LOSO fold)")
    a.set_ylabel("Shape-200 PCC"); a.set_ylim(0.35, 1.02)
    a.set_title(f"(a) MMECG: REM-Net higher on {int((ours > base).sum())}/{len(ours)} folds")
    a.legend(loc="lower right", fontsize=7.5)

    xb = np.arange(1, len(crr) + 1)
    b.plot(xb, crr, "o", color=C_OURS, ms=7)
    b.axhline(crr.mean(), color=C_WARN, ls="--", lw=1.2,
              label=f"mean {crr.mean():.3f}$\\pm${crr.std(ddof=1):.3f}")
    b.set_xticks(xb); b.set_ylim(0.35, 1.02)
    b.set_xlabel("subject-independent fold"); b.set_title("(b) CR-RVS (CW)")
    b.legend(loc="lower right", fontsize=7.5)
    save(fig, "F7_perfold_paired")


# ---------------------------------------------------------------- F8
def fig_ablation():
    rows = list(csv.DictReader(open(EXP / "q1_remnet_multimetric_ablation_gate" / "ablation_gate_summary.csv")))
    arms = [r["arm"] for r in rows]
    lab = {"A0": "A0", "A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4"}
    metrics = [("pcc_mean", "Shape-200 PCC", True), ("r2_mean", "$R^2$", True),
               ("rmse_mean", "RMSE", False), ("mae_mean", "MAE", False)]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.7))
    for ax, (key, title, higher) in zip(axes, metrics):
        v = [float(r[key]) for r in rows]
        cols = [C_MUTE if a in ("A0", "A1") else (C_OURS if a == "A4" else C_BASE) for a in arms]
        ax.bar(range(len(v)), v, color=cols, width=0.62)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels([lab[a] for a in arms], fontsize=8)
        ax.set_title(title + ("  $\\uparrow$" if higher else "  $\\downarrow$"), fontsize=9)
        ax.axvspan(-0.5, 1.5, color="#cccccc", alpha=0.28, zorder=0)
        for i, val in enumerate(v):
            ax.text(i, val + (0.02 if val >= 0 else -0.06), f"{val:.3f}",
                    ha="center", fontsize=6.2)
        lo, hi = min(v + [0]), max(v)
        ax.set_ylim(lo - 0.18 * (hi - lo + 1e-9), hi + 0.20 * (hi - lo + 1e-9))

    fig.suptitle("Controlled ablation gate (3 strict-LOSO folds, single seed)", fontsize=9.5, y=1.16)
    fig.text(0.5, 1.055, "A0 WSST-TCN   |   A1 event-structured   |   A2 no event supervision   |   "
                         "A3 no Shape-200 PCC term   |   A4 final",
             ha="center", fontsize=7.0, color="#333")
    fig.text(0.5, 1.005, "shaded (A0, A1): no morphology head, scored through a different path "
                         "-- not comparable with A2--A4",
             ha="center", fontsize=6.6, color="#777")
    save(fig, "F8_ablation_multimetric")


# ---------------------------------------------------------------- F9
def fig_delta_sys():
    runs = [("q1_align_ablation_fold01_b1_deltasys", "B1: $\\delta_{sys}$ only"),
            ("q1_align_ablation_fold01_b3_deltasys_res", "B3: $\\delta_{sys}$+residual"),
            ("q1_align_ablation_fold01_b4_full", "B4: full")]
    data = []
    for tag, lbl in runs:
        f = EXP / tag / "fold_01" / "results_delta_sys" / "delta_sys_raw.npz"
        j = EXP / tag / "fold_01" / "results_delta_sys" / "delta_sys_analysis.json"
        if f.exists():
            data.append((lbl, np.load(f), json.loads(j.read_text())))
    if not data:
        print("skip F9: no delta_sys outputs"); return

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.2, 2.8))
    lag = data[0][1]["lag_ms"]
    lag = lag[np.isfinite(lag)]
    a.hist(lag, bins=26, color=C_BASE, alpha=0.85, label=f"pseudo-GT lag\n{lag.mean():.0f}$\\pm${lag.std():.0f} ms")
    cols = [C_OURS, "#2ca02c", C_WARN]
    for (lbl, z, _), col in zip(data, cols):
        a.axvline(z["delta_sys"].mean(), color=col, lw=2.0, label=f"{lbl}: {z['delta_sys'].mean():.1f} ms")
    a.set_xlabel("delay (ms)"); a.set_ylabel("windows")
    a.set_title("(a) learned $\\delta_{sys}$ vs physiological lag", fontsize=8.5)
    a.legend(fontsize=6.2, loc="upper right")

    for (lbl, z, _), col in zip(data, cols):
        b.scatter(np.full(len(z["delta_sys"]), 0) + np.random.default_rng(0).normal(0, .06, len(z["delta_sys"]))
                  + cols.index(col), z["delta_sys"], s=5, alpha=0.45, color=col)
    b.set_xticks(range(len(data)))
    b.set_xticklabels([d[0] for d in data], fontsize=6.4)
    b.set_ylabel("$\\delta_{sys}$ (ms)")
    b.set_title("(b) per-window spread $<1$ ms:\ncollapsed to a constant", fontsize=8.5)
    b.set_ylim(0, 16)

    w = 0.36; xs = np.arange(len(data))
    r_lag = [d[2]["corr_delta_sys_vs_pseudo_gt_lag"] for d in data]
    r_hr = [d[2]["corr_delta_sys_vs_hr"] for d in data]
    c.bar(xs - w / 2, r_lag, w, color=C_BASE, label="vs pseudo-GT lag")
    c.bar(xs + w / 2, r_hr, w, color=C_OURS, label="vs heart rate")
    c.axhline(0, color="k", lw=0.8)
    for y in (0.3, -0.3):
        c.axhline(y, color=C_MUTE, ls=":", lw=0.9)
    c.set_xticks(xs); c.set_xticklabels([d[0] for d in data], fontsize=6.4)
    c.set_ylabel("Pearson $r$"); c.set_ylim(-0.55, 0.55)
    c.set_title("(c) no consistent physiological\nassociation (sign flips)", fontsize=8.5)
    c.legend(fontsize=6.4, loc="lower left")
    save(fig, "F9_delta_sys")


# ---------------------------------------------------------------- F10
def fig_robustness():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for tag, lbl, col, mk in [("q1_final_remnet_incumbent_mmecg_seed42", "MMECG (77 GHz FMCW, 50 bins)", C_OURS, "o"),
                              ("q1_final_remnet_incumbent_crrvs_seed42", "CR-RVS (24 GHz CW, 1 channel)", C_WARN, "s")]:
        acc = {}
        for p in sorted(EXP.glob(f"{tag}/fold_*/results_robustness/robustness_shape200.json")):
            d = json.loads(p.read_text())
            for k, v in d["pcc_mean_by_level"].items():
                acc.setdefault(float(k), []).append(v)
        lv = sorted(acc)
        m = [np.mean(acc[k]) for k in lv]
        sd = [np.std(acc[k]) for k in lv]
        a.errorbar(lv, m, yerr=sd, marker=mk, color=col, lw=1.6, ms=5, capsize=2.5, label=lbl)
    a.set_xlabel("additive noise level (relative to per-window std)")
    a.set_ylabel("Shape-200 PCC"); a.set_ylim(0, 1.02)
    a.set_title("(a) input-noise robustness", fontsize=9)
    a.legend(fontsize=6.8, loc="lower left")

    tol = [50, 100, 150]
    for tag, lbl, col, mk in [("q1_final_remnet_incumbent_mmecg_seed42", "MMECG (FMCW)", C_OURS, "o"),
                              ("q1_final_remnet_incumbent_crrvs_seed42", "CR-RVS (CW)", C_WARN, "s")]:
        acc = {t: [] for t in tol}
        hr = []
        for p in sorted(EXP.glob(f"{tag}/fold_*/results_rate/rate_metrics.json")):
            d = json.loads(p.read_text())
            for t in tol:
                acc[t].append(d[f"rpeak_f1_{t}ms"])
            hr.append(d["hr_error_bpm_mean"])
        m = [np.mean(acc[t]) for t in tol]
        sd = [np.std(acc[t]) for t in tol]
        b.errorbar(tol, m, yerr=sd, marker=mk, color=col, lw=1.6, ms=5, capsize=2.5,
                   label=f"{lbl}\nHR err {np.mean(hr):.3f} bpm")
    b.set_xticks(tol); b.set_xlabel("R-peak matching tolerance (ms)")
    b.set_ylabel("R-peak F1"); b.set_ylim(0, 1.0)
    b.set_title("(b) beat localization vs tolerance", fontsize=9)
    b.legend(fontsize=6.8, loc="upper left")
    save(fig, "F10_robustness_timing")


if __name__ == "__main__":
    fig_perfold(); fig_ablation(); fig_delta_sys(); fig_robustness()
