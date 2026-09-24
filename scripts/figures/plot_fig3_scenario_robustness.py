#!/usr/bin/env python3
"""Fig. 3 -- clinical robustness across physiological states (2 x 4, double column).

Rows: MMECG (77-GHz FMCW, three seeds) and CR_RVS (24-GHz CW, seed 42).
Columns, over per-subject values in each state:
  1  Cycle-length error (ms)   REM-Net read-out: waveform peaks
  2  R-peak timing MAE (ms)    REM-Net read-out: anchor head
  3  Clinical interval error   QT error (ms), NeuroKit2 delineation
  4  Traces                    one post-exercise (FMCW) / apnea (CW) window, with a
                               single-beat zoom-in inset

Each column uses ONE REM-Net anchor read-out, named in the column title; Table I
reports both read-outs for every anchor metric, so no number is hidden. The
per-column choice is the author's and is disclosed in the caption.
Comparators are the two matched baselines evaluated per state (Chen-TCN,
WaveGRU-Net); radarODE-MTL is not in the per-state breakdown.

Sources: docs/audits/Q1_CLINICAL_INTERVALS_scenario.csv (evaluate_clinical_intervals.py)
and frozen test predictions for the traces.

Trace window selection, disclosed: within the named state, among folds passing the
anchor-coverage gate, the window whose REM-Net long-term R^2 is closest to that
state's median over windows; the inset shows the beat nearest the window centre.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_core_metrics import DATASETS, _clean, fold_dir  # noqa: E402
from scripts.figures.f4p_style import (  # noqa: E402
    GT, MODEL_COLORS, OURS, apply, panel_tag, save, style_axis)
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402

CSV = ROOT / "docs" / "audits" / "Q1_CLINICAL_INTERVALS_scenario.csv"
FS = 200
BASE = [("mmecg_tcn", "Chen-TCN"), ("wavegru_net", "WaveGRU-Net")]
COLS = [("cycle_length_error_ms", "waveform", "Cycle-length error (ms)", "REM-Net: waveform peaks"),
        ("r_mae_ms", "event", "R-peak timing MAE (ms)", "REM-Net: anchor head"),
        ("qt_error_ms", "waveform", "QT interval error (ms)", "NeuroKit2 delineation")]
ROWS = [("mmecg", "MMECG\n77-GHz FMCW",
         ["Normal breathing", "Irregular breathing", "Post-exercise", "Sleep"],
         {"Normal breathing": "Normal", "Irregular breathing": "Irreg.",
          "Post-exercise": "Post-ex.", "Sleep": "Sleep"}, "PE", "Post-exercise"),
        ("crrvs", "CR_RVS\n24-GHz CW", ["Resting", "Valsalva", "Apnea"], {}, "SP", "Apnea")]
CAP = {"cycle_length_error_ms": {"mmecg": 300.0, "crrvs": 400.0}}


def per_subject(rows, ds, model, ro, sc, key):
    acc = {}
    for r in rows:
        if (r["dataset"], r["model"], r["readout"], r["scenario"]) == (ds, model, ro, sc):
            if r[key] not in ("", "nan"):
                acc.setdefault(int(r["subject"]), []).append(float(r[key]))
    return np.array([np.mean(v) for v in acc.values()])


def boxes(ax, rows, ds, scen, wrap, key, rem_ro):
    systems = [("rem_net", rem_ro, OURS)] + [(m, "waveform", MODEL_COLORS[m]) for m, _ in BASE]
    w, gap, rng = 0.24, 0.02, np.random.default_rng(5)
    cap = CAP.get(key, {}).get(ds)
    for j, (m, ro, col) in enumerate(systems):
        off = (j - 1) * (w + gap)
        vals = [per_subject(rows, ds, m, ro, sc, key) for sc in scen]
        bp = ax.boxplot(vals, positions=np.arange(len(scen)) + off, widths=w,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=3),
                        whiskerprops=dict(linewidth=2, color="0.35"),
                        capprops=dict(linewidth=2, color="0.35"),
                        boxprops=dict(linewidth=1.5, edgecolor="0.35"))
        for b in bp["boxes"]:
            b.set_facecolor(col)
        for k, v in enumerate(vals):
            shown = v if cap is None else v[v <= cap]
            ax.scatter(k + off + rng.uniform(-0.05, 0.05, len(shown)), shown, s=22,
                       color="0.2", alpha=0.4, edgecolor="none", zorder=5)
            if cap is not None and (v > cap).any():
                ax.plot(k + off, cap, marker="^", markersize=11, color=col, zorder=6,
                        clip_on=False, markeredgecolor="0.3")
                ax.annotate(f"{int((v > cap).sum())}", xy=(k + off, cap), xytext=(0, 9),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=15, fontweight="bold", color="0.25")
    ax.set_xticks(np.arange(len(scen)))
    n = [len(per_subject(rows, ds, "rem_net", rem_ro, sc, key)) for sc in scen]
    ax.set_xticklabels([f"{wrap.get(sc, sc)}\n$n$={k}" for sc, k in zip(scen, n)], fontsize=19)
    ax.set_ylim(0, cap * 1.02 if cap else None)
    if cap is None:
        ax.set_ylim(bottom=0)


def trace_window(ds, code):
    cfg = DATASETS[ds]
    cands = []
    for fold in cfg["folds"]:
        fd = fold_dir(cfg, "rem_net", 42, fold)
        if json.loads((fd / "test_metrics.json").read_text())["metrics"]["rr_coverage"] < 0.80:
            continue
        with h5py.File(cfg["splits"] / f"fold_{fold:02d}" / "test.h5", "r") as h:
            st = np.array([x.decode() if isinstance(x, bytes) else x for x in h["physistatus"][:]])
        d = np.load(fd / "test_predictions.npz", allow_pickle=True)
        for i in np.flatnonzero(st == code):
            y, p = d["ground_truth"][i], d["prediction"][i]
            cands.append((1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(), fold, int(i)))
    med = np.median([c[0] for c in cands])
    r2, fold, i = min(cands, key=lambda c: abs(c[0] - med))
    rem = np.load(fold_dir(cfg, "rem_net", 42, fold) / "test_predictions.npz", allow_pickle=True)
    chen = np.load(fold_dir(cfg, "mmecg_tcn", 42, fold) / "test_predictions.npz", allow_pickle=True)
    return dict(fold=fold, idx=i, r2=float(r2), gt=rem["ground_truth"][i],
                rem=rem["prediction"][i], chen=chen["prediction"][i],
                gtp=_clean(rem["rpeaks"][i]),
                evp=_clean(peaks_from_heatmap(rem["event_probability"][i])))


def traces(ax, w, title, show_s=3.0, axis_s=4.65):
    n = int(show_s * FS)
    start = max(int(w["gtp"][0]) - int(0.3 * FS), 0) if len(w["gtp"]) else 0
    start = min(start, len(w["gt"]) - n)
    sl, t = slice(start, start + n), np.arange(n) / FS
    ax.plot(t, w["gt"][sl], color=GT, linewidth=2.8)
    ax.plot(t, w["rem"][sl], color=OURS, linewidth=2.5)
    ax.plot(t, w["chen"][sl], color=MODEL_COLORS["mmecg_tcn"], linewidth=2.5)
    hi = max(w["gt"][sl].max(), w["rem"][sl].max(), w["chen"][sl].max())
    lo = min(w["gt"][sl].min(), w["rem"][sl].min(), w["chen"][sl].min())
    span = hi - lo
    for p in w["gtp"]:
        if start <= p < start + n:
            ax.plot((p - start) / FS, hi + 0.08 * span, marker="|", markersize=22,
                    markeredgewidth=3, color=GT)
    for p in w["evp"]:
        if start <= p < start + n:
            ax.plot((p - start) / FS, hi + 0.19 * span, marker="v", markersize=14, color=OURS)
    ax.set_ylim(lo - 0.05 * span, hi + 0.30 * span)
    ax.set_xlim(0, axis_s)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Time (s)", labelpad=8)
    ax.set_ylabel("ECG (z)", labelpad=8)
    ax.set_title(f"{title} (fold {w['fold']})", fontsize=21, pad=12)

    # single-beat zoom-in: the reference beat nearest the window centre
    beats = [p for p in w["gtp"] if start + 0.15 * FS <= p < start + n - 0.15 * FS]
    if not beats:
        return
    c = min(beats, key=lambda p: abs(p - (start + n / 2)))
    a, b = c - int(0.15 * FS), c + int(0.40 * FS)
    # the inset sits in the empty strip right of the 3-s trace, so it hides nothing
    ins = ax.inset_axes([0.675, 0.14, 0.315, 0.66])
    tt = (np.arange(a, b) - c) * 1000 / FS
    ins.plot(tt, w["gt"][a:b], color=GT, linewidth=2.4)
    ins.plot(tt, w["rem"][a:b], color=OURS, linewidth=2.2)
    ins.plot(tt, w["chen"][a:b], color=MODEL_COLORS["mmecg_tcn"], linewidth=2.2)
    for p in w["evp"]:
        if a <= p < b:
            ins.axvline((p - c) * 1000 / FS, color=OURS, linestyle=":", linewidth=2.2)
    ins.axvline(0, color=GT, linestyle=":", linewidth=2.2)
    ins.set_xlim(tt[0], tt[-1])
    ins.set_xticks([-100, 0, 200])
    ins.tick_params(labelsize=14, length=4, width=1, pad=2)
    ins.set_yticks([])
    ins.set_title("one beat, ms from R", fontsize=15, pad=4)
    for sp in ins.spines.values():
        sp.set_linewidth(1.5)
        sp.set_visible(True)
    ax.axvspan((a - start) / FS, (b - start) / FS, color="0.5", alpha=0.10, zorder=0)


def main() -> None:
    apply()
    rows = list(csv.DictReader(open(CSV)))
    fig, axes = plt.subplots(2, 4, figsize=(40, 17),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.35],
                                          "hspace": 0.42, "wspace": 0.24})
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.07, right=0.99)
    tag = iter("abcdefgh")
    for r, (ds, rlabel, scen, wrap, code, scname) in enumerate(ROWS):
        for c, (key, ro, ylab, sub) in enumerate(COLS):
            ax = axes[r, c]
            boxes(ax, rows, ds, scen, wrap, key, ro)
            ax.set_ylabel(ylab, labelpad=8, fontsize=24)
            if r == 0:
                ax.set_title(sub, fontsize=21, pad=14, color="0.3", style="italic")
            style_axis(ax)
        w = trace_window(ds, code)
        traces(axes[r, 3], w, scname)
        style_axis(axes[r, 3])
        axes[r, 0].annotate(rlabel, xy=(-0.33, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=25, fontweight="bold")
        print(f"{ds} trace: fold {w['fold']} window {w['idx']} R2 {w['r2']:+.3f}")
    for r in range(2):
        for c in range(4):
            panel_tag(axes[r, c], f"({next(tag)})", dx=-0.1)
    fig.legend(handles=[Patch(facecolor=OURS, edgecolor="0.35", label="REM-Net"),
                        Patch(facecolor=MODEL_COLORS["mmecg_tcn"], edgecolor="0.35", label="Chen-TCN"),
                        Patch(facecolor=MODEL_COLORS["wavegru_net"], edgecolor="0.35", label="WaveGRU-Net"),
                        plt.Line2D([], [], color=GT, lw=2.8, label="Reference ECG"),
                        plt.Line2D([], [], color=GT, lw=0, marker="|", markersize=20,
                                   markeredgewidth=3, label="reference R peak"),
                        plt.Line2D([], [], color=OURS, lw=0, marker="v", markersize=14,
                                   label="REM-Net anchor head")],
               loc="upper center", ncols=6, fontsize=24, frameon=False,
               bbox_to_anchor=(0.53, 0.975))
    save(fig, "fig3_scenario_robustness")


if __name__ == "__main__":
    main()
