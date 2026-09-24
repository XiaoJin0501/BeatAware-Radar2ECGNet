#!/usr/bin/env python3
"""Fig. 4 -- anatomy of the long-term reconstruction bottleneck (2 x 2, double column).

(a) Three tracks over one held-out window: REM-Net's reconstruction; a fixed
    training-fold template placed at the REFERENCE anchors (oracle timing); the
    identical template placed at REM-Net's own anchor-head estimates. Only the
    placement differs between the last two tracks.
(b) Signed offset of every matched anchor-head estimate against its reference R peak
    (one-to-one matching within +/-150 ms, all folds and seeds).
(c) Missed-detection rate as the matching tolerance is tightened from +/-150 ms to
    +/-20 ms, for REM-Net's two read-outs and the two strongest comparators.
(d) Sensitivity of long-term R^2 to anchor jitter: the template is placed at the
    reference anchors after adding zero-mean Gaussian jitter of standard deviation
    sigma to every beat. sigma = 0 reproduces the frozen oracle-timing bound; the
    jitter REM-Net actually exhibits is marked from panel (b).

All data are frozen MMECG test predictions (loso_v2, 11 folds). Nothing is trained.
Template construction, synthesis and scoring are imported from
scripts/evaluate_template_timing.py so every drawn number is the scored number.

Window selection in (a), disclosed: among folds passing the anchor-coverage gate,
the fold whose REM-Net long-term R^2 is closest to the benchmark median; within it,
among windows with at least two anchor-head estimates, the window whose
template-at-anchor-head R^2 is closest to that fold's median of the same quantity.
The example is therefore typical of the condition the third track illustrates, not
of the reconstruction in the first track.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_core_metrics import DATASETS, _clean, fold_dir  # noqa: E402
from scripts.evaluate_template_timing import score, synthesise, train_template  # noqa: E402
from scripts.figures.f4p_style import (  # noqa: E402
    ACCENT, GT, MODEL_COLORS, OURS, OURS_WAVE, REFERENCE, SYSTEMS, apply, panel_tag,
    save, style_axis)
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402

CFG = DATASETS["mmecg"]
FS = 200
TOLS_MS = [20, 30, 40, 50, 60, 80, 100, 120, 150]
SIGMAS_MS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
SWEEP = [("rem_net", "event"), ("rem_net", "waveform"), ("radarode_mtl", "waveform"),
         ("mmecg_tcn", "waveform")]


def anchors(d, readout):
    if readout == "event":
        return [_clean(peaks_from_heatmap(e)) for e in d["event_probability"]]
    return [_clean(detect_rpeaks(p, fs=FS)) for p in d["prediction"]]


def match_signed(pred, gt, tol):
    """Greedy nearest one-to-one matching; returns (n_matched, signed offsets in samples)."""
    if gt.size == 0 or pred.size == 0:
        return 0, []
    d = pred[None, :] - gt[:, None]
    a = np.abs(d)
    mg, mp, out = np.zeros(gt.size, bool), np.zeros(pred.size, bool), []
    for f in np.argsort(a.ravel()):
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi] or a[gi, pi] > tol:
            continue
        mg[gi] = mp[pi] = True
        out.append(int(d[gi, pi]))
    return int(mg.sum()), out


def collect():
    """One pass over the frozen predictions: offsets, tolerance sweep, jitter sweep."""
    offsets = []
    sys_offsets = {k: [] for k in SWEEP}
    sweep = {k: {t: [0, 0] for t in TOLS_MS} for k in SWEEP}
    rng = np.random.default_rng(2026)
    r2_sig = {s: [] for s in SIGMAS_MS}
    r2_event = []
    for fold in CFG["folds"]:
        template, fb_rr, level = train_template(CFG["splits"], fold)
        base = None
        for seed in CFG["seeds"]:
            for m, ro in SWEEP:
                d = np.load(fold_dir(CFG, m, seed, fold) / "test_predictions.npz",
                            allow_pickle=True)
                gtp = [_clean(r) for r in d["rpeaks"]]
                pk = anchors(d, ro)
                for p, g in zip(pk, gtp):
                    for t in TOLS_MS:
                        n, off = match_signed(p, g, int(round(t * FS / 1000)))
                        sweep[(m, ro)][t][0] += n
                        sweep[(m, ro)][t][1] += g.size
                        if t == 150:
                            sys_offsets[(m, ro)].extend(off)
                            if (m, ro) == ("rem_net", "event"):
                                offsets.extend(off)
                if (m, ro, seed) == ("rem_net", "event", 42):
                    base = d
                    synth = np.stack([synthesise(p, template, fb_rr, level) for p in pk])
                    r2_event.append(score(synth, d["ground_truth"], d["rpeaks"])["r2"])
        gt, gtp = base["ground_truth"], [_clean(r) for r in base["rpeaks"]]
        for s in SIGMAS_MS:
            jit = []
            for g in gtp:
                j = g + np.round(rng.normal(0, s * FS / 1000, g.size)).astype(int)
                jit.append(_clean(j))
            synth = np.stack([synthesise(p, template, fb_rr, level) for p in jit])
            r2_sig[s].append(score(synth, gt, base["rpeaks"])["r2"])
        print(f"fold {fold:2d} done", flush=True)
    # standard deviation, matching the Gaussian sigma of the injected jitter; the MAD is
    # quantised to whole samples (5 ms) and ties every system at 51.9 ms
    spread = {k: float(np.std(v)) * 1000 / FS for k, v in sys_offsets.items()}
    return (np.asarray(offsets) * 1000 / FS, sweep, spread,
            {s: float(np.mean(v)) for s, v in r2_sig.items()},
            {s: float(np.std(v)) for s, v in r2_sig.items()}, float(np.mean(r2_event)))


def pick_window():
    metrics = {f: json.loads((fold_dir(CFG, "rem_net", 42, f) / "test_metrics.json")
                             .read_text())["metrics"] for f in CFG["folds"]}
    ok = {f: m["r2"] for f, m in metrics.items() if m["rr_coverage"] >= 0.80}
    med = float(np.median(list(ok.values())))
    fold = min(ok, key=lambda f: abs(ok[f] - med))
    d = np.load(fold_dir(CFG, "rem_net", 42, fold) / "test_predictions.npz", allow_pickle=True)
    pred, gt = d["prediction"], d["ground_truth"]
    ev = anchors(d, "event")
    template, fb_rr, level = train_template(CFG["splits"], fold)
    cand = np.array([i for i in range(len(pred)) if len(ev[i]) >= 2])
    q = np.array([r2(synthesise(ev[i], template, fb_rr, level), gt[i]) for i in cand])
    i = int(cand[np.argmin(np.abs(q - np.median(q)))])
    gtp = _clean(d["rpeaks"][i])
    return dict(fold=fold, idx=i, gt=gt[i], rem=pred[i], gtp=gtp, evp=ev[i],
                t_gt=synthesise(gtp, template, fb_rr, level),
                t_ev=synthesise(ev[i], template, fb_rr, level))


def longterm_r2() -> dict:
    import csv
    rows = list(csv.DictReader(open(ROOT / "docs" / "audits" / "Q1_CORE_METRICS_fold.csv")))
    out = {}
    for m, ro in SWEEP:
        per_seed = []
        for s in CFG["seeds"]:
            v = [float(r["longterm_r2"]) for r in rows if r["dataset"] == "mmecg"
                 and r["model"] == m and r["readout"] == ro and r["seed"] == str(s)]
            per_seed.append(np.mean(v))
        out[(m, ro)] = float(np.mean(per_seed))
    return out


def r2(p, y):
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    apply()
    import pickle
    cache = Path("/tmp/claude-1011/fig4_collect_cache.pkl")
    if cache.exists():
        offs, sweep, spread, r2_mean, r2_sd, r2_ev = pickle.loads(cache.read_bytes())
    else:
        offs, sweep, spread, r2_mean, r2_sd, r2_ev = collect()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps((offs, sweep, spread, r2_mean, r2_sd, r2_ev)))
    lt_r2 = longterm_r2()
    w = pick_window()

    fig, axes = plt.subplots(2, 2, figsize=(28, 20),
                             gridspec_kw={"hspace": 0.34, "wspace": 0.2})

    # (a) three tracks ------------------------------------------------------------
    ax = axes[0, 0]
    n = int(4.0 * FS)
    start = max(int(w["gtp"][0]) - int(0.3 * FS), 0) if len(w["gtp"]) else 0
    start = min(start, len(w["gt"]) - n)
    sl, t = slice(start, start + n), np.arange(n) / FS
    tracks = [("REM-Net reconstruction", w["rem"], OURS, "-"),
              ("Template @ reference anchors", w["t_gt"], REFERENCE, "--"),
              ("Template @ REM-Net anchor head", w["t_ev"], ACCENT, "--")]
    span = w["gt"].max() - w["gt"].min()
    step = 1.55 * span
    for k, (lab, sig, col, ls) in enumerate(tracks):
        off = -k * step
        ax.plot(t, w["gt"][sl] + off, color=GT, linewidth=2.6, alpha=0.9)
        ax.plot(t, sig[sl] + off, color=col, linewidth=3.0, linestyle=ls)
        ax.text(0.02, off + w["gt"].max() + 0.12 * span,
                f"{lab}    $R^2$ = {r2(sig, w['gt']):+.2f}",
                fontsize=20, color=col, va="bottom", ha="left", fontweight="bold")
    ax.set_xlim(0, 4.0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (s)", labelpad=10)
    ax.set_title("Same template, different placement", fontsize=24, pad=16)
    ax.text(0.99, w["gt"].max() + 0.12 * span, "black: reference ECG", fontsize=19,
            color=GT, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_ylim(-2 * step + w["gt"].min() - 0.1 * span, w["gt"].max() + 0.45 * span)
    panel_tag(ax, "(a)")

    # (b) signed offsets ------------------------------------------------------------
    ax = axes[0, 1]
    ax.hist(offs, bins=np.arange(-150, 155, 5), color=OURS, alpha=0.8, edgecolor="white",
            linewidth=1.0)
    med = float(np.median(offs))
    rsd = float(np.std(offs))
    ax.axvline(0, color=REFERENCE, linestyle="--", linewidth=3)
    ax.axvline(med, color=ACCENT, linewidth=3)
    ax.axvspan(med - rsd, med + rsd, color=ACCENT, alpha=0.10)
    ax.text(0.03, 0.95, f"median offset  {med:+.1f} ms\ns.d.  {rsd:.1f} ms\n"
                        f"$n$ = {len(offs):,} matched beats", transform=ax.transAxes,
            fontsize=21, va="top", ha="left", color="0.25")
    ax.set_xlabel("Signed anchor offset, anchor head (ms)", labelpad=10)
    ax.set_ylabel("matched beats", labelpad=10)
    ax.set_title("No bias, but wide beat-to-beat jitter", fontsize=24, pad=16)
    style_axis(ax)
    panel_tag(ax, "(b)")

    # (c) tolerance sweep --------------------------------------------------------------
    ax = axes[1, 0]
    ax.axvspan(0, 50, color=ACCENT, alpha=0.07)
    ax.text(33, 96, "sub-50 ms\nclinical timing", fontsize=19, color=ACCENT, va="top",
            ha="center")
    for k in SWEEP:
        spec = SYSTEMS[k]
        mdr = [100 * (1 - sweep[k][tt][0] / max(sweep[k][tt][1], 1)) for tt in TOLS_MS]
        ax.plot(TOLS_MS, mdr, color=spec["color"], linestyle=spec["ls"], linewidth=3.4,
                marker="o", markersize=11, markeredgecolor="white", markeredgewidth=2,
                label=spec["label"])
    ax.set_xlim(15, 155)
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.set_xlabel("Matching tolerance (ms)   (tighter $\\rightarrow$)", labelpad=10)
    ax.set_ylabel(r"Missed-detection rate (%) $\downarrow$", labelpad=10)
    ax.set_title("Detection collapses at clinical timing precision", fontsize=24, pad=16)
    ax.legend(fontsize=19, loc="upper left", frameon=False)
    style_axis(ax)
    panel_tag(ax, "(c)")

    # (d) jitter sensitivity ------------------------------------------------------------
    ax = axes[1, 1]
    s = np.array(SIGMAS_MS, float)
    m = np.array([r2_mean[x] for x in SIGMAS_MS])
    sd = np.array([r2_sd[x] for x in SIGMAS_MS])
    ax.fill_between(s, m - sd, m + sd, color=REFERENCE, alpha=0.15)
    ax.plot(s, m, color=REFERENCE, linewidth=3.6, marker="D", markersize=11,
            label="Template @ jittered reference anchors")
    ax.axhline(0, color="0.55", linewidth=2)
    ax.axhline(0.041, color=OURS, linestyle=":", linewidth=3)
    ax.text(100, 0.058, "best trained system, $R^2$ = 0.041", color=OURS, fontsize=19,
            ha="right", va="bottom")
    # where the jitter curve meets the best trained system: its equivalent jitter
    k = int(np.flatnonzero(m < 0.041)[0])
    s_eq = s[k - 1] + (m[k - 1] - 0.041) / (m[k - 1] - m[k]) * (s[k] - s[k - 1])
    ax.plot(s_eq, 0.041, marker="o", markersize=18, markerfacecolor="white",
            markeredgecolor=OURS, markeredgewidth=3, zorder=7)
    ax.annotate(f"equivalent jitter $\\approx${s_eq:.0f} ms", xy=(s_eq, 0.041),
                xytext=(38, 0.72), fontsize=19, color=OURS, ha="left",
                arrowprops=dict(arrowstyle="->", color=OURS, lw=2.5))
    ax.plot(rsd, r2_ev, marker="*", markersize=26, color=ACCENT, zorder=6,
            label="Template @ REM-Net anchor head")
    # operating points of the trained systems: (anchor offset spread, own long-term R^2)
    for k in SWEEP:
        spec = SYSTEMS[k]
        ax.plot(spread[k], lt_r2[k], marker="o" if k[1] == "event" else "s", markersize=19,
                color=spec["color"], markeredgecolor="white", markeredgewidth=2.5, zorder=8,
                linestyle="none", label=spec["label"])
    ax.set_xlabel(r"Injected anchor jitter $\sigma_\tau$ (ms)", labelpad=10)
    ax.set_ylabel(r"Long-term $R^2$ $\uparrow$", labelpad=10)
    ax.set_title(r"$R^2$ falls from the oracle bound as jitter grows", fontsize=24, pad=16)
    xs = [spread[k] for k in SWEEP]
    ax.annotate("trained systems sit above the curve:\nshrinkage hedges timing error (Eq. 4)",
                xy=(np.mean(xs), 0.07), xytext=(56, 0.36), fontsize=18, color="0.25",
                ha="left", arrowprops=dict(arrowstyle="->", color="0.35", lw=2.2))
    ax.legend(fontsize=17, loc="upper center", frameon=False, ncols=2,
              bbox_to_anchor=(0.5, -0.17), columnspacing=1.2, handletextpad=0.5)
    style_axis(ax)
    panel_tag(ax, "(d)")

    save(fig, "fig4_timing_bottleneck")
    print(f"\n(a) fold {w['fold']} window {w['idx']}")
    print(f"(b) median {med:+.2f} ms, robust spread {rsd:.2f} ms, n = {len(offs)}")
    for k in SWEEP:
        print(f"(c) {k}: MDR@150 {100*(1-sweep[k][150][0]/sweep[k][150][1]):.1f}%  "
              f"MDR@50 {100*(1-sweep[k][50][0]/sweep[k][50][1]):.1f}%  "
              f"MDR@20 {100*(1-sweep[k][20][0]/sweep[k][20][1]):.1f}%")
    print("(d) " + "  ".join(f"{x}ms:{r2_mean[x]:+.3f}" for x in SIGMAS_MS))
    print(f"    template @ REM-Net anchor head: R2 {r2_ev:+.3f}")
    for k in SWEEP:
        print(f"    operating point {k}: spread {spread[k]:.1f} ms, own long-term R2 {lt_r2[k]:+.3f}")
    print(f"    selected window: " + ", ".join(f"{lab.split(' ')[0]}.. R2 {r2(sig, w['gt']):+.3f}"
          for lab, sig in (("rem", w["rem"]), ("t_gt", w["t_gt"]), ("t_ev", w["t_ev"]))))


if __name__ == "__main__":
    main()
