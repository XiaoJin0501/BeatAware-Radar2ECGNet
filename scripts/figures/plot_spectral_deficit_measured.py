#!/usr/bin/env python3
"""Fig. Spectral deficit — reference, a time-domain-loss system, and the MR-STFT system.

Measured, not synthetic: all three columns come from sealed seed-42 test estimates of
the same held-out window. Chen-TCN is trained with a time-domain MSE objective and
REM-Net with the multi-resolution STFT objective, so the figure contrasts the two
objectives on identical data.

The measurement it reports is not the textbook expectation. Across three seeds the
above-10 Hz energy ratio is 0.122 for the MR-STFT system and 0.362 for the MSE
baseline, i.e. the spectral objective yields the *smoothest* estimate of the six
benchmarked systems, and relative high-frequency energy does not track reconstruction
quality: Attention-DCGAN carries the most (0.806) with the worst correlation (0.113).
Section VI reads this together with the timing diagnostics.

Selection is disclosed and deliberately unfavourable-neutral: the fold is the one whose
REM-Net above-10 Hz ratio is closest to the benchmark median, and within it the window
whose REM-Net correlation is closest to that fold's median.

Usage:
  python scripts/figures/plot_spectral_deficit_measured.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scienceplots  # noqa: F401
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "paper" / "figures"
REMNET = ROOT / "experiments_mmecg" / "q1_remnet_continuous_strict_loso_seed42" / "L3"
BASELINE = (ROOT / "experiments_mmecg"
            / "q1_matched_baselines_continuous_strict_loso_seed42" / "mmecg_tcn")
FS = 200.0
ZOOM_S = 2.5                     # time-domain window in the top row
HF_CUT = 10.0
BLACK = "#2B2B2B"                # reference
RED = "#E74C3C"                  # time-domain-loss baseline, and the marked band
BLUE = "#1F449C"                 # MR-STFT system

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.top": True, "ytick.right": True, "savefig.transparent": True,
})


def hf_share(x: np.ndarray) -> float:
    """Fraction of this signal's spectral energy above the cut."""
    freq = np.fft.rfftfreq(len(x), d=1.0 / FS)
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    return float(power[freq > HF_CUT].sum() / power.sum())


def pick_window() -> dict:
    ratios = []
    for fold in range(1, 12):
        m = json.loads((REMNET / f"fold_{fold:02d}" / "test_metrics.json").read_text())["metrics"]
        ratios.append((fold, m["above_10hz_energy_ratio"]))
    target = float(np.median([r for _, r in ratios]))
    fold = min(ratios, key=lambda fr: abs(fr[1] - target))[0]

    rem = np.load(REMNET / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
    base = np.load(BASELINE / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
    gt, pred = rem["ground_truth"], rem["prediction"]
    if not np.array_equal(gt, base["ground_truth"]):
        raise SystemExit("the two runs do not share test windows")
    pc, yc = pred - pred.mean(1, keepdims=True), gt - gt.mean(1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum(1))
    pcc = np.where(den > 1e-12, (pc * yc).sum(1) / np.maximum(den, 1e-12), np.nan)
    index = int(np.nanargmin(np.abs(pcc - np.nanmedian(pcc))))

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))

    return {"fold": fold, "index": index, "gt": gt[index],
            "baseline": base["prediction"][index], "remnet": pred[index],
            "pcc_baseline": corr(base["prediction"][index], gt[index]),
            "pcc_remnet": float(pcc[index])}


def draw(axes, x: np.ndarray, colour: str, title: str, note: str, start: int,
         corr_note: str = "") -> None:
    ax_time, ax_spec = axes
    n = int(ZOOM_S * FS)
    t = np.arange(n) / FS
    ax_time.plot(t, x[start:start + n], color=colour, linewidth=0.9)
    ax_time.set_title(title, color=colour, pad=5)
    ax_time.set_xlim(0, ZOOM_S)
    ax_time.set_ylim(-3.0, 6.3)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_yticks([])
    if corr_note:
        ax_time.text(0.97, 0.93, corr_note, transform=ax_time.transAxes, ha="right",
                     va="top", fontsize=7.5, color=colour)

    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    mesh = ax_spec.pcolormesh(tt, f, 20 * np.log10(np.abs(z) + 1e-6), cmap="viridis",
                              shading="gouraud", vmin=-45, vmax=10, rasterized=True)
    ax_spec.mesh = mesh
    ax_spec.set_xlim(0, len(x) / FS)
    ax_spec.set_ylim(0, 70)
    ax_spec.set_xlabel("Time (s)")
    ax_spec.add_patch(Rectangle((0.06, HF_CUT), len(x) / FS - 0.12, 70 - HF_CUT,
                                fill=False, edgecolor=RED, linewidth=1.0,
                                linestyle=(0, (2.4, 1.4)), zorder=5))
    ax_spec.text(len(x) / FS / 2, 65, note, color="white", fontsize=7.5, ha="center",
                 va="top", zorder=6)


def main() -> None:
    w = pick_window()
    shares = {k: hf_share(w[k]) for k in ("gt", "baseline", "remnet")}
    # Start the zoom at the first reference R peak so all three columns show the same beats.
    start = max(int(np.argmax(w["gt"])) - int(0.3 * FS), 0)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.6),
                             gridspec_kw={"height_ratios": [1.0, 1.25],
                                          "hspace": 0.55, "wspace": 0.18})
    draw(axes[:, 0], w["gt"], BLACK, "Reference ECG",
         f"{100 * shares['gt']:.0f}% of energy above {HF_CUT:.0f} Hz", start)
    draw(axes[:, 1], w["baseline"], RED, "Time-domain loss (Chen-TCN)",
         f"{100 * shares['baseline'] / shares['gt']:.0f}% of the reference share", start,
         f"$\\rho$ = {w['pcc_baseline']:.2f}")
    draw(axes[:, 2], w["remnet"], BLUE, "MR-STFT loss (REM-Net)",
         f"{100 * shares['remnet'] / shares['gt']:.0f}% of the reference share", start,
         f"$\\rho$ = {w['pcc_remnet']:.2f}")
    for ax in axes[1, 1:]:
        ax.set_yticklabels([])
    axes[1, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Amplitude (z)")

    bar = fig.colorbar(axes[1, 2].mesh, ax=axes[1, :].tolist(), fraction=0.022, pad=0.015)
    bar.set_label("Magnitude (dB)", fontsize=9)
    bar.ax.tick_params(labelsize=8, direction="in")
    fig.text(0.5, -0.03,
             f"Held-out subject, fold {w['fold']}, window {w['index']}; window correlation "
             f"{w['pcc_baseline']:.2f} (time-domain loss) and {w['pcc_remnet']:.2f} (MR-STFT). "
             "Across three seeds the above-10 Hz ratio is 0.362 and 0.122 respectively.",
             ha="center", fontsize=7, color="0.35")
    for path in (OUT / "Fig_Spectral_Deficit_Measured.pdf",
                 OUT / "Fig_Spectral_Deficit_Measured.png"):
        fig.savefig(path, bbox_inches="tight", transparent=True, dpi=600)
        print(f"wrote {path}")
    print(f"fold {w['fold']} window {w['index']}: reference HF share {shares['gt']:.3f}; "
          f"baseline {shares['baseline'] / shares['gt']:.3f} of it (PCC {w['pcc_baseline']:.3f}); "
          f"REM-Net {shares['remnet'] / shares['gt']:.3f} of it (PCC {w['pcc_remnet']:.3f})")
    plt.close(fig)


if __name__ == "__main__":
    main()
