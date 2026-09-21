#!/usr/bin/env python3
"""Fig. Spectral deficit — the same contrast, measured on frozen estimates.

The companion to `plot_mechanism_stft.py`. Nothing here is synthetic: the left column
is the reference ECG of a held-out subject and the right column is the seed-42 estimate
of the MR-STFT system for the same window, taken from
`q1_remnet_continuous_strict_loso_seed42`. The window is chosen as the one whose
correlation is closest to that fold's median, so it is representative rather than
favourable, and the fold is the one whose above-10 Hz energy ratio is closest to the
benchmark median (0.122).

This figure is what the paper should show when it discusses spectral fidelity: the
trained system mitigates the deficit, it does not remove it.

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
RUN = ROOT / "experiments_mmecg" / "q1_remnet_continuous_strict_loso_seed42" / "L3"
FS = 200.0
GREY = "#2B2B2B"         # reference
BLUE = "#1F449C"         # estimate
RED = "#E74C3C"          # marked band
HF_BAND = (10.0, 60.0)

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                            "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                            "savefig.transparent": True})


def pick_window() -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """The representative fold (median >10 Hz ratio) and its median-correlation window."""
    folds = []
    for fold in range(1, 12):
        metrics = json.loads((RUN / f"fold_{fold:02d}" / "test_metrics.json").read_text())["metrics"]
        folds.append((fold, metrics["above_10hz_energy_ratio"]))
    target = float(np.median([r for _, r in folds]))
    fold, ratio = min(folds, key=lambda fr: abs(fr[1] - target))

    data = np.load(RUN / f"fold_{fold:02d}" / "test_predictions.npz", allow_pickle=True)
    pred, gt = data["prediction"], data["ground_truth"]
    pc, yc = pred - pred.mean(1, keepdims=True), gt - gt.mean(1, keepdims=True)
    den = np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum(1))
    pcc = np.where(den > 1e-12, (pc * yc).sum(1) / np.maximum(den, 1e-12), np.nan)
    index = int(np.nanargmin(np.abs(pcc - np.nanmedian(pcc))))
    return gt[index], pred[index], fold, index, float(pcc[index])


def hf_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    freq = np.fft.rfftfreq(len(pred), d=1.0 / FS)
    p = np.abs(np.fft.rfft(pred - pred.mean())) ** 2
    g = np.abs(np.fft.rfft(gt - gt.mean())) ** 2
    hi = freq > 10.0
    return float((p[hi].sum() / p.sum()) / (g[hi].sum() / g.sum()))


def draw_column(axes, x: np.ndarray, colour: str, title: str, marker: str) -> None:
    ax_time, ax_spec = axes
    t = np.arange(len(x)) / FS
    ax_time.plot(t, x, color=colour, linewidth=0.8)
    ax_time.set_title(title, color=colour, pad=4)
    ax_time.set_xlim(0, t[-1])
    ax_time.set_ylim(-3.2, 6.5)
    ax_time.set_xticks([]); ax_time.set_yticks([])
    for side in ax_time.spines:
        ax_time.spines[side].set_visible(False)

    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    mesh = ax_spec.pcolormesh(tt, f, 20 * np.log10(np.abs(z) + 1e-6), cmap="viridis",
                              shading="gouraud", vmin=-45, vmax=10, rasterized=True)
    ax_spec.mesh = mesh
    ax_spec.set_xlim(0, t[-1]); ax_spec.set_ylim(0, 70)
    ax_spec.set_xlabel("Time (s)"); ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.add_patch(Rectangle((0.05, HF_BAND[0]), t[-1] - 0.1, HF_BAND[1] - HF_BAND[0],
                                fill=False, edgecolor=RED, linewidth=0.8,
                                linestyle=(0, (2.2, 1.4)), zorder=5))
    ax_spec.text(t[-1] / 2, HF_BAND[1] + 2.5, marker, color=RED, fontsize=5.8,
                 ha="center", va="bottom")


def main() -> None:
    gt, pred, fold, index, pcc = pick_window()
    ratio = hf_ratio(pred, gt)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.0),
                             gridspec_kw={"height_ratios": [1.0, 1.3],
                                          "hspace": 0.28, "wspace": 0.22})
    draw_column(axes[:, 0], gt, GREY, "Reference ECG", "above 10 Hz: reference")
    draw_column(axes[:, 1], pred, BLUE, "Estimate (MR-STFT objective)",
                f"above 10 Hz: {100 * ratio:.0f}% of the reference share")
    bar = fig.colorbar(axes[1, 1].mesh, ax=axes[1, :].tolist(), fraction=0.026, pad=0.015)
    bar.set_label("Magnitude (dB)", fontsize=6)
    bar.ax.tick_params(labelsize=5.5)
    fig.text(0.5, -0.04,
             f"Held-out subject, fold {fold}, window {index}; correlation {pcc:.2f}, the "
             f"median window of that fold. Benchmark median above-10 Hz ratio 0.122.",
             ha="center", fontsize=5.4, color="0.35")
    for path in (OUT / "Fig_Spectral_Deficit_Measured.pdf",
                 OUT / "Fig_Spectral_Deficit_Measured.png"):
        fig.savefig(path, bbox_inches="tight", transparent=True, dpi=600)
        print(f"wrote {path}")
    print(f"fold {fold}, window {index}, PCC {pcc:.3f}, >10 Hz ratio {ratio:.3f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
