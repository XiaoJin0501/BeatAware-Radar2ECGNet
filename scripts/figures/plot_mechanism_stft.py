#!/usr/bin/env python3
"""Fig. Mechanism — why a multi-resolution STFT objective penalises a smoothed beat.

CONCEPTUAL ILLUSTRATION. Both waveforms are synthetic (sums of Gaussians on a
respiration-like sinusoid); neither is a model output. The measured behaviour of the
trained systems is reported in Section VI and in
`docs/audits/Q1_MULTISEED_BENCHMARK_AUDIT.md`: the MR-STFT system still retains only
0.331 of the reference variance and a median above-10 Hz energy ratio of 0.122, i.e.
it mitigates rather than removes the smoothing this figure contrasts. The caption must
say the signals are synthetic.

Layout: 2 x 2. Left column, a time-domain waveform whose R peak is broad and low and
its spectrogram, with the high-frequency band at the R instant dim. Right column, a
sharp R peak and the vertical high-frequency ridge it produces. The dashed boxes mark
the same time-frequency region in both columns.

Usage:
  python scripts/figures/plot_mechanism_stft.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import scienceplots  # noqa: F401  (registers the 'science' and 'ieee' styles)
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

OUT = Path(__file__).resolve().parents[2] / "docs" / "paper" / "figures"
FS = 200.0
DURATION = 4.0
HEART_RATE_HZ = 1.15
RED = "#E74C3C"          # attenuated / missing energy
BLUE = "#1F449C"         # energy preserved
HF_BAND = (18.0, 60.0)   # the band the dashed boxes mark

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                            "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                            "savefig.transparent": True})


def gaussian(t: np.ndarray, centre: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)


def beat(t: np.ndarray, r_time: float, blunt: float) -> np.ndarray:
    """P-QRS-T complex as a sum of Gaussians.

    `blunt` widens and attenuates the whole QRS complex -- the fast part of the beat --
    while P and T, which carry the low-frequency content, are identical in both columns.
    The two spectrograms then differ only above ~15 Hz.
    """
    qrs_scale, qrs_gain = blunt, 1.0 / (0.45 * blunt + 0.55)
    return (gaussian(t, r_time - 0.20, 0.028, 0.14)                            # P
            + gaussian(t, r_time - 0.028 * qrs_scale, 0.010 * qrs_scale,
                       -0.16 * qrs_gain)                                       # Q
            + gaussian(t, r_time, 0.009 * qrs_scale, 1.0 * qrs_gain)           # R
            + gaussian(t, r_time + 0.032 * qrs_scale, 0.013 * qrs_scale,
                       -0.26 * qrs_gain)                                       # S
            + gaussian(t, r_time + 0.22, 0.055, 0.30))                         # T


def waveform(sharp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0, DURATION, 1.0 / FS)
    r_times = np.arange(0.45, DURATION, 1.0 / HEART_RATE_HZ)
    x = 0.05 * np.sin(2 * np.pi * 0.25 * t)                             # respiration
    for r in r_times:
        x = x + beat(t, r, blunt=1.0 if sharp else 4.5)
    return t, x, r_times


def draw_column(axes, sharp: bool, colour: str, title: str) -> None:
    ax_time, ax_spec = axes
    t, x, r_times = waveform(sharp)

    ax_time.plot(t, x, color=colour, linewidth=0.9)
    ax_time.set_title(title, color=colour, pad=4)
    ax_time.set_xlim(0, DURATION)
    ax_time.set_ylim(-0.55, 1.15)
    ax_time.set_xticks([]); ax_time.set_yticks([])
    for side in ax_time.spines:
        ax_time.spines[side].set_visible(False)

    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    magnitude = 20 * np.log10(np.abs(z) + 1e-6)
    mesh = ax_spec.pcolormesh(tt, f, magnitude, cmap="viridis", shading="gouraud",
                              vmin=-80, vmax=-20, rasterized=True)
    ax_spec.mesh = mesh
    ax_spec.set_xlim(0, DURATION)
    ax_spec.set_ylim(0, 70)
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

    # Mark the same time-frequency region in both columns: the QRS instant, high band.
    r = r_times[1]
    ax_spec.add_patch(Rectangle((r - 0.13, HF_BAND[0]), 0.26, HF_BAND[1] - HF_BAND[0],
                                fill=False, edgecolor=RED, linewidth=0.8,
                                linestyle=(0, (2.2, 1.4)), zorder=5))
    note = "high-frequency\nenergy lost" if not sharp else "high-frequency\nenergy preserved"
    ax_spec.annotate(note, xy=(r + 0.15, np.mean(HF_BAND)), xytext=(r + 0.42, 58),
                     color=RED, fontsize=5.6, ha="left", va="top",
                     arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.6))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.1),
                             gridspec_kw={"height_ratios": [1.0, 1.35],
                                          "hspace": 0.30, "wspace": 0.22})
    draw_column(axes[:, 0], sharp=False, colour=RED, title="Time-domain loss only")
    draw_column(axes[:, 1], sharp=True, colour=BLUE, title="Multi-resolution STFT loss")
    bar = fig.colorbar(axes[1, 1].mesh, ax=axes[1, :].tolist(), fraction=0.026, pad=0.015)
    bar.set_label("Magnitude (dB)", fontsize=6)
    bar.ax.tick_params(labelsize=5.5)
    fig.text(0.5, -0.035, "Synthetic signals; conceptual illustration of the objective, "
                          "not a model output.", ha="center", fontsize=5.4, color="0.35")
    for path in (OUT / "Fig_Mechanism_STFT.pdf", OUT / "Fig_Mechanism_STFT.png"):
        fig.savefig(path, bbox_inches="tight", transparent=True, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
