#!/usr/bin/env python3
"""MR-STFT mechanism panel, emitted as four independent parts for manual assembly.

No grid, no shared canvas: each part is its own PDF, cropped to its own content, so
the layout is decided during assembly rather than by matplotlib.

  part1_time_mse.pdf      floating blunt waveform, brick red, no axes
  part2_stft_mse.pdf      its spectrogram, inward ticks, dashed high-frequency box
  part3_time_mrstft.pdf   floating sharp waveform, navy, no axes
  part4_stft_mrstft.pdf   its spectrogram, same scaling and box

The two spectrograms share identical colour limits (-80 to -20 dB), time span and
frequency range, so they remain directly comparable after assembly.

Signals are synthetic and come from `plot_mechanism_stft.py`; the trained MR-STFT
system does not produce the sharp beat in part 3, so the assembled figure's caption
must say the illustration is conceptual.

Usage:
  python scripts/figures/plot_mechanism_parts.py
"""
from __future__ import annotations

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

from scripts.figures.plot_mechanism_stft import FS, waveform  # noqa: E402

OUT = ROOT / "docs" / "paper" / "figures" / "mechanism_parts"
BRICK = "#C0392B"        # time-domain loss, and the marked band in both spectrograms
NAVY = "#2C3E50"         # MR-STFT loss
HF_BAND = (18.0, 60.0)
VMIN, VMAX = -80, -20

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.top": True, "ytick.right": True,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.transparent": True,
})
SAVE = dict(bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)


def _save(fig, name: str) -> None:
    """The part itself, plus a PNG twin for checking the assembly."""
    for path in (OUT / name, OUT / name.replace(".pdf", ".png")):
        fig.savefig(path, **SAVE)
    print(f"wrote {OUT / name}")
    plt.close(fig)


def save_time(x: np.ndarray, t: np.ndarray, colour: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 1.15))
    ax.plot(t, x, color=colour, lw=2.0, solid_capstyle="round")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.6, 1.2)
    ax.axis("off")
    _save(fig, name)


def save_spectrogram(x: np.ndarray, r_time: float, name: str) -> None:
    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.pcolormesh(tt, f, 20 * np.log10(np.abs(z) + 1e-6), cmap="viridis",
                  shading="gouraud", vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_xlim(0, len(x) / FS)
    ax.set_ylim(0, 70)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2.5)
    ax.add_patch(Rectangle((r_time - 0.13, HF_BAND[0]), 0.26, HF_BAND[1] - HF_BAND[0],
                           fill=False, edgecolor=BRICK, linestyle="--", lw=1.0, zorder=5))
    _save(fig, name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sharp, colour, time_name, spec_name in (
        (False, BRICK, "part1_time_mse.pdf", "part2_stft_mse.pdf"),
        (True, NAVY, "part3_time_mrstft.pdf", "part4_stft_mrstft.pdf"),
    ):
        t, x, r_times = waveform(sharp)
        save_time(x, t, colour, time_name)
        save_spectrogram(x, r_times[1], spec_name)
    print(f"\nfour parts in {OUT}; spectrograms share {VMIN} to {VMAX} dB, 0-70 Hz, "
          f"0-{len(x) / FS:.0f} s")


if __name__ == "__main__":
    main()
