#!/usr/bin/env python3
"""Fig_Mechanism_Base — the MR-STFT mechanism panel with no text at all.

A base layer for annotation elsewhere: two synthetic waveforms and their
spectrograms, no titles, no labels, no callouts. The waveforms come from
`plot_mechanism_stft.py`, so the annotated and bare versions cannot drift apart.

The signals are synthetic (see that module's docstring): the trained MR-STFT system
does not produce the sharp beat drawn on the right, and whatever caption is added to
this base must say the illustration is conceptual.

Usage:
  python scripts/figures/plot_mechanism_base.py
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

OUT = ROOT / "docs" / "paper" / "figures"
BRICK = "#C0392B"        # time-domain loss column, and the marked band
NAVY = "#2C3E50"         # MR-STFT column
HF_BAND = (18.0, 60.0)
CMAP = "magma"
BOX = BRICK

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.top": True, "ytick.right": True, "savefig.transparent": True,
})


def draw_column(axes, sharp: bool, colour: str) -> None:
    ax_time, ax_spec = axes
    t, x, r_times = waveform(sharp)

    ax_time.plot(t, x, color=colour, lw=1.5, solid_capstyle="round")
    ax_time.set_xlim(0, t[-1])
    ax_time.set_ylim(-0.55, 1.15)
    ax_time.axis("off")

    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    ax_spec.pcolormesh(tt, f, 20 * np.log10(np.abs(z) + 1e-6), cmap=CMAP,
                       shading="gouraud", vmin=-80, vmax=-20, rasterized=True)
    ax_spec.set_xlim(0, t[-1])
    ax_spec.set_ylim(0, 70)
    for spine in ax_spec.spines.values():
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    ax_spec.tick_params(width=0.5, length=2.5)

    r = r_times[1]
    ax_spec.add_patch(Rectangle((r - 0.13, HF_BAND[0]), 0.26, HF_BAND[1] - HF_BAND[0],
                                fill=False, edgecolor=BOX, linestyle="--", lw=1.0,
                                zorder=5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.4),
                             gridspec_kw={"height_ratios": [1, 2.5],
                                          "hspace": 0.05, "wspace": 0.15})
    draw_column(axes[:, 0], sharp=False, colour=BRICK)
    draw_column(axes[:, 1], sharp=True, colour=NAVY)
    axes[1, 1].set_yticklabels([])
    for path in (OUT / "Fig_Mechanism_Base.pdf", OUT / "Fig_Mechanism_Base.png"):
        fig.savefig(path, bbox_inches="tight", transparent=True, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
