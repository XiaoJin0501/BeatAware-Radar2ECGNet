#!/usr/bin/env python3
"""MR-STFT mechanism panel, emitted as four independent parts for manual assembly.

No shared canvas: each part is its own PDF, so the layout is decided during assembly
rather than by matplotlib.

  part1_time_mse.pdf      blunt waveform in a framed panel
  part2_stft_mse.pdf      its spectrogram, dashed high-frequency box
  part3_time_mrstft.pdf   sharp waveform in the same panel style
  part4_stft_mrstft.pdf   its spectrogram, same scaling and box

**Alignment.** All four parts share one canvas width, one axes rectangle and one
colour-bar slot, and are saved with a fixed bounding box rather than a content-dependent
one. `bbox_inches='tight'` would crop each part to its own contents, and since an
amplitude axis and a frequency axis carry differently wide tick labels, the two rows
would end up a few points apart when stacked. The time panels reserve the colour bar's
slot with an invisible axes, so a waveform placed above a spectrogram lines up on both
the left and the right edge. The transparent background means the reserved margin shows
nothing.

The two spectrograms share colour limits (-80 to -20 dB), time span and frequency
range, and both rows share the same x ticks, so a column stacks cleanly.

Signals are synthetic and come from `plot_mechanism_stft.py`; the trained MR-STFT
system does not produce the sharp beat in part 3, so the assembled figure's caption
must say the illustration is conceptual.

Usage:
  python scripts/figures/plot_mechanism_parts.py
  python scripts/figures/plot_mechanism_parts.py --cmap inferno --box white --suffix _alt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import scienceplots  # noqa: F401
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.figures.plot_mechanism_stft import FS, waveform  # noqa: E402

OUT = ROOT / "docs" / "paper" / "figures" / "mechanism_parts"
BRICK = "#C0392B"        # dashed high-frequency box
INK = "#922B21"          # both waveforms: one deep red, one style
FRAME = "#333333"        # panel frame and ticks
HF_BAND = (18.0, 60.0)
VMIN, VMAX = -80, -20

# One canvas and one plotting rectangle for all four parts. Equal figure size and an
# explicit add_axes rectangle are what make the stacked parts align to the pixel; the
# colour-bar slot is reserved on the waveform parts too, so both outer edges match.
FIG_W, FIG_H = 5.0, 2.8
AXES_RECT = [0.15, 0.15, 0.70, 0.78]        # left, bottom, width, height
CBAR_RECT = [0.875, 0.15, 0.028, 0.78]
XTICKS = [0, 1, 2, 3, 4]                     # identical on both rows

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "font.size": 9, "axes.labelsize": 9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.top": True, "ytick.right": True,
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.transparent": True,
})


def _panel():
    """The shared canvas, the shared plotting rectangle and the shared colour-bar slot."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(AXES_RECT)
    cax = fig.add_axes(CBAR_RECT)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(FRAME)
    ax.tick_params(width=0.5, length=2.5, color=FRAME, labelcolor="black")
    ax.set_xticks(XTICKS)
    ax.set_xlabel("Time (s)")
    return fig, ax, cax


def _save(fig, name: str) -> None:
    """Fixed bounding box, so every part crops to the same width."""
    box = Bbox.from_bounds(0, 0, *fig.get_size_inches())
    for path in (OUT / name, OUT / name.replace(".pdf", ".png")):
        fig.savefig(path, bbox_inches=box, pad_inches=0.05, transparent=True, dpi=600)
    print(f"wrote {OUT / name}")
    plt.close(fig)


def save_time(x: np.ndarray, t: np.ndarray, name: str) -> None:
    fig, ax, cax = _panel()
    cax.set_axis_off()                       # reserve the colour bar's width, draw nothing
    ax.plot(t, x, color=INK, lw=2.0, solid_capstyle="round")
    ax.set_xlim(0, t[-1])
    margin = 0.08 * (x.max() - x.min())
    ax.set_ylim(x.min() - margin, x.max() + margin)
    ax.set_ylabel("Amplitude")
    _save(fig, name)


def save_spectrogram(x: np.ndarray, r_time: float, name: str, cmap: str, box: str) -> None:
    f, tt, z = signal.stft(x, fs=FS, nperseg=64, noverlap=56, window="hann")
    fig, ax, cax = _panel()
    mesh = ax.pcolormesh(tt, f, 20 * np.log10(np.abs(z) + 1e-6), cmap=cmap,
                         shading="gouraud", vmin=VMIN, vmax=VMAX, rasterized=True)
    ax.set_xlim(0, len(x) / FS)
    ax.set_ylim(0, 70)
    ax.set_ylabel("Frequency (Hz)")
    ax.add_patch(Rectangle((r_time - 0.13, HF_BAND[0]), 0.26, HF_BAND[1] - HF_BAND[0],
                           fill=False, edgecolor=box, linestyle="--", lw=1.0, zorder=5))
    bar = fig.colorbar(mesh, cax=cax)
    bar.outline.set_linewidth(0.5)
    bar.outline.set_edgecolor(FRAME)
    bar.ax.tick_params(width=0.5, length=2.5, color=FRAME, labelsize=7)
    _save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmap", choices=("viridis", "magma", "inferno"), default="magma")
    ap.add_argument("--box", default=BRICK, help="dashed-box colour, e.g. white")
    ap.add_argument("--suffix", default="", help="appended to every part name")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sharp, time_name, spec_name in (
        (False, "part1_time_mse", "part2_stft_mse"),
        (True, "part3_time_mrstft", "part4_stft_mrstft"),
    ):
        t, x, r_times = waveform(sharp)
        save_time(x, t, f"{time_name}{args.suffix}.pdf")
        save_spectrogram(x, r_times[1], f"{spec_name}{args.suffix}.pdf", args.cmap, args.box)
    print(f"\nfour parts in {OUT}; canvas {FIG_W} x {FIG_H} in, plotting rectangle "
          f"{AXES_RECT}, colour-bar slot {CBAR_RECT} reserved on every part, "
          f"{args.cmap} at {VMIN} to {VMAX} dB, dashed box {args.box}")


if __name__ == "__main__":
    main()
