#!/usr/bin/env python3
"""Fig. REM-Net architecture -- the continuous (B1) network, with contributions marked.

Every shape, channel count and parameter count is read from the code and from the
frozen seed-42 fold-1 config, not from prose:

  src/models/continuous/ce_net.py      arm B1 = WSSTStem + LegacyDecoder, fuse is None
  .../L3/fold_01/config.json           params 1,438,019 = stem 205,568 + decoder 1,232,451
  wsst8s_subject_pool/subject_*.h5     wsst (N, 50, 71, 240) float16, ecg (N, 1, 1600)

The decoder split is derived from the module definitions: TCN 1,149,440, waveform head
41,345, anchor head 41,666 (sum 1,232,451, exact).

The contribution badges carry only what `docs/PAPER_CLAIM_LEDGER.md` supports. No badge
claims waveform or morphology superiority -- the ledger forbids that sentence.

Usage:
  python scripts/figures/plot_remnet_architecture.py
  python scripts/figures/plot_remnet_architecture.py --no-badges   # bare network
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import scienceplots  # noqa: F401
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "docs" / "paper" / "figures"

# Canvas in hundredths of an inch, so the axes are isotropic and rounded corners
# stay circular regardless of the aspect ratio.
FIG_W, FIG_H = 7.16, 4.60
W, H = FIG_W * 100, FIG_H * 100

NAVY = "#2C3E50"      # shared spine
BLUE = "#1F449C"      # R -- representation
AMBER = "#B9770E"     # E -- event / anchors
RED = "#922B21"       # M -- morphology / waveform
GREY = "#4D4D4D"      # objective
TEAL = "#117A65"      # contribution badges
TINT = {BLUE: "#EAEFF7", NAVY: "#EDF0F2", AMBER: "#FAF1E2", RED: "#F7EAE9"}

plt.style.use(["science", "ieee"])
matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "sans-serif",
    "savefig.transparent": True, "figure.facecolor": "none",
})


def band(ax, x0, x1, y0, y1, label, colour):
    """A soft grouping panel behind a stage, with its band letter above."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=6",
                                facecolor=TINT[colour], edgecolor="none", zorder=0))
    ax.text((x0 + x1) / 2, y1 + 7, label, ha="center", va="bottom",
            fontsize=8, color=colour, fontweight="bold", zorder=3)


def box(ax, x, y, w, h, title, lines, colour, note=None, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                facecolor="white", edgecolor=colour, linewidth=lw,
                                zorder=2))
    ax.text(x + w / 2, y + h - 11, title, ha="center", va="top", fontsize=8.2,
            color=colour, fontweight="bold", zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 26 - i * 12.5, line, ha="center", va="top",
                fontsize=6.9, color="#333333", zorder=3)
    if note:
        ax.text(x + w / 2, y + 6, note, ha="center", va="bottom", fontsize=6.9,
                color=colour, style="italic", zorder=3)


def chip(ax, x, y, w, h, text, colour, above=None, below=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                facecolor="white", edgecolor=colour, linewidth=0.8,
                                linestyle=(0, (3, 1.6)), zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
            color=colour, zorder=3)
    if above:
        ax.text(x + w / 2, y + h + 5, above, ha="center", va="bottom", fontsize=5.9,
                color="#6B6B6B", zorder=3)
    if below:
        ax.text(x + w / 2, y - 5, below, ha="center", va="top", fontsize=6.4,
                color="#444444", zorder=3)


def arrow(ax, p0, p1, colour=NAVY, style="-|>", lw=1.1, dashed=False,
          connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9,
                                 color=colour, linewidth=lw, zorder=4,
                                 shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3.2, 1.8)) if dashed else "solid",
                                 connectionstyle=connection))


def badge(ax, x, y, n):
    ax.add_patch(Circle((x, y), 9.5, facecolor=TEAL, edgecolor="white", linewidth=1.0,
                        zorder=6))
    ax.text(x, y - 0.4, str(n), ha="center", va="center", fontsize=7.2, color="white",
            fontweight="bold", zorder=7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-badges", action="store_true",
                    help="draw the bare network with no contribution callouts")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    spine_y = 305

    band(ax, 8, 318, 226, 360, "R  Representation", BLUE)
    band(ax, 322, 542, 226, 360, "shared temporal model", NAVY)
    band(ax, 552, 712, 196, 420, "E / M  decoupled read-outs", AMBER)

    # ---- representation --------------------------------------------------------
    box(ax, 16, 264, 88, 82, "WSST",
        ["50 range bins", "71 frequencies", "240 frames / 8 s"], BLUE)
    arrow(ax, (104, spine_y), (116, spine_y))

    box(ax, 116, 252, 132, 106, "WSST Stem",
        ["Conv 3x3   50 -> 128", "BN + GELU + MaxPool 2x1", "Conv 3x3  128 -> 128",
         "average-pool frequency"], BLUE, note="0.21 M")
    arrow(ax, (248, spine_y), (260, spine_y))

    chip(ax, 260, 291, 52, 28, "-> 400", BLUE,
         above="linear interp", below="128 x 400\n50 Hz latent")
    arrow(ax, (312, spine_y), (324, spine_y))

    # ---- shared temporal model -------------------------------------------------
    box(ax, 324, 262, 140, 90, "Dilated residual TCN",
        ["5 residual blocks,  k = 7", "128 channels, BN + GELU",
         "receptive field ~ 7.5 s"], NAVY, note="1.15 M")
    ax.text(324, 245, "dilation", ha="left", va="center", fontsize=6.2, color=NAVY)
    for i, d in enumerate((1, 2, 4, 8, 16)):
        cx = 366 + i * 19
        ax.add_patch(FancyBboxPatch((cx - 8, 238), 16, 14,
                                    boxstyle="round,pad=0,rounding_size=2",
                                    facecolor="white", edgecolor=NAVY, linewidth=0.6,
                                    zorder=3))
        ax.text(cx, 244.6, str(d), ha="center", va="center", fontsize=5.8, color=NAVY,
                zorder=4)
    arrow(ax, (464, spine_y), (476, spine_y))

    chip(ax, 476, 291, 52, 28, "-> 1600", NAVY,
         above="linear interp", below="128 x 1600\n200 Hz")

    # ---- fork ------------------------------------------------------------------
    ax.plot([528, 552], [spine_y, spine_y], color=NAVY, lw=1.1, zorder=4)
    ax.plot([552, 552], [248, 375], color=NAVY, lw=1.1, zorder=4)
    ax.add_patch(Circle((552, spine_y), 2.6, color=NAVY, zorder=5))
    arrow(ax, (552, 375), (564, 375))
    arrow(ax, (552, 248), (564, 248))

    box(ax, 564, 338, 128, 78, "Anchor Head",
        ["Conv 5  128 -> 64, GELU", "Conv 5  64 -> 2"], AMBER, note="2 x 1600  logits")
    box(ax, 564, 210, 128, 78, "Waveform Head",
        ["Conv 5  128 -> 64, GELU", "Conv 5  64 -> 1 (linear)"], RED,
        note="1 x 1600  ECG")

    # ---- objective -------------------------------------------------------------
    ax.add_patch(FancyBboxPatch((286, 112), 420, 62,
                                boxstyle="round,pad=0,rounding_size=4",
                                facecolor="white", edgecolor=GREY, linewidth=0.9,
                                linestyle=(0, (3.2, 1.8)), zorder=2))
    ax.text(496, 160, "Training objective", ha="center", va="top", fontsize=8.0,
            color=GREY, fontweight="bold", zorder=3)
    ax.text(496, 144, "L  =  Huber  +  0.20 x BCE (anchors)  +  0.50 x MR-STFT",
            ha="center", va="top", fontsize=7.6, color="#333333", zorder=3)
    ax.text(496, 129, "MR-STFT windows 128 / 256 / 512,   hops 32 / 64 / 128",
            ha="center", va="top", fontsize=6.6, color="#555555", zorder=3)
    arrow(ax, (700, 338), (700, 174), colour=AMBER, style="<|-", dashed=True, lw=0.9)
    arrow(ax, (590, 210), (590, 174), colour=RED, style="<|-", dashed=True, lw=0.9)

    ax.text(16, 196, "Total  1.44 M parameters", ha="left", va="center", fontsize=7.6,
            color=NAVY, fontweight="bold")
    ax.text(16, 183, "stem 0.21 M  +  decoder 1.23 M", ha="left", va="center",
            fontsize=6.9, color="#555555")

    # ---- contribution badges and legend ---------------------------------------
    if not args.no_badges:
        badge(ax, 122, 352, 1)
        badge(ax, 543, 398, 2)
        badge(ax, 292, 168, 3)
        badge(ax, 20, 212, 4)

        items = [
            (1, "Synchrosqueezed input.", "One stem over 50 range bins; frequency is "
             "pooled, time is kept."),
            (2, "Factorized read-outs.", "Anchors and waveform are separate heads on "
             "one representation."),
            (3, "Spectral objective (L3).", "Selected by the registered loss gate; "
             "median >10 Hz ratio 0.122."),
            (4, "Lightweight.", "1.44 M against 41.6 M for the multitask ODE "
             "comparator."),
        ]
        for i, (n, headline, tail) in enumerate(items):
            col, row = i % 2, i // 2
            x = 20 + col * 352
            y = 72 - row * 26
            ax.add_patch(Circle((x, y), 7.4, facecolor=TEAL, edgecolor="none", zorder=6))
            ax.text(x, y - 0.3, str(n), ha="center", va="center", fontsize=6.2,
                    color="white", fontweight="bold", zorder=7)
            ax.text(x + 13, y + 4.5, headline, ha="left", va="center", fontsize=7.0,
                    color=TEAL, fontweight="bold")
            ax.text(x + 13, y - 5.5, tail, ha="left", va="center", fontsize=6.5,
                    color="#444444")

        ax.text(20, 14, "Contributions are protocol- and design-level. No panel of this "
                "figure claims superior waveform morphology.", ha="left", va="center",
                fontsize=6.2, color="#8A8A8A", style="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"Fig_REMNet_Architecture{args.suffix}"
    for path in (OUT / f"{stem}.pdf", OUT / f"{stem}.png"):
        fig.savefig(path, transparent=True, dpi=600)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
