"""Shared plotting style for the REM-Net manuscript figures.

Follows the conventions of github.com/ChenLiu-1996/figures4papers: draw large, set
large fonts, thick lines, hide the top and right spines, thick axis lines, and
frameless legends. Figures are authored at roughly three times their printed size
and scaled down by LaTeX, which keeps strokes and labels crisp in print.

Palette: soft tints for the comparison systems and one saturated navy for the
proposed model, so the eye finds "ours" without a legend lookup. Reference levels
(the radar-free template floor and the oracle-timing bound) are drawn in a neutral
grey and always dashed, because they are properties of the measurement rather than
results of a system.

Every figure writes PDF (for LaTeX), PNG and JPG (quality 95).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "paper" / "sections" / "figures"

# Proposed system, then comparison systems in descending benchmark order.
OURS = "#0F4D92"
MODEL_COLORS = {
    "rem_net":               OURS,
    "mmecg_tcn":             "#E9A6A1",
    "radarode_mtl":          "#8BCF8B",
    "cross_modal_conformer": "#F2C46D",
    "wavegru_net":           "#A9C8E8",
    "attention_dcgan":       "#C9A9D6",
}
MODEL_LABELS = {
    "rem_net":               "REM-Net (ours)",
    "mmecg_tcn":             "Chen-TCN",
    "radarode_mtl":          "radarODE-MTL",
    "cross_modal_conformer": "Cross-Modal Conformer",
    "wavegru_net":           "WaveGRU-Net",
    "attention_dcgan":       "Attention-DCGAN",
}
MODEL_ORDER = ["rem_net", "mmecg_tcn", "radarode_mtl",
               "cross_modal_conformer", "wavegru_net", "attention_dcgan"]

REFERENCE = "#4D4D4D"     # template floor and oracle-timing bound
GT = "#2B2B2B"            # reference ECG in waveform panels
ACCENT = "#C0392B"        # the one thing a panel wants you to look at

# REM-Net has two anchor read-outs that are never merged (docs/paper/TERMINOLOGY.md):
# the anchor head is primary and drawn solid; the waveform-derived read-out is drawn
# in a lighter tint with hatching (boxes) or dashes (lines).
OURS_WAVE = "#7FA3D1"
SYSTEMS = {
    ("rem_net", "event"):     dict(label="REM-Net, anchor head", color=OURS, hatch="", ls="-"),
    ("rem_net", "waveform"):  dict(label="REM-Net, waveform peaks", color=OURS_WAVE, hatch="//", ls="--"),
    ("mmecg_tcn", "waveform"): dict(label="Chen-TCN", color=MODEL_COLORS["mmecg_tcn"], hatch="", ls="-"),
    ("wavegru_net", "waveform"): dict(label="WaveGRU-Net", color=MODEL_COLORS["wavegru_net"], hatch="", ls="-"),
    ("radarode_mtl", "waveform"): dict(label="radarODE-MTL", color=MODEL_COLORS["radarode_mtl"], hatch="", ls="-"),
}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 24,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 3,
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 (IEEE requirement)
        "ps.fonttype": 42,
    })


def style_axis(ax, ticks: bool = True) -> None:
    if ticks:
        ax.tick_params(labelsize=24, length=8, width=1.5)


def panel_tag(ax, text: str, dx: float = -0.06, dy: float = 1.04) -> None:
    """The (a) / (b) tag, placed outside the axes in the figures4papers manner."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=30,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)
    # JPG for submission systems that reject PNG; rendered from the 300-dpi PNG
    from PIL import Image
    Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.jpg", quality=95,
                                                         subsampling=0)
    print(f"wrote {OUT / f'{name}.jpg'}")
