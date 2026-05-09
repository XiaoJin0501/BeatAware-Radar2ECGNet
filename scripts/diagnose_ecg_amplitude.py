"""
diagnose_ecg_amplitude.py — D3: per-window ECG amplitude distribution by subject.

If per-window (max - min) of z-score ECG varies a lot across windows, then
per-window min-max normalization (currently done in MMECGWindowedH5Dataset) is
flattening absolute amplitude information across windows. R-peak height in the
[0,1] target then encodes ratio-to-window-max rather than physiological
amplitude — which is harder to learn consistently.

Reads samplewise/train.h5 directly (no model needed).

Output:
  experiments_mmecg/diagnostics/D3_ecg_amplitude_histogram.png
  experiments_mmecg/diagnostics/D3_ecg_amplitude_summary.csv
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

H5_PATH = Path("/home/qhh2237/Datasets/MMECG/processed/samplewise/train.h5")
OUT_DIR = ROOT / "experiments_mmecg" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[D3] reading {H5_PATH}")
    with h5py.File(H5_PATH, "r") as hf:
        ecg = hf["ecg"][:]                     # [N, 1, 1600] float32 z-score
        subj = hf["subject_id"][:].astype(int) # [N]
        phys = hf["physistatus"][:]             # [N] bytes
    N = ecg.shape[0]

    # per-window amplitude in z-score domain
    amp = ecg[:, 0, :].max(axis=-1) - ecg[:, 0, :].min(axis=-1)  # [N]
    scenes = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in phys])
    print(f"[D3] {N} windows, subjects {sorted(set(subj.tolist()))}, scenes {sorted(set(scenes))}")

    # ── per-subject histogram grid
    sub_unique = sorted(set(subj.tolist()))
    n = len(sub_unique)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.4),
                             constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]

    rows_csv = []
    global_min, global_max = float(amp.min()), float(amp.max())

    for ax, s in zip(axes, sub_unique):
        a = amp[subj == s]
        ax.hist(a, bins=30, color="#4C72B0", alpha=0.85)
        ax.set_title(f"Subject {s}  (n={len(a)})", fontsize=9)
        ax.set_xlim(global_min, global_max)
        ax.axvline(np.median(a), color="red", lw=1, label=f"med={np.median(a):.2f}")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.tick_params(labelsize=7)
        rows_csv.append({
            "subject_id": int(s),
            "n_windows": int(len(a)),
            "amp_median": float(np.median(a)),
            "amp_mean":   float(a.mean()),
            "amp_std":    float(a.std()),
            "amp_min":    float(a.min()),
            "amp_max":    float(a.max()),
            "amp_iqr":    float(np.percentile(a, 75) - np.percentile(a, 25)),
            "cv":         float(a.std() / (a.mean() + 1e-8)),
        })

    for ax in axes[len(sub_unique):]:
        ax.axis("off")

    fig.suptitle(
        "D3: per-window ECG amplitude (z-score domain max−min) by subject\n"
        "Wide distributions ⇒ per-window min-max normalization compresses physiological R-peak amplitude inconsistently.",
        fontsize=10,
    )
    out_png = OUT_DIR / "D3_ecg_amplitude_histogram.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[D3] saved {out_png}")

    df = pd.DataFrame(rows_csv).sort_values("subject_id")
    out_csv = OUT_DIR / "D3_ecg_amplitude_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[D3] saved {out_csv}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
