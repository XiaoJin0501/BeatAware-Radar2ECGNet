"""
diagnose_rcg_bin_correlation.py — D1: per range-bin RCG–ECG correlation heatmap.

For each window in samplewise/val.h5 we compute Pearson correlation between
each of the 50 range bins and the (z-score) ECG. We then aggregate by
(subject_id, scene) and plot a heatmap.

Two correlation views:
  (a) raw H5 RCG (already 0.5-20 Hz bandpass per dataset construction)
  (b) heart-band filtered RCG (0.8-3.5 Hz Butterworth filtfilt) — mirrors the
      narrow_bandpass option in the loader

Output:
  experiments_mmecg/diagnostics/D1_rcg_bin_correlation_raw.png
  experiments_mmecg/diagnostics/D1_rcg_bin_correlation_narrow.png
  experiments_mmecg/diagnostics/D1_rcg_bin_correlation_summary.csv
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

H5_PATH = Path("/home/qhh2237/Datasets/MMECG/processed/samplewise/val.h5")
OUT_DIR = ROOT / "experiments_mmecg" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FS = 200.0


def _per_window_corr(rcg_win: np.ndarray, ecg_win: np.ndarray) -> np.ndarray:
    """rcg_win:(50,L), ecg_win:(L,) → (50,) Pearson per bin."""
    rc = rcg_win - rcg_win.mean(axis=-1, keepdims=True)
    ec = ecg_win - ecg_win.mean()
    num = (rc * ec).sum(axis=-1)
    den = np.sqrt((rc ** 2).sum(axis=-1) * (ec ** 2).sum() + 1e-12)
    return num / den


def _bandpass(x: np.ndarray, lo: float = 0.8, hi: float = 3.5) -> np.ndarray:
    b, a = scipy.signal.butter(4, [lo, hi], btype="band", fs=FS)
    return scipy.signal.filtfilt(b, a, x, axis=-1)


def _heatmap(matrix: np.ndarray, row_labels: list[str], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(row_labels) + 1.2))
    vmax = float(np.abs(matrix).max())
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(0, 50, 5))
    ax.set_xlabel("Range bin (0..49)")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="mean |Pearson|")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"[D1] reading {H5_PATH}")
    with h5py.File(H5_PATH, "r") as hf:
        rcg = hf["rcg"][:]   # [N, 50, 1600] z-score per channel
        ecg = hf["ecg"][:, 0, :]  # [N, 1600] z-score
        subj = hf["subject_id"][:].astype(int)
        phys = hf["physistatus"][:]
    N = rcg.shape[0]
    scenes = np.array([p.decode() if isinstance(p, bytes) else str(p) for p in phys])
    print(f"[D1] {N} windows; computing per-window per-bin correlations…")

    # ── (a) raw RCG ↔ ECG per-window correlation
    corr_raw = np.zeros((N, 50), dtype=np.float32)
    for i in range(N):
        corr_raw[i] = _per_window_corr(rcg[i], ecg[i])

    # ── (b) heart-band filtered (0.8-3.5 Hz)
    print("[D1] applying 0.8-3.5 Hz heart-band filter…")
    rcg_nb = np.empty_like(rcg)
    for i in range(N):
        rcg_nb[i] = _bandpass(rcg[i])
    corr_nb = np.zeros((N, 50), dtype=np.float32)
    for i in range(N):
        corr_nb[i] = _per_window_corr(rcg_nb[i], ecg[i])

    # ── aggregate by (subject, scene) — use mean of |corr| (sign can flip per window)
    keys = list(zip(subj.tolist(), scenes.tolist()))
    groups = sorted(set(keys), key=lambda x: (x[0], x[1]))
    raw_mat = np.zeros((len(groups), 50), dtype=np.float32)
    nb_mat = np.zeros_like(raw_mat)
    rows_csv = []
    for gi, (s, sc) in enumerate(groups):
        idxs = [i for i, k in enumerate(keys) if k == (s, sc)]
        raw_mat[gi] = np.abs(corr_raw[idxs]).mean(axis=0)
        nb_mat[gi] = np.abs(corr_nb[idxs]).mean(axis=0)
        rows_csv.append({
            "subject_id": int(s),
            "scene": sc,
            "n_windows": len(idxs),
            "raw_max_bin":   int(raw_mat[gi].argmax()),
            "raw_max_corr":  float(raw_mat[gi].max()),
            "raw_mean_corr": float(raw_mat[gi].mean()),
            "nb_max_bin":    int(nb_mat[gi].argmax()),
            "nb_max_corr":   float(nb_mat[gi].max()),
            "nb_mean_corr":  float(nb_mat[gi].mean()),
        })

    row_labels = [f"S{s} {sc}" for (s, sc) in groups]
    _heatmap(raw_mat, row_labels,
             "D1(a): mean |Pearson(rcg_bin, ecg)| — raw RCG (0.5-20 Hz)",
             OUT_DIR / "D1_rcg_bin_correlation_raw.png")
    _heatmap(nb_mat, row_labels,
             "D1(b): mean |Pearson(rcg_bin, ecg)| — heart-band 0.8-3.5 Hz",
             OUT_DIR / "D1_rcg_bin_correlation_narrow.png")

    df = pd.DataFrame(rows_csv)
    out_csv = OUT_DIR / "D1_rcg_bin_correlation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[D1] saved {out_csv}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
