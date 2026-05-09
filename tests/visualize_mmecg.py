"""
visualize_mmecg.py — MMECG H5 数据集诊断可视化

用法：
  # LOSO fold 1 的 train split，显示 3 个 segment
  python tests/visualize_mmecg.py --fold 1 --split train --n 3

  # LOSO fold 1 的 test split（含峰值标注）
  python tests/visualize_mmecg.py --fold 1 --split test --n 2

  # Samplewise 协议的 val split
  python tests/visualize_mmecg.py --protocol samplewise --split val --n 4

  # 仅打印统计，不显示图形
  python tests/visualize_mmecg.py --fold 1 --split train --n 100 --no-plot
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOSO_DIR = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
SW_DIR   = Path("/home/qhh2237/Datasets/MMECG/processed/samplewise")
STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}
PHYSISTATUS_MAP = {"NB": 0, "IB": 1, "SP": 2, "PE": 3}


def _resolve_h5(protocol: str, fold: int, split: str) -> Path:
    if protocol == "samplewise":
        return SW_DIR / f"{split}.h5"
    else:
        return LOSO_DIR / f"fold_{fold:02d}" / f"{split}.h5"


def print_stats(h5_path: Path, n_samples: int) -> None:
    print(f"\nH5 file: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        N = hf["rcg"].shape[0]
        print(f"  Total segments : {N}")
        print(f"  Showing        : {n_samples}")

        rcg = hf["rcg"][:n_samples]
        ecg = hf["ecg"][:n_samples]
        subj = hf["subject_id"][:n_samples]
        phys = hf["physistatus"][:n_samples]

        print(f"\n  rcg  shape={rcg.shape} dtype={rcg.dtype}")
        print(f"       min={rcg.min():.3f}  max={rcg.max():.3f}  "
              f"mean={rcg.mean():.3f}  std={rcg.std():.3f}")

        print(f"\n  ecg  shape={ecg.shape} dtype={ecg.dtype}")
        print(f"       min={ecg.min():.3f}  max={ecg.max():.3f}  "
              f"mean={ecg.mean():.3f}  std={ecg.std():.3f}  "
              f"[H5=z-score; loader applies min-max→[0,1]]")

        print(f"\n  subject_ids : {sorted(set(subj.tolist()))}")
        decoded = [p.decode() if isinstance(p, bytes) else str(p) for p in phys]
        unique, counts = np.unique(decoded, return_counts=True)
        print(f"  physistatus : { {k: int(v) for k,v in zip(unique, counts)} }")

        if "rpeak_indices" in hf:
            n_rpeaks = [len(hf["rpeak_indices"][i]) for i in range(min(n_samples, N))]
            print(f"\n  R-peak counts (first {n_samples}): "
                  f"min={min(n_rpeaks)} max={max(n_rpeaks)} "
                  f"mean={np.mean(n_rpeaks):.1f}")

        if "delineation_valid" in hf:
            dv = hf["delineation_valid"][:n_samples]
            print(f"  delineation_valid : {int(dv.sum())}/{n_samples} valid")


def plot_segments(h5_path: Path, n_plot: int, include_peaks: bool) -> None:
    with h5py.File(h5_path, "r") as hf:
        N = hf["rcg"].shape[0]
        indices = np.linspace(0, N - 1, min(n_plot, N), dtype=int)

        for idx in indices:
            rcg_seg  = hf["rcg"][idx]    # (50, 1600)
            ecg_seg  = hf["ecg"][idx, 0] # (1600,)
            rpeak_ds = hf["rpeak_indices"]
            r_peaks  = rpeak_ds[idx].astype(np.int32) if "rpeak_indices" in hf else None
            subj  = int(hf["subject_id"][idx])
            phys_b = hf["physistatus"][idx]
            state = phys_b.decode() if isinstance(phys_b, bytes) else str(phys_b)

            has_qst = include_peaks and all(
                k in hf for k in ("q_indices", "s_indices", "tpeak_indices")
            )
            if has_qst:
                q_idx = hf["q_indices"][idx].astype(np.int32)
                s_idx = hf["s_indices"][idx].astype(np.int32)
                t_idx = hf["tpeak_indices"][idx].astype(np.int32)

            fig, axes = plt.subplots(2, 1, figsize=(14, 5),
                                     gridspec_kw={"height_ratios": [2, 1]})
            fig.suptitle(f"Segment {idx} | Subject {subj} | State: {state}", fontsize=11)

            # ── radar heatmap ────────────────────────────────────────────────
            ax = axes[0]
            im = ax.imshow(rcg_seg, aspect="auto", origin="lower",
                           cmap="RdBu_r", vmin=-3, vmax=3,
                           extent=[0, 8, 0, 50])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Range bin")
            ax.set_title("RCG (z-score, 50 channels)")
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

            # ── ECG + peaks ──────────────────────────────────────────────────
            ax2 = axes[1]
            t = np.linspace(0, 8, 1600)
            ax2.plot(t, ecg_seg, color="steelblue", lw=0.8, label="ECG")

            if r_peaks is not None and len(r_peaks) > 0:
                rp = r_peaks[(r_peaks >= 0) & (r_peaks < 1600)]
                ax2.vlines(rp / 200, 0, 1, color="red", lw=0.7,
                           alpha=0.6, label="R peaks")

            if has_qst:
                for arr, color, label in [
                    (q_idx, "purple", "Q"), (s_idx, "green", "S"),
                    (t_idx, "orange", "T"),
                ]:
                    valid = arr[(arr >= 0) & (arr < 1600)]
                    if len(valid):
                        ax2.vlines(valid / 200, 0, 0.8, color=color,
                                   lw=0.6, alpha=0.5, label=label)

            ax2.set_xlim(0, 8)
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.legend(loc="upper right", fontsize=7, ncol=4)

            plt.tight_layout()
            plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     type=int,  default=1,
                        help="LOSO fold index 1-based (ignored for samplewise)")
    parser.add_argument("--split",    type=str,  default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--protocol", type=str,  default="loso",
                        choices=["loso", "samplewise"])
    parser.add_argument("--n",        type=int,  default=3,
                        help="Number of segments to visualize (stats always use all)")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Print stats only, skip plotting")
    args = parser.parse_args()

    h5_path = _resolve_h5(args.protocol, args.fold, args.split)
    if not h5_path.exists():
        print(f"H5 file not found: {h5_path}")
        sys.exit(1)

    # stats over all (up to 500 for speed)
    with h5py.File(h5_path, "r") as hf:
        N_total = hf["rcg"].shape[0]
    print_stats(h5_path, min(N_total, 500))

    if not args.no_plot:
        include_peaks = (args.split == "test")
        try:
            plot_segments(h5_path, n_plot=args.n, include_peaks=include_peaks)
        except Exception as e:
            print(f"Plot failed (headless server?): {e}")
            print("Re-run with --no-plot to skip plotting.")


if __name__ == "__main__":
    main()
