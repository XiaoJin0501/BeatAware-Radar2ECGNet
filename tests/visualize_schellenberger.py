"""
visualize_dataset.py — 数据集可视化诊断脚本

功能：
  1. 打印各 NPY 文件的 shape / dtype / 值域
  2. 用 matplotlib 画出前 5 秒波形（ECG / radar_raw / radar_phase / rpeak mask）
  3. 保存图片到 tests/output/

用法：
  python tests/visualize_dataset.py [--dataset_dir dataset] [--subject GDN0001]
                                    [--scenario resting] [--seg_idx 0]
                                    [--out_dir tests/output]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无 GUI 后端
import matplotlib.pyplot as plt
import numpy as np


FS = 200          # 200 Hz
PLOT_SECS = 5     # 前 5 秒
N_PLOT = FS * PLOT_SECS   # 1000 个采样点


def print_array_info(name: str, arr: np.ndarray) -> None:
    print(f"  {name:<22s}  shape={str(arr.shape):<20s}  dtype={str(arr.dtype):<10s}"
          f"  min={arr.min():.4f}  max={arr.max():.4f}")


def visualize(dataset_dir: Path, subject: str, scenario: str,
              seg_idx: int, out_dir: Path) -> None:
    seg_dir = dataset_dir / subject / scenario / "segments"
    if not seg_dir.exists():
        raise FileNotFoundError(f"分段目录不存在: {seg_dir}")

    # ── 加载 ──────────────────────────────────────────────────────────────
    files = {
        "radar_raw":        seg_dir / "radar_raw.npy",
        "radar_phase":      seg_dir / "radar_phase.npy",
        "ecg":              seg_dir / "ecg.npy",
        "rpeak":            seg_dir / "rpeak.npy",
        "radar_spec_input": seg_dir / "radar_spec_input.npy",
        "radar_spec_loss":  seg_dir / "radar_spec_loss.npy",
    }

    arrays = {}
    print(f"\n{'='*65}")
    print(f"受试者: {subject}  场景: {scenario}  分段索引: {seg_idx}")
    print(f"{'='*65}")
    print("\n[Array Info]")
    for key, path in files.items():
        if not path.exists():
            print(f"  {key:<22s}  *** 文件不存在 ***")
            continue
        arr = np.load(path)
        arrays[key] = arr
        print_array_info(key, arr)

    N_segs = arrays["ecg"].shape[0]
    print(f"\n  总分段数: {N_segs}")

    if seg_idx >= N_segs:
        raise IndexError(f"seg_idx={seg_idx} 超出范围（共 {N_segs} 段）")

    # ── 取出目标分段，提取前 5 秒 ─────────────────────────────────────────
    ecg        = arrays["ecg"][seg_idx, 0, :N_PLOT]          # (1000,)
    radar_raw  = arrays["radar_raw"][seg_idx, 0, :N_PLOT]
    radar_ph   = arrays["radar_phase"][seg_idx, 0, :N_PLOT]
    rpeak      = arrays["rpeak"][seg_idx, 0, :N_PLOT]
    t          = np.arange(N_PLOT) / FS                      # 时间轴（秒）

    # ── 画图 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Subject: {subject}  |  Scenario: {scenario}  |  Seg #{seg_idx}  |  前 {PLOT_SECS}s",
        fontsize=13, fontweight="bold"
    )

    # 1. ECG
    ax = axes[0]
    ax.plot(t, ecg, color="#E74C3C", linewidth=0.9)
    ax.set_ylabel("ECG (norm)", fontsize=10)
    ax.set_title("ECG — check for QRS peaks", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2. R-peak Gaussian Mask
    ax = axes[1]
    ax.fill_between(t, rpeak, alpha=0.6, color="#E67E22")
    ax.plot(t, rpeak, color="#E67E22", linewidth=0.8)
    ax.set_ylabel("R-peak mask", fontsize=10)
    ax.set_title("R-peak Gaussian label — check alignment with QRS", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Radar Phase (filtered)
    ax = axes[2]
    ax.plot(t, radar_ph, color="#2980B9", linewidth=0.9)
    ax.set_ylabel("phase (rad)", fontsize=10)
    ax.set_title("Radar Phase (bandpass 0.5-10 Hz) — check breathing + heartbeat", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Radar Raw (unfiltered)
    ax = axes[3]
    ax.plot(t, radar_raw, color="#27AE60", linewidth=0.9)
    ax.set_ylabel("phase (rad)", fontsize=10)
    ax.set_title("Radar Raw (ellipse corrected, unfiltered) — check breathing trend", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}_{scenario}_seg{seg_idx:03d}_waveform.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[图片已保存] {out_path}")

    # ── Spec 可视化（radar_spec_input，第 0 分段）────────────────────────
    if "radar_spec_input" in arrays:
        spec = arrays["radar_spec_input"][seg_idx, 0]    # (33, T)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        im = ax2.imshow(spec, aspect="auto", origin="lower",
                        cmap="viridis",
                        extent=[0, spec.shape[1] / FS * 8, 0, 10])
        plt.colorbar(im, ax=ax2, label="Magnitude")
        ax2.set_xlabel("Time (s)", fontsize=10)
        ax2.set_ylabel("Frequency (Hz)", fontsize=10)
        ax2.set_title(
            f"Radar Spec Input — {subject} / {scenario} / Seg #{seg_idx}  "
            f"shape={spec.shape}", fontsize=10
        )
        out_path2 = out_dir / f"{subject}_{scenario}_seg{seg_idx:03d}_spec.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[图片已保存] {out_path2}")

    # ── 多受试者 / 多场景概览（ECG 叠加图）──────────────────────────────
    subjects_sample = sorted([d.name for d in dataset_dir.iterdir()
                               if d.is_dir() and d.name.startswith("GDN")])[:6]
    fig3, axes3 = plt.subplots(len(subjects_sample), 1,
                                figsize=(14, 2.5 * len(subjects_sample)),
                                sharex=True)
    if len(subjects_sample) == 1:
        axes3 = [axes3]
    fig3.suptitle("First 6 Subjects — ECG (Resting, Seg #0, first 5s)", fontsize=12)

    for ax3, subj in zip(axes3, subjects_sample):
        p = dataset_dir / subj / "resting" / "segments" / "ecg.npy"
        if p.exists():
            e = np.load(p)[0, 0, :N_PLOT]
            ax3.plot(t, e, linewidth=0.8, color="#C0392B")
            ax3.set_ylabel(subj, fontsize=9, rotation=0, labelpad=55)
        else:
            ax3.text(0.5, 0.5, "无数据", ha="center", va="center",
                     transform=ax3.transAxes)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.3)

    axes3[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    out_path3 = out_dir / "multi_subject_ecg_overview.png"
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[图片已保存] {out_path3}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="数据集可视化诊断")
    parser.add_argument("--dataset_dir", type=Path,
                        default=project_root / "dataset")
    parser.add_argument("--subject",  type=str, default="GDN0001")
    parser.add_argument("--scenario", type=str, default="resting")
    parser.add_argument("--seg_idx",  type=int, default=0)
    parser.add_argument("--out_dir",  type=Path,
                        default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()

    visualize(args.dataset_dir, args.subject, args.scenario,
              args.seg_idx, args.out_dir)


if __name__ == "__main__":
    main()
