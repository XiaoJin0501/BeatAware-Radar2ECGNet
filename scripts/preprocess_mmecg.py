"""
preprocess_mmecg.py — MMECG 数据集预处理脚本

从 MMECG.h5 提取滑窗片段并保存为 NPY 格式，供 MMECGDataset 加载。

输出结构：
    dataset_mmecg/
        subject_1/
            rcg.npy     [N, 50, 1600]  float32
            ecg.npy     [N, 1,  1600]  float32  (per-window min-max [0,1])
            rpeak.npy   [N, 1,  1600]  float32  (Gaussian soft label, σ=5)
            meta.npy    [N, 2]         int32    (subject_id, state_code)
        ...
        subject_11/
        metadata_mmecg.json

state_code: NB=0, IB=1, SP=2, PE=3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, sosfilt

# ── 常量 ─────────────────────────────────────────────────────────────────────
H5_PATH     = Path("/home/qhh2237/Datasets/MMECG/MMECG.h5")
OUT_ROOT    = Path("dataset_mmecg")
FS          = 200        # Hz
WIN_LEN     = 1600       # samples (8 s)
STRIDE      = 800        # samples (50% overlap)
RPEAK_SIGMA = 5          # Gaussian soft label σ (= 25ms @ 200Hz)
STATE_MAP   = {"NB": 0, "IB": 1, "SP": 2, "PE": 3}


# ── 信号处理工具 ──────────────────────────────────────────────────────────────
def bandpass(x: np.ndarray, lo: float = 0.5, hi: float = 40.0, fs: int = FS):
    """零相位 Butterworth 带通滤波（沿最后一维）。"""
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfilt(sos, x, axis=-1)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """对 (C, T) 数组逐通道 z-score 归一化，消除 range bin 间幅度差异。"""
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((x - mu) / std).astype(np.float32)


def gaussian_mask(indices: np.ndarray, length: int, sigma: float = RPEAK_SIGMA) -> np.ndarray:
    """将 R 峰索引转换为 Gaussian 软标签。"""
    mask = np.zeros(length, dtype=np.float32)
    for idx in indices:
        lo = max(0, int(idx) - 3 * int(sigma))
        hi = min(length, int(idx) + 3 * int(sigma) + 1)
        t = np.arange(lo, hi)
        mask[lo:hi] += np.exp(-0.5 * ((t - idx) / sigma) ** 2)
    return np.clip(mask, 0.0, 1.0)


def detect_rpeaks(ecg_1d: np.ndarray, fs: int = FS):
    """NeuroKit2 R 峰检测，返回局部窗口内索引。"""
    try:
        _, info = nk.ecg_peaks(ecg_1d, sampling_rate=fs, method="pantompkins1985")
        return info["ECG_R_Peaks"]
    except Exception:
        return np.array([], dtype=int)


def normalize_ecg(ecg_1d: np.ndarray) -> np.ndarray:
    """Per-window min-max 归一化到 [0, 1]。"""
    lo, hi = ecg_1d.min(), ecg_1d.max()
    if hi - lo < 1e-8:
        return np.zeros_like(ecg_1d, dtype=np.float32)
    return ((ecg_1d - lo) / (hi - lo)).astype(np.float32)


# ── 主处理函数 ─────────────────────────────────────────────────────────────────
def process_record(rcg_raw: np.ndarray, ecg_raw: np.ndarray,
                   subject_id: int, state: str):
    """
    输入:
        rcg_raw  (T, 50)  — 原始 RCG
        ecg_raw  (T, 1)   — 原始 ECG
    返回:
        rcg_wins   [N, 50, 1600]
        ecg_wins   [N, 1,  1600]
        rpeak_wins [N, 1,  1600]
        meta_wins  [N, 2]         (subject_id, state_code)
    """
    T = rcg_raw.shape[0]
    ecg_1d = ecg_raw[:, 0].astype(np.float64)

    # 对 ECG 做带通滤波（保留心脏波形完整形态，0.5–40 Hz）
    ecg_bp = bandpass(ecg_1d, lo=0.5, hi=40.0)

    # 对 RCG 每个 range bin 做宽带滤波（0.5–40 Hz）
    # 宽带保留 RCG 中的心脏时序信息（含相位），模型从中学习向完整ECG的映射。
    # 窄带(0.8-3.5Hz)虽然提升低频相关性，但会去除波形形态的高频信息，
    # 导致模型无法重建ECG的QRS形态，实际表现更差。
    rcg_T  = rcg_raw.T.astype(np.float64)            # (50, T)
    rcg_bp = bandpass(rcg_T, lo=0.5, hi=40.0)        # (50, T)

    # 逐通道 z-score：消除各 range bin 间 ~5x 的幅度差，
    # 使 FMCWRangeEncoder 的 SE 注意力按信号质量而非幅度选择通道
    rcg_bp = zscore_per_channel(rcg_bp)              # (50, T)  float32

    rcg_wins, ecg_wins, rpeak_wins, meta_wins = [], [], [], []
    state_code = STATE_MAP.get(state, 0)

    n_wins = (T - WIN_LEN) // STRIDE + 1
    for i in range(n_wins):
        s = i * STRIDE
        e = s + WIN_LEN

        # RCG 片段：float32
        rcg_w = rcg_bp[:, s:e].astype(np.float32)   # (50, 1600)

        # ECG 片段：归一化
        ecg_w = normalize_ecg(ecg_bp[s:e])           # (1600,)

        # R 峰检测 → Gaussian mask
        peaks = detect_rpeaks(ecg_w, fs=FS)
        rp_mask = gaussian_mask(peaks, WIN_LEN, RPEAK_SIGMA)  # (1600,)

        rcg_wins.append(rcg_w)
        ecg_wins.append(ecg_w[np.newaxis, :])    # (1, 1600)
        rpeak_wins.append(rp_mask[np.newaxis, :])  # (1, 1600)
        meta_wins.append([subject_id, state_code])

    if len(rcg_wins) == 0:
        return None

    return (
        np.stack(rcg_wins,    axis=0),   # (N, 50, 1600)
        np.stack(ecg_wins,    axis=0),   # (N,  1, 1600)
        np.stack(rpeak_wins,  axis=0),   # (N,  1, 1600)
        np.array(meta_wins,   dtype=np.int32),  # (N, 2)
    )


def main(args):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 按 subject_id 收集所有片段
    subject_data: dict[int, dict] = {}

    with h5py.File(H5_PATH, "r") as f:
        group_names = sorted(f.keys())
        print(f"Total records in H5: {len(group_names)}")

        for name in group_names:
            grp = f[name]
            sid   = int(grp.attrs["subject_id"])
            state = str(grp.attrs["state"])
            rcg   = grp["RCG"][()]    # (35505, 50)
            ecg   = grp["ECG"][()]    # (35505,  1)

            result = process_record(rcg, ecg, sid, state)
            if result is None:
                continue
            rcg_w, ecg_w, rp_w, meta_w = result

            if sid not in subject_data:
                subject_data[sid] = {"rcg": [], "ecg": [], "rpeak": [], "meta": []}
            subject_data[sid]["rcg"].append(rcg_w)
            subject_data[sid]["ecg"].append(ecg_w)
            subject_data[sid]["rpeak"].append(rp_w)
            subject_data[sid]["meta"].append(meta_w)

            if not args.quiet:
                print(f"  [{name}] subject={sid} state={state} wins={len(rcg_w)}")

    # 保存每个受试者的 NPY 文件
    subject_ids = sorted(subject_data.keys())
    total_wins = 0
    summary = {}

    for sid in subject_ids:
        d = subject_data[sid]
        rcg_arr   = np.concatenate(d["rcg"],   axis=0)
        ecg_arr   = np.concatenate(d["ecg"],   axis=0)
        rp_arr    = np.concatenate(d["rpeak"], axis=0)
        meta_arr  = np.concatenate(d["meta"],  axis=0)

        out_dir = OUT_ROOT / f"subject_{sid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "rcg.npy",   rcg_arr)
        np.save(out_dir / "ecg.npy",   ecg_arr)
        np.save(out_dir / "rpeak.npy", rp_arr)
        np.save(out_dir / "meta.npy",  meta_arr)

        print(f"subject_{sid:2d}: {len(rcg_arr):5d} windows  -> {out_dir}")
        total_wins += len(rcg_arr)
        summary[sid] = int(len(rcg_arr))

    # LOSO fold 分配（fold_i 留出 subject i+1 作测试集）
    loso_folds = {i: {"test": subject_ids[i], "train": [s for s in subject_ids if s != subject_ids[i]]}
                  for i in range(len(subject_ids))}

    meta_json = {
        "subject_ids":  subject_ids,
        "n_subjects":   len(subject_ids),
        "n_folds":      len(subject_ids),
        "win_len":      WIN_LEN,
        "stride":       STRIDE,
        "fs":           FS,
        "total_windows": total_wins,
        "windows_per_subject": summary,
        "state_map":    STATE_MAP,
        "loso_folds":   loso_folds,
    }
    json_path = OUT_ROOT / "metadata_mmecg.json"
    with open(json_path, "w") as jf:
        json.dump(meta_json, jf, indent=2)

    print(f"\nDone. Total windows: {total_wins}")
    print(f"Metadata written to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(args)
