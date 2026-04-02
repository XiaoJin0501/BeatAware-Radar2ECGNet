"""
step2_ecg_processing.py — ECG 信号预处理

流程：
  ECG 原始信号 (2000Hz, tfm_ecg2)
  → NeuroKit2 清洗（去噪/基线校正）
  → Pan-Tompkins R峰检测
  → 降采样至 200Hz (factor=10)
  → R峰索引在 200Hz 下重新映射

输出（每受试者每场景）：
  ecg_clean.npy      # [L_200], float32, 清洗+降采样后ECG（原始尺度，未归一化）
  rpeak_indices.npy  # [M],     int32,   R峰在200Hz下的索引

注意：
  - 归一化放到 step4_segment_save.py（per-segment），这里保留原始幅度
  - 高斯Mask也在 step4 分段后生成（避免跨段边界问题）

用法：
  python step2_ecg_processing.py [--raw_dir DIR] [--out_dir DIR]
"""

import argparse
import logging
from pathlib import Path

import neurokit2 as nk
import numpy as np
from scipy.signal import decimate

from data_preprocessing.utils.mat_loader import load_mat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FS = 200
DECIMATE_FACTOR = 10
VALID_SCENARIOS = {"Resting", "Valsalva", "Apnea"}


def process_ecg(ecg_raw: np.ndarray, fs: int = 2000) -> dict:
    """
    对单段 ECG 信号执行完整预处理流程。

    Parameters
    ----------
    ecg_raw : ndarray, shape (N,)
        原始 ECG 信号（tfm_ecg2），采样率 fs
    fs : int
        原始采样率（默认 2000Hz）

    Returns
    -------
    dict:
        ecg_clean      : ndarray, shape (N // DECIMATE_FACTOR,), float32
                         清洗 + 降采样后 ECG（200Hz），保留原始幅度范围
        rpeak_indices  : ndarray, shape (M,), int32
                         R 峰在 200Hz 下的索引
        rpeak_indices_orig : ndarray, shape (M,), int32
                         R 峰在原始采样率下的索引（调试用）
        n_rpeaks       : int
    """
    ecg_raw = np.asarray(ecg_raw, dtype=np.float64)

    # 1. NeuroKit2 ECG 清洗（在原始采样率下进行，效果更好）
    ecg_cleaned = nk.ecg_clean(ecg_raw, sampling_rate=fs, method="neurokit")

    # 2. R峰检测（在原始采样率下，精度最高）
    try:
        _, rpeaks_info = nk.ecg_peaks(
            ecg_cleaned, sampling_rate=fs, method="neurokit"
        )
        rpeaks_orig = rpeaks_info["ECG_R_Peaks"].astype(np.int64)
    except Exception as e:
        logger.warning(f"neurokit R峰检测失败，尝试 pantompkins: {e}")
        try:
            _, rpeaks_info = nk.ecg_peaks(
                ecg_cleaned, sampling_rate=fs, method="pantompkins"
            )
            rpeaks_orig = rpeaks_info["ECG_R_Peaks"].astype(np.int64)
        except Exception as e2:
            logger.error(f"R峰检测完全失败: {e2}")
            rpeaks_orig = np.array([], dtype=np.int64)

    # 3. 降采样 ECG 到 200Hz
    ecg_200 = _decimate_ecg(ecg_cleaned, factor=DECIMATE_FACTOR)

    # 4. R峰索引映射到 200Hz（round后clip到合法范围）
    if len(rpeaks_orig) > 0:
        rpeaks_200 = np.round(rpeaks_orig / DECIMATE_FACTOR).astype(np.int32)
        rpeaks_200 = np.clip(rpeaks_200, 0, len(ecg_200) - 1)
        # 去重（降采样可能使相邻峰映射到同一位置）
        rpeaks_200 = np.unique(rpeaks_200)
    else:
        rpeaks_200 = np.array([], dtype=np.int32)

    return {
        "ecg_clean": ecg_200.astype(np.float32),
        "rpeak_indices": rpeaks_200.astype(np.int32),
        "rpeak_indices_orig": rpeaks_orig.astype(np.int32),
        "n_rpeaks": len(rpeaks_200),
    }


def _decimate_ecg(signal: np.ndarray, factor: int) -> np.ndarray:
    """ECG 降采样，分步执行避免数值不稳定（factor=10=2×5）。"""
    out = signal.copy()
    for step in _factorize(factor):
        out = decimate(out, step, zero_phase=True)
    return out


def _factorize(n: int) -> list[int]:
    """将因子分解为不超过13的质因子序列。"""
    factors = []
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    return factors if factors else [1]


def process_subject_scenario(
    mat_path: Path,
    out_dir: Path,
) -> bool:
    """
    处理单个受试者的单个场景 ECG，保存到 out_dir/<subject>/<scenario>/。

    Returns
    -------
    bool: 是否处理成功
    """
    try:
        data = load_mat(mat_path)
    except Exception as e:
        logger.error(f"加载失败 {mat_path}: {e}")
        return False

    subject = data["subject"]
    scenario = data["scenario"]

    if scenario not in VALID_SCENARIOS:
        logger.debug(f"跳过场景 {scenario}（{mat_path.name}）")
        return True

    save_dir = out_dir / subject / scenario.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "ecg_clean.npy").exists():
        logger.info(f"已存在，跳过: {save_dir}")
        return True

    logger.info(f"处理 ECG: {subject} / {scenario} ...")

    result = process_ecg(data["tfm_ecg2"], fs=data["fs_ecg"])

    np.save(save_dir / "ecg_clean.npy", result["ecg_clean"])
    np.save(save_dir / "rpeak_indices.npy", result["rpeak_indices"])

    logger.info(
        f"  已保存: ecg_clean{result['ecg_clean'].shape}, "
        f"R峰数={result['n_rpeaks']}, "
        f"平均心率≈{result['n_rpeaks'] / (len(result['ecg_clean']) / TARGET_FS) * 60:.1f} bpm"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="ECG 信号预处理")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/home/qhh2237/Datasets/Med_Radar"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset"),
    )
    args = parser.parse_args()

    mat_files = sorted(args.raw_dir.rglob("*.mat"))
    logger.info(f"共找到 {len(mat_files)} 个 .mat 文件")

    success, failed = 0, 0
    for mat_path in mat_files:
        ok = process_subject_scenario(mat_path, args.out_dir)
        if ok:
            success += 1
        else:
            failed += 1

    logger.info(f"完成：成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()
