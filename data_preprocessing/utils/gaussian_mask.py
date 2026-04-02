"""
gaussian_mask.py — 生成高斯软标签 R 峰 Mask

用高斯曲线替代二值化的 R 峰标注，使模型对峰值位置的学习更平滑。
σ = 5 采样点 @ 200 Hz = 25 ms，覆盖 QRS 尖峰的典型宽度。
"""

import numpy as np


def generate_gaussian_mask(
    r_peaks_idx: np.ndarray,
    signal_len: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    以每个 R 峰为中心生成高斯软标签，多个峰叠加后 clip 到 [0, 1]。

    Parameters
    ----------
    r_peaks_idx : ndarray, shape (M,), int
        R 峰在目标采样率（200Hz）下的索引位置
    signal_len : int
        信号长度（采样点数）
    sigma : float
        高斯标准差（采样点），默认 5（@200Hz = 25ms）

    Returns
    -------
    mask : ndarray, shape (signal_len,), float32
        高斯软标签，值域 [0, 1]
    """
    mask = np.zeros(signal_len, dtype=np.float64)
    r_peaks_idx = np.asarray(r_peaks_idx, dtype=np.int64)

    t = np.arange(signal_len, dtype=np.float64)
    for idx in r_peaks_idx:
        if 0 <= idx < signal_len:
            mask += np.exp(-0.5 * ((t - idx) / sigma) ** 2)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)
