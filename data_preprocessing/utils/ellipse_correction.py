"""
ellipse_correction.py — I/Q 椭圆失真校正

六端口连续波雷达输出的 I/Q 信号通常存在：
  1. DC 偏移（直流分量叠加）
  2. 幅度不平衡（I/Q 两路增益不同）
  3. 正交误差（I/Q 相位差偏离 90°）

三步校正使 IQ 轨迹从椭圆还原为圆形，提高反正切解调的准确性。
"""

import numpy as np


def ellipse_correction(
    I: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    三步 I/Q 椭圆校正：去DC → 幅度均衡 → 正交性校正。

    Parameters
    ----------
    I, Q : ndarray, shape (N,)
        原始雷达 I/Q 信号

    Returns
    -------
    I_corr, Q_corr : ndarray, shape (N,)
        校正后的 I/Q 信号，均值为0，IQ轨迹近似圆形
    """
    I = np.asarray(I, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Step 1: 去除 DC 偏移
    I = I - np.mean(I)
    Q = Q - np.mean(Q)

    # Step 2: 幅度均衡（令 Q 的标准差与 I 相同）
    std_I = np.std(I)
    std_Q = np.std(Q)
    if std_Q > 1e-12:
        Q = Q * (std_I / std_Q)

    # Step 3: 正交性校正（SVD 法去除 I/Q 相关性）
    # 将 [I, Q] 视为 2D 点集，用 SVD 对齐主轴到坐标轴
    iq = np.stack([I, Q], axis=0)  # (2, N)
    # 协方差矩阵的 SVD
    cov = np.cov(iq)               # (2, 2)
    _, _, Vt = np.linalg.svd(cov)
    # 将 IQ 旋转到主成分坐标系（去除相关性）
    iq_corr = Vt @ iq              # (2, N)
    I_corr = iq_corr[0]
    Q_corr = iq_corr[1]

    # 重新做幅度均衡（SVD 旋转后方差可能不等）
    std_I2 = np.std(I_corr)
    std_Q2 = np.std(Q_corr)
    if std_Q2 > 1e-12 and std_I2 > 1e-12:
        Q_corr = Q_corr * (std_I2 / std_Q2)

    return I_corr, Q_corr
