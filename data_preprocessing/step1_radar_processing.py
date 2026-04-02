"""
step1_radar_processing.py — 雷达信号预处理

流程：
  I/Q 原始信号 (2000Hz)
  → 椭圆校正
  → 反正切解调 + 相位展开
  → 带通滤波 (0.5~10Hz, 4阶Butterworth)
  → 降采样至 200Hz (factor=10)
  → 两种 STFT（用途不同）

输出（每受试者每场景的处理结果）：
  radar_raw.npy         # [L_200], 椭圆校正后相位，未滤波，200Hz
  radar_phase.npy       # [L_200], 带通滤波后相位，200Hz
  radar_spec_input.npy  # [1, 33, T], 细粒度单分辨率STFT（nperseg=64），模型输入用
  radar_spec_loss.npy   # [3, F, T], 多分辨率STFT（nperseg=128/256/512），STFT Loss用

用法：
  python step1_radar_processing.py [--raw_dir DIR] [--out_dir DIR]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, decimate
from scipy.signal import stft as scipy_stft

from data_preprocessing.utils.ellipse_correction import ellipse_correction
from data_preprocessing.utils.mat_loader import load_mat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 固定参数
TARGET_FS = 200
DECIMATE_FACTOR = 10          # 2000 → 200 Hz
BANDPASS_LOW = 0.5            # Hz
BANDPASS_HIGH = 10.0          # Hz
BANDPASS_ORDER = 4

# 多分辨率 STFT 参数（@200Hz）— 用于 STFT Loss 计算
STFT_LOSS_CONFIGS = [
    {"nperseg": 128, "noverlap": 64,  "nfft": 128},   # QRS精细分辨
    {"nperseg": 256, "noverlap": 128, "nfft": 256},   # P-T波
    {"nperseg": 512, "noverlap": 256, "nfft": 512},   # RR间期节律
]

# 细粒度 STFT 参数（@200Hz）— 用于模型输入（Exp B radar_spec 表征）
# nperseg=64, noverlap=56, stride=8 → T≈196帧, F=33 bins（8s段上）
STFT_INPUT_CONFIG = {"nperseg": 64, "noverlap": 56, "nfft": 64}

VALID_SCENARIOS = {"Resting", "Valsalva", "Apnea"}


def process_radar(I: np.ndarray, Q: np.ndarray, fs: int = 2000) -> dict:
    """
    对单段 I/Q 信号执行完整雷达处理流程。

    Parameters
    ----------
    I, Q : ndarray, shape (N,)
        原始雷达 I/Q 信号，采样率 fs
    fs : int
        原始采样率（默认2000Hz）

    Returns
    -------
    dict:
        radar_raw        : ndarray, shape (N//DECIMATE_FACTOR,)   未滤波相位，200Hz
        radar_phase      : ndarray, shape (N//DECIMATE_FACTOR,)   滤波后相位，200Hz
        radar_spec_input : ndarray, shape (1, 33, T)              细粒度STFT，模型输入用
        radar_spec_loss  : ndarray, shape (3, F_min, T_min)       多分辨率STFT，Loss用
    """
    # 1. 椭圆校正
    I_c, Q_c = ellipse_correction(I, Q)

    # 2. 反正切解调 + 相位展开
    phase = np.unwrap(np.arctan2(Q_c, I_c))

    # 3. 降采样（先做抗混叠低通，再降采样）— 得到 radar_raw（未滤波）
    radar_raw = _decimate(phase, factor=DECIMATE_FACTOR, fs=fs)

    # 4. 带通滤波（在原始采样率下滤波，避免降采样引入边界效应）
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW, BANDPASS_HIGH],
                 btype="bandpass", fs=fs, output="sos")
    phase_filtered = sosfiltfilt(sos, phase)

    # 5. 降采样到 200Hz — radar_phase
    radar_phase = _decimate(phase_filtered, factor=DECIMATE_FACTOR, fs=fs)

    # 6a. 细粒度 STFT（模型输入用）
    radar_spec_input = _compute_stft_input(radar_phase, fs=TARGET_FS)

    # 6b. 多分辨率 STFT（STFT Loss 用）
    radar_spec_loss = _compute_stft_loss(radar_phase, fs=TARGET_FS)

    return {
        "radar_raw": radar_raw.astype(np.float32),
        "radar_phase": radar_phase.astype(np.float32),
        "radar_spec_input": radar_spec_input.astype(np.float32),
        "radar_spec_loss": radar_spec_loss.astype(np.float32),
    }


def _decimate(signal: np.ndarray, factor: int, fs: int) -> np.ndarray:
    """
    安全降采样：对大因子分解为多步，避免数值不稳定。
    factor=10 = 2×5，分两步执行。
    """
    # 分解因子
    steps = _factorize(factor)
    out = signal.copy()
    for step in steps:
        out = decimate(out, step, zero_phase=True)
    return out


def _factorize(n: int) -> list[int]:
    """将降采样因子分解为不超过13的质因子序列（scipy decimate推荐<=13）。"""
    factors = []
    for p in [2, 3, 5, 7, 11, 13]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    return factors if factors else [1]


def _stft_magnitude_db(signal: np.ndarray, cfg: dict, fs: int) -> np.ndarray:
    """计算单组参数的 STFT 幅度谱（dB），返回 shape (F, T)。"""
    _, _, Zxx = scipy_stft(
        signal,
        fs=fs,
        nperseg=cfg["nperseg"],
        noverlap=cfg["noverlap"],
        nfft=cfg["nfft"],
        boundary=None,
        padded=False,
    )
    return 20 * np.log10(np.abs(Zxx) + 1e-8)


def _compute_stft_input(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """
    细粒度单分辨率 STFT，用于模型输入（radar_spec 表征实验）。

    参数：nperseg=64, noverlap=56, stride=8 @200Hz
    对 1600 点段：T = (1600-64)//8 + 1 = 196，F = 33

    Returns
    -------
    ndarray, shape (1, 33, T)
    """
    mag_db = _stft_magnitude_db(signal, STFT_INPUT_CONFIG, fs)
    return mag_db[np.newaxis]  # (1, F, T)


def _compute_stft_loss(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """
    多分辨率 STFT，用于 STFT Loss 计算（3种表征实验统一使用）。

    Returns
    -------
    ndarray, shape (3, F_min, T_min)
        3个分辨率的幅度谱（dB），频率和时间维度对齐到最小公约尺寸。
    """
    specs = [_stft_magnitude_db(signal, cfg, fs) for cfg in STFT_LOSS_CONFIGS]

    min_T = min(s.shape[1] for s in specs)
    min_F = min(s.shape[0] for s in specs)
    specs_aligned = [s[:min_F, :min_T] for s in specs]

    return np.stack(specs_aligned, axis=0)  # (3, F, T)


def process_subject_scenario(
    mat_path: Path,
    out_dir: Path,
) -> bool:
    """
    处理单个受试者的单个场景，保存到 out_dir/<subject>/<scenario>/。

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
        return True  # 不是错误，只是不处理

    save_dir = out_dir / subject / scenario.lower()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已处理
    if (save_dir / "radar_raw.npy").exists():
        logger.info(f"已存在，跳过: {save_dir}")
        return True

    logger.info(f"处理: {subject} / {scenario} ...")

    result = process_radar(data["radar_i"], data["radar_q"], fs=data["fs_radar"])

    np.save(save_dir / "radar_raw.npy",        result["radar_raw"])
    np.save(save_dir / "radar_phase.npy",      result["radar_phase"])
    np.save(save_dir / "radar_spec_input.npy", result["radar_spec_input"])
    np.save(save_dir / "radar_spec_loss.npy",  result["radar_spec_loss"])

    logger.info(
        f"  已保存: radar_raw{result['radar_raw'].shape}, "
        f"radar_phase{result['radar_phase'].shape}, "
        f"radar_spec_input{result['radar_spec_input'].shape}, "
        f"radar_spec_loss{result['radar_spec_loss'].shape}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="雷达信号预处理")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("/home/qhh2237/Datasets/Med_Radar"),
        help="原始 .mat 数据集根目录",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset"),
        help="输出目录",
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
