"""
mat_loader.py — 加载单个 .mat 文件并返回标准化字段字典

实际数据集字段（已探索确认）：
  - radar_i, radar_q : float64, shape (N, 1), 2000 Hz
  - tfm_ecg2         : float64, shape (N, 1), 2000 Hz（与雷达等长，天然对齐）
  - fs_radar         : uint16, scalar, 2000
  - fs_ecg           : uint16, scalar, 2000
  - measurement_info : 包含日期/场景/受试者ID
"""

from pathlib import Path

import h5py
import numpy as np
import scipy.io


def load_mat(mat_path: str | Path) -> dict:
    """
    加载单个 .mat 文件，返回标准化字段字典。

    自动检测 .mat 格式版本（v5 用 scipy，v7.3 用 h5py）。

    Parameters
    ----------
    mat_path : str or Path
        .mat 文件路径

    Returns
    -------
    dict with keys:
        radar_i   : ndarray, shape (N,), float64, 2000 Hz
        radar_q   : ndarray, shape (N,), float64, 2000 Hz
        tfm_ecg2  : ndarray, shape (N,), float64, 2000 Hz
        fs_radar  : int, 雷达采样率（通常 2000）
        fs_ecg    : int, ECG 采样率（通常 2000）
        subject   : str, 如 "GDN0001"
        scenario  : str, 如 "Resting"

    Raises
    ------
    ValueError
        若必要字段缺失
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"文件不存在: {mat_path}")

    data = _load_raw(mat_path)
    return _parse(data, mat_path)


def _load_raw(mat_path: Path) -> dict:
    """尝试 scipy（v5/v6），失败则用 h5py（v7.3）。"""
    try:
        return scipy.io.loadmat(str(mat_path))
    except Exception:
        pass

    # v7.3（HDF5格式）
    result = {}
    with h5py.File(str(mat_path), "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                result[key] = obj[()]
    return result


def _parse(data: dict, mat_path: Path) -> dict:
    """从原始字典中提取并标准化所需字段。"""
    required = {"radar_i", "radar_q", "tfm_ecg2"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"{mat_path.name} 缺少字段: {missing}")

    def flatten(arr) -> np.ndarray:
        """将 (N,1) 或 (1,N) 降为 (N,) 的 float64。"""
        arr = np.asarray(arr, dtype=np.float64).squeeze()
        if arr.ndim != 1:
            raise ValueError(f"字段形状异常，squeeze后仍为 {arr.shape}")
        return arr

    fs_radar = int(data["fs_radar"].flat[0]) if "fs_radar" in data else 2000
    fs_ecg = int(data["fs_ecg"].flat[0]) if "fs_ecg" in data else 2000

    # 从 measurement_info 解析受试者和场景（若存在）
    subject, scenario = _parse_measurement_info(data, mat_path)

    return {
        "radar_i": flatten(data["radar_i"]),
        "radar_q": flatten(data["radar_q"]),
        "tfm_ecg2": flatten(data["tfm_ecg2"]),
        "fs_radar": fs_radar,
        "fs_ecg": fs_ecg,
        "subject": subject,
        "scenario": scenario,
    }


def _parse_measurement_info(data: dict, mat_path: Path) -> tuple[str, str]:
    """
    解析受试者ID和场景名。

    受试者ID：**始终从文件名解析**（measurement_info 中存在拼写错误，不可信）
    场景名：优先从 measurement_info 读取，失败时从文件名解析

    文件名格式：GDN0001_1_Resting.mat
    """
    # 受试者ID 始终从文件名提取（measurement_info 不可信，存在如 GND0006、GDN0025_no02 等脏数据）
    stem = mat_path.stem   # "GDN0001_1_Resting"
    parts = stem.split("_")
    subject = parts[0]     # "GDN0001"

    # 场景名：优先从 measurement_info 读取
    scenario = None
    if "measurement_info" in data:
        try:
            info = data["measurement_info"]
            info_flat = info.flatten()
            scenario_raw = str(info_flat[1].flat[0])
            # 只接受合法场景名（防止字段内容异常）
            if scenario_raw in {"Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"}:
                scenario = scenario_raw
        except Exception:
            pass

    # 从文件名回退
    if scenario is None:
        scenario = parts[-1] if len(parts) >= 3 else "Unknown"

    return subject, scenario
