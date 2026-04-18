"""
step2b_delineate.py — P/T 波位置提取（增量预处理）

在 step2 已完成 R 峰检测的基础上，对现有 ecg_clean.npy 运行
NeuroKit2 的 ecg_delineate() 提取 P/T 波峰索引，生成：
  pwave_indices.npy  # [M_p] int32，P 波峰在 200Hz 下的索引（失败时为空数组）
  twave_indices.npy  # [M_t] int32，T 波峰在 200Hz 下的索引（失败时为空数组）

delineation 成功率：
  Resting   ≈ 95%+ 正常心律，成功率高
  Valsalva  ≈ 80%（深呼吸致波形扭曲）
  Apnea     ≈ 80%（憋气致低振幅 P 波）
失败时保存空数组，step4 会生成全零 mask 并设 valid=False。

用法：
  python data_preprocessing/step2b_delineate.py --dataset_dir dataset
"""

import argparse
import logging
from pathlib import Path

import neurokit2 as nk
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FS     = 200
VALID_SCENARIOS = ["resting", "valsalva", "apnea"]


def delineate_ecg(ecg_200: np.ndarray, rpeak_indices: np.ndarray) -> dict:
    """
    对 200Hz ECG 运行 ecg_delineate() 提取 P/T 波峰位置。

    Parameters
    ----------
    ecg_200       : ndarray, shape (L,), float32  200Hz ECG 信号
    rpeak_indices : ndarray, shape (M,), int32    R 峰索引（200Hz）

    Returns
    -------
    dict:
        pwave_indices : ndarray, shape (M_p,), int32  P 波峰（空数组=失败）
        twave_indices : ndarray, shape (M_t,), int32  T 波峰（空数组=失败）
        success       : bool
    """
    if len(rpeak_indices) < 2:
        logger.warning("R 峰数量不足（<2），跳过 delineation")
        return {
            "pwave_indices": np.array([], dtype=np.int32),
            "twave_indices": np.array([], dtype=np.int32),
            "success": False,
        }

    try:
        _, waves = nk.ecg_delineate(
            ecg_200.astype(np.float64),
            rpeak_indices.astype(int),
            sampling_rate=TARGET_FS,
            method="dwt",
            show=False,
        )

        # waves 是 DataFrame，提取列并去除 NaN
        p_col = waves.get("ECG_P_Peaks", None)
        t_col = waves.get("ECG_T_Peaks", None)

        def _extract_valid_indices(col) -> np.ndarray:
            if col is None:
                return np.array([], dtype=np.int32)
            arr = np.asarray(col, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            return valid.astype(np.int32)

        p_idx = _extract_valid_indices(p_col)
        t_idx = _extract_valid_indices(t_col)

        success = len(p_idx) > 0 and len(t_idx) > 0
        return {
            "pwave_indices": p_idx,
            "twave_indices": t_idx,
            "success": success,
        }

    except Exception as e:
        logger.warning(f"ecg_delineate 失败: {e}")
        return {
            "pwave_indices": np.array([], dtype=np.int32),
            "twave_indices": np.array([], dtype=np.int32),
            "success": False,
        }


def process_scenario(scenario_dir: Path) -> bool:
    """
    对单个受试者×场景目录执行 delineation，写出 pwave/twave 索引文件。

    scenario_dir 下需已有：
        ecg_clean.npy       (step2 输出)
        rpeak_indices.npy   (step2 输出)

    写出：
        pwave_indices.npy
        twave_indices.npy

    Returns True on success, False on skip.
    """
    ecg_path   = scenario_dir / "ecg_clean.npy"
    rpeak_path = scenario_dir / "rpeak_indices.npy"

    if not (ecg_path.exists() and rpeak_path.exists()):
        logger.debug(f"step2 文件不存在，跳过: {scenario_dir}")
        return False

    # 已处理则跳过
    if (scenario_dir / "pwave_indices.npy").exists():
        logger.debug(f"已存在，跳过: {scenario_dir}")
        return True

    logger.info(f"Delineating: {scenario_dir.parent.name}/{scenario_dir.name} ...")

    ecg_200       = np.load(ecg_path)
    rpeak_indices = np.load(rpeak_path)

    result = delineate_ecg(ecg_200, rpeak_indices)

    np.save(scenario_dir / "pwave_indices.npy", result["pwave_indices"])
    np.save(scenario_dir / "twave_indices.npy", result["twave_indices"])

    status = "OK" if result["success"] else "FAILED(empty)"
    logger.info(
        f"  P波峰: {len(result['pwave_indices'])} | "
        f"T波峰: {len(result['twave_indices'])} | {status}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="P/T 波位置提取（增量）")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="step1/step2 输出目录（默认：项目根目录/dataset）",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        logger.error(f"dataset_dir 不存在: {dataset_dir}")
        return

    total, success, skipped = 0, 0, 0
    for subject_dir in sorted(dataset_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("GDN"):
            continue
        for scenario in VALID_SCENARIOS:
            scenario_dir = subject_dir / scenario
            if not scenario_dir.is_dir():
                continue
            total += 1
            ok = process_scenario(scenario_dir)
            if ok:
                success += 1
            else:
                skipped += 1

    logger.info(f"完成：处理 {success} 个场景，跳过 {skipped} 个（共 {total}）")


if __name__ == "__main__":
    main()
