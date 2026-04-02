"""
step4_segment_save.py — 分段、归一化、保存 NPY + 5-Fold CV 划分

流程：
  读取 step1/step2 的处理结果（radar_raw/phase/spec + ecg_clean + rpeak_indices）
  读取 step3 的 qc_report.json（确定哪些受试者通过QC）
  → 滑窗分段（window=1600, stride=800）
  → 每段 ECG 做 min-max 归一化到 [0,1]
  → 每段生成高斯 R峰 Mask（σ=5）
  → 按受试者/场景保存为 NPY
  → 生成 metadata.json（含5-Fold CV 划分）

输出目录：
  dataset/
    GDN0001/
      resting/
        radar_raw.npy    # [N, 1, 1600], float32
        radar_phase.npy  # [N, 1, 1600], float32
        radar_spec_input.npy  # [N, 1, 33, T_seg], float32，模型输入用
        radar_spec_loss.npy   # [N, 3, F, T_seg],  float32，STFT Loss用
        ecg.npy          # [N, 1, 1600], float32, 归一化到[0,1]
        rpeak.npy        # [N, 1, 1600], float32, 高斯软标签
      valsalva/ ...
    ...
    metadata.json

用法：
  python step4_segment_save.py [--dataset_dir DIR] [--qc_report PATH]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from data_preprocessing.utils.gaussian_mask import generate_gaussian_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW_LEN = 1600    # 8s @ 200Hz
STRIDE = 800         # 50% 重叠
GAUSSIAN_SIGMA = 5.0
TARGET_FS = 200
N_FOLDS = 5
RANDOM_SEED = 42
VALID_SCENARIOS = ["resting", "valsalva", "apnea"]


def segment_signal(signal: np.ndarray) -> np.ndarray:
    """
    对一维信号做滑窗分段。

    Parameters
    ----------
    signal : ndarray, shape (L,)

    Returns
    -------
    ndarray, shape (N, WINDOW_LEN)
        N = (L - WINDOW_LEN) // STRIDE + 1
    """
    segments = []
    L = len(signal)
    start = 0
    while start + WINDOW_LEN <= L:
        segments.append(signal[start : start + WINDOW_LEN])
        start += STRIDE
    if not segments:
        return np.empty((0, WINDOW_LEN), dtype=np.float32)
    return np.stack(segments, axis=0).astype(np.float32)


def normalize_ecg_segments(ecg_segs: np.ndarray) -> np.ndarray:
    """
    Per-segment min-max 归一化，将每段 ECG 映射到 [0, 1]。

    Parameters
    ----------
    ecg_segs : ndarray, shape (N, L)

    Returns
    -------
    ndarray, shape (N, L), float32
    """
    out = np.empty_like(ecg_segs, dtype=np.float32)
    for i in range(len(ecg_segs)):
        seg = ecg_segs[i].astype(np.float64)
        vmin, vmax = seg.min(), seg.max()
        if vmax - vmin > 1e-8:
            out[i] = ((seg - vmin) / (vmax - vmin)).astype(np.float32)
        else:
            out[i] = np.zeros_like(seg, dtype=np.float32)
    return out


def generate_rpeak_segments(
    rpeak_indices: np.ndarray,
    signal_len: int,
) -> np.ndarray:
    """
    对整段信号的 R峰生成高斯Mask，再滑窗分段。

    Parameters
    ----------
    rpeak_indices : ndarray, shape (M,)
        R峰在整段信号（200Hz）中的全局索引
    signal_len : int
        信号总长度

    Returns
    -------
    ndarray, shape (N, WINDOW_LEN), float32
    """
    full_mask = generate_gaussian_mask(rpeak_indices, signal_len, sigma=GAUSSIAN_SIGMA)
    segs = segment_signal(full_mask)
    return segs


def segment_spec(spec: np.ndarray) -> np.ndarray:
    """
    对 STFT 谱图做时间轴分段，与 radar_phase 的分段对应。

    spec : ndarray, shape (3, F, T_full)
    返回: ndarray, shape (N, 3, F, T_seg)
        T_seg 由 STFT 的时间分辨率与 WINDOW_LEN 的对应关系决定
    """
    # STFT 的时间帧数与信号长度的比值
    T_full = spec.shape[2]
    # 估算帧率：T_full / signal_len（近似）
    # 用 segment_signal 相同的起始索引按比例截取
    # 实际上此处直接对 spec 的 T 轴做类似滑窗
    # 注意：STFT 的 T 维和信号的 L 维存在固定倍率关系
    # 我们在这里采用对齐原则：每个分段对应 spec 的一个时间切片
    # 这在 dataset.py 里也可以按需裁剪；这里保存整段spec，分段在dataset.py中处理
    # 返回 shape: (1, 3, F, T_full) 表示整段spec（不在此步分段）
    return spec[np.newaxis]  # (1, 3, F, T_full)


def process_scenario(
    scenario_dir: Path,
    save_dir: Path,
) -> int:
    """
    处理单个场景目录，分段后保存 NPY。

    Returns
    -------
    int: 分段数量，失败返回 -1
    """
    required_files = ["radar_raw.npy", "radar_phase.npy",
                      "radar_spec_input.npy", "radar_spec_loss.npy",
                      "ecg_clean.npy", "rpeak_indices.npy"]
    for f in required_files:
        if not (scenario_dir / f).exists():
            logger.warning(f"缺少文件 {f}，跳过: {scenario_dir}")
            return -1

    radar_raw = np.load(scenario_dir / "radar_raw.npy")
    radar_phase = np.load(scenario_dir / "radar_phase.npy")
    radar_spec_input = np.load(scenario_dir / "radar_spec_input.npy")  # (1, 33, T)
    radar_spec_loss  = np.load(scenario_dir / "radar_spec_loss.npy")   # (3, F, T)
    ecg_clean = np.load(scenario_dir / "ecg_clean.npy")
    rpeak_indices = np.load(scenario_dir / "rpeak_indices.npy")

    # 长度对齐（雷达和ECG应等长，若有微小差异取最短）
    min_len = min(len(radar_raw), len(radar_phase), len(ecg_clean))
    radar_raw = radar_raw[:min_len]
    radar_phase = radar_phase[:min_len]
    ecg_clean = ecg_clean[:min_len]
    rpeak_indices = rpeak_indices[rpeak_indices < min_len]

    # 分段
    raw_segs = segment_signal(radar_raw)       # (N, 1600)
    phase_segs = segment_signal(radar_phase)   # (N, 1600)
    ecg_segs = segment_signal(ecg_clean)       # (N, 1600)

    N = len(raw_segs)
    if N == 0:
        logger.warning(f"分段数为0，跳过: {scenario_dir}")
        return 0

    # ECG 归一化
    ecg_segs_norm = normalize_ecg_segments(ecg_segs)

    # 高斯 Mask
    rpeak_segs = generate_rpeak_segments(rpeak_indices, min_len)
    # 确保与分段数一致（rpeak_segs数量不足时为数据问题，直接报错）
    assert len(rpeak_segs) == N, (
        f"rpeak_segs 数量 {len(rpeak_segs)} ≠ raw_segs 数量 {N}，数据分段逻辑存在问题"
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "radar_raw.npy",   raw_segs[:, np.newaxis, :])    # (N,1,1600)
    np.save(save_dir / "radar_phase.npy", phase_segs[:, np.newaxis, :])  # (N,1,1600)
    np.save(save_dir / "ecg.npy",         ecg_segs_norm[:, np.newaxis, :])  # (N,1,1600)
    np.save(save_dir / "rpeak.npy",       rpeak_segs[:, np.newaxis, :])  # (N,1,1600)

    # STFT spec: 整段保存（N个分段共享同一个spec，在dataset.py按需截取）
    # 这里为了与其他NPY对齐，每个分段截取对应的时间帧
    # spec形状 (3, F, T)，T对应整段信号
    # 简单处理：将spec按时间轴分段，每段取对应时间帧
    _save_spec_segments(radar_spec_input, min_len, N, save_dir, fname="radar_spec_input.npy")
    _save_spec_segments(radar_spec_loss,  min_len, N, save_dir, fname="radar_spec_loss.npy")

    logger.info(f"  {save_dir.parent.name}/{save_dir.name}: {N}个分段")
    return N


def _save_spec_segments(
    spec: np.ndarray,
    signal_len: int,
    n_segs: int,
    save_dir: Path,
    fname: str = "radar_spec.npy",
) -> None:
    """
    将 STFT spec (3, F, T_full) 按分段切割，保存为 (N, 3, F, T_seg)。

    spec 的时间帧数 T 与信号采样点数的比值 = T / signal_len。
    每个分段的采样点范围 [start, start+WINDOW_LEN]，
    对应 spec 的帧范围 [t_start, t_start+T_seg]。
    """
    _, F, T_full = spec.shape
    t_ratio = T_full / signal_len  # 帧/采样点

    t_window = max(1, round(WINDOW_LEN * t_ratio))
    t_stride = max(1, round(STRIDE * t_ratio))

    spec_segs = []
    t_start = 0
    for _ in range(n_segs):
        t_end = t_start + t_window
        if t_end > T_full:
            # 用最后一段补齐
            seg = spec[:, :, T_full - t_window : T_full]
        else:
            seg = spec[:, :, t_start : t_end]
        spec_segs.append(seg)
        t_start += t_stride

    spec_arr = np.stack(spec_segs, axis=0)  # (N, 3, F, T_seg)
    np.save(save_dir / fname, spec_arr.astype(np.float32))


def build_fold_assignments(passed_subjects: list[str]) -> dict:
    """
    用 KFold 对通过QC的受试者按受试者划分5折。

    Returns
    -------
    dict: {"fold_0": [...], "fold_1": [...], ...}
        每折包含该折的测试集受试者列表
    """
    subjects = sorted(passed_subjects)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_assignments = {}
    indices = np.arange(len(subjects))

    for fold_idx, (_, test_idx) in enumerate(kf.split(indices)):
        fold_name = f"fold_{fold_idx}"
        fold_assignments[fold_name] = [subjects[i] for i in test_idx]

    return fold_assignments


def main():
    parser = argparse.ArgumentParser(description="分段保存 NPY + 生成 metadata.json")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset"),
    )
    parser.add_argument(
        "--qc_report",
        type=Path,
        default=Path("/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset/qc_report.json"),
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=None,
        help="step1/step2 的处理结果目录（默认与 dataset_dir 相同）",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir or args.dataset_dir

    # 读取 QC 报告，确定通过的受试者
    if not args.qc_report.exists():
        logger.error(f"QC 报告不存在: {args.qc_report}，请先运行 step3_qc.py")
        return

    with open(args.qc_report, encoding="utf-8") as f:
        qc_report = json.load(f)

    passed_subjects = qc_report.get("passed_subjects", [])
    removed_subjects = qc_report.get("removed_subjects", [])
    logger.info(f"通过QC: {len(passed_subjects)} 人，剔除: {len(removed_subjects)} 人")
    logger.info(f"剔除受试者: {removed_subjects}")

    # 分段处理
    segment_counts = {}
    for subject_id in passed_subjects:
        subject_dir = processed_dir / subject_id
        if not subject_dir.exists():
            logger.warning(f"受试者目录不存在: {subject_dir}")
            continue

        segment_counts[subject_id] = {}
        for scenario in VALID_SCENARIOS:
            scenario_dir = subject_dir / scenario
            if not scenario_dir.exists():
                continue

            save_dir = args.dataset_dir / subject_id / scenario / "segments"
            n = process_scenario(scenario_dir, save_dir)
            if n > 0:
                segment_counts[subject_id][scenario] = n

    # 5-Fold CV 划分
    fold_assignments = build_fold_assignments(passed_subjects)

    # 生成 metadata.json
    metadata = {
        "target_fs": TARGET_FS,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "total_subjects_initial": qc_report.get("summary", {}).get("total_subjects", 30),
        "qc_removed": removed_subjects,
        "final_subjects": passed_subjects,
        "fold_assignments": {
            "seed": RANDOM_SEED,
            "folds": fold_assignments,
        },
        "segment_counts": segment_counts,
        "total_segments": {
            scenario: sum(
                counts.get(scenario, 0)
                for counts in segment_counts.values()
            )
            for scenario in VALID_SCENARIOS
        },
    }

    metadata_path = args.dataset_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    total = sum(metadata["total_segments"].values())
    logger.info(f"完成！总分段数: {total}")
    logger.info(f"各场景: {metadata['total_segments']}")
    logger.info(f"metadata.json 保存至: {metadata_path}")


if __name__ == "__main__":
    main()
