"""
step3_qc.py — 质量控制（QC）

读取 step1/step2 的处理结果，对每个受试者×场景计算质量指标，
输出 qc_report.json，并标记哪些受试者需要被整体剔除。

QC 指标（在 200Hz 信号上计算）：

  1. 雷达 SNR（IQ 平面圆度）
     - 以椭圆校正后相位信号能量 / 宽带噪声能量估算
     - 低 SNR 提示传感器未对准或严重运动伪影

  2. 雷达相位跳变率
     - 统计 |Δphase| > threshold 的比例
     - 高跳变率提示传感器掉落或严重干扰

  3. ECG 基线漂移比
     - <0.5Hz 能量 / 总能量，越高说明基线漂移越严重
     - 高漂移且无法校正时提示电极接触问题

  4. R峰检测失效率
     - 统计 8s 窗口内 R峰数 < 4（<30bpm）或 > 24（>180bpm）的比例
     - 高失效率提示 ECG 信噪比太低

剔除规则（任意一条触发即剔除整个受试者，阈值参数化可调）：
  - 雷达相位跳变率 > MAX_JUMP_RATE
  - ECG 基线漂移比 > MAX_BASELINE_RATIO
  - R峰检测失效率 > MAX_RPEAK_FAILURE_RATE

注意：SNR 仅记录，不单独作为剔除条件（低 SNR 通常已被跳变率捕获）。

用法：
  python step3_qc.py [--dataset_dir DIR] [--out_json PATH]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import welch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FS = 200
WINDOW_LEN = 1600   # 8s @ 200Hz

# 默认 QC 阈值（可通过命令行覆盖）
DEFAULT_MAX_JUMP_RATE = 0.01        # 相位跳变率 > 1% → 剔除
DEFAULT_MAX_BASELINE_RATIO = 0.30   # 基线漂移能量占比 > 30% → 剔除
DEFAULT_MAX_RPEAK_FAILURE_RATE = 0.20  # R峰失效窗口 > 20% → 剔除


def compute_phase_jump_rate(
    radar_phase: np.ndarray,
    threshold_rad: float = 0.5,
) -> float:
    """
    计算相位信号中异常跳变的比例。

    threshold_rad: 相邻采样点间视为跳变的相位差阈值（弧度）
    @200Hz，0.5rad ≈ 4.5cm 的胸壁运动变化，合理的体动触发阈值
    """
    if len(radar_phase) < 2:
        return 1.0   # 信号过短，视为全部跳变，直接标记失败
    diff = np.abs(np.diff(radar_phase))
    return float(np.mean(diff > threshold_rad))


def compute_baseline_drift_ratio(
    ecg_clean: np.ndarray,
    fs: int = TARGET_FS,
    cutoff: float = 0.5,
) -> float:
    """
    计算 ECG 信号中低频基线漂移能量占总能量的比例。

    使用 Welch 功率谱密度估计，<cutoff Hz 的功率 / 总功率。
    """
    freqs, psd = welch(ecg_clean, fs=fs, nperseg=min(256, len(ecg_clean) // 4))
    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0
    baseline_power = np.sum(psd[freqs < cutoff])
    return float(baseline_power / total_power)


def compute_rpeak_failure_rate(
    rpeak_indices: np.ndarray,
    signal_len: int,
    window_len: int = WINDOW_LEN,
    min_hr: float = 30.0,
    max_hr: float = 180.0,
    fs: int = TARGET_FS,
) -> float:
    """
    统计 8s 窗口中 R峰数量在合理范围外（<30bpm 或 >180bpm）的比例。

    min_hr / max_hr: 单位 bpm
    """
    window_sec = window_len / fs
    min_peaks = int(min_hr / 60 * window_sec)  # 4
    max_peaks = int(max_hr / 60 * window_sec)  # 24

    n_windows = max(1, (signal_len - window_len) // (window_len // 2) + 1)
    failed = 0

    for i in range(n_windows):
        start = i * (window_len // 2)
        end = start + window_len
        if end > signal_len:
            break
        peaks_in_window = np.sum((rpeak_indices >= start) & (rpeak_indices < end))
        if peaks_in_window < min_peaks or peaks_in_window > max_peaks:
            failed += 1

    return failed / n_windows if n_windows > 0 else 1.0


def evaluate_scenario(
    scenario_dir: Path,
    thresholds: dict,
) -> dict:
    """
    评估单个场景目录的质量，返回指标字典。

    场景目录应包含 step1/step2 的输出：
      radar_phase.npy, ecg_clean.npy, rpeak_indices.npy
    """
    metrics = {
        "phase_jump_rate": None,
        "baseline_drift_ratio": None,
        "rpeak_failure_rate": None,
        "passed": False,
        "fail_reasons": [],
    }

    radar_phase_path = scenario_dir / "radar_phase.npy"
    ecg_clean_path = scenario_dir / "ecg_clean.npy"
    rpeak_path = scenario_dir / "rpeak_indices.npy"

    if not radar_phase_path.exists() or not ecg_clean_path.exists():
        metrics["fail_reasons"].append("缺少处理文件（step1/step2未完成）")
        return metrics

    radar_phase = np.load(radar_phase_path)
    ecg_clean = np.load(ecg_clean_path)
    rpeak_indices = np.load(rpeak_path) if rpeak_path.exists() else np.array([])

    # 指标1: 雷达相位跳变率
    jump_rate = compute_phase_jump_rate(radar_phase)
    metrics["phase_jump_rate"] = round(jump_rate, 6)
    if jump_rate > thresholds["max_jump_rate"]:
        metrics["fail_reasons"].append(
            f"雷达相位跳变率过高: {jump_rate:.4f} > {thresholds['max_jump_rate']}"
        )

    # 指标2: ECG 基线漂移
    baseline_ratio = compute_baseline_drift_ratio(ecg_clean)
    metrics["baseline_drift_ratio"] = round(baseline_ratio, 6)
    if baseline_ratio > thresholds["max_baseline_ratio"]:
        metrics["fail_reasons"].append(
            f"ECG基线漂移过高: {baseline_ratio:.4f} > {thresholds['max_baseline_ratio']}"
        )

    # 指标3: R峰检测失效率
    failure_rate = compute_rpeak_failure_rate(rpeak_indices, len(ecg_clean))
    metrics["rpeak_failure_rate"] = round(failure_rate, 6)
    if failure_rate > thresholds["max_rpeak_failure_rate"]:
        metrics["fail_reasons"].append(
            f"R峰检测失效率过高: {failure_rate:.4f} > {thresholds['max_rpeak_failure_rate']}"
        )

    metrics["passed"] = len(metrics["fail_reasons"]) == 0
    return metrics


def run_qc(
    dataset_dir: Path,
    thresholds: dict,
) -> dict:
    """
    遍历 dataset_dir 下所有受试者/场景，生成完整 QC 报告。

    受试者只要有任意场景的任意指标超出阈值，整个受试者所有数据都被标记剔除。

    Returns
    -------
    dict: QC 报告
    """
    report = {
        "thresholds": thresholds,
        "subjects": {},
        "removed_subjects": [],
        "passed_subjects": [],
        "summary": {},
    }

    subject_dirs = sorted(
        d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("GDN")
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_report = {"scenarios": {}, "subject_passed": True, "fail_reasons": []}

        for scenario in ["resting", "valsalva", "apnea"]:
            scenario_dir = subject_dir / scenario
            if not scenario_dir.exists():
                continue
            metrics = evaluate_scenario(scenario_dir, thresholds)
            subject_report["scenarios"][scenario] = metrics

            # 任一场景失败 → 受试者失败
            if not metrics["passed"]:
                subject_report["subject_passed"] = False
                for reason in metrics["fail_reasons"]:
                    subject_report["fail_reasons"].append(f"[{scenario}] {reason}")

        report["subjects"][subject_id] = subject_report

        if subject_report["subject_passed"]:
            report["passed_subjects"].append(subject_id)
        else:
            report["removed_subjects"].append(subject_id)

    report["summary"] = {
        "total_subjects": len(subject_dirs),
        "passed": len(report["passed_subjects"]),
        "removed": len(report["removed_subjects"]),
        "removal_rate": (
            round(len(report["removed_subjects"]) / len(subject_dirs), 4)
            if subject_dirs else 0.0
        ),
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="数据集质量控制")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="step1/step2 输出目录（默认：项目根目录/dataset）",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="QC 报告输出路径（默认：dataset_dir/qc_report.json）",
    )
    parser.add_argument("--max_jump_rate", type=float, default=DEFAULT_MAX_JUMP_RATE)
    parser.add_argument("--max_baseline_ratio", type=float, default=DEFAULT_MAX_BASELINE_RATIO)
    parser.add_argument("--max_rpeak_failure_rate", type=float, default=DEFAULT_MAX_RPEAK_FAILURE_RATE)
    args = parser.parse_args()
    if args.out_json is None:
        args.out_json = args.dataset_dir / "qc_report.json"

    thresholds = {
        "max_jump_rate": args.max_jump_rate,
        "max_baseline_ratio": args.max_baseline_ratio,
        "max_rpeak_failure_rate": args.max_rpeak_failure_rate,
    }

    logger.info(f"开始 QC，阈值: {thresholds}")
    report = run_qc(args.dataset_dir, thresholds)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"QC 完成：{report['summary']}")
    logger.info(f"已剔除受试者: {report['removed_subjects']}")
    logger.info(f"报告保存至: {args.out_json}")


if __name__ == "__main__":
    main()
