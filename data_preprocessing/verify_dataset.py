"""
verify_dataset.py — 数据集完整性校验

检查内容：
  1. metadata.json 存在且字段完整
  2. 每个通过QC的受试者/场景的 NPY 文件存在
  3. NPY shape 正确（N, 1, 1600），spec 为（N, 3, F, T）
  4. ECG 值域在 [0, 1]
  5. rpeak.npy 值域在 [0, 1]
  6. 输出片段数统计表（每受试者/场景）

用法：
  python verify_dataset.py [--dataset_dir DIR]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW_LEN = 1600
VALID_SCENARIOS = ["resting", "valsalva", "apnea"]
EXPECTED_FILES = ["radar_raw.npy", "radar_phase.npy",
                  "radar_spec_input.npy", "radar_spec_loss.npy",
                  "ecg.npy", "rpeak.npy"]


def verify_npy(path: Path, expected_shape_suffix: tuple, name: str) -> list[str]:
    """
    检查单个 NPY 文件。

    expected_shape_suffix: 除第0维（N）外的期望形状，如 (1, 1600)
    """
    errors = []
    if not path.exists():
        errors.append(f"文件不存在: {path.name}")
        return errors

    arr = np.load(path)
    if arr.ndim < len(expected_shape_suffix) + 1:
        errors.append(f"{name} 维度不足: {arr.shape}")
        return errors

    actual_suffix = arr.shape[1:]
    if actual_suffix != expected_shape_suffix:
        errors.append(f"{name} shape 异常: 期望 (N,{expected_shape_suffix})，实际 {arr.shape}")

    if name == "ecg":
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin < -0.01 or vmax > 1.01:
            errors.append(f"{name} 值域超出[0,1]: [{vmin:.4f}, {vmax:.4f}]")

    if name == "rpeak":
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin < -0.01 or vmax > 1.01:
            errors.append(f"{name} 值域超出[0,1]: [{vmin:.4f}, {vmax:.4f}]")

    return errors


def verify_dataset(dataset_dir: Path) -> bool:
    """
    全量校验数据集。返回 True 表示通过，False 表示有错误。
    """
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        logger.error("metadata.json 不存在，请先运行 step4_segment_save.py")
        return False

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    passed_subjects = metadata.get("final_subjects", [])
    segment_counts = metadata.get("segment_counts", {})

    logger.info(f"验证 {len(passed_subjects)} 名受试者...")

    all_errors = []
    stats_rows = []

    # 打印表头
    print(f"\n{'受试者':<12} {'Resting':>10} {'Valsalva':>10} {'Apnea':>10} {'合计':>8} {'状态':>6}")
    print("-" * 60)

    for subject_id in sorted(passed_subjects):
        subject_dir = dataset_dir / subject_id
        row_errors = []
        counts = {"resting": 0, "valsalva": 0, "apnea": 0}

        for scenario in VALID_SCENARIOS:
            scenario_dir = subject_dir / scenario / "segments"
            if not scenario_dir.exists():
                continue

            # 加载分段数
            ecg_path = scenario_dir / "ecg.npy"
            if not ecg_path.exists():
                row_errors.append(f"{scenario}/ecg.npy 不存在")
                continue
            ecg_arr = np.load(ecg_path)
            N = ecg_arr.shape[0]
            counts[scenario] = N

            # 校验各文件
            checks = [
                ("radar_raw", (1, WINDOW_LEN)),
                ("radar_phase", (1, WINDOW_LEN)),
                ("ecg", (1, WINDOW_LEN)),
                ("rpeak", (1, WINDOW_LEN)),
            ]
            for fname, expected_sfx in checks:
                errs = verify_npy(scenario_dir / f"{fname}.npy", expected_sfx, fname)
                row_errors.extend([f"{scenario}/{e}" for e in errs])

            # radar_spec_input: (N, 1, 33, T_seg)
            spec_in_path = scenario_dir / "radar_spec_input.npy"
            if spec_in_path.exists():
                s = np.load(spec_in_path)
                if s.ndim != 4 or s.shape[1] != 1 or s.shape[2] != 33:
                    row_errors.append(f"{scenario}/radar_spec_input shape 异常: {s.shape}，期望(N,1,33,T)")
            else:
                row_errors.append(f"{scenario}/radar_spec_input.npy 不存在")

            # radar_spec_loss: (N, 3, F, T_seg)
            spec_loss_path = scenario_dir / "radar_spec_loss.npy"
            if spec_loss_path.exists():
                s = np.load(spec_loss_path)
                if s.ndim != 4 or s.shape[1] != 3:
                    row_errors.append(f"{scenario}/radar_spec_loss shape 异常: {s.shape}，期望(N,3,F,T)")
            else:
                row_errors.append(f"{scenario}/radar_spec_loss.npy 不存在")

            # 一致性：各文件分段数应相同
            for fname in ["radar_raw.npy", "radar_phase.npy", "rpeak.npy",
                          "radar_spec_input.npy", "radar_spec_loss.npy"]:
                p = scenario_dir / fname
                if p.exists():
                    n = np.load(p).shape[0]
                    if n != N:
                        row_errors.append(f"{scenario}/{fname} 分段数 {n} ≠ ecg.npy {N}")

        total = sum(counts.values())
        status = "✗" if row_errors else "✓"
        print(
            f"{subject_id:<12} {counts['resting']:>10} {counts['valsalva']:>10} "
            f"{counts['apnea']:>10} {total:>8} {status:>6}"
        )
        stats_rows.append((subject_id, counts, total))
        all_errors.extend([f"[{subject_id}] {e}" for e in row_errors])

    # 汇总行
    total_resting = sum(r[1]["resting"] for r in stats_rows)
    total_valsalva = sum(r[1]["valsalva"] for r in stats_rows)
    total_apnea = sum(r[1]["apnea"] for r in stats_rows)
    grand_total = total_resting + total_valsalva + total_apnea
    print("-" * 60)
    print(
        f"{'合计':<12} {total_resting:>10} {total_valsalva:>10} "
        f"{total_apnea:>10} {grand_total:>8}"
    )

    # Fold 分布统计
    fold_assignments = metadata.get("fold_assignments", {}).get("folds", {})
    if fold_assignments:
        print(f"\n5-Fold 划分（seed={metadata.get('random_seed', 42)}）：")
        for fold_name, fold_subjects in sorted(fold_assignments.items()):
            fold_segs = sum(
                sum(segment_counts.get(s, {}).values()) for s in fold_subjects
            )
            print(f"  {fold_name}: {len(fold_subjects)} 人，约 {fold_segs} 分段")

    if all_errors:
        print(f"\n[错误] 共 {len(all_errors)} 个问题：")
        for e in all_errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\n[通过] 数据集校验完成，总分段: {grand_total}")
        return True


def main():
    parser = argparse.ArgumentParser(description="数据集完整性校验")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/home/qhh2237/Projects/BeatAware-Radar2ECGNet/dataset"),
    )
    args = parser.parse_args()

    ok = verify_dataset(args.dataset_dir)
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
