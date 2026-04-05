"""
summarize_ablation.py — 消融实验横向汇总

读取 experiments/ 下各实验的 test_summary.csv + config.json，
拼成论文 Table 2 格式的对比表，并输出 bar chart。

用法：
  # 自动发现 experiments/ 下所有已完成测试的实验
  python scripts/summarize_ablation.py

  # 只汇总指定实验
  python scripts/summarize_ablation.py --exps ExpA ExpB_raw ExpB_phase ExpB_spec

  # 指定排序指标（默认 mae）
  python scripts/summarize_ablation.py --sort_by pcc --ascending False

输出：
  experiments/ablation_summary.csv   ← 机器可读汇总表
  experiments/ablation_summary.png   ← 各指标 bar chart
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 实验计划（用于排序和分组，顺序即论文表格顺序）
# 若未在此列表中则追加到末尾
# =============================================================================

PLANNED_ORDER = [
    "ExpA",
    "ExpB_raw", "ExpB_phase", "ExpB_spec",
    "ExpC",
    "ExpD1", "ExpD2", "ExpD3", "ExpD4", "ExpD5",
]

# 在表格中展示的配置字段（从 config.json 读取）
CONFIG_FIELDS = ["input_type", "use_pam", "alpha", "beta"]

# 展示的指标字段（从 test_summary.csv 的 mean 行读取）
METRIC_FIELDS = [
    "mae", "rmse", "pcc", "prd", "rpeak_f1",
    "dtw", "rr_interval_mae", "qrs_width_mae", "qt_interval_mae", "pr_interval_mae",
]

# 指标越小越好（用于 bar chart 颜色标注）
LOWER_IS_BETTER = {"mae", "rmse", "prd"}


# =============================================================================
# 工具函数
# =============================================================================

def load_mean_metrics(exp_dir: Path) -> dict | None:
    """
    从 test_summary.csv 读取 mean 行；
    若不存在则尝试 test_summary.json。
    返回 None 表示该实验尚未完成测试。
    """
    csv_path  = exp_dir / "test_summary.csv"
    json_path = exp_dir / "test_summary.json"

    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("fold", "")).strip().lower() == "mean":
                    return {k: _try_float(v) for k, v in row.items() if k != "fold"}
        return None

    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        mean = data.get("mean", {})
        return {k: _try_float(v) for k, v in mean.items() if k != "fold"} or None

    return None


def load_config(exp_dir: Path) -> dict:
    """读取实验配置；若不存在返回空字典。"""
    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _try_float(v) -> float | str:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _fmt(v, field: str) -> str:
    if isinstance(v, float):
        if field in ("pcc", "rpeak_f1"):
            return f"{v:.4f}"
        if field in ("prd",):
            return f"{v:.2f}"
        return f"{v:.4f}"
    return str(v)


def _sort_key(tag: str) -> int:
    try:
        return PLANNED_ORDER.index(tag)
    except ValueError:
        return len(PLANNED_ORDER)


# =============================================================================
# 主逻辑
# =============================================================================

def collect_results(exp_dir_root: Path, requested: list[str] | None) -> list[dict]:
    """
    扫描实验目录，返回已完成测试的实验结果列表，每项包含：
        tag, config_fields..., metric_fields...
    """
    if requested:
        candidates = [exp_dir_root / tag for tag in requested]
    else:
        candidates = sorted(
            [p for p in exp_dir_root.iterdir() if p.is_dir()],
            key=lambda p: _sort_key(p.name),
        )

    rows = []
    for exp_dir in candidates:
        if not exp_dir.is_dir():
            print(f"[WARN] 目录不存在，跳过: {exp_dir}")
            continue

        metrics = load_mean_metrics(exp_dir)
        if metrics is None:
            print(f"[SKIP] {exp_dir.name}：尚未完成测试（无 test_summary.csv）")
            continue

        cfg = load_config(exp_dir)

        row = {"exp_tag": exp_dir.name}
        for f in CONFIG_FIELDS:
            row[f] = cfg.get(f, "—")
        for f in METRIC_FIELDS:
            row[f] = metrics.get(f, float("nan"))

        rows.append(row)

    return rows


def print_table(rows: list[dict]) -> None:
    """在终端打印格式化的对比表。"""
    if not rows:
        print("没有可汇总的实验结果。")
        return

    # 列宽
    tag_w   = max(len(r["exp_tag"]) for r in rows) + 2
    cfg_w   = {f: max(len(f), max(len(str(r[f])) for r in rows)) + 2 for f in CONFIG_FIELDS}
    met_w   = {f: max(len(f), 8) + 2 for f in METRIC_FIELDS}

    header = (
        f"{'Exp':<{tag_w}}"
        + "".join(f"{f:<{cfg_w[f]}}" for f in CONFIG_FIELDS)
        + "".join(f"{f:>{met_w[f]}}" for f in METRIC_FIELDS)
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  消融实验汇总")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        line = (
            f"{r['exp_tag']:<{tag_w}}"
            + "".join(f"{str(r[f]):<{cfg_w[f]}}" for f in CONFIG_FIELDS)
            + "".join(f"{_fmt(r[f], f):>{met_w[f]}}" for f in METRIC_FIELDS)
        )
        print(line)

    print(sep)
    print()


def save_csv(rows: list[dict], out_path: Path) -> None:
    """保存汇总 CSV。"""
    if not rows:
        return
    fieldnames = ["exp_tag"] + CONFIG_FIELDS + METRIC_FIELDS
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"汇总 CSV 已保存: {out_path}")


def save_chart(rows: list[dict], out_path: Path) -> None:
    """
    绘制各指标 bar chart，每个指标一个子图。
    最优值（最小或最大）用深色高亮，其余用浅色。
    """
    if len(rows) < 1:
        return

    n_metrics = len(METRIC_FIELDS)
    tags = [r["exp_tag"] for r in rows]
    x = np.arange(len(tags))

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle("Ablation Study — Metric Comparison", fontsize=13, y=1.01)

    for ax, field in zip(axes, METRIC_FIELDS):
        vals = [r[field] if isinstance(r[field], float) else float("nan") for r in rows]
        valid = [v for v in vals if not np.isnan(v)]

        if not valid:
            ax.set_title(field)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        best = min(valid) if field in LOWER_IS_BETTER else max(valid)
        colors = ["#2C7BB6" if (not np.isnan(v) and v == best) else "#ABD9E9"
                  for v in vals]

        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)

        # 数值标注
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                label = f"{v:.3f}" if field != "prd" else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(valid) * 0.01,
                        label, ha="center", va="bottom", fontsize=7.5)

        ax.set_title(field.upper(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.18)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 方向注释
        direction = "↓ lower is better" if field in LOWER_IS_BETTER else "↑ higher is better"
        ax.set_xlabel(direction, fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart 已保存: {out_path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总消融实验结果，生成对比表与 bar chart"
    )
    parser.add_argument(
        "--exps", nargs="*", default=None,
        help="指定汇总的 exp_tag 列表（默认自动发现所有已完成实验）"
    )
    parser.add_argument(
        "--exp_dir", type=str, default="experiments",
        help="实验根目录（默认 experiments/）"
    )
    parser.add_argument(
        "--sort_by", type=str, default="mae",
        choices=METRIC_FIELDS,
        help="按哪个指标排序（默认 mae）"
    )
    parser.add_argument(
        "--ascending", type=lambda x: x.lower() != "false", default=True,
        help="升序排列（默认 True；若 sort_by=pcc 建议设为 False）"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    exp_dir_root = project_root / args.exp_dir

    if not exp_dir_root.exists():
        print(f"[ERROR] 实验目录不存在: {exp_dir_root}")
        sys.exit(1)

    rows = collect_results(exp_dir_root, args.exps)

    if not rows:
        print("没有找到任何已完成测试的实验。请先运行 scripts/test.py。")
        sys.exit(0)

    # 排序
    rows.sort(
        key=lambda r: r[args.sort_by] if isinstance(r[args.sort_by], float) else float("inf"),
        reverse=not args.ascending,
    )

    print_table(rows)

    out_csv   = exp_dir_root / "ablation_summary.csv"
    out_chart = exp_dir_root / "ablation_summary.png"
    save_csv(rows, out_csv)
    save_chart(rows, out_chart)


if __name__ == "__main__":
    main()
