"""
plot_subject_metrics.py — 30 受试者指标柱状图

从各 fold 的 test_metrics_by_subject.csv 合并，生成：
  1. 30-subject 总览图（PCC / MAE / RMSE，按受试者排列）
  2. D5 泛化实验图（resting / valsalva / apnea 分组条形图）

用法：
  # 绘制指定实验的所有受试者指标
  python scripts/plot_subject_metrics.py --exp_tag ExpB_phase

  # 绘制 D5 泛化实验（跨场景分组）
  python scripts/plot_subject_metrics.py --exp_tag ExpD5_generalization --d5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# 数据加载：合并所有 fold 的 per-subject CSV
# =============================================================================

def load_all_subject_metrics(exp_root: Path) -> pd.DataFrame:
    """
    读取 exp_root/fold_*/results/test_metrics_by_subject.csv，合并成单个 DataFrame。

    返回带列：fold, subject, scenario, n_samples, mae, rmse, pcc, prd, rpeak_f1, ...
    """
    dfs = []
    for csv_path in sorted(exp_root.glob("fold_*/results/test_metrics_by_subject.csv")):
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARN] Cannot read {csv_path}: {e}")

    if not dfs:
        raise FileNotFoundError(
            f"No test_metrics_by_subject.csv found under {exp_root}.\n"
            "Run test.py first."
        )

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# 全受试者总览图（单场景 or 场景均值）
# =============================================================================

def plot_subject_overview(
    df:       pd.DataFrame,
    save_dir: Path,
    exp_tag:  str,
    metrics:  list[str] = ("pcc", "mae", "rmse"),
    scenario: str | None = None,
) -> None:
    """
    三行子图：每行对应一个指标（PCC/MAE/RMSE），横轴为受试者 ID，纵轴为指标值。

    scenario : 限定场景（None = 对所有场景取均值）
    """
    if scenario:
        data = df[df["scenario"] == scenario].copy()
        title_suffix = f"  [scenario: {scenario}]"
    else:
        # 对每个受试者的所有场景取均值（按 n_samples 加权）
        num_cols = [m for m in metrics if m in df.columns]
        data = df.groupby("subject")[num_cols].mean().reset_index()
        title_suffix = "  [all scenarios, avg]"

    if "subject" not in data.columns:
        print("[WARN] No subject column found, skip overview plot.")
        return

    subjects = sorted(data["subject"].unique())
    n_sub = len(subjects)

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(14, n_sub * 0.6), 3.5 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle(
        f"{exp_tag} — Per-Subject Metrics{title_suffix}\n"
        f"(n={n_sub} subjects)",
        fontsize=11, y=1.01,
    )

    for ax, metric in zip(axes, metrics):
        if metric not in data.columns:
            ax.set_visible(False)
            continue

        vals = []
        for sub in subjects:
            row = data[data["subject"] == sub]
            vals.append(float(row[metric].mean()) if len(row) > 0 else float("nan"))

        colors = ["#2980B9"] * n_sub
        # 高亮最差受试者（MAE/RMSE 最大，或 PCC 最小）
        if metric in ("mae", "rmse", "prd"):
            worst = int(np.nanargmax(vals))
        else:
            worst = int(np.nanargmin(vals))
        colors[worst] = "#E74C3C"

        ax.bar(range(n_sub), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.axhline(float(np.nanmean(vals)), color="#888", lw=1, linestyle="--",
                   label=f"Mean={np.nanmean(vals):.4f}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)

    axes[-1].set_xticks(range(n_sub))
    axes[-1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Subject ID", fontsize=9)

    plt.tight_layout()
    fname = f"subject_overview{'_' + scenario if scenario else ''}.png"
    out_path = save_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# D5 跨场景分组柱状图
# =============================================================================

def plot_d5_scenario_bars(
    df:       pd.DataFrame,
    save_dir: Path,
    exp_tag:  str,
    metrics:  list[str] = ("pcc", "mae", "rmse"),
) -> None:
    """
    D5 泛化实验专用图：
      横轴 = 受试者 ID，每个受试者有 1-3 组条形（resting/valsalva/apnea），
      纵轴 = 指标值。
    """
    scenarios = sorted(df["scenario"].unique())
    subjects  = sorted(df["subject"].unique())
    n_sub = len(subjects)
    n_sc  = len(scenarios)

    scenario_colors = {
        "resting":  "#2980B9",
        "valsalva": "#27AE60",
        "apnea":    "#E74C3C",
    }
    default_colors = ["#2980B9", "#27AE60", "#E74C3C", "#F39C12"]

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(16, n_sub * 0.8), 4.0 * len(metrics)),
                             sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle(
        f"{exp_tag} — D5 Cross-Scenario Generalization\n"
        f"Per-Subject × Scenario  (n={n_sub} subjects, {n_sc} scenarios)",
        fontsize=11, y=1.01,
    )

    bar_w = 0.8 / max(n_sc, 1)
    x = np.arange(n_sub)

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        for si, sc in enumerate(scenarios):
            vals = []
            for sub in subjects:
                row = df[(df["subject"] == sub) & (df["scenario"] == sc)]
                vals.append(float(row[metric].mean()) if len(row) > 0 else float("nan"))

            color = scenario_colors.get(sc, default_colors[si % len(default_colors)])
            offset = (si - n_sc / 2 + 0.5) * bar_w
            ax.bar(x + offset, vals, width=bar_w * 0.9,
                   color=color, label=sc, alpha=0.85, edgecolor="white", linewidth=0.4)

        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
    axes[-1].set_xlabel("Subject ID", fontsize=9)

    plt.tight_layout()
    out_path = save_dir / "subject_d5_scenarios.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-subject metrics")
    parser.add_argument("--exp_tag", required=True, help="Experiment tag (e.g. ExpB_phase)")
    parser.add_argument("--exp_dir", default="experiments", help="Experiments root dir")
    parser.add_argument("--d5", action="store_true",
                        help="D5 mode: plot grouped bars by scenario per subject")
    parser.add_argument("--scenario", default=None,
                        help="Filter by scenario (resting/valsalva/apnea). "
                             "None = average across all scenarios.")
    parser.add_argument("--metrics", nargs="+", default=["pcc", "mae", "rmse"],
                        help="Metrics to plot (default: pcc mae rmse)")
    args = parser.parse_args()

    exp_root = Path(args.exp_dir) / args.exp_tag
    if not exp_root.exists():
        print(f"Experiment directory not found: {exp_root}")
        sys.exit(1)

    save_dir = exp_root / "subject_plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading per-subject metrics from {exp_root} ...")
    df = load_all_subject_metrics(exp_root)
    print(f"  Loaded {len(df)} rows, "
          f"{df['subject'].nunique()} subjects, "
          f"{df['scenario'].nunique()} scenarios")

    if args.d5:
        print("Plotting D5 cross-scenario grouped bars ...")
        plot_d5_scenario_bars(df, save_dir, args.exp_tag, metrics=args.metrics)

    print("Plotting per-subject overview ...")
    plot_subject_overview(df, save_dir, args.exp_tag,
                          metrics=args.metrics, scenario=args.scenario)

    # 如果有多个场景，顺便也各画一张
    if not args.scenario:
        for sc in sorted(df["scenario"].unique()):
            plot_subject_overview(df, save_dir, args.exp_tag,
                                  metrics=args.metrics, scenario=sc)

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
