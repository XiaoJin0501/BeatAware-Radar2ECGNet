"""
plot_paper_figures.py — 生成论文所需全部图表

输出目录：experiments/paper_figures/
  Fig1_ablation_bar.pdf/png      — 消融实验对比 bar chart
  Fig2_waveform_qual.pdf/png     — 波形质量可视化（好/差受试者对比）
  Fig3_subject_scatter.pdf/png   — 每受试者 PCC scatter plot
  Fig4_scenario_radar.pdf/png    — 三场景雷达图对比

用法：
  python scripts/plot_paper_figures.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 全局风格 ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

EXP_DIR   = Path("experiments")
OUT_DIR   = EXP_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 颜色方案
C_A = "#95A5A6"   # baseline     — 灰
C_B = "#58D68D"   # +PAM only    — 绿
C_C = "#5DADE2"   # +EMD only    — 蓝
C_D = "#E74C3C"   # Full (+PAM+EMD) — 红
C_GT   = "#2980B9"
C_PRED = "#E74C3C"


# =============================================================================
# 工具函数
# =============================================================================

def read_test_summary_mean(exp_tag: str) -> dict:
    p = EXP_DIR / exp_tag / "test_summary.csv"
    if not p.exists():
        return {}
    with open(p) as f:
        for row in csv.DictReader(f):
            if str(row.get("fold", "")).strip().lower() == "mean":
                return {k: float(v) for k, v in row.items()
                        if k != "fold" and v not in ("", "nan")}
    return {}


def read_subject_metrics(exp_tag: str) -> list[dict]:
    rows = []
    for fold in range(5):
        p = EXP_DIR / exp_tag / f"fold_{fold}" / "results" / "test_metrics_by_subject.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def read_scenario_metrics(exp_tag: str) -> dict[str, list]:
    """返回 {scenario: [pcc_fold0, pcc_fold1, ...]}"""
    result = {"resting": [], "valsalva": [], "apnea": []}
    for fold in range(5):
        p = EXP_DIR / exp_tag / f"fold_{fold}" / "results" / "test_metrics_by_scenario.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                sc = row["scenario"]
                if sc in result:
                    result[sc].append(float(row["pcc"]))
    return result


# =============================================================================
# Fig 1 — 消融实验 Bar Chart
# =============================================================================

def plot_ablation_bar():
    models = [
        ("ModelA_baseline",  "Model A\n(Baseline)",  C_A),
        ("ModelB_pam_only",  "Model B\n(+PAM)",       C_B),
        ("ModelC_ki_pa",     "Model C\n(+EMD)",       C_C),
        ("ModelD_full",      "Model D\n(Full)",        C_D),
    ]
    metrics = [
        ("pcc",      "PCC ↑",    True),
        ("mae",      "MAE ↓",    False),
        ("rpeak_f1", "F1 ↑",     True),
        ("prd",      "PRD (%) ↓",False),
    ]

    data = {}
    for tag, label, color in models:
        m = read_test_summary_mean(tag)
        data[label] = (m, color)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    fig.suptitle("Ablation Study — 5-Fold Mean Test Metrics", fontsize=11, y=1.02)

    x  = np.arange(4)
    w  = 0.55
    labels = [label for _, label, _ in models]

    for ax, (key, ylabel, higher_better) in zip(axes, metrics):
        vals   = [data[l][0].get(key, float("nan")) for l in labels]
        colors = [data[l][1] for l in labels]

        valid  = [v for v in vals if not np.isnan(v)]
        best   = max(valid) if higher_better else min(valid)

        bars = ax.bar(x, vals, width=w, color=colors,
                      edgecolor="white", linewidth=0.6)

        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            fmt   = f"{v:.3f}" if key != "prd" else f"{v:.1f}"
            weight = "bold" if v == best else "normal"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(valid) * 0.015,
                    fmt, ha="center", va="bottom",
                    fontsize=8, fontweight=weight)

        # 星号标注最优
        best_idx = vals.index(best)
        ax.text(x[best_idx], best + max(valid) * 0.06,
                "★", ha="center", va="bottom", fontsize=10, color="#F39C12")

        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["A", "B", "C", "D"], fontsize=9)
        ax.set_ylim(0, max(valid) * 1.22)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # 图例
    patches = [mpatches.Patch(color=c, label=l.replace("\n", " "))
               for _, l, c in models]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig1_ablation_bar.{ext}")
    plt.close(fig)
    print(f"[Fig1] Saved: {OUT_DIR}/Fig1_ablation_bar.{{png,pdf}}")


# =============================================================================
# Fig 2 — 波形质量可视化
# =============================================================================

def plot_waveform_qual():
    """
    从 fold_0（好受试者）和 fold_3（差受试者）各选 3 个代表性样本，
    展示 GT vs Pred 波形和功率谱。
    """
    from src.data.dataset import RadarECGDataset
    from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
    from src.utils.metrics import compute_waveform_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_fold(fold_idx, n_samples=4):
        ckpt_path = EXP_DIR / "ModelD_full" / f"fold_{fold_idx}" / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            return None, None, None
        ckpt = torch.load(ckpt_path, map_location=device)
        model = BeatAwareRadar2ECGNet(
            input_type="phase", C=64, d_state=16,
            dropout=0.0, use_pam=True, use_emd=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ds = RadarECGDataset(
            "dataset", fold_idx=fold_idx, split="val",
            input_type="phase", scenarios=["resting", "valsalva", "apnea"],
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=n_samples, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        with torch.no_grad():
            pred, _ = model(batch["radar"].to(device))
        return (batch["ecg"].squeeze(1).cpu().numpy(),
                pred.squeeze(1).cpu().numpy(),
                batch.get("scenario", ["unknown"] * n_samples))

    # ── 布局：2 行（fold_0 / fold_3），每行 4 列 ────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    fig.suptitle("ECG Reconstruction Quality: Model D (Full)",
                 fontsize=12, y=1.01)

    t = np.arange(1600) / 200.0   # 时间轴（秒）

    row_info = [
        (0, "Fold 0 — High-quality subjects"),
        (3, "Fold 3 — Challenging subjects"),
    ]

    for row_idx, (fold, row_title) in enumerate(row_info):
        gt, pred, scenarios = load_fold(fold, n_samples=4)
        if gt is None:
            continue

        axes[row_idx, 0].set_ylabel(row_title, fontsize=9, labelpad=6)

        # 计算 per-sample PCC
        gt_t   = torch.tensor(gt).unsqueeze(1)
        pred_t = torch.tensor(pred).unsqueeze(1)
        m = compute_waveform_metrics(pred_t, gt_t)

        for col in range(4):
            ax = axes[row_idx, col]
            ax.plot(t, gt[col],   color=C_GT,   lw=0.9, alpha=0.9, label="GT")
            ax.plot(t, pred[col], color=C_PRED,  lw=0.9, alpha=0.85, label="Pred",
                    linestyle="--")
            ax.set_xlim(0, 8)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.tick_params(labelsize=7)
            sc_label = scenarios[col] if isinstance(scenarios, list) else "—"
            ax.set_title(f"Sample {col+1} ({sc_label})", fontsize=8.5)
            ax.grid(True, alpha=0.2, linewidth=0.4)

    # 图例
    leg_lines = [
        plt.Line2D([0], [0], color=C_GT,   lw=1.2, label="GT ECG"),
        plt.Line2D([0], [0], color=C_PRED, lw=1.2, linestyle="--", label="Predicted"),
    ]
    fig.legend(handles=leg_lines, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig2_waveform_qual.{ext}")
    plt.close(fig)
    print(f"[Fig2] Saved: {OUT_DIR}/Fig2_waveform_qual.{{png,pdf}}")


# =============================================================================
# Fig 3 — Per-subject PCC Scatter Plot
# =============================================================================

def plot_subject_scatter():
    rows = read_subject_metrics("ModelD_full")

    # 按 subject 聚合（取各场景均值）
    subject_data = {}
    for r in rows:
        subj = r["subject"]
        pcc  = float(r["pcc"]) if r["pcc"] not in ("", "nan") else np.nan
        mae  = float(r["mae"]) if r["mae"] not in ("", "nan") else np.nan
        if subj not in subject_data:
            subject_data[subj] = {"pcc": [], "mae": [], "scenarios": []}
        if not np.isnan(pcc):
            subject_data[subj]["pcc"].append(pcc)
            subject_data[subj]["mae"].append(mae)
            subject_data[subj]["scenarios"].append(r["scenario"])

    subjects = sorted(subject_data.keys())
    mean_pcc = [np.mean(subject_data[s]["pcc"]) for s in subjects]
    mean_mae = [np.mean(subject_data[s]["mae"]) for s in subjects]

    # 颜色：GDN0022 红色标注
    bar_colors = ["#E74C3C" if s == "GDN0022" else "#5DADE2" for s in subjects]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                             gridspec_kw={"hspace": 0.5})
    fig.suptitle("Per-Subject Performance — Model D (Full), 5-Fold CV",
                 fontsize=11, y=1.01)

    x = np.arange(len(subjects))

    # PCC
    ax = axes[0]
    bars = ax.bar(x, mean_pcc, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.nanmean(mean_pcc), color="#E67E22", lw=1.2,
               linestyle="--", label=f"Mean PCC = {np.nanmean(mean_pcc):.3f}")
    ax.set_ylabel("PCC ↑", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("GDN", "") for s in subjects],
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, frameon=False)
    # 标注 GDN0022
    idx22 = subjects.index("GDN0022") if "GDN0022" in subjects else -1
    if idx22 >= 0:
        ax.annotate("GDN0022\n(outlier)", xy=(idx22, mean_pcc[idx22]),
                    xytext=(idx22 + 1.5, mean_pcc[idx22] + 0.08),
                    fontsize=7.5, color="#E74C3C",
                    arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=0.8))

    # MAE
    ax = axes[1]
    ax.bar(x, mean_mae, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axhline(np.nanmean(mean_mae), color="#E67E22", lw=1.2,
               linestyle="--", label=f"Mean MAE = {np.nanmean(mean_mae):.3f}")
    ax.set_ylabel("MAE ↓", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("GDN", "") for s in subjects],
                       rotation=45, ha="right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, frameon=False)

    # 图例
    patches = [
        mpatches.Patch(color="#5DADE2", label="Normal subject"),
        mpatches.Patch(color="#E74C3C", label="Low-quality signal (GDN0022)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_subject_scatter.{ext}")
    plt.close(fig)
    print(f"[Fig3] Saved: {OUT_DIR}/Fig3_subject_scatter.{{png,pdf}}")


# =============================================================================
# Fig 4 — 三场景 PCC 对比（radar chart + bar）
# =============================================================================

def plot_scenario_comparison():
    exps = [
        ("ModelA_baseline", "Model A", C_A),
        ("ModelB_pam_only", "Model B", C_B),
        ("ModelC_ki_pa",    "Model C", C_C),
        ("ModelD_full",     "Model D", C_D),
    ]
    scenarios = ["resting", "valsalva", "apnea"]
    sc_labels  = ["Resting", "Valsalva", "Apnea"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4),
                             gridspec_kw={"wspace": 0.35})
    fig.suptitle("Scenario-wise PCC Comparison (5-Fold Mean)", fontsize=11, y=1.01)

    # ── 左图：grouped bar ───────────────────────────────────────────────────
    ax = axes[0]
    n_sc   = len(scenarios)
    n_mod  = len(exps)
    width  = 0.22
    x      = np.arange(n_sc)

    for i, (tag, label, color) in enumerate(exps):
        sc_data = read_scenario_metrics(tag)
        means = [np.mean(sc_data[sc]) if sc_data[sc] else np.nan
                 for sc in scenarios]
        offset = (i - n_mod / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width=width, label=label,
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sc_labels, fontsize=9)
    ax.set_ylabel("PCC ↑", fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_title("Grouped Bar Chart", fontsize=9)

    # ── 右图：折线图（Resting / Valsalva / Apnea 逐场景趋势）──────────────
    ax = axes[1]
    for tag, label, color in exps:
        sc_data = read_scenario_metrics(tag)
        means = [np.mean(sc_data[sc]) if sc_data[sc] else np.nan
                 for sc in scenarios]
        ax.plot(sc_labels, means, marker="o", color=color,
                label=label, lw=1.8, markersize=6)
        for xi, v in enumerate(means):
            if not np.isnan(v):
                ax.text(xi, v + 0.012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, color=color)

    ax.set_ylabel("PCC ↑", fontsize=10)
    ax.set_ylim(0.4, 0.85)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_title("Trend across Scenarios", fontsize=9)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig4_scenario_comparison.{ext}")
    plt.close(fig)
    print(f"[Fig4] Saved: {OUT_DIR}/Fig4_scenario_comparison.{{png,pdf}}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output: {OUT_DIR.resolve()}\n")

    print("→ Fig 1: Ablation bar chart")
    plot_ablation_bar()

    print("→ Fig 2: Waveform quality visualization")
    try:
        plot_waveform_qual()
    except Exception as e:
        print(f"  [WARN] Fig2 skipped: {e}")

    print("→ Fig 3: Per-subject scatter plot")
    plot_subject_scatter()

    print("→ Fig 4: Scenario comparison")
    plot_scenario_comparison()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
