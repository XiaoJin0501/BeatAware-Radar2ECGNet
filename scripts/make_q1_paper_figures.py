"""make_q1_paper_figures.py — publication-quality figures for the Q1 paper (G2).

Figures:
  F1  T1 MMECG morphology comparison (Shape-200 PCC by method, tiered).
  F2  T3 CR_RVS external validation (Block A local vs Block B reported-only).
  F3  Multi-seed stability (per-fold Shape-200 PCC, 3 seeds, MMECG + CR_RVS).
  F4  Route-C mechanism (oracle vs predicted-anchor gap; 3-fold generalization).
  F5  Representative ECG beat-morphology examples (pred vs GT).

Numbers for F1/F2/F4 are the locked, audited values (see docs/T1_*, T3_*,
ASSEMBLY_DESIGN). F3/F5 read the committed G1 outputs / a G1 checkpoint.
Outputs: experiments_mmecg/q1_paper_figures/ (regenerable; script is the artifact).
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments_mmecg" / "q1_paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})

TIER_COLOR = {"proposed": "#1f77b4", "adapted": "#ff7f0e", "baseline": "#7f7f7f",
              "reported": "#2ca02c", "prior": "#9467bd"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def f1_mmecg_morphology():
    # method, PCC, err(±), tier  (locked: T1 Block A; proposed multi-seed)
    rows = [
        ("REM-Net (proposed)", 0.885, 0.002, "proposed"),
        ("radarODE-MTL (adapted)", 0.728, 0.133, "adapted"),
        ("REM-Event (ablation)", 0.279, 0.151, "baseline"),
        ("WSST-TCN (repr.)", 0.273, 0.151, "baseline"),
        ("RSSRNet (adapted)", 0.262, 0.107, "adapted"),
        ("AirECG (adapted)", 0.180, 0.062, "adapted"),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows],
            color=[TIER_COLOR[r[3]] for r in rows], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Shape-200 PCC (11-fold strict LOSO, fold-macro)")
    ax.set_title("MMECG beat-morphology reconstruction (T1)")
    ax.set_xlim(0, 1.0)
    for yi, r in zip(y, rows):
        ax.text(r[1] + r[2] + 0.01, yi, f"{r[1]:.3f}", va="center", fontsize=9)
    ax.text(0.99, 0.02, "BeatAware (LOSO+calib) shown separately (Block B)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, style="italic")
    _save(fig, "F1_mmecg_morphology")


def f2_crrvs_external():
    # Block A local (subject-independent) + Block B reported-only (diff protocol)
    methods = ["REM-Net\n(proposed)", "CW-STFT+TCN\n(local)", "VMD-CBNeT\n(reported)", "CNN+LSTM\n(reported)"]
    pcc = [0.886, 0.550, 0.856, 0.647]; pcc_err = [0.001, 0.075, 0, 0]
    rmse = [0.104, 0.174, 0.112, 0.127]
    colors = [TIER_COLOR["proposed"], TIER_COLOR["baseline"], TIER_COLOR["reported"], TIER_COLOR["reported"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    x = np.arange(len(methods))
    a1.bar(x, pcc, yerr=pcc_err, color=colors, capsize=3)
    a1.set_xticks(x); a1.set_xticklabels(methods, fontsize=8); a1.set_ylabel("PCC"); a1.set_ylim(0, 1)
    a1.set_title("CR_RVS PCC"); a1.axvline(1.5, ls="--", c="k", alpha=0.4)
    for xi, v in zip(x, pcc): a1.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    a2.bar(x, rmse, color=colors)
    a2.set_xticks(x); a2.set_xticklabels(methods, fontsize=8); a2.set_ylabel("RMSE"); a2.set_title("CR_RVS RMSE")
    a2.axvline(1.5, ls="--", c="k", alpha=0.4)
    fig.suptitle("CR_RVS external validation (T3): Block A local 5-fold | Block B reported-only (diff. protocol)",
                 fontsize=10)
    fig.text(0.5, 0.005, "Left of dashed line: our subject-independent 5-fold. Right: reported-only, different protocol (not same-protocol comparison).",
             ha="center", fontsize=7.5, style="italic")
    _save(fig, "F2_crrvs_external")


def _perfold(tag, folds):
    out = []
    for f in folds:
        c = glob.glob(str(ROOT / f"experiments_mmecg/{tag}/fold_{f:02d}/results_shape_head/*.json"))
        if c:
            j = json.load(open(c[0])); v = j.get("pcc_mean", j.get("pcc"))
            if isinstance(v, (int, float)): out.append((f, v))
    return out


def f3_multiseed_stability():
    seeds = [42, 123, 2024]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, ds, folds, tagf in [(a1, "MMECG", range(1, 12), "mmecg"), (a2, "CR_RVS", range(1, 6), "crrvs")]:
        allmeans = []
        for i, s in enumerate(seeds):
            pts = _perfold(f"q1_g1_remnet_{tagf}_seed{s}", folds)
            if not pts: continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.scatter(xs, ys, label=f"seed {s}", alpha=0.7)
            allmeans.append(np.mean(ys))
        if allmeans:
            m = np.mean(allmeans); sd = np.std(allmeans, ddof=1)
            ax.axhline(m, ls="--", c="k", alpha=0.6)
            ax.text(0.98, 0.03, f"mean {m:.3f} ± {sd:.3f} (across seeds)",
                    transform=ax.transAxes, ha="right", fontsize=9)
        ax.set_title(f"{ds} Shape-200 per-fold PCC"); ax.set_xlabel("fold"); ax.set_ylabel("PCC")
        ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8)
    fig.suptitle("Multi-seed stability (G1): tight across-seed clustering", fontsize=11)
    _save(fig, "F3_multiseed_stability")


def f4_route_c_mechanism():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    # Panel A: oracle vs predicted-anchor gap (fold01)
    labels = ["dense\nfull-window", "predicted-anchor\n(WSST)", "predicted-anchor\n(rcg)",
              "predicted-anchor\n(fusion)", "oracle-anchor"]
    vals = [0.39, 0.29, 0.42, 0.51, 0.96]
    cols = ["#7f7f7f", "#ff7f0e", "#ff7f0e", "#1f77b4", "#2ca02c"]
    a1.bar(np.arange(len(labels)), vals, color=cols)
    a1.set_xticks(np.arange(len(labels))); a1.set_xticklabels(labels, fontsize=7.5)
    a1.set_ylabel("continuous OLA PCC (fold-01)"); a1.set_ylim(0, 1.0)
    a1.set_title("Continuous gap is anchor localization, not morphology")
    for i, v in enumerate(vals): a1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    # Panel B: 3-fold generalization (fusion predicted-anchor OLA)
    folds = ["fold_01", "fold_02", "fold_03"]; ola = [0.48, 0.02, 0.03]; miss = [0.16, 0.49, 0.38]
    x = np.arange(3); w = 0.38
    a2.bar(x - w / 2, ola, w, label="pred-anchor OLA PCC", color="#1f77b4")
    a2.bar(x + w / 2, miss, w, label="missed-beat rate", color="#d62728")
    a2.set_xticks(x); a2.set_xticklabels(folds); a2.set_ylim(0, 1.0)
    a2.set_title("Anchor precision does NOT generalize (LOSO)"); a2.legend(fontsize=8)
    fig.suptitle("Route-C mechanism analysis (discovery): oracle bound high, predicted-anchor precision limited",
                 fontsize=10)
    _save(fig, "F4_route_c_mechanism")


def f5_morphology_examples():
    """Overlay predicted vs GT center-cycle morphology from a G1 checkpoint."""
    try:
        import torch
        from configs.mmecg_config import MMECGConfig
        from src.data.mmecg_dataset import build_loso_wsst_loaders_h5
        from src.models.model_factory import build_mmecg_model
        from scripts.train_mmecg import _shape200_targets
        cfg = MMECGConfig(); cfg.model_arch = "rem_align_mtl"; cfg.use_alignment = False
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        run = ROOT / "experiments_mmecg/q1_g1_remnet_mmecg_seed42/fold_01"
        saved = json.load(open(run / "config.json"))
        for k in ("baseline_hidden", "dropout", "n_range_bins", "use_alignment", "wsst_suffix"):
            if k in saved: setattr(cfg, k, saved[k])
        _, _, test = build_loso_wsst_loaders_h5(1, cfg.loso_h5_dir, batch_size=64,
                                                num_workers=2, balanced_sampling=False,
                                                balance_by="subject", target_norm="minmax",
                                                wsst_suffix="_wsst.h5")
        model = build_mmecg_model(cfg).to(dev)
        ck = torch.load(run / "checkpoints/best_shape200.pt", map_location=dev)
        state = ck.get("model_state_dict") or ck.get("model_state") or ck.get("model")
        model.load_state_dict(state); model.eval()
        batch = next(iter(test))
        with torch.no_grad():
            ecg_out, _ = model(batch["radar"].to(dev))
        pred = ecg_out.shape200_pred.cpu().numpy()[:, 0]
        gt = _shape200_targets(batch["ecg"].to(dev), batch["cycle_bounds"].to(dev)).cpu().numpy()[:, 0]
        # pick 4 examples spanning PCC
        def pcc(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))
        pccs = np.array([pcc(pred[i], gt[i]) for i in range(len(pred))])
        order = np.argsort(pccs)
        picks = [order[int(q * (len(order) - 1))] for q in (0.95, 0.7, 0.4, 0.1)]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
        for ax, idx in zip(axes, picks):
            ax.plot(gt[idx], label="GT", c="k", lw=1.5)
            ax.plot(pred[idx], label="REM-Net", c="#1f77b4", lw=1.5, alpha=0.8)
            ax.set_title(f"PCC={pccs[idx]:.3f}", fontsize=9); ax.set_xticks([])
        axes[0].legend(fontsize=8)
        fig.suptitle("Representative MMECG beat-morphology reconstructions (REM-Net vs GT, Shape-200)", fontsize=10)
        _save(fig, "F5_morphology_examples")
    except Exception as e:
        print("F5 skipped (inference unavailable):", e)


if __name__ == "__main__":
    f1_mmecg_morphology()
    f2_crrvs_external()
    f3_multiseed_stability()
    f4_route_c_mechanism()
    f5_morphology_examples()
    print("figures in", OUT)
