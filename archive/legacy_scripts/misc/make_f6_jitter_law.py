"""make_f6_jitter_law.py -- F6: the anchor-jitter fidelity law (paper centerpiece).

Panel A: ECG beat power spectrum -> RMS bandwidth beta -> critical jitter sigma_c.
Panel B: jitter-fidelity transfer function. Closed-form prediction
         rho(sigma) = sum S(f) exp(-2 pi^2 f^2 sigma^2) / sum S(f)
         vs (i) controlled jitter injection on real MMECG ECG and
            (ii) ten independent radar anchor-estimation experiments.

Analysis-only: reads frozen dataset + committed anchor_metrics.json. No training.
"""
import glob, json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments_mmecg" / "q1_paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
LOSO = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
FS = 200.0
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})


def beat_spectrum():
    beats = []
    for fp in sorted(LOSO.glob("fold_*/test.h5")):
        with h5py.File(fp, "r") as h:
            E, R = h["ecg"][:, 0, :], h["rpeak_indices"][:]
            for i in range(E.shape[0]):
                for k in np.asarray(R[i]).ravel().astype(int):
                    a, b = k - 40, k + 60          # -200 ms .. +300 ms
                    if a >= 0 and b <= E.shape[1]:
                        s = E[i, a:b].astype(np.float64)
                        beats.append(s - s.mean())
    beats = np.asarray(beats)
    S = (np.abs(np.fft.rfft(beats, axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(beats.shape[1], 1 / FS)
    return f, S, beats.shape[0]


def jitter_injection(sigmas_ms, seed=0):
    """Ground-truth ECG with beats displaced by N(0, sigma); PCC vs original."""
    rng = np.random.default_rng(seed)
    ecgs, rpks = [], []
    for fp in sorted(LOSO.glob("fold_*/test.h5")):
        with h5py.File(fp, "r") as h:
            E, R = h["ecg"][:, 0, :], h["rpeak_indices"][:]
            for i in range(E.shape[0]):
                ecgs.append(E[i].astype(np.float64))
                rpks.append(np.asarray(R[i]).ravel().astype(int))
    out = []
    for sm in sigmas_ms:
        pccs = []
        for x, r in zip(ecgs, rpks):
            if len(r) < 3:
                continue
            y, w = np.zeros_like(x), np.zeros_like(x)
            bnd = np.concatenate(([0], (r[:-1] + r[1:]) // 2, [len(x)]))
            for j in range(len(r)):
                a, b = bnd[j], bnd[j + 1]
                d = int(round(rng.normal(0, sm / 1000 * FS)))
                a2, b2 = np.clip(a + d, 0, len(x)), np.clip(b + d, 0, len(x))
                seg = x[a:b][: b2 - a2]
                y[a2:a2 + len(seg)] += seg
                w[a2:a2 + len(seg)] += 1
            m = w > 0
            if m.sum() < 100:
                continue
            y[m] /= w[m]
            pccs.append(np.corrcoef(x[m], y[m])[0, 1])
        out.append(np.mean(pccs))
    return np.asarray(out)


def anchor_experiments():
    rows = []
    for p in sorted(ROOT.glob("experiments_mmecg/q1_anchor_*/fold_*/results_anchor/anchor_metrics.json")):
        d = json.loads(p.read_text())
        v = d.get("variants", {}).get("soft")
        if not v:
            continue
        rows.append((p.parts[-4].replace("q1_anchor_", ""),
                     v["anchor_mae_ms_median"], v["anchor_f1_100ms"],
                     v["predicted_anchor_ola_pcc"], v["oracle_anchor_ola_pcc"]))
    return rows


def main():
    f, S, nbeats = beat_spectrum()
    beta = float(np.sqrt((S * f ** 2).sum() / S.sum()))
    sigma_c = 1000 / (2 * np.pi * beta)

    def rho(sig_ms):
        sig_ms = np.atleast_1d(np.asarray(sig_ms, dtype=float))
        return np.array([(S * np.exp(-2 * np.pi ** 2 * f ** 2 * (s / 1000) ** 2)).sum() / S.sum()
                         for s in sig_ms])

    grid = np.linspace(0, 130, 400)
    sig_meas = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100], dtype=float)
    meas = jitter_injection(sig_meas)
    anchors = anchor_experiments()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.1))

    # --- Panel A: beat spectrum ---
    Sn = S / S.max()
    axA.semilogy(f, Sn, color="#1f77b4", lw=1.6)
    axA.axvline(beta, color="#d62728", ls="--", lw=1.4,
                label=rf"$\beta$ = {beta:.1f} Hz (RMS bandwidth)")
    axA.set_xlim(0, 50)
    axA.set_ylim(1e-5, 2)
    axA.set_xlabel("Frequency (Hz)")
    axA.set_ylabel("Normalized power spectral density")
    axA.set_title(f"(a) MMECG beat spectrum (n = {nbeats:,} beats)")
    axA.legend(loc="upper right", fontsize=9)
    axA.text(0.97, 0.55, rf"$\sigma_c=\dfrac{{1}}{{2\pi\beta}}$ = {sigma_c:.0f} ms",
             transform=axA.transAxes, ha="right", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#d6b656"))

    # --- Panel B: transfer function ---
    axB.plot(grid, rho(grid), color="#d62728", lw=2.2, zorder=3,
             label="closed-form prediction (no free parameters)")
    axB.plot(sig_meas, meas, "o-", color="#1f77b4", ms=5, lw=1.4, zorder=4,
             label="jitter injection on real ECG")
    x = np.array([r[1] / 0.6745 for r in anchors])
    y = np.array([r[3] for r in anchors])
    axB.scatter(x, y, marker="^", s=64, color="#2ca02c", ec="k", lw=0.5, zorder=5,
                label="radar anchor estimators ($n$=10)")
    orc = np.mean([r[4] for r in anchors])
    axB.axhline(orc, color="#7f7f7f", ls=":", lw=1.6, zorder=2)
    axB.text(128, orc - 0.10, f"oracle anchors ({orc:.2f}) - morphology is not the limit",
             ha="right", fontsize=9, color="#555")
    axB.axvspan(0, sigma_c, color="#2ca02c", alpha=0.10, zorder=0)
    axB.text(sigma_c / 2, -0.155, r"$\sigma<\sigma_c$", ha="center", fontsize=8.5, color="#2ca02c")
    axB.annotate("no radar anchor estimator\nreaches this regime",
                 xy=(sigma_c, 0.62), xytext=(38, 0.78), fontsize=9, color="#b8860b",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#b8860b", lw=1.4,
                                 connectionstyle="arc3,rad=0.2"))
    axB.set_xlim(0, 130)
    axB.set_ylim(-0.2, 1.34)
    axB.set_xlabel(r"anchor jitter $\sigma$ (ms)")
    axB.set_ylabel("continuous ECG fidelity (PCC)")
    axB.set_title("(b) The anchor-jitter fidelity law")
    axB.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F6_jitter_law.{ext}", bbox_inches="tight")
    plt.close(fig)

    stats = {
        "n_beats": nbeats, "beta_hz": beta, "sigma_c_ms": sigma_c,
        "theory_pcc_0.9_sigma_ms": float(grid[np.argmin(np.abs(rho(grid) - 0.9))]),
        "theory_pcc_0.5_sigma_ms": float(grid[np.argmin(np.abs(rho(grid) - 0.5))]),
        "injection": {float(s): float(m) for s, m in zip(sig_meas, meas)},
        "anchor_runs": [
            {"run": r[0], "mae_median_ms": r[1], "f1_100ms": r[2],
             "ola_pred": r[3], "ola_oracle": r[4],
             "bound_jitter_only": float(rho(r[1] / 0.6745)[0]),
             "bound_x_f1": float(rho(r[1] / 0.6745)[0] * r[2])} for r in anchors],
    }
    b = np.array([s["bound_x_f1"] for s in stats["anchor_runs"]])
    stats["corr_bound_vs_observed"] = float(np.corrcoef(b, y)[0, 1])
    (OUT / "F6_jitter_law_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote F6_jitter_law.pdf/.png  beta={beta:.2f} Hz  sigma_c={sigma_c:.1f} ms  "
          f"corr={stats['corr_bound_vs_observed']:.3f}")


if __name__ == "__main__":
    main()
