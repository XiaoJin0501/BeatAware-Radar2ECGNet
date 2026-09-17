#!/usr/bin/env python3
"""analyze_delta_sys.py -- C3 delay analysis for the alignment branch.

Analysis-only: loads a frozen aligned checkpoint, extracts the learned systematic
electromechanical delay `delta_sys` on held-out test windows, and relates it to
independent physiological references, per docs/METRIC_CONTRACT.md Sec. "Delay
Analysis Metrics (C3)":

  (a) delta_sys vs pseudo-GT lag  -- cross-correlation lag between the radar
      mechanical envelope and the ground-truth R-peak train. Pseudo-GT is used
      for validation only; it never enters training or inference.
  (b) delta_sys vs heart rate     -- PEP is expected to shorten as HR rises.
  (c) delta_sys distribution      -- per-fold and per-subject spread.

No training, no checkpoint writes.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig                      # noqa: E402
from src.data.mmecg_dataset import build_loso_wsst_loaders_h5     # noqa: E402
from src.models.model_factory import build_mmecg_model            # noqa: E402

FS = 200.0
MAX_LAG_MS = 300.0


def load_model(cfg, run_dir, device, ckpt_name="best.pt"):
    saved = json.loads((run_dir / "config.json").read_text())
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = build_mmecg_model(cfg).to(device)
    ck = torch.load(run_dir / "checkpoints" / ckpt_name, map_location=device, weights_only=True)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()
    print(f"  loaded epoch={ck.get('epoch','?')} val_pcc={float(ck.get('val_pcc',float('nan'))):.4f}")
    return model, cfg


def mechanical_envelope(rcg: np.ndarray) -> np.ndarray:
    """Dominant-bin mechanical activity: |d/dt| of the highest-variance range bin."""
    bin_idx = int(np.argmax(rcg.var(axis=1)))
    x = rcg[bin_idx].astype(np.float64)
    d = np.abs(np.diff(x, prepend=x[0]))
    d -= d.mean()
    return d


def xcorr_lag_ms(env: np.ndarray, rpeaks: np.ndarray, n: int) -> float:
    """Lag (ms) at which the R-peak train best predicts mechanical activity.

    Positive lag = mechanical event follows the electrical R-peak (PEP-like)."""
    train = np.zeros(n)
    r = rpeaks[(rpeaks >= 0) & (rpeaks < n)]
    if len(r) < 2:
        return np.nan
    train[r] = 1.0
    train -= train.mean()
    maxlag = int(MAX_LAG_MS / 1000 * FS)
    best, best_lag = -np.inf, np.nan
    ev = env - env.mean()
    den = np.linalg.norm(ev) * np.linalg.norm(train) + 1e-8
    for lag in range(0, maxlag + 1):
        c = float(np.dot(ev[lag:], train[: n - lag])) / den
        if c > best:
            best, best_lag = c, lag
    return best_lag / FS * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = MMECGConfig()
    run_dir = Path(cfg.exp_dir) / args.exp_tag / f"fold_{args.fold_idx:02d}"
    print(f"[delta_sys] {args.exp_tag} fold_{args.fold_idx:02d}")
    model, cfg = load_model(cfg, run_dir, device, args.checkpoint)

    _, _, test_loader = build_loso_wsst_loaders_h5(
        fold_idx=args.fold_idx, loso_dir=cfg.loso_h5_dir,
        batch_size=int(getattr(cfg, "batch_size", 16)),
        num_workers=int(getattr(cfg, "num_workers", 4)),
        balanced_sampling=False,
        target_norm=getattr(cfg, "target_norm", "minmax"),
        wsst_suffix=getattr(cfg, "wsst_suffix", "_wsst.h5"),
    )

    # --- learned delay, in dataset order ---
    d_sys, d_res, subjects, states = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["radar"].to(device))
            ecg = out[0] if isinstance(out, (tuple, list)) else out
            if not hasattr(ecg, "delta_sys"):
                raise AttributeError("checkpoint has no delta_sys; is the alignment branch enabled?")
            d_sys.append(ecg.delta_sys.squeeze(1).mean(-1).cpu().numpy())
            d_res.append(ecg.delta_residual.squeeze(1).abs().mean(-1).cpu().numpy())
            subjects.append(np.asarray(batch["subject"]))
            states.append(np.asarray(batch["state"]))
    d_sys = np.concatenate(d_sys); d_res = np.concatenate(d_res)
    subjects = np.concatenate(subjects); states = np.concatenate(states)

    # --- independent physiological references from the raw fold file ---
    fold_dir = Path(cfg.loso_h5_dir) / f"fold_{args.fold_idx:02d}"
    with h5py.File(fold_dir / "test.h5", "r") as h:
        rcg_all = h["rcg"][:]
        rpk_all = h["rpeak_indices"][:]
    n_win = min(len(d_sys), rcg_all.shape[0])
    lag_ms, hr_bpm = np.full(n_win, np.nan), np.full(n_win, np.nan)
    for i in range(n_win):
        r = np.asarray(rpk_all[i]).ravel().astype(int)
        lag_ms[i] = xcorr_lag_ms(mechanical_envelope(rcg_all[i]), r, rcg_all.shape[2])
        if len(r) >= 2:
            hr_bpm[i] = 60.0 / (np.mean(np.diff(r)) / FS)
    d_sys, d_res = d_sys[:n_win], d_res[:n_win]
    subjects, states = subjects[:n_win], states[:n_win]

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
            return float("nan"), int(m.sum())
        return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())

    r_lag, n_lag = corr(d_sys, lag_ms)
    r_hr, n_hr = corr(d_sys, hr_bpm)

    res = {
        "exp_tag": args.exp_tag, "fold": args.fold_idx, "checkpoint": args.checkpoint,
        "n_windows": int(n_win),
        "delta_sys_ms": {"mean": float(np.mean(d_sys)), "std": float(np.std(d_sys)),
                         "min": float(np.min(d_sys)), "max": float(np.max(d_sys)),
                         "abs_mean": float(np.mean(np.abs(d_sys)))},
        "delta_residual_abs_ms_mean": float(np.mean(d_res)),
        "pseudo_gt_lag_ms": {"mean": float(np.nanmean(lag_ms)), "std": float(np.nanstd(lag_ms))},
        "hr_bpm": {"mean": float(np.nanmean(hr_bpm)), "std": float(np.nanstd(hr_bpm))},
        "corr_delta_sys_vs_pseudo_gt_lag": r_lag, "n_pairs_lag": n_lag,
        "corr_delta_sys_vs_hr": r_hr, "n_pairs_hr": n_hr,
        "per_subject": {int(s): {"n": int((subjects == s).sum()),
                                 "delta_sys_mean_ms": float(d_sys[subjects == s].mean())}
                        for s in np.unique(subjects)},
    }
    out_dir = run_dir / "results_delta_sys"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_sys_analysis.json").write_text(json.dumps(res, indent=1))
    np.savez(out_dir / "delta_sys_raw.npz", delta_sys=d_sys, delta_res=d_res,
             lag_ms=lag_ms, hr_bpm=hr_bpm, subject=subjects, state=states)

    print(f"  delta_sys      = {res['delta_sys_ms']['mean']:+.2f} +- {res['delta_sys_ms']['std']:.2f} ms "
          f"(|mean|={res['delta_sys_ms']['abs_mean']:.2f}, range [{res['delta_sys_ms']['min']:.1f}, {res['delta_sys_ms']['max']:.1f}])")
    print(f"  |delta_res|    = {res['delta_residual_abs_ms_mean']:.2f} ms")
    print(f"  pseudo-GT lag  = {res['pseudo_gt_lag_ms']['mean']:.1f} +- {res['pseudo_gt_lag_ms']['std']:.1f} ms")
    print(f"  HR             = {res['hr_bpm']['mean']:.1f} +- {res['hr_bpm']['std']:.1f} bpm")
    print(f"  corr(delta_sys, pseudo-GT lag) = {r_lag:+.3f}  (n={n_lag})")
    print(f"  corr(delta_sys, HR)            = {r_hr:+.3f}  (n={n_hr})")
    print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
