"""evaluate_remnet_rate_metrics.py — rate/localization metrics on the FROZEN REM-Net.

Inference-only on the frozen seed-42 final-test checkpoints (no retraining). For each
fold it loads `checkpoints/best.pt`, runs the model on the fold test set, and reports
beat-localization metrics from the model's EVENT HEAD (R-mask):
  - R-peak F1 @ 50/100/150 ms
  - RR-interval error (ms) and heart-rate error (bpm) from matched beats
Results are written to `<run>/fold_XX/results_rate/rate_metrics.json` (traceable to the
final run directory). Honest support for the localization-precision discovery, not a
strength claim.

Example:
  python scripts/evaluate_remnet_rate_metrics.py --exp_tag q1_final_remnet_incumbent_mmecg_seed42 --folds 1-11 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.models.model_factory import build_mmecg_model  # noqa: E402
from scripts.train_anchor_head import peaks_from_heatmap, match_counts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FS = 200
WIN = 1600
MS = 1000.0 / FS


def _matched_rr_hr(pred, gt, tol):
    """RR error (ms) and HR error (bpm) over matched consecutive beats."""
    if pred.size < 2 or gt.size < 2:
        return None, None
    d = np.abs(gt[:, None] - pred[None, :]); order = np.argsort(d.ravel())
    mg = np.zeros(gt.size, bool); mp = np.zeros(pred.size, bool); pairs = []
    for f in order:
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi]:
            continue
        if d[gi, pi] <= tol:
            mg[gi] = mp[pi] = True; pairs.append((gi, pi))
    if len(pairs) < 2:
        return None, None
    pairs.sort()
    rr_err = []
    for (g0, p0), (g1, p1) in zip(pairs[:-1], pairs[1:]):
        rr_err.append(abs((gt[g1] - gt[g0]) - (pred[p1] - pred[p0])) * MS)
    gt_rr = np.mean(np.diff([gt[g] for g, _ in pairs])) * MS / 1000.0
    pr_rr = np.mean(np.diff([pred[p] for _, p in pairs])) * MS / 1000.0
    hr_err = abs(60.0 / max(pr_rr, 1e-6) - 60.0 / max(gt_rr, 1e-6))
    return float(np.mean(rr_err)) if rr_err else None, float(hr_err)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--folds", default="1-11", help="e.g. 1-11 or 1,2,3")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    if "-" in args.folds:
        a, b = args.folds.split("-"); folds = list(range(int(a), int(b) + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]
    dev = torch.device(args.device)
    tols = {50: 10, 100: 20, 150: 30}

    agg = {f"f1_{k}ms": [] for k in tols}
    agg["rr_error_ms"] = []; agg["hr_error_bpm"] = []
    for fold in folds:
        run = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{fold:02d}"
        saved = json.load(open(run / "config.json"))
        cfg = MMECGConfig()
        for k in ("model_arch", "baseline_hidden", "dropout", "n_range_bins",
                  "use_alignment", "wsst_suffix", "loso_h5_dir"):
            if k in saved:
                setattr(cfg, k, saved[k])
        model = build_mmecg_model(cfg).to(dev)
        ck = torch.load(run / "checkpoints" / args.checkpoint, map_location=dev)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        fold_dir = Path(cfg.loso_h5_dir) / f"fold_{fold:02d}"
        wsst = fold_dir / f"test{saved.get('wsst_suffix', '_wsst.h5')}"
        TP = {k: 0 for k in tols}; FP = dict(TP); FN = dict(TP)
        rr_list = []; hr_list = []
        with h5py.File(fold_dir / "test.h5", "r") as h5, h5py.File(wsst, "r") as wh5, torch.no_grad():
            n = int(h5["ecg"].shape[0])
            for lo in range(0, n, 32):
                idx = list(range(lo, min(lo + 32, n)))
                radar = torch.from_numpy(np.asarray(wh5["wsst"][idx], np.float32)).to(dev)
                _, peak_preds = model(radar)
                rmask = peak_preds[0].cpu().numpy()[:, 0]  # event-head R-mask
                for k, i in enumerate(idx):
                    gt = np.sort(np.unique(np.asarray(h5["rpeak_indices"][i], np.int64)))
                    gt = gt[(gt >= 0) & (gt < WIN)].astype(float)
                    pr = peaks_from_heatmap(rmask[k]).astype(float)
                    for tms, tol in tols.items():
                        tp, fp, fn = match_counts(pr, gt, tol)
                        TP[tms] += tp; FP[tms] += fp; FN[tms] += fn
                    rr, hr = _matched_rr_hr(pr, gt, tols[100])
                    if rr is not None:
                        rr_list.append(rr); hr_list.append(hr)

        def f1(k):
            p = TP[k] / (TP[k] + FP[k] + 1e-8); r = TP[k] / (TP[k] + FN[k] + 1e-8)
            return float(2 * p * r / (p + r + 1e-8))
        res = {"exp_tag": args.exp_tag, "fold": fold, "checkpoint": args.checkpoint,
               "source": "event_head_r_mask",
               "rpeak_f1_50ms": f1(50), "rpeak_f1_100ms": f1(100), "rpeak_f1_150ms": f1(150),
               "rr_error_ms_mean": float(np.mean(rr_list)) if rr_list else None,
               "hr_error_bpm_mean": float(np.mean(hr_list)) if hr_list else None,
               "n_windows_with_rr": len(rr_list)}
        out = run / "results_rate"; out.mkdir(exist_ok=True)
        (out / "rate_metrics.json").write_text(json.dumps(res, indent=2))
        for k in tols:
            agg[f"f1_{k}ms"].append(f1(k))
        if rr_list:
            agg["rr_error_ms"].append(np.mean(rr_list)); agg["hr_error_bpm"].append(np.mean(hr_list))
        print(f"fold_{fold:02d}: F1@100={f1(100):.3f} F1@150={f1(150):.3f} "
              f"RR={np.mean(rr_list):.1f}ms HR={np.mean(hr_list):.1f}bpm" if rr_list else f"fold_{fold:02d}: F1@100={f1(100):.3f}")

    print("\n=== AGGREGATE (fold-macro) ===")
    for k, v in agg.items():
        if v:
            print(f"  {k}: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
