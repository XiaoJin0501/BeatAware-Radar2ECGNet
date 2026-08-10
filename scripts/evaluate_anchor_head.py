"""evaluate_anchor_head.py — Stage-1 anchor localization metrics + coupling PCC.

Metrics (route C, docs/ASSEMBLY_DESIGN.md):
  - Anchor MAE (ms): median matched predicted-anchor to GT R distance.
  - Anchor F1 @ 50/100/150 ms.
  - RR error (ms) from predicted anchors.
  - Predicted-anchor vs Oracle-anchor per-window OLA PCC (morphology given = GT
    center cycle), isolating the effect of anchor quality on continuous ECG.

Example:
  python scripts/evaluate_anchor_head.py --exp_tag q1_anchor_head_fold01 --fold_idx 1 --device cuda:0
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
from src.models.remalign import AnchorHeatmapNet  # noqa: E402
from scripts.train_anchor_head import match_counts, peaks_from_heatmap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FS = 200
WIN = 1600


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(x)


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if n <= 0 or x.size == 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    if x.size == 1:
        return np.full(n, x[0], dtype=np.float32)
    xp = np.linspace(0.0, 1.0, x.size)
    return np.interp(np.linspace(0.0, 1.0, n), xp, x).astype(np.float32)


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1); b = b.reshape(-1)
    if a.size < 2:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-8
    return float((a * b).sum() / d)


def _central_cycle(rpeaks: np.ndarray, length: int = WIN) -> tuple[int, int]:
    r = np.sort(np.unique(rpeaks[(rpeaks >= 0) & (rpeaks < length)]))
    if r.size >= 2:
        c = length // 2
        pairs = [(int(a), int(b)) for a, b in zip(r[:-1], r[1:]) if b > a + 10]
        if pairs:
            inside = [(a, b) for a, b in pairs if a <= c <= b]
            return inside[0] if inside else min(pairs, key=lambda p: abs((p[0] + p[1]) / 2 - c))
    return 700, 900


def _assemble(anchors: np.ndarray, template: np.ndarray, length: int = WIN):
    a = np.sort(np.unique(anchors[(anchors >= 0) & (anchors < length)]))
    if a.size < 2 or template.size < 2:
        return None, None
    out = np.zeros(length, np.float64); w = np.zeros(length, np.float64)
    for s, e in zip(a[:-1], a[1:]):
        seg = _resample(template, int(e - s))
        out[s:e] += seg; w[s:e] += 1.0
    mask = w > 0
    out[mask] /= w[mask]
    return out.astype(np.float32), mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loso_h5_dir", default=None)
    ap.add_argument("--wsst_suffix", default="_wsst.h5")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = MMECGConfig()
    loso_dir = Path(args.loso_h5_dir or cfg.loso_h5_dir)
    fold_dir = loso_dir / f"fold_{args.fold_idx:02d}"
    run_dir = ROOT / cfg.exp_dir / args.exp_tag / f"fold_{args.fold_idx:02d}"
    device = torch.device(args.device)

    model = AnchorHeatmapNet(in_channels=cfg.n_range_bins, hidden=cfg.baseline_hidden,
                             dropout=cfg.dropout, out_len=cfg.signal_len).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "anchor_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    test_h5 = fold_dir / "test.h5"
    test_wsst = fold_dir / f"test{args.wsst_suffix}"
    ms = 1000.0 / FS

    tols = {50: int(50 * FS / 1000), 100: int(100 * FS / 1000), 150: int(150 * FS / 1000)}
    TP = {k: 0 for k in tols}; FP = {k: 0 for k in tols}; FN = {k: 0 for k in tols}
    match_err_ms: list[float] = []
    rr_err_ms: list[float] = []
    pcc_pred: list[float] = []
    pcc_oracle: list[float] = []

    with h5py.File(test_h5, "r") as h5, h5py.File(test_wsst, "r") as wh5, torch.no_grad():
        n = int(h5["ecg"].shape[0])
        for lo in range(0, n, args.batch_size):
            hi = min(lo + args.batch_size, n)
            idx = list(range(lo, hi))
            radar = torch.from_numpy(np.asarray(wh5["wsst"][idx], dtype=np.float32)).to(device)
            heat = model(radar).cpu().numpy()[:, 0]
            for k, i in enumerate(idx):
                ecg = np.asarray(h5["ecg"][i], dtype=np.float32).reshape(-1)
                gt = np.sort(np.unique(np.asarray(h5["rpeak_indices"][i], dtype=np.int64)))
                gt = gt[(gt >= 0) & (gt < WIN)]
                pred = peaks_from_heatmap(heat[k])
                # F1 at each tolerance
                for tol_ms, tol in tols.items():
                    tp, fp, fn = match_counts(pred, gt, tol)
                    TP[tol_ms] += tp; FP[tol_ms] += fp; FN[tol_ms] += fn
                # matched localization error (within 150 ms)
                if pred.size and gt.size:
                    d = np.abs(gt[:, None] - pred[None, :])
                    order = np.argsort(d.ravel()); mg = np.zeros(gt.size, bool); mp = np.zeros(pred.size, bool)
                    for f in order:
                        gi, pi = divmod(int(f), pred.size)
                        if mg[gi] or mp[pi]:
                            continue
                        if d[gi, pi] <= tols[150]:
                            mg[gi] = mp[pi] = True
                            match_err_ms.append(float(d[gi, pi]) * ms)
                # RR error (mean RR per window)
                if pred.size >= 2 and gt.size >= 2:
                    rr_err_ms.append(abs(float(np.mean(np.diff(pred))) - float(np.mean(np.diff(gt)))) * ms)
                # coupling PCC: morphology given = GT central cycle
                a, b = _central_cycle(gt)
                template = _norm01(ecg[a:b]) if b > a else _norm01(ecg[700:900])
                gt_ecg = _norm01(ecg)
                for anchors, bucket in ((pred, pcc_pred), (gt, pcc_oracle)):
                    asm, mask = _assemble(anchors, template)
                    if asm is not None and mask.sum() > 50:
                        bucket.append(_pcc(asm[mask], gt_ecg[mask]))

    def f1(k):
        p = TP[k] / (TP[k] + FP[k] + 1e-8); r = TP[k] / (TP[k] + FN[k] + 1e-8)
        return float(2 * p * r / (p + r + 1e-8))

    summary = {
        "exp_tag": args.exp_tag, "fold_idx": int(args.fold_idx), "num_windows": int(n),
        "anchor_f1_50ms": f1(50), "anchor_f1_100ms": f1(100), "anchor_f1_150ms": f1(150),
        "anchor_mae_ms_median": float(np.median(match_err_ms)) if match_err_ms else float("nan"),
        "anchor_mae_ms_mean": float(np.mean(match_err_ms)) if match_err_ms else float("nan"),
        "rr_error_ms_mean": float(np.mean(rr_err_ms)) if rr_err_ms else float("nan"),
        "predicted_anchor_ola_pcc": float(np.nanmean(pcc_pred)) if pcc_pred else float("nan"),
        "oracle_anchor_ola_pcc": float(np.nanmean(pcc_oracle)) if pcc_oracle else float("nan"),
        "note": "Per-window assembly with morphology GIVEN (GT central cycle); isolates anchor quality. Oracle uses GT anchors, predicted uses AnchorHeatmapNet anchors.",
    }
    out_dir = run_dir / "results_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anchor_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
