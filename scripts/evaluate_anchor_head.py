"""evaluate_anchor_head.py — Stage-1/1b anchor localization metrics + coupling PCC.

Reports, for several anchor post-processing variants, the effect on localization
and on continuous assembly (morphology GIVEN = GT central cycle, so the only
variable is anchor quality). See docs/ASSEMBLY_DESIGN.md.

Variants:
  - raw            : integer peaks from the heatmap.
  - soft           : + soft-argmax sub-sample refinement.
  - soft+offset    : + global PEP offset (estimated on VAL, applied to test),
                     i.e. shift predicted anchors to the electrical-R timeline.

Example:
  python scripts/evaluate_anchor_head.py --exp_tag q1_anchor_head_fold01b --fold_idx 1 --device cuda:0
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
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FS = 200
WIN = 1600
MS = 1000.0 / FS


def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(x)


def _resample(x, n):
    x = np.asarray(x, dtype=np.float32)
    if n <= 0 or x.size == 0:
        return np.zeros(max(n, 0), dtype=np.float32)
    if x.size == 1:
        return np.full(n, x[0], dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, x.size), x).astype(np.float32)


def _pcc(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    if a.size < 2:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-8
    return float((a * b).sum() / d)


def _central_cycle(rpeaks, length=WIN):
    r = np.sort(np.unique(rpeaks[(rpeaks >= 0) & (rpeaks < length)]))
    if r.size >= 2:
        c = length // 2
        pairs = [(int(a), int(b)) for a, b in zip(r[:-1], r[1:]) if b > a + 10]
        if pairs:
            inside = [(a, b) for a, b in pairs if a <= c <= b]
            return inside[0] if inside else min(pairs, key=lambda p: abs((p[0] + p[1]) / 2 - c))
    return 700, 900


def _soft_argmax(heat, peaks, win=8):
    """Refine integer peaks to sub-sample centroids over a +/- win window."""
    y = np.asarray(heat, dtype=np.float32)
    out = []
    for p in peaks:
        lo, hi = max(0, int(p) - win), min(y.size, int(p) + win + 1)
        w = np.clip(y[lo:hi], 0, None)
        if w.sum() < 1e-6:
            out.append(float(p))
        else:
            idx = np.arange(lo, hi)
            out.append(float((idx * w).sum() / w.sum()))
    return np.asarray(out, dtype=np.float32)


def _peaks_param(y, thr_rel=0.5, thr_floor=0.3, min_dist=40):
    y = np.asarray(y, np.float32).reshape(-1)
    if y.size < 3:
        return np.empty(0, np.int64)
    ymax = float(y.max())
    if ymax < 1e-6:
        return np.empty(0, np.int64)
    thr = max(thr_floor, thr_rel * ymax)
    cand = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] >= thr)) + 1
    out = []
    for p in cand.tolist():
        if not out or p - out[-1] >= min_dist:
            out.append(int(p))
        elif y[p] > y[out[-1]]:
            out[-1] = int(p)
    return np.asarray(out, np.int64)


def _anchors(heat, soft, thr_rel=0.5, thr_floor=0.3):
    p = _peaks_param(heat, thr_rel, thr_floor)
    if p.size == 0:
        return p.astype(np.float32)
    return _soft_argmax(heat, p) if soft else p.astype(np.float32)


def _match(pred, gt, tol):
    if gt.size == 0 and pred.size == 0:
        return 0, 0, 0, [], []
    if gt.size == 0:
        return 0, int(pred.size), 0, [], []
    if pred.size == 0:
        return 0, 0, int(gt.size), [], []
    d = np.abs(gt[:, None] - pred[None, :])
    order = np.argsort(d.ravel()); mg = np.zeros(gt.size, bool); mp = np.zeros(pred.size, bool)
    tp = 0; signed = []; pairs = []
    for f in order:
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi]:
            continue
        if d[gi, pi] <= tol:
            mg[gi] = mp[pi] = True; tp += 1
            signed.append(float(gt[gi] - pred[pi]))
            pairs.append((gi, pi))
    return tp, int((~mp).sum()), int((~mg).sum()), signed, pairs


def _assemble(anchors, template, length=WIN):
    a = np.sort(np.unique(np.round(anchors).astype(int)))
    a = a[(a >= 0) & (a < length)]
    if a.size < 2 or template.size < 2:
        return None, None
    out = np.zeros(length); w = np.zeros(length)
    for s, e in zip(a[:-1], a[1:]):
        out[s:e] += _resample(template, int(e - s)); w[s:e] += 1.0
    mask = w > 0; out[mask] /= w[mask]
    return out.astype(np.float32), mask


def _load_split(model, h5_path, wsst_path, device, batch=32):
    """Return per-window (heatmap, gt_peaks, ecg)."""
    items = []
    with h5py.File(h5_path, "r") as h5, h5py.File(wsst_path, "r") as wh5, torch.no_grad():
        n = int(h5["ecg"].shape[0])
        for lo in range(0, n, batch):
            idx = list(range(lo, min(lo + batch, n)))
            r = torch.from_numpy(np.asarray(wh5["wsst"][idx], dtype=np.float32)).to(device)
            heat = model(r).cpu().numpy()[:, 0]
            for k, i in enumerate(idx):
                gt = np.sort(np.unique(np.asarray(h5["rpeak_indices"][i], dtype=np.int64)))
                gt = gt[(gt >= 0) & (gt < WIN)].astype(np.float32)
                items.append((heat[k], gt, np.asarray(h5["ecg"][i], dtype=np.float32).reshape(-1)))
    return items


def _estimate_offset(items, soft, thr_rel=0.5, thr_floor=0.3):
    """Median signed (gt - pred) over matched pairs within 150 ms."""
    signed = []
    for heat, gt, _ in items:
        pred = _anchors(heat, soft, thr_rel, thr_floor)
        _, _, _, s, _ = _match(pred, gt, int(150 * FS / 1000))
        signed.extend(s)
    return float(np.median(signed)) if signed else 0.0


def _metrics(items, soft, offset, thr_rel=0.5, thr_floor=0.3):
    tols = {50: 50 * FS / 1000, 100: 100 * FS / 1000, 150: 150 * FS / 1000}
    TP = {k: 0 for k in tols}; FP = dict(TP); FN = dict(TP)
    mae = []; rr_crude = []; rr_matched = []; signed_bias = []
    gt_total = 0; pred_total = 0; miss100 = 0; extra100 = 0
    pcc_pred = []; pcc_oracle = []
    for heat, gt, ecg in items:
        pred = _anchors(heat, soft, thr_rel, thr_floor) + offset
        gt_total += int(gt.size); pred_total += int(pred.size)
        pairs100 = []
        for tms, tol in tols.items():
            tp, fp, fn, s, pairs = _match(pred, gt, tol)
            TP[tms] += tp; FP[tms] += fp; FN[tms] += fn
            if tms == 150:
                mae.extend([abs(v) * MS for v in s])
            if tms == 100:
                signed_bias.extend([v * MS for v in s]); miss100 += fn; extra100 += fp; pairs100 = pairs
        # crude window mean-RR (count-sensitive)
        if pred.size >= 2 and gt.size >= 2:
            rr_crude.append(abs(float(np.mean(np.diff(pred))) - float(np.mean(np.diff(gt)))) * MS)
        # matched-pair RR (robust to miss/extra): consecutive matched beats only
        if len(pairs100) >= 2:
            ps = sorted(pairs100, key=lambda x: x[0])
            for (g0, p0), (g1, p1) in zip(ps[:-1], ps[1:]):
                rr_matched.append(abs((gt[g1] - gt[g0]) - (pred[p1] - pred[p0])) * MS)
        a, b = _central_cycle(gt); tmpl = _norm01(ecg[a:b]) if b > a else _norm01(ecg[700:900])
        gtn = _norm01(ecg)
        asm, m = _assemble(pred, tmpl)
        if asm is not None and m.sum() > 50:
            pcc_pred.append(_pcc(asm[m], gtn[m]))
        asm_o, mo = _assemble(gt, tmpl)
        if asm_o is not None and mo.sum() > 50:
            pcc_oracle.append(_pcc(asm_o[mo], gtn[mo]))

    def f1(k):
        p = TP[k] / (TP[k] + FP[k] + 1e-8); r = TP[k] / (TP[k] + FN[k] + 1e-8)
        return float(2 * p * r / (p + r + 1e-8))
    return {
        "anchor_f1_50ms": f1(50), "anchor_f1_100ms": f1(100), "anchor_f1_150ms": f1(150),
        "anchor_mae_ms_median": float(np.median(mae)) if mae else float("nan"),
        "signed_bias_ms_median": float(np.median(signed_bias)) if signed_bias else float("nan"),
        "missed_rate_100ms": float(miss100 / (gt_total + 1e-8)),
        "extra_rate_100ms": float(extra100 / (pred_total + 1e-8)),
        "rr_error_ms_crude": float(np.mean(rr_crude)) if rr_crude else float("nan"),
        "rr_error_ms_matched": float(np.mean(rr_matched)) if rr_matched else float("nan"),
        "predicted_anchor_ola_pcc": float(np.nanmean(pcc_pred)) if pcc_pred else float("nan"),
        "oracle_anchor_ola_pcc": float(np.nanmean(pcc_oracle)) if pcc_oracle else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold_idx", type=int, required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--loso_h5_dir", default=None)
    ap.add_argument("--wsst_suffix", default="_wsst.h5")
    ap.add_argument("--soft_argmax_win", type=int, default=8)
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

    test_items = _load_split(model, fold_dir / "test.h5", fold_dir / f"test{args.wsst_suffix}", device)
    val_items = _load_split(model, fold_dir / "val.h5", fold_dir / f"val{args.wsst_suffix}", device)
    offset = _estimate_offset(val_items, soft=True)  # estimated on VAL (leakage-safe)

    variants = {
        "raw": _metrics(test_items, soft=False, offset=0.0),
        "soft": _metrics(test_items, soft=True, offset=0.0),
        "soft+offset": _metrics(test_items, soft=True, offset=offset),
    }
    summary = {
        "exp_tag": args.exp_tag, "fold_idx": int(args.fold_idx),
        "num_test_windows": len(test_items),
        "pep_offset_samples_from_val": offset, "pep_offset_ms_from_val": offset * MS,
        "variants": variants,
        "note": "Assembly morphology GIVEN (GT central cycle); isolates anchor quality. Offset estimated on VAL and applied to test (leakage-safe). Sign: +offset shifts predicted anchors earlier toward the electrical-R timeline.",
    }
    out_dir = run_dir / "results_anchor"; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anchor_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
