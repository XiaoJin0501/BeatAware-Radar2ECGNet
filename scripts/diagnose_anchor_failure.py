"""diagnose_anchor_failure.py — why does anchor recall collapse on folds 02/03?

Three cheap diagnostics (no training), using the trained fusion anchor checkpoints:
  1. train vs val vs test recall@100ms  -> overfitting vs underfitting.
  2. peak-threshold sweep on a fold's test -> threshold/NMS vs model-not-predicting.
  3. per-fold signal quality: model confidence at GT beats, HR, beats/window,
     physistatus mix, rcg amplitude -> is fold_01 just an easy subject?

Writes experiments_mmecg/q1_anchor_failure_diag/diagnosis.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.models.remalign import AnchorHeatmapFusionNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FS = 200
WIN = 1600
cfg = MMECGConfig()
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _peaks(y, thr_rel=0.5, thr_floor=0.3, min_dist=40):
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


def _recall_fp(pred, gt, tol=20):
    if gt.size == 0:
        return 0, 0, 0
    if pred.size == 0:
        return 0, 0, int(gt.size)
    d = np.abs(gt[:, None] - pred[None, :]); order = np.argsort(d.ravel())
    mg = np.zeros(gt.size, bool); mp = np.zeros(pred.size, bool); tp = 0
    for f in order:
        gi, pi = divmod(int(f), pred.size)
        if mg[gi] or mp[pi]:
            continue
        if d[gi, pi] <= tol:
            mg[gi] = mp[pi] = True; tp += 1
    return tp, int((~mp).sum()), int((~mg).sum())


def _load_model(fold):
    tag = "q1_anchor_fusion_fold01" if fold == 1 else f"q1_anchor_fusion_fold{fold:02d}"
    m = AnchorHeatmapFusionNet(in_channels=cfg.n_range_bins, hidden=cfg.baseline_hidden,
                               dropout=cfg.dropout, n_blocks=6, out_len=cfg.signal_len).to(DEV)
    ck = torch.load(ROOT / "experiments_mmecg" / tag / f"fold_{fold:02d}" / "checkpoints/anchor_best.pt", map_location=DEV)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def _heatmaps(model, fold, split, limit=None, batch=32):
    fd = Path(cfg.loso_h5_dir) / f"fold_{fold:02d}"
    out = []
    with h5py.File(fd / f"{split}.h5", "r") as h, h5py.File(fd / f"{split}_wsst.h5", "r") as w:
        n = int(h["ecg"].shape[0]); n = min(n, limit) if limit else n
        for lo in range(0, n, batch):
            idx = list(range(lo, min(lo + batch, n)))
            rcg = np.asarray(h["rcg"][idx], np.float32)
            rcg = (rcg - rcg.mean(2, keepdims=True)) / (rcg.std(2, keepdims=True) + 1e-6)
            wsst = torch.from_numpy(np.asarray(w["wsst"][idx], np.float32)).to(DEV)
            heat = model(wsst, torch.from_numpy(rcg).to(DEV)).cpu().numpy()[:, 0]
            for k, i in enumerate(idx):
                gt = np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64)))
                gt = gt[(gt >= 0) & (gt < WIN)]
                out.append((heat[k], gt, np.asarray(h["rcg"][i], np.float32),
                            _dec(h["physistatus"][i])))
    return out


def _dec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _recall_at(items, thr_rel=0.5, thr_floor=0.3, min_dist=40, tol=20):
    TP = FN = FP = 0
    for heat, gt, _, _ in items:
        p = _peaks(heat, thr_rel, thr_floor, min_dist)
        tp, fp, fn = _recall_fp(p.astype(float), gt.astype(float), tol)
        TP += tp; FP += fp; FN += fn
    return TP / (TP + FN + 1e-8), FP / (TP + FP + 1e-8)  # recall, FP-rate


def main():
    diag = {}

    # Step 1: train/val/test recall@100 using each fold's own fusion model.
    step1 = {}
    for fold in (1, 2, 3):
        m = _load_model(fold)
        row = {}
        for split, lim in (("train", 400), ("val", None), ("test", None)):
            items = _heatmaps(m, fold, split, limit=lim)
            rec, fpr = _recall_at(items)
            row[split] = {"recall_100ms": round(rec, 3), "fp_rate": round(fpr, 3), "n": len(items)}
        step1[f"fold_{fold}"] = row
        print(f"[step1] fold_{fold}: " + " ".join(f"{s}={row[s]['recall_100ms']}" for s in ("train", "val", "test")))
    diag["step1_train_val_test_recall"] = step1

    # Step 2: threshold sweep on fold_02 test.
    m2 = _load_model(2)
    test2 = _heatmaps(m2, 2, "test")
    sweep = {}
    for tr in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
        rec, fpr = _recall_at(test2, thr_rel=tr, thr_floor=min(tr, 0.05))
        sweep[f"thr_rel_{tr}"] = {"recall_100ms": round(rec, 3), "fp_rate": round(fpr, 3)}
        print(f"[step2] fold_02 thr_rel={tr}: recall={rec:.3f} fp_rate={fpr:.3f}")
    diag["step2_threshold_sweep_fold02_test"] = sweep

    # Step 3: per-fold test signal quality.
    step3 = {}
    for fold in (1, 2, 3):
        m = _load_model(fold)
        items = _heatmaps(m, fold, "test")
        conf = []; hr = []; beats = []; amp = []; phys = Counter()
        for heat, gt, rcg, ps in items:
            beats.append(len(gt)); phys[ps] += 1
            amp.append(float(np.mean(np.std(rcg, axis=1))))
            if gt.size:
                conf.append(float(np.mean(heat[gt])))  # model response AT GT beats
            if gt.size >= 2:
                hr.append(60.0 * FS / float(np.mean(np.diff(gt))))
        step3[f"fold_{fold}"] = {
            "mean_conf_at_gt": round(float(np.mean(conf)), 3) if conf else None,
            "mean_hr_bpm": round(float(np.mean(hr)), 1) if hr else None,
            "mean_beats_per_window": round(float(np.mean(beats)), 2),
            "mean_rcg_amp": round(float(np.mean(amp)), 4),
            "physistatus": dict(phys),
        }
        print(f"[step3] fold_{fold}: conf@GT={step3[f'fold_{fold}']['mean_conf_at_gt']} "
              f"HR={step3[f'fold_{fold}']['mean_hr_bpm']} beats/win={step3[f'fold_{fold}']['mean_beats_per_window']} "
              f"phys={dict(phys)}")
    diag["step3_fold_signal_quality"] = step3

    out = ROOT / "experiments_mmecg" / "q1_anchor_failure_diag"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosis.json").write_text(json.dumps(diag, indent=2))
    print("\nWROTE", out / "diagnosis.json")


if __name__ == "__main__":
    main()
