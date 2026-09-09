#!/usr/bin/env python3
"""Full-support C0-C5 re-evaluation. Spec: q1_cgd_reeval_fullsupport.json

Evaluation only. No training, no test split, and the frozen evaluator's scoring and
assembly DEFINITIONS are imported unchanged -- this file only loads the new models and
supplies crop-derived morphology, because B1/B2a have no shape head.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
# FROZEN definitions, imported unchanged:
from scripts.evaluate_continuous_gap_decomposition import (  # noqa: E402
    _metrics, _resample, _affine_fit, _assemble_record, ARMS, DESC_ARMS, WIN)
from src.models.continuous.ce_net import CENet  # noqa: E402
import scripts.train_continuous_ecg_pilot as TR  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso")


def _windows(arm: str, fold: int, split: str, device) -> list[dict]:
    assert split != "test", "test split is prohibited"
    run = ROOT / "experiments_mmecg/q1_continuous_ecg_b0_b2_pilot" / arm / f"fold_{fold:02d}"
    model = CENet(arm).to(device)
    ck = torch.load(run / "checkpoints" / "best_composite.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    med, iqr = TR.fit_scaler(fold)                      # TRAIN-fold only, frozen rule

    wp = DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5"
    with h5py.File(wp, "r") as w:
        a = dict(w.attrs)
        assert a["support"] == "full_window_samples_0_1600" and float(a["supported_fraction_of_target"]) == 1.0
        wsst = np.asarray(w["wsst"][:])
    with h5py.File(DATA / f"fold_{fold:02d}" / f"{split}.h5", "r") as h:
        ecg = np.asarray(h["ecg"][:, 0], np.float64)
        rec = [x.decode() if isinstance(x, bytes) else str(x) for x in h["record_id"][:]]
        start = np.asarray(h["start_idx"][:], np.int64)
        rpk = [np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64))) for i in range(len(ecg))]

    out = []
    with torch.no_grad():
        for lo in range(0, len(ecg), 32):
            hi = min(lo + 32, len(ecg))
            x = torch.from_numpy(wsst[lo:hi].astype(np.float32)).to(device)
            y, ev = model(x)
            # predictions are in scaled space -> back to RAW ECG units (deployable inverse)
            yr = y.cpu().numpy()[:, 0].astype(np.float64) * iqr + med
            rm = torch.sigmoid(ev).cpu().numpy()[:, 0].astype(np.float64)
            for k, i in enumerate(range(lo, hi)):
                rp = rpk[i]; rp = rp[(rp >= 0) & (rp < WIN)]
                # CROP-DERIVED morphology: the model has no shape head (see spec)
                shape = np.zeros(200)
                if len(rp) >= 2:
                    pr = np.stack([rp[:-1], rp[1:]], 1)
                    j = int(np.argmin(np.abs(pr.mean(1) - WIN // 2)))
                    a0, b0 = int(pr[j, 0]), int(pr[j, 1])
                    if b0 > a0 + 10:
                        shape = _resample(yr[k, a0:b0], 200)
                out.append({"record": rec[i], "start": int(start[i]), "ecg": ecg[i],
                            "rpeaks_local": rp, "full": yr[k], "shape": shape, "rmask": rm[k]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="B1,B2a")
    ap.add_argument("--folds", default="1,5,9")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    rows = []
    for arm in a.arms.split(","):
        for fold in [int(f) for f in a.folds.split(",")]:
            tr = _windows(arm, fold, "train", dev)
            # frozen affine rule, fitted on TRAIN windows only
            from scripts.evaluate_continuous_gap_decomposition import _fit_fixed_affine
            full_aff, shape_aff = _fit_fixed_affine(tr); del tr
            va = _windows(arm, fold, "val", dev)
            by = defaultdict(list)
            for w in va: by[w["record"]].append(w)
            for recid, ws in sorted(by.items()):
                R = _assemble_record(ws, full_aff, shape_aff)     # FROZEN assembly
                gt = R["gt"]
                valid = {x: ~np.isnan(R["arms"][x]) & ~np.isnan(gt) for x in ARMS}
                common = np.logical_and.reduce([valid[x] for x in ARMS])
                row = {"model": arm, "fold": fold, "record": recid,
                       "total_samples": R["total"], "common_support_ratio": float(common.sum()/max(R["total"],1)),
                       "common_support_samples": int(common.sum()),
                       "n_cycles_matched": R["n_cycles_matched"], "n_cycles_total": R["n_cycles_total"]}
                for x in ARMS:
                    nat = valid[x]
                    row[f"{x}_native_cov"] = float(nat.sum()/max(R["total"],1))
                    for k, v in _metrics(R["arms"][x][nat], gt[nat]).items(): row[f"{x}_native_{k}"] = v
                    for k, v in _metrics(R["arms"][x][common], gt[common]).items(): row[f"{x}_common_{k}"] = v
                for x in DESC_ARMS:
                    nat = ~np.isnan(R["arms"][x]) & ~np.isnan(gt)
                    row[f"{x}_native_cov"] = float(nat.sum()/max(R["total"],1))
                    for k, v in _metrics(R["arms"][x][nat], gt[nat]).items(): row[f"{x}_native_{k}"] = v
                rows.append(row)
            print(f"  {arm} fold{fold}: {len(by)} records, common support "
                  f"{np.mean([r['common_support_ratio'] for r in rows if r['model']==arm and r['fold']==fold]):.3f}")
    out = ROOT / "experiments_mmecg/q1_cgd_reeval_fullsupport"; out.mkdir(parents=True, exist_ok=True)
    (out / "reeval_metrics.json").write_text(json.dumps({
        "spec": "experiments/specs/q1_cgd_reeval_fullsupport.json",
        "tier": "pilot", "publishable": False, "split": "validation", "dataset": "MMECG",
        "morphology_source": "CROP-DERIVED from the full-window output (B1/B2a have no shape head)",
        "isolation": "NOT comparable with q1_continuous_gap_decomposition_audit (different morphology source and input support)",
        "records": rows}, indent=2) + "\n")
    print(f"wrote {out/'reeval_metrics.json'} ({len(rows)} records)")


if __name__ == "__main__":
    main()
