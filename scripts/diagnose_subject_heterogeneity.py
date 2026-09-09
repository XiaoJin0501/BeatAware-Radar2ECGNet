#!/usr/bin/env python3
"""Per-record polarity / lag / amplitude diagnostics for the loso_v2 B1 baseline.

Reports every record separately. Deliberately computes NO cross-subject mean:
subjects differ so much that an average would describe none of them.
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet          # noqa: E402
import scripts.train_continuous_ecg_pilot as TR          # noqa: E402
from src.utils.metrics import detect_rpeaks             # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
MAXLAG = 200          # +/- 1.0 s at 200 Hz


def pcc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum()/d) if d > 1e-12 else np.nan


def best_lag(p, y, maxlag=MAXLAG):
    """PCC maximised over integer lag; returns (best_pcc, lag_samples)."""
    pc = p - p.mean(); yc = y - y.mean()
    n = len(p)
    c = np.correlate(pc, yc, mode="full")
    denom = np.sqrt((pc*pc).sum() * (yc*yc).sum())
    if denom < 1e-12: return np.nan, 0
    c = c/denom
    lags = np.arange(-(n-1), n)
    m = np.abs(lags) <= maxlag
    cc, ll = c[m], lags[m]
    i = int(np.argmax(np.abs(cc)))
    return float(cc[i]), int(ll[i])


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TR._set_split_root("loso_v2")
    out = {}
    for fold in (1, 5, 9):
        run = ROOT/"experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"/f"fold_{fold:02d}"
        m = CENet("B1").to(dev)
        ck = torch.load(run/"checkpoints/best_composite.pt", map_location=dev, weights_only=True)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        med, iqr = TR.fit_scaler(fold)
        with h5py.File(DATA/f"fold_{fold:02d}/val_wsst8s.h5") as w: X = np.asarray(w["wsst"][:])
        with h5py.File(DATA/f"fold_{fold:02d}/val.h5") as h:
            Y = np.asarray(h["ecg"][:,0], np.float64)
            rec = [x.decode() if isinstance(x,bytes) else str(x) for x in h["record_id"][:]]
            subj = int(h["subject_id"][0])
            phys = [x.decode() if isinstance(x,bytes) else str(x) for x in h["physistatus"][:]]
        P = []
        with torch.no_grad():
            for lo in range(0, len(Y), 32):
                x = torch.from_numpy(X[lo:lo+32].astype(np.float32)).to(dev)
                P.append(m(x)[0].cpu().numpy()[:,0].astype(np.float64)*iqr + med)
        P = np.concatenate(P)
        byrec = defaultdict(list)
        for i,r in enumerate(rec): byrec[r].append(i)
        print(f"\n===== fold {fold}  validation subject S{subj}  "
              f"{len(Y)} windows / {len(byrec)} records =====")
        print(f"{'record':<14}{'state':<6}{'n':>5}{'PCC@0':>9}{'bestPCC':>9}{'lag_ms':>8}"
              f"{'polarity':>10}{'var_ratio':>10}{'F1@100':>8}")
        recs=[]
        for r, idx in sorted(byrec.items()):
            p = P[idx].ravel(); y = Y[idx].ravel()
            z = pcc(p, y); bp, bl = best_lag(p, y)
            neg = float(np.mean([pcc(P[i], Y[i]) < 0 for i in idx]))
            vr = float(np.mean([P[i].var()/max(Y[i].var(),1e-12) for i in idx]))
            tp=fp=fn=0
            for i in idx:
                gt = detect_rpeaks(Y[i], fs=200); pk = detect_rpeaks(P[i], fs=200)
                gt=np.asarray(gt); pk=np.asarray(pk)
                used=set()
                for g in gt:
                    d=np.abs(pk-g) if len(pk) else np.array([])
                    ok=[j for j in np.argsort(d) if j not in used and d[j]<=20] if len(d) else []
                    if ok: used.add(ok[0]); tp+=1
                    else: fn+=1
                fp += max(len(pk)-len(used),0)
            pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f1=2*pr*rc/max(pr+rc,1e-8)
            print(f"{r:<14}{phys[idx[0]]:<6}{len(idx):>5}{z:>9.3f}{bp:>9.3f}{bl*5:>8.0f}"
                  f"{neg:>10.2f}{vr:>10.3f}{f1:>8.3f}")
            recs.append(dict(record=r, state=phys[idx[0]], n=len(idx), pcc0=z, best_pcc=bp,
                             lag_ms=bl*5, frac_windows_negative=neg, var_ratio=vr, f1_100=f1))
        out[f"fold{fold}_S{subj}"] = recs
    Path("experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/subject_diagnostics.json"
         ).write_text(json.dumps(out, indent=2)+"\n")
    print("\nwrote subject_diagnostics.json  (no cross-subject mean computed, by design)")


if __name__ == "__main__":
    main()
