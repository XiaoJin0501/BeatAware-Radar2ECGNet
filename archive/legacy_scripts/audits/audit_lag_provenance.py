#!/usr/bin/env python3
"""Lag provenance audit on FROZEN checkpoints. No training, no new modules.

Measures four SIGNED lags independently of the oracle-PCC search:
  L1 radar anchor  -> GT      (physical electromechanical delay)
  L2 event head    -> GT      (model's own timing readout)
  L3 waveform peak -> GT      (timing implied by the reconstruction)
  L4 waveform      -> event head (INTERNAL consistency; needs no ground truth)

L4 is the deployable candidate: if the waveform is shifted relative to the event
head, it can be corrected at test time without any GT.

Also: cross-correlation peak ambiguity, start/end index sanity, HR and state
relationships, and a stratified correction comparison
(none / fixed-global / event-head-derived / oracle).
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
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
FS, WIN, MAXLAG = 200, 1600, 120        # +/- 600 ms


def med_iqr_mad(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0: return dict(n=0)
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=len(x), median=float(m), iqr=float(q3-q1),
                mad=float(np.median(np.abs(x-m))), p05=float(np.percentile(x,5)),
                p95=float(np.percentile(x,95)))


def peak_offset(pred_pk, gt_pk, tol=MAXLAG):
    """Signed median (pred - nearest gt) in samples. Independent of any PCC search."""
    if len(pred_pk) == 0 or len(gt_pk) == 0: return np.nan
    d = []
    for p in pred_pk:
        k = np.argmin(np.abs(gt_pk - p))
        if abs(gt_pk[k] - p) <= tol: d.append(p - gt_pk[k])
    return float(np.median(d)) if d else np.nan


def mech_envelope(rcg):
    b = int(np.argmax(rcg.var(axis=1)))
    x = rcg[b].astype(np.float64)
    e = np.abs(np.diff(x, prepend=x[0])); return e - e.mean()


def xcorr_lag_amb(a, b, maxlag=MAXLAG):
    """Best lag plus an AMBIGUITY ratio: |second peak| / |best peak|, where the
    second peak is taken outside +/-150 ms of the best. Near 1 = ambiguous."""
    ac, bc = a - a.mean(), b - b.mean()
    n = len(a); den = np.sqrt((ac*ac).sum()*(bc*bc).sum())
    if den < 1e-12: return np.nan, np.nan
    c = np.correlate(ac, bc, mode="full")/den
    lags = np.arange(-(n-1), n); m = np.abs(lags) <= maxlag
    cc, ll = c[m], lags[m]
    i = int(np.argmax(np.abs(cc))); best = ll[i]
    far = np.abs(ll - best) > 30
    amb = float(np.max(np.abs(cc[far]))/max(abs(cc[i]), 1e-12)) if far.any() else np.nan
    return float(best), amb


def pcc_at(p, y, lag):
    if lag > 0:   p2, y2 = p[lag:], y[:len(y)-lag]
    elif lag < 0: p2, y2 = p[:len(p)+lag], y[-lag:]
    else:         p2, y2 = p, y
    if len(p2) < 10: return np.nan
    a, b = p2-p2.mean(), y2-y2.mean()
    d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 1e-12 else np.nan


def collect(fold, dev):
    run = ROOT/"experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"/f"fold_{fold:02d}"
    m = CENet("B1").to(dev)
    ck = torch.load(run/"checkpoints/best_composite.pt", map_location=dev, weights_only=True)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    med, iqr = TR.fit_scaler(fold)
    out = []
    for split in ("train", "val"):
        with h5py.File(DATA/f"fold_{fold:02d}/{split}_wsst8s.h5") as w: X = np.asarray(w["wsst"][:])
        with h5py.File(DATA/f"fold_{fold:02d}/{split}.h5") as h:
            Y = np.asarray(h["ecg"][:,0], np.float64); RC = h["rcg"]
            rec=[x.decode() if isinstance(x,bytes) else str(x) for x in h["record_id"][:]]
            ph =[x.decode() if isinstance(x,bytes) else str(x) for x in h["physistatus"][:]]
            sid=np.asarray(h["subject_id"][:]); st=np.asarray(h["start_idx"][:]); en=np.asarray(h["end_idx"][:])
            rpk=[np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64))) for i in range(len(Y))]
            for lo in range(0, len(Y), 32):
                hi=min(lo+32,len(Y))
                x=torch.from_numpy(X[lo:hi].astype(np.float32)).to(dev)
                with torch.no_grad(): yv, ev = m(x)
                P=yv.cpu().numpy()[:,0].astype(np.float64)*iqr+med
                E=torch.sigmoid(ev).cpu().numpy()[:,0].astype(np.float64)
                rc=np.asarray(RC[lo:hi], np.float32)
                for k,i in enumerate(range(lo,hi)):
                    g=rpk[i]; g=g[(g>=0)&(g<WIN)].astype(float)
                    if len(g)<2: continue
                    wp=np.asarray(detect_rpeaks(P[k], fs=FS), float)
                    epk=np.asarray(peaks_from_heatmap(E[k]), float)
                    rl,ra=xcorr_lag_amb(mech_envelope(rc[k]),
                                        np.histogram(g, bins=WIN, range=(0,WIN))[0].astype(float))
                    hr=60.0*FS/np.mean(np.diff(g)) if len(g)>=2 else np.nan
                    out.append(dict(split=split, fold=fold, subject=int(sid[i]), record=rec[i],
                        state=ph[i], start=int(st[i]), end=int(en[i]), hr=float(hr),
                        L1_radar_gt=rl*1000/FS if np.isfinite(rl) else np.nan, L1_amb=ra,
                        L2_event_gt=peak_offset(epk,g)*1000/FS if len(epk) else np.nan,
                        L3_wave_gt=peak_offset(wp,g)*1000/FS if len(wp) else np.nan,
                        L4_wave_event=peak_offset(wp,epk)*1000/FS if len(wp) and len(epk) else np.nan,
                        pcc0=pcc_at(P[k],Y[i],0), win=i))
    return out, None


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TR._set_split_root("loso_v2")
    allw=[]
    for f in (1,5,9):
        w,_=collect(f,dev); allw+=w
        print(f"  collected fold {f}: {len(w)} windows")
    Path("experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/lag_provenance.json").write_text(
        json.dumps(allw, indent=1)+"\n")
    print(f"\nwrote lag_provenance.json ({len(allw)} windows)")


if __name__ == "__main__":
    main()
