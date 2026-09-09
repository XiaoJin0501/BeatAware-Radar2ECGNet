#!/usr/bin/env python3
"""Radar-only lag-predictability probe on a FROZEN B1.

Spec: experiments/specs/q1_lag_predictability_probe.json

Validation-time prediction consumes ONLY radar (WSST, rcg) and B1's own outputs.
The validation ECG is opened solely to SCORE, never to produce a prediction.
Lag labels come exclusively from TRAIN B1-vs-TRAIN-GT.
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch
from scipy import signal as sps
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet          # noqa: E402
import scripts.train_continuous_ecg_pilot as TR          # noqa: E402
from src.utils.metrics import detect_rpeaks             # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
FS, WIN, LAGS = 200, 1600, np.arange(-100, 101)
ALPHAS = np.logspace(-3, 4, 20)


def shift(p, l):
    if l > 0: return np.concatenate([p[l:], np.repeat(p[-1], l)])
    if l < 0: return np.concatenate([np.repeat(p[0], -l), p[:l]])
    return p


def pcc(a, b):
    a = a-a.mean(); b = b-b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 1e-12 else np.nan


def metrics(p, y):
    e = p-y; q1, q3 = np.percentile(y, [25, 75]); rng = max(q3-q1, 1e-8)
    yc = y-y.mean()
    return dict(pcc=pcc(p, y), rmse=float(np.sqrt((e*e).mean())),
                nrmse=float(np.sqrt((e*e).mean())/rng),
                r2=float(1-(e*e).sum()/max((yc*yc).sum(), 1e-12)))


def f1_at(P, Y, tol_ms=50):
    tol = int(tol_ms*FS/1000); tp=fp=fn=0
    for p, y in zip(P, Y):
        g = np.asarray(detect_rpeaks(y, fs=FS), float); k = np.asarray(detect_rpeaks(p, fs=FS), float)
        used = set()
        for gg in g:
            if len(k):
                d = np.abs(k-gg); c = [j for j in np.argsort(d) if j not in used and d[j] <= tol]
            else: c = []
            if c: used.add(c[0]); tp += 1
            else: fn += 1
        fp += max(len(k)-len(used), 0)
    pr = tp/max(tp+fp, 1); rc = tp/max(tp+fn, 1)
    return 2*pr*rc/max(pr+rc, 1e-8)


def mech_env(rcg):
    b = int(np.argmax(rcg.var(axis=1))); x = rcg[b].astype(np.float64)
    e = np.abs(np.diff(x, prepend=x[0])); return e-e.mean(), b


def radar_features(rcg, pred):
    """RADAR-ONLY + B1's OWN OUTPUT. Never touches ECG."""
    env, b = mech_env(rcg)
    f, ps = sps.welch(rcg[b].astype(np.float64), fs=FS, nperseg=256)
    tot = max(ps.sum(), 1e-12)
    bands = [ps[(f >= lo) & (f < hi)].sum()/tot for lo, hi in ((0.8,3.5),(3.5,10),(10,20),(20,50))]
    ac = np.correlate(env, env, mode="full")[len(env)-1:]
    ac = ac/max(ac[0], 1e-12)
    seg = ac[int(0.4*FS):int(1.5*FS)]
    rr = (int(np.argmax(seg))+int(0.4*FS))/FS if len(seg) else np.nan
    pk = np.asarray(detect_rpeaks(pred, fs=FS), float)          # B1's own output
    tr = np.zeros(WIN); tr[pk.astype(int)[(pk >= 0) & (pk < WIN)]] = 1.0; tr -= tr.mean()
    c = np.correlate(env-env.mean(), tr, mode="full")
    ll = np.arange(-(WIN-1), WIN); m = np.abs(ll) <= 100
    r2p = float(ll[m][np.argmax(np.abs(c[m]))])                  # radar -> prediction lag
    hr = 60.0*FS/np.mean(np.diff(pk)) if len(pk) >= 2 else 0.0
    return np.array([b, float(rcg[b].var()), *bands, rr if np.isfinite(rr) else 0.0,
                     r2p, hr, float(env.std()), float(np.median(np.abs(env)))], float)


def run_fold(fold, dev):
    run = ROOT/"experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"/f"fold_{fold:02d}"
    m = CENet("B1").to(dev)
    ck = torch.load(run/"checkpoints/best_composite.pt", map_location=dev, weights_only=True)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)          # frozen, explicitly
    med, iqr = TR.fit_scaler(fold)
    D = {}
    for split in ("train", "val"):
        with h5py.File(DATA/f"fold_{fold:02d}/{split}_wsst8s.h5") as w: X = np.asarray(w["wsst"][:])
        with h5py.File(DATA/f"fold_{fold:02d}/{split}.h5") as h:
            Y = np.asarray(h["ecg"][:, 0], np.float64)
            RC = np.asarray(h["rcg"][:], np.float32)
            rec = [x.decode() if isinstance(x, bytes) else str(x) for x in h["record_id"][:]]
            sid = int(h["subject_id"][0])
        P, Z, F = [], [], []
        with torch.no_grad():
            for lo in range(0, len(Y), 32):
                hi = min(lo+32, len(Y))
                x = torch.from_numpy(X[lo:hi].astype(np.float32)).to(dev)
                lat = m.wsst_stem(x)                                   # frozen latent
                yv, _ = m(x)
                pr = yv.cpu().numpy()[:, 0].astype(np.float64)*iqr+med
                P.append(pr)
                Z.append(np.concatenate([lat.mean(-1).cpu().numpy(), lat.std(-1).cpu().numpy()], 1))
                for k in range(hi-lo): F.append(radar_features(RC[lo+k], pr[k]))
        D[split] = dict(P=np.concatenate(P), Y=Y, Z=np.concatenate(Z), F=np.asarray(F),
                        rec=rec, sid=sid)
    # ---- lag labels from TRAIN ONLY ----
    tr = D["train"]
    lab = np.array([LAGS[int(np.argmax([pcc(shift(tr["P"][i], l), tr["Y"][i]) for l in LAGS]))]
                    for i in range(len(tr["Y"]))], float)
    return m, D, lab


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TR._set_split_root("loso_v2")
    out = {}
    for fold in (1, 5, 9):
        _, D, lab = run_fold(fold, dev)
        tr, va = D["train"], D["val"]
        preds = {}
        preds["A1_train_median"] = np.full(len(va["Y"]), float(np.median(lab)))
        for name, key in (("A2_handcrafted", "F"), ("A3_frozen_feature_ridge", "Z")):
            Xtr = np.nan_to_num(tr[key]); Xva = np.nan_to_num(va[key])
            mu, sd = Xtr.mean(0), Xtr.std(0)+1e-8
            r = RidgeCV(alphas=ALPHAS).fit((Xtr-mu)/sd, lab)          # TRAIN labels only
            preds[name] = r.predict((Xva-mu)/sd)
        # oracle upper bounds (use val GT -> UPPER BOUND ONLY)
        owin = np.array([LAGS[int(np.argmax([pcc(shift(va["P"][i], l), va["Y"][i]) for l in LAGS]))]
                         for i in range(len(va["Y"]))], float)
        byrec = defaultdict(list)
        for i, r in enumerate(va["rec"]): byrec[r].append(i)
        orec = np.zeros(len(va["Y"]))
        for r, idx in byrec.items(): orec[idx] = np.median(owin[idx])
        arms = {"A0_zero": np.zeros(len(va["Y"])), **preds,
                "U1_oracle_record": orec, "U2_oracle_window": owin}
        res = {}
        for a, lv in arms.items():
            Pc = np.stack([shift(va["P"][i], int(round(lv[i]))) for i in range(len(va["Y"]))])
            mm = [metrics(Pc[i], va["Y"][i]) for i in range(len(va["Y"]))]
            res[a] = dict(lag_mae_ms=float(np.mean(np.abs(lv-owin))*1000/FS),
                          pcc=float(np.nanmean([x["pcc"] for x in mm])),
                          r2=float(np.nanmean([x["r2"] for x in mm])),
                          nrmse=float(np.nanmean([x["nrmse"] for x in mm])),
                          f1_50=float(f1_at(Pc, va["Y"], 50)))
        out[f"S{va['sid']}"] = res
        print(f"\n=== fold {fold}  validation subject S{va['sid']}  ({len(va['Y'])} windows) ===")
        print(f"{'arm':<26}{'lagMAE_ms':>10}{'PCC':>9}{'dPCC':>8}{'R2':>8}{'NRMSE':>8}{'F1@50':>8}")
        for a, r in res.items():
            print(f"{a:<26}{r['lag_mae_ms']:>10.1f}{r['pcc']:>9.3f}{r['pcc']-res['A0_zero']['pcc']:>+8.3f}"
                  f"{r['r2']:>8.3f}{r['nrmse']:>8.3f}{r['f1_50']:>8.3f}")
    Path("experiments_mmecg/q1_lag_predictability_probe").mkdir(parents=True, exist_ok=True)
    Path("experiments_mmecg/q1_lag_predictability_probe/results.json").write_text(json.dumps(out, indent=2)+"\n")
    print("\n================= GATE =================")
    for a in ("A1_train_median", "A2_handcrafted", "A3_frozen_feature_ridge"):
        d = [out[s][a]["pcc"]-out[s]["A0_zero"]["pcc"] for s in out]
        allup = all(x > 0 for x in d); mean = float(np.mean(d))
        r2ok = all(out[s][a]["r2"] >= out[s]["A0_zero"]["r2"] for s in out)
        nrok = all(out[s][a]["nrmse"] <= out[s]["A0_zero"]["nrmse"] for s in out)
        f1ok = all(out[s][a]["f1_50"] >= out[s]["A0_zero"]["f1_50"]-1e-9 for s in out)
        print(f"  {a:<26} dPCC {[f'{x:+.3f}' for x in d]}  mean {mean:+.3f}  "
              f"all_up={allup} R2ok={r2ok} NRMSEok={nrok} F1ok={f1ok}  "
              f"-> {'PASS' if (allup and mean>=0.08 and r2ok and nrok and f1ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
