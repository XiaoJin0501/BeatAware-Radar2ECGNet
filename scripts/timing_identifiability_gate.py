#!/usr/bin/env python3
"""Timing-identifiability gate: T0b (WSST) vs T1 (phase-preserving native-rate rcg).

Spec: experiments/specs/q1_timing_identifiability_gate.json
R-peak heatmap ONLY. No ECG reconstruction. Validation ECG is used solely to score.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data.mmecg_dataset import _gaussian_mask       # noqa: E402
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
FS, WIN = 200, 1600
BUDGET = dict(epochs=80, patience=20, lr=1e-4, wd=1e-4, bs=16, seed=42)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class _Blk(nn.Module):
    def __init__(self, c, d, k=9):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d, bias=False), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, k, padding=d*(k-1)//2, dilation=d, bias=False), nn.BatchNorm1d(c), nn.GELU())
    def forward(self, x): return x + self.f(x)


class T1Net(nn.Module):
    """Native-rate, phase-preserving. Stride 1 throughout: time resolution never reduced."""
    def __init__(self, in_ch=50, c=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, c, 9, padding=4, bias=False), nn.BatchNorm1d(c), nn.GELU())
        self.body = nn.Sequential(*[_Blk(c, d) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Conv1d(c, 1, 9, padding=4)
    def forward(self, x): return self.head(self.body(self.stem(x)))


class T0bNet(nn.Module):
    """Same head and depth, but consuming WSST magnitude -> must upsample 240 -> 1600."""
    def __init__(self, in_bins=50, c=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_bins, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.body = nn.Sequential(*[_Blk(c, d) for d in (1, 2, 4, 8, 16, 32, 64)])
        self.head = nn.Conv1d(c, 1, 9, padding=4)
    def forward(self, x):
        h = self.pool(self.stem(x)).squeeze(2)
        h = F.interpolate(h, size=WIN, mode="linear", align_corners=False)
        return self.head(self.body(h))


class DS(Dataset):
    def __init__(self, fold, split, arm):
        assert split != "test"
        self.arm = arm
        with h5py.File(DATA/f"fold_{fold:02d}/{split}.h5") as h:
            self.rcg = np.asarray(h["rcg"][:], np.float32) if arm == "T1" else None
            self.rpk = [np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64)))
                        for i in range(h["ecg"].shape[0])]
            self.n = h["ecg"].shape[0]
            self.rec = [x.decode() if isinstance(x, bytes) else str(x) for x in h["record_id"][:]]
            self.sid = int(h["subject_id"][0])
        if arm != "T1":
            with h5py.File(DATA/f"fold_{fold:02d}/{split}_wsst8s.h5") as w:
                self.wsst = np.asarray(w["wsst"][:])
    def __len__(self): return self.n
    def __getitem__(self, i):
        rp = self.rpk[i]; rp = rp[(rp >= 0) & (rp < WIN)]
        y = torch.from_numpy(_gaussian_mask(rp, WIN, 4.0)[None].astype(np.float32))
        x = torch.from_numpy(self.rcg[i]) if self.arm == "T1" else torch.from_numpy(self.wsst[i].astype(np.float32))
        return {"x": x, "y": y, "idx": i}


def peak_stats(pred_logits, rpk_list, idxs):
    """F1 at three tolerances plus signed/absolute error and failure rate."""
    cnt = {t: [0, 0, 0] for t in (25, 50, 100)}
    signed, fails = [], 0
    for lg, i in zip(pred_logits, idxs):
        pk = np.asarray(peaks_from_heatmap(1/(1+np.exp(-lg))), float)
        g = rpk_list[i]; g = g[(g >= 0) & (g < WIN)].astype(float)
        if len(g) == 0: continue
        if len(pk) == 0: fails += 1
        matched = 0
        for t in cnt:
            tol = t*FS/1000; used = set(); tp = 0
            for gg in g:
                if len(pk):
                    d = np.abs(pk-gg); c = [j for j in np.argsort(d) if j not in used and d[j] <= tol]
                else: c = []
                if c: used.add(c[0]); tp += 1
            cnt[t][0] += tp; cnt[t][1] += max(len(pk)-len(used), 0); cnt[t][2] += len(g)-tp
            if t == 50: matched = tp
        if matched == 0: fails += 1 if len(pk) else 0
        for p in pk:
            k = int(np.argmin(np.abs(g-p)))
            if abs(g[k]-p) <= 100*FS/1000: signed.append((p-g[k])*1000/FS)
    out = {}
    for t, (tp, fp, fn) in cnt.items():
        pr = tp/max(tp+fp, 1); rc = tp/max(tp+fn, 1)
        out[f"f1_{t}"] = 2*pr*rc/max(pr+rc, 1e-8)
    s = np.asarray(signed) if signed else np.array([np.nan])
    out.update(signed_median=float(np.median(s)), mae=float(np.mean(np.abs(s))),
               iqr=float(np.subtract(*np.percentile(np.abs(s), [75, 25]))) if len(s) > 1 else np.nan,
               failure_rate=fails/max(len(idxs), 1))
    return out


def run(arm, fold, dev, epochs):
    set_seed(BUDGET["seed"]+fold)
    tr, va = DS(fold, "train", arm), DS(fold, "val", arm)
    g = torch.Generator(); g.manual_seed(BUDGET["seed"]+fold)
    ltr = DataLoader(tr, batch_size=BUDGET["bs"], shuffle=True, drop_last=True, generator=g)
    lva = DataLoader(va, batch_size=BUDGET["bs"], shuffle=False)
    net = (T1Net() if arm == "T1" else T0bNet()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=BUDGET["lr"], weight_decay=BUDGET["wd"])
    best, bad, bstate, bstats = -1, 0, None, None
    ep_times = []
    for ep in range(1, epochs+1):
        _t_ep = time.time()
        net.train()
        for b in ltr:
            loss = F.binary_cross_entropy_with_logits(net(b["x"].to(dev)), b["y"].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); L, I = [], []
        with torch.no_grad():
            for b in lva:
                L.append(net(b["x"].to(dev)).cpu().numpy()[:, 0]); I += b["idx"].tolist()
        st = peak_stats(np.concatenate(L), va.rpk, I)
        ep_times.append(time.time()-_t_ep)
        if ep <= 3 or ep % 10 == 0:
            print(f"      {arm} fold{fold} ep{ep:3d}: F1@50 {st['f1_50']:.3f}  "
                  f"[{ep_times[-1]:.1f}s/epoch]", flush=True)
        if st["f1_50"] > best:
            best, bad = st["f1_50"], 0
            bstats = dict(st, epoch=ep); bstate = {k: v.cpu() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= BUDGET["patience"]: break
    return bstats, va.sid, sum(p.numel() for p in net.parameters()), float(np.mean(ep_times)), len(ep_times)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=BUDGET["epochs"])
    ap.add_argument("--smoke", action="store_true",
                    help="reduced run; ALWAYS writes to the _smoke tree, never the formal one")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    res = defaultdict(dict)
    for arm in ("T0b", "T1"):
        for fold in (1, 5, 9):
            t0 = time.time(); st, sid, npar, s_ep, n_ep = run(arm, fold, dev, a.epochs)
            res[arm][f"S{sid}"] = dict(st, params=npar, sec_per_epoch=s_ep, epochs_run=n_ep)
            print(f"  {arm} fold{fold} S{sid}: F1@50 {st['f1_50']:.3f} F1@25 {st['f1_25']:.3f} "
                  f"signed {st['signed_median']:+.1f}ms MAE {st['mae']:.1f}ms | "
                  f"{n_ep} epochs, {s_ep:.1f}s/epoch, RUN TOTAL {time.time()-t0:.0f}s")
    n_done = sum(len(v) for v in res.values())
    smoke = a.smoke or a.epochs < BUDGET["epochs"] or n_done < 6
    tag = "q1_timing_identifiability_gate" + ("_smoke" if smoke else "")
    out = ROOT/"experiments_mmecg"/tag; out.mkdir(parents=True, exist_ok=True)
    name = "results.json" if not smoke else f"results_partial_{n_done}of6_{a.epochs}ep.json"
    (out/name).write_text(json.dumps(res, indent=2)+"\n")
    if smoke:
        print(f"\n  [SMOKE/PARTIAL] {n_done}/6 runs, {a.epochs} epochs -> {out/name}")
        print("  The formal directory is untouched; this is NOT a gate result.")
    print("\n" + "="*74)
    print(f"{'subj':>6}{'arm':>6}{'F1@25':>8}{'F1@50':>8}{'F1@100':>8}{'signed':>9}{'MAE':>8}{'IQR':>8}{'fail':>7}")
    for s in res["T1"]:
        for arm in ("T0b", "T1"):
            r = res[arm][s]
            print(f"{s:>6}{arm:>6}{r['f1_25']:>8.3f}{r['f1_50']:>8.3f}{r['f1_100']:>8.3f}"
                  f"{r['signed_median']:>+9.1f}{r['mae']:>8.1f}{r['iqr']:>8.1f}{r['failure_rate']:>7.3f}")
    d = [res["T1"][s]["f1_50"]-res["T0b"][s]["f1_50"] for s in res["T1"]]
    mae_ok = all(res["T1"][s]["mae"] <= 20 for s in res["T1"])
    s2 = res["T1"].get("S2", {}).get("signed_median", 99)
    print("\n=============== GATE ===============")
    print(f"  all 3 subjects improve F1@50 : {all(x>0 for x in d)}   {[f'{x:+.3f}' for x in d]}")
    print(f"  mean dF1@50 >= 0.10          : {np.mean(d)>=0.10}   (mean {np.mean(d):+.3f})")
    print(f"  MAE <= 20 ms all subjects    : {mae_ok}   ({[round(res['T1'][s]['mae'],1) for s in res['T1']]})")
    print(f"  S2 systematic offset gone    : {abs(s2)<15}   (median signed {s2:+.1f} ms)")
    ok = all(x>0 for x in d) and np.mean(d)>=0.10 and mae_ok and abs(s2)<15
    print(f"\n  GATE: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
