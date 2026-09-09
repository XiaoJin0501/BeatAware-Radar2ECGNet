#!/usr/bin/env python3
"""Train/evaluate one arm of q1_continuous_ecg_b0_b2_pilot. VALIDATION ONLY.

Spec:     experiments/specs/q1_continuous_ecg_b0_b2_pilot.json
Contract: docs/CONTINUOUS_ECG_TASK_CONTRACT.md

Hard invariants (abort, never degrade silently):
  * consumed WSST must declare full-window support;
  * the test split is never opened;
  * checkpoint selection uses the frozen full-waveform composite, never Shape-200;
  * B1/B2* use a linear head against a TRAIN-fold robust scaler;
  * formal runs cannot use smoke parameters, and never overwrite an existing run.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet, param_report, WIN  # noqa: E402
from src.data.mmecg_dataset import _gaussian_mask  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402
from scripts.train_anchor_head import match_counts, peaks_from_heatmap  # noqa: E402

PROCESSED = Path("/home/qhh2237/Datasets/MMECG/processed")
# The original `loso` split is PROTOCOL-INVALID: only 2 distinct validation subjects
# across 11 folds (docs/MMECG_SPLIT_PROVENANCE_AUDIT.md). loso_v2 is the repaired split.
BROKEN_SPLIT_ROOTS = {"loso"}
DATA = PROCESSED / "loso_v2"
FS = 200
BUDGET = dict(epochs=80, patience=20, lr=1e-4, weight_decay=1e-4, batch_size=16, seed=42)
LEGACY_TARGET_ARMS = ("B0",)
COMPOSITE_W = dict(pcc=0.40, nrmse=0.25, r2=0.20, f1=0.15)
EVENT_W = 0.20                       # fixed, IDENTICAL across arms (not an ablated factor)
R_SIGMA, QRS_SIGMA = 4.0, 16.0
QRS_HALF = int(0.050 * FS)           # R +/- 50 ms
STT_LO, STT_HI = int(0.080 * FS), int(0.400 * FS)   # R+80 ms .. R+400 ms
REGION_DEF = {"qrs": "R +/- 50 ms", "stt": "R+80 ms .. R+400 ms"}


def effective_seed(fold: int) -> int:
    """Same for every arm within a fold, so arms are paired; varies across folds."""
    return BUDGET["seed"] + fold


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def _set_split_root(name: str) -> None:
    global DATA
    if name in BROKEN_SPLIT_ROOTS:
        raise SystemExit(
            f"Refusing split_root='{name}': it has only 2 distinct validation subjects across 11 folds "
            "and is marked PROTOCOL-INVALID (docs/MMECG_SPLIT_PROVENANCE_AUDIT.md). Use loso_v2.")
    DATA = PROCESSED / name
    if not DATA.exists():
        raise SystemExit(f"split_root not found: {DATA}")


def _val_sha(fold: int) -> str:
    """Hash of the validation file, recorded per run so duplicate-validation splits
    can never again go unnoticed."""
    import hashlib
    h = hashlib.sha256()
    with open(DATA / f"fold_{fold:02d}" / "val.h5", "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


class Fold(Dataset):
    """Fold data held IN RAM.

    The HDF5-backed version was the bottleneck: ~269 s/epoch at 0% GPU utilisation
    with DataLoader workers saturated on lzf decompression. A fold is only ~4 GiB as
    float16, against >100 GiB available, so the whole split is preloaded once and
    indexed from memory. With no decompression per item, `num_workers=0` is correct
    and avoids duplicating the array across forked workers.
    """

    def __init__(self, fold: int, split: str, arm: str, scaler: tuple[float, float] | None, limit=None):
        assert split != "test", "test split is prohibited in this pilot"
        self.h5p = DATA / f"fold_{fold:02d}" / f"{split}.h5"
        self.wp = DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5"
        self.arm, self.scaler, self.needs_rcg = arm, scaler, (arm == "B2b")
        with h5py.File(self.wp, "r") as w:
            a = dict(w.attrs)
            assert a["support"] == "full_window_samples_0_1600", f"{self.wp} is not full-support"
            assert float(a["supported_fraction_of_target"]) == 1.0
            n = w["wsst"].shape[0]
            self.n = min(limit, n) if limit else n
            t0 = time.time()
            self.wsst = np.asarray(w["wsst"][: self.n])            # float16 in RAM
        with h5py.File(self.h5p, "r") as h:
            self.ecg = np.asarray(h["ecg"][: self.n, 0], np.float32)
            self.rpk = [np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64)))
                        for i in range(self.n)]
            self.rcg = (np.asarray(h["rcg"][: self.n], np.float32) if self.needs_rcg else None)
        gib = (self.wsst.nbytes + self.ecg.nbytes + (self.rcg.nbytes if self.rcg is not None else 0)) / 2 ** 30
        print(f"    preloaded {split} fold{fold}: {self.n} windows, {gib:.2f} GiB, {time.time()-t0:.0f}s")

    def __len__(self): return self.n

    def __getitem__(self, i):
        wsst = self.wsst[i].astype(np.float32)
        ecg = self.ecg[i]
        if self.arm in LEGACY_TARGET_ARMS:
            lo, hi = float(ecg.min()), float(ecg.max())
            y = (ecg - lo) / max(hi - lo, 1e-8)
            inv = np.float32([max(hi - lo, 1e-8), lo])       # ORACLE inverse
        else:
            med, iqr = self.scaler
            y = (ecg - med) / iqr
            inv = np.float32([iqr, med])                     # DEPLOYABLE inverse
        rp = self.rpk[i]
        rp = rp[(rp >= 0) & (rp < WIN)]
        ev = np.stack([_gaussian_mask(rp, WIN, R_SIGMA), _gaussian_mask(rp, WIN, QRS_SIGMA)])
        out = {"wsst": torch.from_numpy(wsst), "y": torch.from_numpy(y[None]),
               "raw": torch.from_numpy(ecg[None]), "inv": torch.from_numpy(inv),
               "event": torch.from_numpy(ev.astype(np.float32)),
               "n_rpeaks": len(rp),
               "rpeaks": torch.from_numpy(np.pad(rp, (0, 40 - len(rp)), constant_values=-1)[:40])}
        if self.needs_rcg:
            out["rcg"] = torch.from_numpy(self.rcg[i])
        return out


def fit_scaler(fold: int) -> tuple[float, float]:
    """Median / IQR over the ENTIRE training fold. Never validation, never test."""
    with h5py.File(DATA / f"fold_{fold:02d}" / "train.h5", "r") as h:
        x = np.asarray(h["ecg"][:, 0], np.float64).reshape(-1)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max(q3 - q1, 1e-6))


def waveform_metrics(pred, gt) -> dict:
    rows = []
    for p, y in zip(pred, gt):
        e = p - y; pc, yc = p - p.mean(), y - y.mean()
        den = np.sqrt((pc ** 2).sum() * (yc ** 2).sum())
        rmse = float(np.sqrt((e ** 2).mean()))
        q1, q3 = np.percentile(y, [25, 75])
        rows.append((float((pc * yc).sum() / den) if den > 1e-12 else np.nan, rmse,
                     rmse / max(q3 - q1, 1e-8), float(np.abs(e).mean()),
                     1.0 - (e ** 2).sum() / max((yc ** 2).sum(), 1e-12)))
    a = np.asarray(rows, float)
    return {k: float(np.nanmean(a[:, i])) for i, k in
            enumerate(("pcc", "rmse", "nrmse", "mae", "r2"))}


def region_errors(pred, gt, rps) -> dict:
    qrs, stt = [], []
    for p, y, r in zip(pred, gt, rps):
        for idx in r.astype(int):
            a, b = max(0, idx - QRS_HALF), min(WIN, idx + QRS_HALF)
            if b > a: qrs.append(np.abs(p[a:b] - y[a:b]).mean())
            a2, b2 = min(WIN, idx + STT_LO), min(WIN, idx + STT_HI)
            if b2 > a2: stt.append(np.abs(p[a2:b2] - y[a2:b2]).mean())
    return {"qrs_region_mae": float(np.mean(qrs)) if qrs else float("nan"),
            "stt_region_mae": float(np.mean(stt)) if stt else float("nan")}


def _centre_cycle(rp):
    if len(rp) < 2: return None
    pr = np.stack([rp[:-1], rp[1:]], 1)
    j = int(np.argmin(np.abs(pr.mean(1) - WIN // 2)))
    a, b = int(pr[j, 0]), int(pr[j, 1])
    return (a, b) if b > a + 10 else None


def shape200_pcc(pred, gt, rps) -> float:
    """AUXILIARY morphology diagnostic. Never used for selection."""
    v = []
    for p, y, r in zip(pred, gt, rps):
        cb = _centre_cycle(r.astype(int))
        if cb is None: continue
        a, b = cb
        f = lambda z: np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, b - a), z[a:b])
        ps, ys = f(p), f(y)
        n = lambda z: (z - z.min()) / max(z.max() - z.min(), 1e-8)
        ps, ys = n(ps), n(ys)
        pc, yc = ps - ps.mean(), ys - ys.mean()
        den = np.sqrt((pc ** 2).sum() * (yc ** 2).sum())
        if den > 1e-12: v.append(float((pc * yc).sum() / den))
    return float(np.mean(v)) if v else float("nan")


def timing_metrics(pred, evmask, rps) -> dict:
    wc = {t: [0, 0, 0] for t in (50, 100, 150)}
    ec = {t: [0, 0, 0] for t in (50, 100, 150)}
    rr, hr = [], []
    for p, em, g in zip(pred, evmask, rps):
        wp = detect_rpeaks(p, fs=FS)                    # waveform-derived readout
        ep = peaks_from_heatmap(em)                     # event-head readout
        for t in wc:
            tol = int(round(t * FS / 1000))
            for store, pk in ((wc, wp), (ec, ep)):
                a, b, c = match_counts(pk, g, tol)
                store[t][0] += a; store[t][1] += b; store[t][2] += c
        if len(wp) >= 2 and len(g) >= 2:
            pr_rr = float(np.mean(np.diff(np.sort(wp)))) * 1000 / FS
            gt_rr = float(np.mean(np.diff(np.sort(g)))) * 1000 / FS
            rr.append(abs(pr_rr - gt_rr))
            hr.append(abs(60000 / max(pr_rr, 1e-6) - 60000 / max(gt_rr, 1e-6)))
    def f1(c):
        o = {}
        for t, (tp, fp, fn) in c.items():
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            o[t] = 2 * pr * rc / max(pr + rc, 1e-8)
        return o
    w, e = f1(wc), f1(ec)
    return {**{f"f1_{t}ms": w[t] for t in w},
            **{f"event_f1_{t}ms": e[t] for t in e},
            "rr_error_ms": float(np.mean(rr)) if rr else float("nan"),
            "hr_error_bpm": float(np.mean(hr)) if hr else float("nan"),
            "rr_coverage": len(rr) / max(len(pred), 1)}


def composite(m: dict) -> float:
    return (COMPOSITE_W["pcc"] * m["pcc"] + COMPOSITE_W["nrmse"] * (1 - min(m["nrmse"], 1.0))
            + COMPOSITE_W["r2"] * max(m["r2"], 0.0) + COMPOSITE_W["f1"] * m["f1_100ms"])


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    """All arms scored in RAW ECG units via each arm's own inverse, so metrics are
    comparable. B0's inverse is ORACLE (its target is defined by GT min/max)."""
    model.eval(); P, G, EV, RP = [], [], [], []
    for b in loader:
        kw = {"rcg": b["rcg"].to(dev)} if "rcg" in b else {}
        y, ev = model(b["wsst"].to(dev), **kw)
        inv = b["inv"].numpy()
        P.append(y.cpu().numpy()[:, 0] * inv[:, 0:1] + inv[:, 1:2])
        G.append(b["raw"].numpy()[:, 0])
        EV.append(torch.sigmoid(ev).cpu().numpy()[:, 0])
        for j in range(len(b["raw"])):
            r = b["rpeaks"][j].numpy(); RP.append(r[r >= 0].astype(float))
    P, G, EV = np.concatenate(P), np.concatenate(G), np.concatenate(EV)
    m = waveform_metrics(P, G)
    m.update(timing_metrics(P, EV, RP))
    m.update(region_errors(P, G, RP))
    m["shape200_pcc_auxiliary"] = shape200_pcc(P, G, RP)
    m["composite"] = composite(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["B0", "B1", "B2a", "B2a-wide", "B2b"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="reduced run; writes to a SEPARATE tree")
    ap.add_argument("--smoke_epochs", type=int, default=1)
    ap.add_argument("--smoke_limit", type=int, default=48)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split_root", default="loso_v2")
    ap.add_argument("--exp_tag", default=None,
                    help="output tree name; defaults to the pilot tag, or <tag>_smoke in smoke mode")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    # ---- formal / smoke parameter isolation: a formal run CANNOT be shrunk ----
    base_tag = a.exp_tag or "q1_continuous_ecg_b0_b2_pilot"
    if a.smoke:
        epochs, limit, tag = a.smoke_epochs, a.smoke_limit, base_tag + "_smoke"
    else:
        epochs, limit, tag = BUDGET["epochs"], None, base_tag

    out = ROOT / "experiments_mmecg" / tag / a.arm / f"fold_{a.fold:02d}"
    if (out / "best.json").exists() and not a.overwrite:
        raise SystemExit(f"Refusing to overwrite a completed run at {out}. Pass --overwrite to replace it.")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    _set_split_root(a.split_root)
    seed = effective_seed(a.fold)
    set_seed(seed)
    dev = torch.device(a.device)
    scaler = None if a.arm in LEGACY_TARGET_ARMS else fit_scaler(a.fold)
    tr, va = Fold(a.fold, "train", a.arm, scaler, limit), Fold(a.fold, "val", a.arm, scaler, limit)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=BUDGET["batch_size"], shuffle=True, num_workers=0,
                     drop_last=True, generator=g)
    lva = DataLoader(va, batch_size=BUDGET["batch_size"], shuffle=False, num_workers=0)

    model = CENet(a.arm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=BUDGET["lr"], weight_decay=BUDGET["weight_decay"])
    # The commit that last touched EXECUTABLE code, not bare HEAD: later
    # spec/registry-only commits must not appear to change the implementation.
    CODE_PATHS = ["src/", "scripts/", "tests/", "configs/"]
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--"] + CODE_PATHS, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = head = "unknown"
    (out / "config.json").write_text(json.dumps({
        "arm": a.arm, "fold": a.fold, "mode": "smoke" if a.smoke else "formal",
        "epochs": epochs, "limit": limit, "effective_seed": seed,
        "seed_rule": "BUDGET.seed + fold; identical across arms within a fold (paired)",
        "cudnn_deterministic": True, "split_used": ["train", "val"], "test_accessed": False,
        "input_pipeline": "RAM-preloaded per fold (HDF5 was the bottleneck at ~269 s/epoch, 0% GPU)",
        "num_workers": 0,
        "scaler_median_iqr": scaler, "scaler_source": "ENTIRE training fold",
        "budget": BUDGET, "params": param_report(model),
        "composite_weights": COMPOSITE_W, "event_loss_weight": EVENT_W,
        "region_definitions": REGION_DEF, "rpeak_readout": "waveform-derived",
        "metric_space": "raw ECG units for ALL arms (per-arm inverse applied)",
        "b0_inverse_is_oracle": a.arm in LEGACY_TARGET_ARMS,
        "wsst": "full_window_samples_0_1600",
        "split_root": a.split_root, "split_dir": str(DATA),
        "val_file_sha256": _val_sha(a.fold),
        "code_commit": commit, "code_commit_is": "last commit touching src/scripts/tests/configs",
        "runtime_head": head,
    }, indent=2, default=str) + "\n")

    best, best_ep, bad, hist = -1e9, -1, 0, []
    for ep in range(1, epochs + 1):
        model.train(); tot = nb = 0; t0 = time.time()
        for b in ltr:
            kw = {"rcg": b["rcg"].to(dev)} if "rcg" in b else {}
            y, ev = model(b["wsst"].to(dev), **kw)
            loss = (F.huber_loss(y, b["y"].to(dev))
                    + EVENT_W * F.binary_cross_entropy_with_logits(ev, b["event"].to(dev)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        m = evaluate(model, lva, dev)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
        print(f"  ep{ep:3d} loss {tot/max(nb,1):.4f} | PCC {m['pcc']:.4f} NRMSE {m['nrmse']:.4f} "
              f"R2 {m['r2']:+.3f} F1@100 {m['f1_100ms']:.3f} evF1 {m['event_f1_100ms']:.3f} "
              f"| S {m['composite']:.4f} ({time.time()-t0:.0f}s)")
        if m["composite"] > best:
            best, best_ep, bad = m["composite"], ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "composite": best, "arm": a.arm, "fold": a.fold},
                       out / "checkpoints" / "best_composite.pt")
        else:
            bad += 1
            if bad >= BUDGET["patience"]:
                print(f"  early stop at ep{ep}"); break
    (out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    (out / "best.json").write_text(json.dumps({
        "arm": a.arm, "fold": a.fold, "mode": "smoke" if a.smoke else "formal",
        "best_epoch": best_ep, "best_composite": best, "code_commit": commit,
        "runtime_head": head, "max_epochs": epochs, "patience": BUDGET["patience"],
        "hit_epoch_cap": best_ep == epochs, "stopped_at_epoch": len(hist),
        "early_stopped": len(hist) < epochs,
        "selected_by": "full-waveform validation composite (Shape-200 excluded)",
        "metrics": hist[best_ep - 1] if best_ep > 0 else None}, indent=2) + "\n")
    print(f"[done] {a.arm} fold{a.fold} ({'smoke' if a.smoke else 'formal'}): "
          f"best S={best:.4f} @ep{best_ep} -> {out}")


if __name__ == "__main__":
    main()
