#!/usr/bin/env python3
"""Train/evaluate one arm of q1_continuous_ecg_b0_b2_pilot. VALIDATION ONLY.

Spec: experiments/specs/q1_continuous_ecg_b0_b2_pilot.json
Contract: docs/CONTINUOUS_ECG_TASK_CONTRACT.md

Hard invariants enforced at startup (abort, never degrade silently):
  * the consumed WSST declares full-window support;
  * no test file is ever opened;
  * checkpoint selection uses the frozen full-waveform composite, never Shape-200;
  * B1/B2* use a linear head against a TRAIN-fold robust scaler.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import h5py, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.ce_net import CENet, param_report, WIN  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402
from scripts.train_anchor_head import match_counts, peaks_from_heatmap  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
FS = 200
BUDGET = dict(epochs=80, patience=20, lr=1e-4, weight_decay=1e-4, batch_size=16, seed=42)
LEGACY_TARGET_ARMS = ("B0",)          # per-window min-max + sigmoid
COMPOSITE_W = dict(pcc=0.40, nrmse=0.25, r2=0.20, f1=0.15)


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Fold(Dataset):
    def __init__(self, fold: int, split: str, arm: str, scaler: tuple[float, float] | None, limit=None):
        assert split != "test", "test split is prohibited in this pilot"
        self.h5p = DATA / f"fold_{fold:02d}" / f"{split}.h5"
        self.wp = DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5"
        with h5py.File(self.wp, "r") as w:
            a = dict(w.attrs)
            assert a["support"] == "full_window_samples_0_1600", f"{self.wp} is not full-support"
            assert float(a["supported_fraction_of_target"]) == 1.0
            n = w["wsst"].shape[0]
        self.n = min(limit, n) if limit else n
        self.arm, self.scaler, self.needs_rcg = arm, scaler, (arm == "B2b")
        self._h = self._w = None

    def __len__(self): return self.n

    def _open(self):
        if self._h is None:
            self._h = h5py.File(self.h5p, "r"); self._w = h5py.File(self.wp, "r")

    def __getitem__(self, i):
        self._open()
        wsst = np.asarray(self._w["wsst"][i], np.float32)
        ecg = np.asarray(self._h["ecg"][i, 0], np.float32)
        if self.arm in LEGACY_TARGET_ARMS:                      # legacy per-window min-max
            lo, hi = float(ecg.min()), float(ecg.max())
            y = (ecg - lo) / max(hi - lo, 1e-8)
            inv = np.float32([max(hi - lo, 1e-8), lo])          # ORACLE inverse (uses GT min/max)
        else:                                                   # train-fold global robust scaler
            med, iqr = self.scaler
            y = (ecg - med) / iqr
            inv = np.float32([iqr, med])                        # DEPLOYABLE inverse (train constants)
        out = {"wsst": torch.from_numpy(wsst), "y": torch.from_numpy(y[None]),
               "raw": torch.from_numpy(ecg[None]), "inv": torch.from_numpy(inv)}
        if self.needs_rcg:
            out["rcg"] = torch.from_numpy(np.asarray(self._h["rcg"][i], np.float32))
        rp = np.sort(np.unique(np.asarray(self._h["rpeak_indices"][i], np.int64)))
        rp = rp[(rp >= 0) & (rp < WIN)]
        out["n_rpeaks"] = len(rp)
        out["rpeaks"] = torch.from_numpy(np.pad(rp, (0, 32 - len(rp)), constant_values=-1)[:32])
        return out


def fit_scaler(fold: int, cap: int = 800) -> tuple[float, float]:
    """Median / IQR from TRAIN fold ECG only. Never validation, never test."""
    with h5py.File(DATA / f"fold_{fold:02d}" / "train.h5", "r") as h:
        n = h["ecg"].shape[0]
        idx = np.linspace(0, n - 1, min(cap, n)).astype(int)
        x = np.concatenate([np.asarray(h["ecg"][int(i), 0], np.float64) for i in idx])
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return float(med), float(max(q3 - q1, 1e-6))


def waveform_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    rows = []
    for p, y in zip(pred, gt):
        e = p - y
        pc, yc = p - p.mean(), y - y.mean()
        den = np.sqrt((pc ** 2).sum() * (yc ** 2).sum())
        pcc = float((pc * yc).sum() / den) if den > 1e-12 else np.nan
        rmse = float(np.sqrt((e ** 2).mean()))
        q1, q3 = np.percentile(y, [25, 75]); iqr = max(q3 - q1, 1e-8)
        r2 = 1.0 - (e ** 2).sum() / max((yc ** 2).sum(), 1e-12)
        rows.append((pcc, rmse, rmse / iqr, float(np.abs(e).mean()), r2))
    a = np.asarray(rows, float)
    for i, k in enumerate(("pcc", "rmse", "nrmse", "mae", "r2")):
        out[k] = float(np.nanmean(a[:, i]))
    return out


def rpeak_f1(pred: np.ndarray, gts: list[np.ndarray], tol_ms=(50, 100, 150)) -> dict:
    c = {t: [0, 0, 0] for t in tol_ms}
    for p, g in zip(pred, gts):
        pk = detect_rpeaks(p, fs=FS)          # WAVEFORM-DERIVED readout
        for t in tol_ms:
            tp, fp, fn = match_counts(pk, g, int(round(t * FS / 1000)))
            c[t][0] += tp; c[t][1] += fp; c[t][2] += fn
    out = {}
    for t, (tp, fp, fn) in c.items():
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"f1_{t}ms"] = 2 * pr * rc / max(pr + rc, 1e-8)
    return out


def composite(m: dict) -> float:
    return (COMPOSITE_W["pcc"] * m["pcc"] + COMPOSITE_W["nrmse"] * (1 - min(m["nrmse"], 1.0))
            + COMPOSITE_W["r2"] * max(m["r2"], 0.0) + COMPOSITE_W["f1"] * m["f1_100ms"])


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    """All arms are scored in RAW ECG UNITS so metrics are comparable across arms.

    Each arm's prediction is mapped back through its own inverse transform. B1/B2*
    use train-fitted constants (deployable). B0 has NO deployable inverse -- its
    per-window min-max target is defined by GT min/max -- so B0 is scored with an
    ORACLE inverse and flagged. That handicap is exactly what B1 is meant to remove,
    so it must be visible rather than hidden inside incomparable NRMSE units.
    """
    model.eval(); P, G, RP = [], [], []
    for b in loader:
        kw = {"rcg": b["rcg"].to(dev)} if "rcg" in b else {}
        y, _ = model(b["wsst"].to(dev), **kw)
        inv = b["inv"].numpy()                                   # (B,2): [scale, offset]
        pred_raw = y.cpu().numpy()[:, 0] * inv[:, 0:1] + inv[:, 1:2]
        P.append(pred_raw); G.append(b["raw"].numpy()[:, 0])
        for j in range(len(b["raw"])):
            r = b["rpeaks"][j].numpy(); RP.append(r[r >= 0].astype(float))
    P, G = np.concatenate(P), np.concatenate(G)
    m = waveform_metrics(P, G); m.update(rpeak_f1(P, RP)); m["composite"] = composite(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["B0", "B1", "B2a", "B2a-wide", "B2b"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=BUDGET["epochs"])
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    ap.add_argument("--exp_tag", default="q1_continuous_ecg_b0_b2_pilot")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    set_seed(BUDGET["seed"] + a.fold)
    dev = torch.device(a.device)

    scaler = None if a.arm in LEGACY_TARGET_ARMS else fit_scaler(a.fold)
    tr = Fold(a.fold, "train", a.arm, scaler, a.limit)
    va = Fold(a.fold, "val", a.arm, scaler, a.limit)
    bs = BUDGET["batch_size"]
    ltr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)
    lva = DataLoader(va, batch_size=bs, shuffle=False, num_workers=2)

    model = CENet(a.arm).to(dev)
    pr = param_report(model)
    opt = torch.optim.AdamW(model.parameters(), lr=BUDGET["lr"], weight_decay=BUDGET["weight_decay"])
    out = ROOT / "experiments_mmecg" / (a.exp_tag + ("_smoke" if a.smoke else "")) / a.arm / f"fold_{a.fold:02d}"
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"arm": a.arm, "fold": a.fold, "split_used": ["train", "val"], "test_accessed": False,
         "scaler_median_iqr": scaler, "budget": BUDGET, "params": pr,
         "composite_weights": COMPOSITE_W, "rpeak_readout": "waveform-derived",
         "metric_space": "raw ECG units for ALL arms (per-arm inverse applied)",
         "b0_inverse_is_oracle": a.arm in LEGACY_TARGET_ARMS,
         "wsst": "full_window_samples_0_1600", "smoke": a.smoke}, indent=2, default=str) + "\n")

    best, best_ep, bad, hist = -1e9, -1, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = nb = 0
        t0 = time.time()
        for b in ltr:
            kw = {"rcg": b["rcg"].to(dev)} if "rcg" in b else {}
            y, _ = model(b["wsst"].to(dev), **kw)
            loss = F.huber_loss(y, b["y"].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        m = evaluate(model, lva, dev)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
        print(f"  ep{ep:3d} loss {tot/max(nb,1):.4f} | val PCC {m['pcc']:.4f} NRMSE {m['nrmse']:.4f} "
              f"R2 {m['r2']:+.3f} F1@100 {m['f1_100ms']:.3f} | S {m['composite']:.4f} ({time.time()-t0:.0f}s)")
        if m["composite"] > best:
            best, best_ep, bad = m["composite"], ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "composite": best},
                       out / "checkpoints" / "best_composite.pt")
        else:
            bad += 1
            if bad >= BUDGET["patience"]:
                print(f"  early stop at ep{ep}"); break
    (out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    (out / "best.json").write_text(json.dumps(
        {"arm": a.arm, "fold": a.fold, "best_epoch": best_ep, "best_composite": best,
         "selected_by": "full-waveform validation composite (no Shape-200)",
         "metrics": hist[best_ep - 1] if best_ep > 0 else None}, indent=2) + "\n")
    print(f"[done] {a.arm} fold{a.fold}: best S={best:.4f} @ep{best_ep} -> {out}")


if __name__ == "__main__":
    main()
