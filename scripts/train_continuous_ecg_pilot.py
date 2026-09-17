#!/usr/bin/env python3
"""Train/evaluate one arm of q1_continuous_ecg_b0_b2_pilot. VALIDATION ONLY.

Spec:     experiments/specs/q1_continuous_ecg_b0_b2_pilot.json
Contract: docs/CONTINUOUS_ECG_TASK_CONTRACT.md

Hard invariants (abort, never degrade silently):
  * consumed WSST must declare full-window support;
  * the test split is never opened, except for the registered final-candidate
    mode after validation checkpoint selection has completed;
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
from src.models.continuous.ce_net import (  # noqa: E402
    CENet, param_report, reset_module_parameters_paired, WIN,
)
from src.data.mmecg_dataset import _gaussian_mask  # noqa: E402
from src.data.wsst8s_subject_pool import SubjectWSSTPool  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402
from src.losses.losses import MultiResolutionSTFTLoss  # noqa: E402
from src.utils.atomic_state import (  # noqa: E402
    atomic_json, atomic_npz, atomic_torch_save, capture_rng_state,
    restore_rng_state, transition_state,
)
from scripts.train_anchor_head import match_counts, peaks_from_heatmap  # noqa: E402

PROCESSED = Path("/home/qhh2237/Datasets/MMECG/processed")
SUBJECT_POOL = PROCESSED / "wsst8s_subject_pool"
# The original `loso` split is PROTOCOL-INVALID: only 2 distinct validation subjects
# across 11 folds (docs/MMECG_SPLIT_PROVENANCE_AUDIT.md). loso_v2 is the repaired split.
BROKEN_SPLIT_ROOTS = {"loso"}
DATA = PROCESSED / "loso_v2"
FS = 200
BUDGET = dict(epochs=100, patience=15, lr=1e-4, weight_decay=1e-4, batch_size=16, seed=42)
LEGACY_TARGET_ARMS = ("B0",)
COMPOSITE_W = dict(pcc=0.40, nrmse=0.25, r2=0.20, f1=0.15)
EVENT_W = 0.20                       # fixed, IDENTICAL across arms (not an ablated factor)
R_SIGMA, QRS_SIGMA = 4.0, 16.0
QRS_HALF = int(0.050 * FS)           # R +/- 50 ms
STT_LO, STT_HI = int(0.080 * FS), int(0.400 * FS)   # R+80 ms .. R+400 ms
REGION_DEF = {"qrs": "R +/- 50 ms", "stt": "R+80 ms .. R+400 ms"}
LOSS_WEIGHTS = {"L1": 0.30, "L2": 0.10, "L3": 0.50}
MRSTFT_CONFIG = {"fft_sizes": [128, 256, 512], "hop_sizes": [32, 64, 128],
                  "win_lengths": [128, 256, 512]}
FINAL_CONTINUOUS_TAG = "q1_remnet_continuous_strict_loso_seed42"


def is_registered_final_mode(final_test: bool, smoke: bool, arm: str,
                             loss_arm: str, tag: str) -> bool:
    return final_test and not smoke and arm == "B1" and loss_arm == "L3" and tag == FINAL_CONTINUOUS_TAG


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


def _split_sha(fold: int, split: str) -> str:
    """Hash a split file so subject/split provenance remains machine-auditable."""
    import hashlib
    h = hashlib.sha256()
    with open(DATA / f"fold_{fold:02d}" / f"{split}.h5", "rb") as fh:
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

    def __init__(self, fold: int, split: str, arm: str, scaler: tuple[float, float] | None,
                 limit=None, allow_test: bool = False, wsst_source: str = "per_split"):
        if split == "test" and not allow_test:
            raise AssertionError("test split requires registered final-candidate mode")
        self.h5p = DATA / f"fold_{fold:02d}" / f"{split}.h5"
        self.wp = DATA / f"fold_{fold:02d}" / f"{split}_wsst8s.h5"
        self.arm, self.scaler = arm, scaler
        self.needs_rcg = arm in ("B2b", "P0-naive", "P1-gated")
        t0 = time.time()
        if wsst_source == "subject_pool":
            full_wsst = SubjectWSSTPool(SUBJECT_POOL).load_for_split(self.h5p)
            n = len(full_wsst)
            self.n = min(limit, n) if limit else n
            self.wsst = full_wsst[:self.n]
            self.wp = SUBJECT_POOL
        elif wsst_source == "per_split":
            with h5py.File(self.wp, "r") as w:
                a = dict(w.attrs)
                assert a["support"] == "full_window_samples_0_1600", f"{self.wp} is not full-support"
                assert float(a["supported_fraction_of_target"]) == 1.0
                n = w["wsst"].shape[0]
                self.n = min(limit, n) if limit else n
                self.wsst = np.asarray(w["wsst"][: self.n])        # float16 in RAM
        else:
            raise ValueError(f"Unknown wsst_source={wsst_source}")
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


def smoothing_diagnostics(pred: np.ndarray, gt: np.ndarray, fs: int = FS) -> dict:
    """Direct tests for amplitude compression and high-frequency smoothing.

    Ratios are computed per window and then averaged, so large-amplitude windows
    cannot dominate a fold.  A value below one means the prediction carries less
    variance / relative >10-Hz energy than its paired ground truth.
    """
    eps = 1e-12
    var_ratio = np.var(pred, axis=1) / np.maximum(np.var(gt, axis=1), eps)
    freq = np.fft.rfftfreq(pred.shape[1], d=1.0 / fs)
    pp = np.abs(np.fft.rfft(pred - pred.mean(1, keepdims=True), axis=1)) ** 2
    gg = np.abs(np.fft.rfft(gt - gt.mean(1, keepdims=True), axis=1)) ** 2
    hi = freq > 10.0
    pfrac = pp[:, hi].sum(1) / np.maximum(pp.sum(1), eps)
    gfrac = gg[:, hi].sum(1) / np.maximum(gg.sum(1), eps)
    hf_ratio = pfrac / np.maximum(gfrac, eps)
    return {"amplitude_variance_ratio": float(np.nanmean(var_ratio)),
            "above_10hz_energy_ratio": float(np.nanmean(hf_ratio))}


def auxiliary_waveform_loss(arm: str, pred: torch.Tensor, target: torch.Tensor,
                            mrstft: MultiResolutionSTFTLoss | None = None) -> torch.Tensor:
    """One and only one registered loss-side intervention."""
    if arm == "L0":
        return pred.new_zeros(())
    if arm == "L1":
        pc = pred - pred.mean(dim=-1, keepdim=True)
        tc = target - target.mean(dim=-1, keepdim=True)
        rho = (pc * tc).sum(-1) / (
            torch.sqrt((pc.square().sum(-1) * tc.square().sum(-1)).clamp_min(1e-12)))
        return LOSS_WEIGHTS[arm] * (1.0 - rho).mean()
    if arm == "L2":
        return LOSS_WEIGHTS[arm] * F.l1_loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))
    if arm == "L3":
        if mrstft is None:
            raise ValueError("L3 requires the locked MR-STFT module")
        return LOSS_WEIGHTS[arm] * mrstft(pred, target)
    raise ValueError(f"unknown loss arm: {arm}")


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
def evaluate(model, loader, dev, compute_shape200: bool = True,
             return_outputs: bool = False):
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
    m.update(smoothing_diagnostics(P, G))
    m.update(timing_metrics(P, EV, RP))
    m.update(region_errors(P, G, RP))
    if compute_shape200:
        m["shape200_pcc_auxiliary"] = shape200_pcc(P, G, RP)
    m["composite"] = composite(m)
    if return_outputs:
        return m, {"prediction": P, "ground_truth": G, "event_probability": EV,
                   "rpeaks": np.asarray(RP, dtype=object)}
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True,
                    choices=["B0", "B1", "B2a", "B2a-wide", "B2b", "P0-naive", "P1-gated"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="reduced run; writes to a SEPARATE tree")
    ap.add_argument("--smoke_epochs", type=int, default=1)
    ap.add_argument("--smoke_limit", type=int, default=48)
    ap.add_argument("--output_root", default=None,
                    help="smoke-only output override for isolated recovery tests")
    ap.add_argument("--interrupt_after_epoch", type=int, default=None,
                    help="smoke-only controlled interruption after atomic epoch save")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split_root", default="loso_v2")
    ap.add_argument("--exp_tag", default=None,
                    help="output tree name; defaults to the pilot tag, or <tag>_smoke in smoke mode")
    ap.add_argument("--loss_arm", choices=["L0", "L1", "L2", "L3"], default="L0",
                    help="registered single-term loss ablation; L0 preserves the original objective")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wsst_source", choices=["per_split", "subject_pool"], default="per_split")
    ap.add_argument(
        "--final_test", action="store_true",
        help="registered final-candidate mode: select on validation, then open test once")
    a = ap.parse_args()

    # ---- formal / smoke parameter isolation: a formal run CANNOT be shrunk ----
    base_tag = a.exp_tag or "q1_continuous_ecg_b0_b2_pilot"
    official_final = is_registered_final_mode(
        a.final_test, a.smoke, a.arm, a.loss_arm, base_tag)
    if a.final_test and not official_final:
        raise SystemExit(
            "--final_test is restricted to registered B1/L3 experiment "
            "q1_remnet_continuous_strict_loso_seed42")
    if official_final and a.wsst_source != "subject_pool":
        raise SystemExit("Official 11-fold final candidate requires --wsst_source subject_pool")
    if a.smoke:
        epochs, limit, tag = a.smoke_epochs, a.smoke_limit, base_tag + "_smoke"
    else:
        epochs, limit, tag = BUDGET["epochs"], None, base_tag
    if (a.output_root or a.interrupt_after_epoch is not None) and not a.smoke:
        raise SystemExit("--output_root/--interrupt_after_epoch are smoke-only")

    completed_b1_loss_gate = a.arm == "B1" and a.loss_arm in ("L1", "L2", "L3")
    native_rcg_gate = a.arm in ("B2a-wide", "B2b") and a.loss_arm == "L3"
    if a.loss_arm != "L0" and not (completed_b1_loss_gate or native_rcg_gate):
        raise SystemExit(
            "Registered auxiliary losses are restricted to B1/L1-L3 or "
            "B2a-wide/B2b with locked L3")
    if native_rcg_gate and base_tag != "q1_native_rcg_resolution_gate":
        raise SystemExit(
            "B2a-wide/B2b L3 is registered only under q1_native_rcg_resolution_gate")
    if native_rcg_gate:
        run_arm = {"B2a-wide": "C0_capacity_matched_wsst",
                   "B2b": "C1_native_rcg_dual_stream"}[a.arm]
    else:
        run_arm = a.arm if a.loss_arm == "L0" else a.loss_arm
    output_root = Path(a.output_root) if a.output_root else ROOT / "experiments_mmecg" / tag
    out = output_root / run_arm / f"fold_{a.fold:02d}"
    state_path = out / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state["stage"] == "COMPLETE":
        print(f"[sealed] COMPLETE fold at {out}; test cannot be reopened")
        return
    if (out / "best.json").exists() and state is None and not a.overwrite:
        raise SystemExit(
            f"Refusing legacy output without a resumable state machine at {out}. "
            "Archive it or explicitly pass --overwrite.")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if state is None:
        state = transition_state(state_path, "TRAINING", fold=a.fold, arm=a.arm,
                                 loss_arm=a.loss_arm, test_accessed=False)

    _set_split_root(a.split_root)
    seed = effective_seed(a.fold)
    set_seed(seed)
    dev = torch.device(a.device)
    scaler = None if a.arm in LEGACY_TARGET_ARMS else fit_scaler(a.fold)
    tr = Fold(a.fold, "train", a.arm, scaler, limit, wsst_source=a.wsst_source)
    va = Fold(a.fold, "val", a.arm, scaler, limit, wsst_source=a.wsst_source)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=BUDGET["batch_size"], shuffle=True, num_workers=0,
                     drop_last=True, generator=g)
    lva = DataLoader(va, batch_size=BUDGET["batch_size"], shuffle=False, num_workers=0)

    model = CENet(a.arm)
    paired_decoder_seed = None
    if native_rcg_gate:
        # Variant-specific stems consume different RNG streams. Re-initialise
        # the byte-identical decoder from a component-specific paired seed.
        paired_decoder_seed = seed + 10_000
        reset_module_parameters_paired(model.decoder, paired_decoder_seed)
    model = model.to(dev)
    mrstft = MultiResolutionSTFTLoss(**MRSTFT_CONFIG).to(dev) if a.loss_arm == "L3" else None
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
    config = {
        "arm": a.arm, "fold": a.fold, "mode": "smoke" if a.smoke else "formal",
        "epochs": epochs, "limit": limit, "effective_seed": seed,
        "seed_rule": "BUDGET.seed + fold; identical across arms within a fold (paired)",
        "cudnn_deterministic": True, "split_used": ["train", "val"], "test_accessed": False,
        "final_candidate_mode": official_final,
        "wsst_source": a.wsst_source,
        "input_pipeline": "RAM-preloaded per fold (HDF5 was the bottleneck at ~269 s/epoch, 0% GPU)",
        "num_workers": 0,
        "scaler_median_iqr": scaler, "scaler_source": "ENTIRE training fold",
        "budget": BUDGET, "params": param_report(model),
        "composite_weights": COMPOSITE_W, "event_loss_weight": EVENT_W,
        "loss_arm": a.loss_arm, "loss_weight": LOSS_WEIGHTS.get(a.loss_arm, 0.0),
        "mrstft_config": MRSTFT_CONFIG if a.loss_arm == "L3" else None,
        "native_rcg_resolution_gate": native_rcg_gate,
        "paired_decoder_seed": paired_decoder_seed,
        "region_definitions": REGION_DEF, "rpeak_readout": "waveform-derived",
        "metric_space": "raw ECG units for ALL arms (per-arm inverse applied)",
        "b0_inverse_is_oracle": a.arm in LEGACY_TARGET_ARMS,
        "wsst": "full_window_samples_0_1600",
        "split_root": a.split_root, "split_dir": str(DATA),
        "val_file_sha256": _split_sha(a.fold, "val"),
        "code_commit": commit, "code_commit_is": "last commit touching src/scripts/tests/configs",
        "runtime_head": head,
    }
    config_path = out / "config.json"
    if config_path.exists() and not a.overwrite:
        prior = json.loads(config_path.read_text())
        identity = ("arm", "fold", "loss_arm", "budget", "effective_seed",
                    "split_root", "wsst_source")
        mismatch = [key for key in identity if prior.get(key) != config.get(key)]
        if mismatch:
            raise SystemExit(f"Resume contract mismatch in {mismatch}: {out}")
    else:
        atomic_json(config_path, config)

    scheduler = None  # Frozen protocol uses constant LR; state records this explicitly.
    best, best_ep, bad, hist, initial_components = -1e9, -1, 0, [], None
    start_epoch = 1
    last_state_path = out / "checkpoints" / "last_state.pt"
    if state["stage"] == "TRAINING" and last_state_path.exists() and not a.overwrite:
        saved = torch.load(last_state_path, map_location=dev, weights_only=False)
        expected = {"fold": a.fold, "arm": a.arm, "loss_arm": a.loss_arm,
                    "max_epochs": epochs, "patience": BUDGET["patience"]}
        mismatch = [key for key, value in expected.items() if saved.get(key) != value]
        if mismatch:
            raise SystemExit(f"Checkpoint resume contract mismatch in {mismatch}: {last_state_path}")
        model.load_state_dict(saved["model_state_dict"])
        opt.load_state_dict(saved["optimizer_state_dict"])
        if scheduler is not None and saved["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(saved["scheduler_state_dict"])
        best, best_ep = float(saved["best_val"]), int(saved["best_epoch"])
        bad, hist = int(saved["patience_counter"]), saved["history"]
        initial_components = saved["initial_loss_components"]
        restore_rng_state(saved["rng_state"], g)
        start_epoch = int(saved["epoch"]) + 1
        print(f"[resume] fold{a.fold:02d} from epoch {start_epoch} "
              f"(best={best:.4f}@{best_ep}, patience={bad})")

    if state["stage"] == "TRAINING":
        for ep in range(start_epoch, epochs + 1):
            model.train(); tot = nb = 0; t0 = time.time()
            for b in ltr:
                kw = {"rcg": b["rcg"].to(dev)} if "rcg" in b else {}
                y, ev = model(b["wsst"].to(dev), **kw)
                target, event_target = b["y"].to(dev), b["event"].to(dev)
                base_loss = F.huber_loss(y, target)
                event_loss = EVENT_W * F.binary_cross_entropy_with_logits(ev, event_target)
                aux_loss = auxiliary_waveform_loss(a.loss_arm, y, target, mrstft)
                loss = base_loss + event_loss + aux_loss
                if initial_components is None:
                    initial_components = {"huber": float(base_loss.detach()),
                                          "weighted_event_bce": float(event_loss.detach()),
                                          "weighted_auxiliary": float(aux_loss.detach()),
                                          "total": float(loss.detach())}
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            m = evaluate(model, lva, dev, compute_shape200=not a.arm.startswith("P"))
            hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), **m})
            print(f"  ep{ep:3d} loss {tot/max(nb,1):.4f} | PCC {m['pcc']:.4f} NRMSE {m['nrmse']:.4f} "
                  f"R2 {m['r2']:+.3f} F1@100 {m['f1_100ms']:.3f} evF1 {m['event_f1_100ms']:.3f} "
                  f"| S {m['composite']:.4f} ({time.time()-t0:.0f}s)")
            if m["composite"] > best:
                best, best_ep, bad = m["composite"], ep, 0
                atomic_torch_save(out / "checkpoints" / "best_composite.pt", {
                    "model_state_dict": model.state_dict(), "epoch": ep,
                    "composite": best, "arm": a.arm, "fold": a.fold})
            else:
                bad += 1
            atomic_torch_save(last_state_path, {
                "epoch": ep, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "patience_counter": bad, "best_val": best, "best_epoch": best_ep,
                "history": hist, "initial_loss_components": initial_components,
                "rng_state": capture_rng_state(g), "fold": a.fold, "arm": a.arm,
                "loss_arm": a.loss_arm, "max_epochs": epochs,
                "patience": BUDGET["patience"],
            })
            if a.interrupt_after_epoch == ep:
                print(f"[controlled-interrupt] after atomic epoch {ep}")
                raise SystemExit(75)
            if bad >= BUDGET["patience"]:
                print(f"  early stop at ep{ep}"); break
        atomic_json(out / "history.json", hist)
        atomic_json(out / "initial_loss_components.json", initial_components)
        atomic_json(out / "best.json", {
            "arm": a.arm, "loss_arm": a.loss_arm, "fold": a.fold,
            "mode": "smoke" if a.smoke else "formal",
            "best_epoch": best_ep, "best_composite": best, "code_commit": commit,
            "runtime_head": head, "max_epochs": epochs, "patience": BUDGET["patience"],
            "hit_epoch_cap": best_ep == epochs, "stopped_at_epoch": len(hist),
            "early_stopped": len(hist) < epochs,
            "selected_by": "full-waveform validation composite (Shape-200 excluded)",
            "metrics": hist[best_ep - 1] if best_ep > 0 else None})
        state = transition_state(state_path, "CHECKPOINT_FROZEN", fold=a.fold,
                                 arm=a.arm, loss_arm=a.loss_arm,
                                 best_epoch=best_ep, test_accessed=False)
    else:
        frozen = json.loads((out / "best.json").read_text())
        best, best_ep, hist = (float(frozen["best_composite"]),
                               int(frozen["best_epoch"]),
                               json.loads((out / "history.json").read_text()))

    if official_final:
        # The held-out test split is constructed only after the validation-best
        # epoch is frozen. Test metrics cannot influence training or selection.
        if state["stage"] == "CHECKPOINT_FROZEN":
            state = transition_state(state_path, "TEST_EVALUATING", fold=a.fold,
                                     arm=a.arm, loss_arm=a.loss_arm,
                                     best_epoch=best_ep, test_accessed=True)
        if state["stage"] != "TEST_EVALUATING":
            raise RuntimeError(f"Test access denied in state {state['stage']}")
        checkpoint = torch.load(out / "checkpoints" / "best_composite.pt",
                                map_location=dev, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        te = Fold(a.fold, "test", a.arm, scaler, allow_test=True,
                  wsst_source=a.wsst_source)
        lte = DataLoader(te, batch_size=BUDGET["batch_size"], shuffle=False, num_workers=0)
        test_metrics, arrays = evaluate(
            model, lte, dev, compute_shape200=False, return_outputs=True)
        atomic_npz(out / "test_predictions.npz", arrays)
        atomic_json(out / "test_metrics.json", {
            "arm": a.arm,
            "loss_arm": a.loss_arm,
            "fold": a.fold,
            "checkpoint_epoch": best_ep,
            "checkpoint_selected_by": "validation composite only",
            "test_opened_after_selection": True,
            "test_file_sha256": _split_sha(a.fold, "test"),
            "aggregation_within_fold": "per-window metrics averaged within held-out test subject",
            "shape200_excluded": True,
            "metrics": test_metrics,
        })
        config = json.loads(config_path.read_text())
        config.update({"split_used": ["train", "val", "test"],
                       "test_accessed": True,
                       "test_access_rule": "once, after validation checkpoint selection",
                       "test_file_sha256": _split_sha(a.fold, "test")})
        atomic_json(config_path, config)
        state = transition_state(state_path, "COMPLETE", fold=a.fold,
                                 arm=a.arm, loss_arm=a.loss_arm,
                                 best_epoch=best_ep, test_accessed=True,
                                 test_metrics="test_metrics.json",
                                 test_predictions="test_predictions.npz")
        print(f"[test] B1 fold{a.fold}: PCC={test_metrics['pcc']:.4f} "
              f"R2={test_metrics['r2']:+.4f} -> {out / 'test_metrics.json'}")
    print(f"[done] {a.arm} fold{a.fold} ({'smoke' if a.smoke else 'formal'}): "
          f"best S={best:.4f} @ep{best_ep} -> {out}")


if __name__ == "__main__":
    main()
