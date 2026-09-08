#!/usr/bin/env python3
"""Continuous-gap decomposition audit.

Spec: experiments/specs/q1_continuous_gap_decomposition_audit.json

DIAGNOSTIC ONLY.  Assembles predicted beat morphology into a continuous trace and
scores it against the ORIGINAL ECG in native amplitude.  `_shape200_targets` and
`normalize_01` are never applied to the ground truth here -- that defect is what
this audit supersedes (retracted claim D1).

Hard rules enforced in code:
  * fixed affine is fitted on TRAINING folds only;
  * unpredicted beats are never filled;
  * K2 only recovers genuine centre-cycle predictions;
  * cross-arm comparison is reported on COMMON SUPPORT, alongside native coverage;
  * contrasts are pairwise, never an additive decomposition.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402
from src.models.model_factory import build_mmecg_model  # noqa: E402

WIN = 1600
# Primary controlled counterfactual contrast: identical window set, common support.
ARMS = ("C0", "C1", "C2", "C3", "C4")
# Descriptive only. Autonomous predicted-anchor pipeline, iterated over PREDICTED
# cycles with no GT-derived window selection. Reported on its own native coverage
# and NEVER folded into the common-support intersection or the primary contrast.
DESC_ARMS = ("C5",)


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(x)), x)


def _affine_fit(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Least-squares gain/offset mapping pred -> gt."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(pred) < 2:
        return 1.0, 0.0
    var = pred.var()
    if var < 1e-12:
        return 0.0, float(gt.mean())
    g = float(((pred - pred.mean()) * (gt - gt.mean())).mean() / var)
    return g, float(gt.mean() - g * pred.mean())


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    if len(pred) < 2:
        return {k: float("nan") for k in ("pcc", "rmse", "nrmse", "mae")}
    err = pred - gt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rng = float(gt.max() - gt.min())
    pc, gc = pred - pred.mean(), gt - gt.mean()
    den = float(np.sqrt((pc ** 2).sum() * (gc ** 2).sum()))
    return {
        "pcc": float((pc * gc).sum() / den) if den > 1e-12 else float("nan"),
        "rmse": rmse,
        "nrmse": rmse / rng if rng > 1e-12 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
    }


def _load_model(run: Path, device, checkpoint: str):
    saved = json.loads((run / "config.json").read_text())
    cfg = MMECGConfig()
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = build_mmecg_model(cfg).to(device)
    ck = torch.load(run / "checkpoints" / checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg, saved


def _infer(model, cfg, fold_dir: Path, split: str, device, limit: int | None = None):
    """Run the frozen model over one split; return per-window arrays + metadata.

    When `limit` is set the subsample is EVENLY SPACED over the split, not the
    first N windows: the first N would be biased toward the earliest records.
    """
    wsst = fold_dir / f"{split}{cfg.wsst_suffix}"
    out = []
    with h5py.File(fold_dir / f"{split}.h5", "r") as h5, h5py.File(wsst, "r") as wh5:
        total = int(h5["ecg"].shape[0])
        sel = (list(range(total)) if limit is None or limit >= total
               else np.linspace(0, total - 1, limit).astype(int).tolist())
        n = len(sel)
        for lo in range(0, n, 32):
            idx = sorted(set(sel[lo:min(lo + 32, n)]))
            radar = torch.from_numpy(np.asarray(wh5["wsst"][idx], np.float32)).to(device)
            with torch.no_grad():
                full, events = model(radar)
            full_np = full.cpu().numpy()[:, 0]
            shape_np = full.shape200_pred.cpu().numpy()[:, 0]
            rmask_np = events[0].cpu().numpy()[:, 0]
            for k, i in enumerate(idx):
                rp = np.sort(np.unique(np.asarray(h5["rpeak_indices"][i], np.int64)))
                rp = rp[(rp >= 0) & (rp < WIN)]
                out.append({
                    "record": h5["record_id"][i].decode() if isinstance(h5["record_id"][i], bytes) else str(h5["record_id"][i]),
                    "start": int(h5["start_idx"][i]),
                    "ecg": h5["ecg"][i, 0].astype(np.float64),
                    "rpeaks_local": rp,
                    "full": full_np[k].astype(np.float64),
                    "shape": shape_np[k].astype(np.float64),
                    "rmask": rmask_np[k],
                })
    return out


def _global_cycles(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Union of global GT R-peaks for one record -> consecutive-pair cycles."""
    peaks = np.unique(np.concatenate([w["start"] + w["rpeaks_local"] for w in windows]))
    if len(peaks) < 2:
        return peaks, np.empty((0, 2), dtype=np.int64)
    return peaks, np.stack([peaks[:-1], peaks[1:]], axis=1)


def _centre_cycle(cycles: np.ndarray, centre: int) -> int:
    """Index of the cycle containing `centre`, else the nearest; -1 if none."""
    if len(cycles) == 0:
        return -1
    inside = np.where((cycles[:, 0] <= centre) & (centre < cycles[:, 1]))[0]
    if len(inside):
        return int(inside[0])
    return int(np.argmin(np.abs(cycles.mean(axis=1) - centre)))


def _fit_fixed_affine(windows: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit (full-head, shape-head) affines on TRAIN windows only."""
    fp, fg, sp, sg = [], [], [], []
    for w in windows:
        fp.append(w["full"]); fg.append(w["ecg"])
        if len(w["rpeaks_local"]) >= 2:
            pairs = np.stack([w["rpeaks_local"][:-1], w["rpeaks_local"][1:]], axis=1)
            j = _centre_cycle(pairs, WIN // 2)
            if j >= 0:
                a, b = int(pairs[j, 0]), int(pairs[j, 1])
                if b > a + 10:
                    sp.append(_resample(w["shape"], b - a)); sg.append(w["ecg"][a:b])
    full_affine = _affine_fit(np.concatenate(fp), np.concatenate(fg))
    shape_affine = _affine_fit(np.concatenate(sp), np.concatenate(sg)) if sp else (1.0, 0.0)
    return full_affine, shape_affine


def _assemble_record(windows: list[dict], full_aff, shape_aff) -> dict:
    """Build every arm for one record. Returns traces, masks and bookkeeping."""
    peaks, cycles = _global_cycles(windows)
    total = max(int(w["start"]) + WIN for w in windows)
    gt = np.full(total, np.nan)
    for w in windows:
        gt[w["start"]:w["start"] + WIN] = w["ecg"]

    # ---- C0: dense full-window head; per-sample nearest-window-centre dedup ----
    c0 = np.full(total, np.nan); c0_dist = np.full(total, np.inf)
    for w in windows:
        s = w["start"]; idx = np.arange(s, s + WIN)
        d = np.abs(idx - (s + WIN // 2))
        better = d < c0_dist[idx]
        c0[idx[better]] = w["full"][better]
        c0_dist[idx[better]] = d[better]
    c0 = full_aff[0] * c0 + full_aff[1]

    # ---- candidate centre-cycle predictions, keyed by GLOBAL GT cycle ----
    cand = defaultdict(list)
    for w in windows:
        if len(w["rpeaks_local"]) < 2:
            continue
        centre = w["start"] + WIN // 2
        j = _centre_cycle(cycles, centre)
        if j < 0:
            continue
        a, b = int(cycles[j, 0]), int(cycles[j, 1])
        if b <= a + 10 or a < w["start"] or b > w["start"] + WIN:
            continue  # cycle not fully inside this window -> not a genuine centre-cycle prediction
        cand[j].append((abs(centre - (a + b) // 2), w))

    # ---- predicted-anchor cycles from the event head (global) ----
    pred_peaks = np.unique(np.concatenate(
        [w["start"] + peaks_from_heatmap(w["rmask"]).astype(np.int64) for w in windows]
    )) if windows else np.empty(0, dtype=np.int64)
    pred_cycles = (np.stack([pred_peaks[:-1], pred_peaks[1:]], axis=1)
                   if len(pred_peaks) >= 2 else np.empty((0, 2), dtype=np.int64))

    def place(keys, use_pred_bounds, oracle_affine):
        trace = np.full(total, np.nan)
        for j in keys:
            _, w = min(cand[j], key=lambda t: t[0])
            if use_pred_bounds:
                centre = w["start"] + WIN // 2
                jp = _centre_cycle(pred_cycles, centre)
                if jp < 0:
                    continue
                a, b = int(pred_cycles[jp, 0]), int(pred_cycles[jp, 1])
            else:
                a, b = int(cycles[j, 0]), int(cycles[j, 1])
            if b <= a + 10 or a < 0 or b > total:
                continue
            seg = _resample(w["shape"], b - a)
            tgt = gt[a:b]
            if oracle_affine:
                if np.isnan(tgt).any():
                    continue
                g, o = _affine_fit(seg, tgt)
            else:
                g, o = shape_aff
            trace[a:b] = g * seg + o
        return trace

    k1 = sorted(j for j in cand if len(cand[j]) >= 1)          # deduplicated centre cycles
    k2 = k1                                                     # genuine centre cycles only

    # ---- C5: autonomous predicted pipeline (DESCRIPTIVE, outside the contrast) ----
    # Iterates PREDICTED cycles directly; window selection never consults GT.
    # Uses the project's standing peak rule (thr=max(0.3,0.5*ymax), min_dist=40);
    # no threshold is tuned here.
    pred_cand = defaultdict(list)
    for w in windows:
        centre = w["start"] + WIN // 2
        jp = _centre_cycle(pred_cycles, centre)
        if jp < 0:
            continue
        a, b = int(pred_cycles[jp, 0]), int(pred_cycles[jp, 1])
        if b <= a + 10 or a < w["start"] or b > w["start"] + WIN:
            continue
        pred_cand[jp].append((abs(centre - (a + b) // 2), w))
    c5 = np.full(total, np.nan)
    for jp, lst in pred_cand.items():
        _, w = min(lst, key=lambda t: t[0])
        a, b = int(pred_cycles[jp, 0]), int(pred_cycles[jp, 1])
        if b <= a + 10 or a < 0 or b > total:
            continue
        c5[a:b] = shape_aff[0] * _resample(w["shape"], b - a) + shape_aff[1]

    arms = {
        "C0": c0,
        "C1": place(k1, False, False), "C2": place(k1, False, True),
        "C3": place(k1, True, False),  "C4": place(k1, True, True),
        "C5": c5,
    }
    return {
        "gt": gt, "arms": arms, "total": total,
        "n_cycles_total": int(len(cycles)), "n_cycles_matched": int(len(k1)),
        "n_pred_cycles_total": int(len(pred_cycles)), "n_pred_cycles_matched": int(len(pred_cand)),
        "candidates_per_cycle": np.bincount([len(v) for v in cand.values()]).tolist() if cand else [],
        "k1_eq_k2": k1 == k2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", default="q1_final_remnet_incumbent_mmecg_seed42")
    ap.add_argument("--folds", default="1")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--phase", default="phase_1_validation_smoke")
    ap.add_argument("--limit", type=int, default=None, help="smoke only: cap windows")
    ap.add_argument("--train_affine_windows", type=int, default=400,
                help="evenly-spaced TRAIN windows used to fit the fixed affine (frozen protocol constant)")
    ap.add_argument("--out_tag", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.phase.startswith("phase_1") and args.split != "val":
        raise SystemExit("Phase 1 is validation-only (spec).")
    folds = ([int(x) for x in args.folds.split(",")] if "," in args.folds
             else list(range(*map(lambda v: int(v), args.folds.split("-")))) if "-" in args.folds
             else [int(args.folds)])
    if "-" in args.folds:
        a, b = args.folds.split("-"); folds = list(range(int(a), int(b) + 1))

    device = torch.device(args.device)
    all_rows = []
    for fold in folds:
        run = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{fold:02d}"
        model, cfg, saved = _load_model(run, device, args.checkpoint)
        fold_dir = Path(cfg.loso_h5_dir) / f"fold_{fold:02d}"

        train_w = _infer(model, cfg, fold_dir, "train", device, limit=args.train_affine_windows)
        full_aff, shape_aff = _fit_fixed_affine(train_w)
        del train_w
        print(f"[fold {fold:02d}] fixed affine  full=(g={full_aff[0]:.4f}, o={full_aff[1]:.4f})"
              f"  shape=(g={shape_aff[0]:.4f}, o={shape_aff[1]:.4f})   [TRAIN-fitted]")

        eval_w = _infer(model, cfg, fold_dir, args.split, device, limit=args.limit)
        by_rec = defaultdict(list)
        for w in eval_w:
            by_rec[w["record"]].append(w)

        for rec, ws in sorted(by_rec.items()):
            R = _assemble_record(ws, full_aff, shape_aff)
            gt = R["gt"]
            valid = {a: ~np.isnan(R["arms"][a]) & ~np.isnan(gt) for a in ARMS}
            common = np.logical_and.reduce([valid[a] for a in ARMS])
            row = {
                "fold": fold, "record": rec, "split": args.split,
                "total_samples": R["total"],
                "common_support_samples": int(common.sum()),
                "common_support_ratio": float(common.sum() / max(R["total"], 1)),
                "n_cycles_total": R["n_cycles_total"], "n_cycles_matched": R["n_cycles_matched"],
                "candidates_per_cycle_histogram": R["candidates_per_cycle"],
                "k2_equals_k1_genuine_centre_only": R["k1_eq_k2"],
            }
            for a in DESC_ARMS:
                nat = ~np.isnan(R["arms"][a]) & ~np.isnan(gt)
                row[f"{a}_native_coverage_ratio"] = float(nat.sum() / max(R["total"], 1))
                row[f"{a}_native_covered_samples"] = int(nat.sum())
                for k, v in _metrics(R["arms"][a][nat], gt[nat]).items():
                    row[f"{a}_native_{k}"] = v
            for a in ARMS:
                nat = valid[a]
                row[f"{a}_native_coverage_ratio"] = float(nat.sum() / max(R["total"], 1))
                row[f"{a}_native_covered_samples"] = int(nat.sum())
                for k, v in _metrics(R["arms"][a][nat], gt[nat]).items():
                    row[f"{a}_native_{k}"] = v
                for k, v in _metrics(R["arms"][a][common], gt[common]).items():
                    row[f"{a}_common_{k}"] = v
            all_rows.append(row)
            print(f"  {rec}: common={row['common_support_ratio']:.3f} "
                  f"| C5(desc) nat={row['C5_native_coverage_ratio']:.2f}/pcc={row['C5_native_pcc']:.3f} | "
                  + "  ".join(f"{a} nat={row[f'{a}_native_coverage_ratio']:.2f}/"
                              f"pcc={row[f'{a}_common_pcc']:.3f}" for a in ARMS))

    out_tag = args.out_tag or f"q1_continuous_gap_decomposition_{args.phase}"
    out = ROOT / "experiments_mmecg" / out_tag
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": "experiments/specs/q1_continuous_gap_decomposition_audit.json",
        "phase": args.phase, "tier": "smoke" if args.phase.startswith("phase_1") else "final",
        "publishable": False, "dataset": "MMECG", "split": args.split,
        "exp_tag": args.exp_tag, "checkpoint": args.checkpoint, "folds": folds,
        "framing": ("Pairwise contrasts on common support. NOT an additive decomposition: "
                    "contrasts are non-orthogonal and do not sum to the C0-to-C2 gap."),
        "arms": {"C0": "dense full-window + fixed affine", "C1": "GT anchors + fixed affine",
                 "C2": "GT anchors + oracle per-cycle affine", "C3": "predicted anchors + fixed affine",
                 "C4": "predicted anchors + oracle per-cycle affine"},
        "descriptive_arms": {"C5": "AUTONOMOUS predicted-anchor pipeline + fixed affine; iterated over "
                                   "predicted cycles with no GT-derived window selection. Descriptive "
                                   "deployment behaviour ONLY. Not part of the primary contrast, not on "
                                   "common support, never differenced against C0-C4."},
        "records": all_rows,
    }
    (out / "decomposition_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out/'decomposition_metrics.json'}  ({len(all_rows)} records)")


if __name__ == "__main__":
    main()
