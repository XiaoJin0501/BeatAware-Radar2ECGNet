#!/usr/bin/env python3
"""Train the calibration-conditioned REM adapter (validation-only pilot).

The B1 radar backbone is frozen.  For every record, windows ending in the first
20 seconds form the paired calibration prefix and windows starting afterwards
form the query set.  Test data is never opened.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.calibrated_rem import CalibrationConditionedREM  # noqa: E402
from src.models.continuous.ce_net import CENet  # noqa: E402
from scripts.train_continuous_ecg_pilot import waveform_metrics, timing_metrics, composite  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
BASE = ROOT / "experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"
OUT = ROOT / "experiments_mmecg/q1_calibration_conditioned_rem_pilot"
FS, CUTOFF, EPOCHS, PATIENCE, BATCH = 200, 4000, 80, 20, 16
ARMS = {
    "neither": (False, False), "event_only": (True, False),
    "morphology_only": (False, True), "full": (True, True),
}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def pcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x, y = pred - pred.mean(-1, keepdim=True), target - target.mean(-1, keepdim=True)
    corr = (x * y).sum(-1) / (x.square().sum(-1).sqrt() * y.square().sum(-1).sqrt()).clamp_min(1e-8)
    return 1 - corr.mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    deriv = F.l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
    return F.huber_loss(pred, target) + 0.30 * pcc_loss(pred, target) + 0.10 * deriv


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()


def freeze_predictions(fold: int, split: str, device: torch.device) -> dict:
    """Run the frozen B1 once and retain only compact predictions/labels in RAM."""
    base_dir = BASE / f"fold_{fold:02d}"
    cfg = json.loads((base_dir / "config.json").read_text())
    med, iqr = map(float, cfg["scaler_median_iqr"])
    h5p, wp = DATA / f"fold_{fold:02d}/{split}.h5", DATA / f"fold_{fold:02d}/{split}_wsst8s.h5"
    with h5py.File(h5p, "r") as h:
        records = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in h["record_id"][:]])
        start, end = np.asarray(h["start_idx"][:]), np.asarray(h["end_idx"][:])
        gt = np.asarray(h["ecg"][:, 0], np.float32)
        rpeaks = [np.sort(np.unique(np.asarray(h["rpeak_indices"][i], np.int64))) for i in range(len(gt))]
    backbone = CENet("B1").to(device)
    ck = torch.load(base_dir / "checkpoints/best_composite.pt", map_location=device, weights_only=True)
    backbone.load_state_dict(ck["model_state_dict"]); backbone.eval()
    preds, events = [], []
    with h5py.File(wp, "r") as w, torch.no_grad():
        assert w.attrs["support"] == "full_window_samples_0_1600"
        for lo in range(0, len(gt), 32):
            x = torch.from_numpy(np.asarray(w["wsst"][lo:lo + 32], np.float32)).to(device)
            p, e = backbone(x)
            preds.append((p * iqr + med).cpu()); events.append(e.cpu())
    del backbone; torch.cuda.empty_cache()
    return {"record": records, "start": start, "end": end, "gt": torch.from_numpy(gt[:, None]),
            "pred": torch.cat(preds), "event": torch.cat(events), "rpeaks": rpeaks}


def record_groups(data: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = []
    for rec in sorted(set(data["record"].tolist())):
        own = data["record"] == rec
        cal = np.flatnonzero(own & (data["end"] <= CUTOFF))
        qry = np.flatnonzero(own & (data["start"] >= CUTOFF))
        if len(cal) and len(qry): groups.append((rec, cal, qry))
    if not groups: raise RuntimeError("No record has sample-disjoint calibration and query windows")
    return groups


def forward_record(model, data, cal, qry, device, train: bool):
    cp, cg = data["pred"][cal].to(device), data["gt"][cal].to(device)
    order = np.random.permutation(qry) if train else qry
    for lo in range(0, len(order), BATCH):
        idx = order[lo:lo + BATCH]
        yield idx, model(data["pred"][idx].to(device), data["event"][idx].to(device), cp, cg)


@torch.no_grad()
def evaluate(model, data, groups, device) -> dict:
    model.eval(); pred, gt, ev, rp = [], [], [], []
    for _, cal, qry in groups:
        for idx, (out, _) in forward_record(model, data, cal, qry, device, False):
            pred.append(out.cpu().numpy()[:, 0]); gt.append(data["gt"][idx].numpy()[:, 0])
            ev.append(torch.sigmoid(data["event"][idx]).numpy()[:, 0]); rp.extend(data["rpeaks"][i] for i in idx)
    pred, gt, ev = np.concatenate(pred), np.concatenate(gt), np.concatenate(ev)
    metrics = waveform_metrics(pred, gt); metrics.update(timing_metrics(pred, ev, rp))
    metrics["composite"] = composite(metrics)
    metrics["n_query_windows"] = len(pred)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA is required")
    seed = 42 + args.fold; seed_all(seed); device = torch.device("cuda")
    epochs, patience = (2, 2) if args.smoke else (EPOCHS, PATIENCE)
    tag = "_smoke" if args.smoke else args.arm
    out = OUT / tag / f"fold_{args.fold:02d}"
    if (out / "best.json").exists() and not args.overwrite: raise SystemExit(f"Refusing overwrite: {out}")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    print("Precomputing frozen B1 train/validation outputs...", flush=True)
    train, val = freeze_predictions(args.fold, "train", device), freeze_predictions(args.fold, "val", device)
    tg, vg = record_groups(train), record_groups(val)
    use_event, use_morph = ARMS[args.arm]
    model = CalibrationConditionedREM(use_event=use_event, use_morphology=use_morph).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {"experiment_id": "q1_calibration_conditioned_rem_pilot", "tier": "smoke" if args.smoke else "pilot",
              "arm": args.arm, "fold": args.fold, "seed": seed, "epochs": epochs, "patience": patience,
              "calibration_seconds": 20, "split": "loso_v2", "test_accessed": False,
              "train_records": len(tg), "val_records": len(vg), "code_commit": commit,
              "val_sha256": sha256(DATA / f"fold_{args.fold:02d}/val.h5"),
              "loss": "Huber + 0.30*(1-PCC) + 0.10*derivative-L1",
              "selection": "validation query-only continuous composite; Shape-200 absent"}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, best_ep, bad, hist = -1e9, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train(); total = batches = 0; t0 = time.time()
        random.shuffle(tg)
        for _, cal, qry in tg:
            for idx, (pred, _) in forward_record(model, train, cal, qry, device, True):
                loss = loss_fn(pred, train["gt"][idx].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
                total += float(loss); batches += 1
        metrics = evaluate(model, val, vg, device)
        row = {"epoch": ep, "train_loss": total / batches, **metrics}; hist.append(row)
        print(f"ep={ep:02d} loss={row['train_loss']:.4f} PCC={row['pcc']:.4f} RMSE={row['rmse']:.4f} "
              f"R2={row['r2']:+.4f} F1@100={row['f1_100ms']:.3f} S={row['composite']:.4f} sec={time.time()-t0:.1f}", flush=True)
        if row["composite"] > best:
            best, best_ep, bad = row["composite"], ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep}, out / "checkpoints/best_composite.pt")
        else:
            bad += 1
            if bad >= patience: break
    (out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    result = {"best_epoch": best_ep, "stopped_at_epoch": len(hist), "best_composite": best,
              "metrics": hist[best_ep - 1], "early_stopped": len(hist) < epochs,
              "shape200_computed": False}
    (out / "best.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DONE " + json.dumps(result), flush=True)


if __name__ == "__main__": main()
