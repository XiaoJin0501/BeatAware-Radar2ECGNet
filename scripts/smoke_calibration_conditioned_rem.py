#!/usr/bin/env python3
"""Stage-0 train-only overfit check for the calibration-conditioned REM adapter.

This is a smoke run, not experiment evidence.  It opens one source-training
record, uses its first 20 seconds as calibration, and fits a sample-disjoint
query batch.  Validation and test files are never opened.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.continuous.calibrated_rem import CalibrationConditionedREM  # noqa: E402
from src.models.continuous.ce_net import CENet  # noqa: E402

DATA = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2")
BASE = ROOT / "experiments_mmecg/q1_continuous_ecg_b1_lososv2_baseline/B1"
OUT = ROOT / "experiments_mmecg/q1_calibration_conditioned_rem_pilot/_smoke/stage0.json"
FS = 200


def pcc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x = pred - pred.mean(dim=-1, keepdim=True)
    y = target - target.mean(dim=-1, keepdim=True)
    return ((x * y).sum(-1) / (x.square().sum(-1).sqrt() * y.square().sum(-1).sqrt()).clamp_min(1e-8)).mean()


def objective(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    derivative = F.l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])
    return F.huber_loss(pred, target) + 0.30 * (1 - pcc(pred, target)) + 0.10 * derivative


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    fold_dir = DATA / f"fold_{args.fold:02d}"
    base_dir = BASE / f"fold_{args.fold:02d}"
    cfg = json.loads((base_dir / "config.json").read_text())
    iqr, med = float(cfg["scaler_median_iqr"][1]), float(cfg["scaler_median_iqr"][0])
    with h5py.File(fold_dir / "train.h5", "r") as h:
        records = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in h["record_id"][:]])
        start, end = np.asarray(h["start_idx"][:]), np.asarray(h["end_idx"][:])
        chosen = next(r for r in sorted(set(records))
                      if np.sum((records == r) & (end <= 20 * FS)) and np.sum((records == r) & (start >= 20 * FS)) >= 8)
        cal_idx = np.flatnonzero((records == chosen) & (end <= 20 * FS))
        qry_idx = np.flatnonzero((records == chosen) & (start >= 20 * FS))[:16]
        cal_gt = torch.from_numpy(np.asarray(h["ecg"][cal_idx, 0], np.float32))[:, None]
        qry_gt = torch.from_numpy(np.asarray(h["ecg"][qry_idx, 0], np.float32))[:, None]
    with h5py.File(fold_dir / "train_wsst8s.h5", "r") as h:
        assert h.attrs["support"] == "full_window_samples_0_1600"
        cal_x = torch.from_numpy(np.asarray(h["wsst"][cal_idx], np.float32))
        qry_x = torch.from_numpy(np.asarray(h["wsst"][qry_idx], np.float32))

    backbone = CENet("B1").to(device)
    ckpt = torch.load(base_dir / "checkpoints/best_composite.pt", map_location=device, weights_only=True)
    backbone.load_state_dict(ckpt["model_state_dict"]); backbone.eval()
    with torch.no_grad():
        cal_pred, _ = backbone(cal_x.to(device))
        qry_pred, qry_event = backbone(qry_x.to(device))
        cal_pred = cal_pred * iqr + med
        qry_pred = qry_pred * iqr + med
        cal_gt, qry_gt = cal_gt.to(device), qry_gt.to(device)

    model = CalibrationConditionedREM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    with torch.no_grad():
        initial, aux = model(qry_pred, qry_event, cal_pred, cal_gt)
        initial_loss, initial_pcc = float(objective(initial, qry_gt)), float(pcc(initial, qry_gt))
    for step in range(1, args.steps + 1):
        pred, aux = model(qry_pred, qry_event, cal_pred, cal_gt)
        loss = objective(pred, qry_gt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % 50 == 0:
            print(f"step={step} loss={float(loss):.6f} pcc={float(pcc(pred, qry_gt)):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        final, aux = model(qry_pred, qry_event, cal_pred, cal_gt)
        final_loss, final_pcc = float(objective(final, qry_gt)), float(pcc(final, qry_gt))
        residual_std = float(aux["residual"].std())
    result = {
        "tier": "smoke", "fold": args.fold, "split_opened": "train only",
        "record": chosen, "calibration_windows": len(cal_idx), "query_windows": len(qry_idx),
        "steps": args.steps, "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_reduction_fraction": 1 - final_loss / initial_loss,
        "initial_pcc": initial_pcc, "final_pcc": final_pcc,
        "residual_std": residual_std,
    }
    result["pass"] = result["loss_reduction_fraction"] >= 0.5 and residual_std > 1e-5
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["pass"]:
        raise SystemExit("Stage-0 gate failed")


if __name__ == "__main__":
    main()
