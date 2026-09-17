#!/usr/bin/env python3
"""Fixed-budget 20% supervised calibration probe; validation only."""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.models.continuous.calibrated_rem import CalibrationConditionedREM  # noqa: E402
from scripts.train_calibration_conditioned_rem import (  # noqa: E402
    DATA, GUIDED_OUT, apply_lag_affine_frontend, evaluate, forward_record,
    freeze_predictions, loss_fn, seed_all,
)
from scripts.train_continuous_ecg_pilot import waveform_metrics, timing_metrics, composite  # noqa: E402

OUT = ROOT / "experiments_mmecg/q1_fewshot20_calibrated_rem_probe"


def fraction_groups(data: dict, fraction: float):
    groups, audit = [], []
    for rec in sorted(set(data["record"].tolist())):
        own = data["record"] == rec
        duration = int(data["end"][own].max())
        cutoff = int(round(duration * fraction))
        cal = np.flatnonzero(own & (data["end"] <= cutoff))
        qry = np.flatnonzero(own & (data["start"] >= cutoff))
        if len(cal) and len(qry):
            groups.append((rec, cal, qry))
            audit.append({"record": rec, "duration_samples": duration, "cutoff_sample": cutoff,
                          "calibration_windows": len(cal), "query_windows": len(qry)})
    if not groups: raise RuntimeError("No valid 20/80 record partitions")
    return groups, audit


@torch.no_grad()
def collect_record(model, data, group, device, stores):
    _, cal, qry = group
    model.eval()
    for idx, (pred, _) in forward_record(model, data, cal, qry, device, False):
        stores[0].append(pred.cpu().numpy()[:, 0])
        stores[1].append(data["gt"][idx].numpy()[:, 0])
        stores[2].append(torch.sigmoid(data["event"][idx]).numpy()[:, 0])
        stores[3].extend(data["rpeaks"][i] for i in idx)


def metrics(stores):
    p, g, e = (np.concatenate(stores[i]) for i in range(3))
    m = waveform_metrics(p, g); m.update(timing_metrics(p, e, stores[3])); m["composite"] = composite(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=0.20)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    if (args.fraction, args.steps) != (0.20, 100): raise SystemExit("Formal probe is locked to fraction=0.20, steps=100")
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    seed_all(42 + args.fold); device = torch.device("cuda")
    data = freeze_predictions(args.fold, "val", device)
    groups, partition = fraction_groups(data, args.fraction)
    frontend = apply_lag_affine_frontend(data, groups)

    source = CalibrationConditionedREM().to(device)
    ck = torch.load(GUIDED_OUT / f"full/fold_{args.fold:02d}/checkpoints/best_composite.pt",
                    map_location=device, weights_only=True)
    source.load_state_dict(ck["model_state_dict"])
    conventional = CalibrationConditionedREM().to(device)  # identity by zero-init
    control = evaluate(conventional, data, groups, device)
    source_metrics = evaluate(source, data, groups, device)

    stores = ([], [], [], [])
    record_rows = []
    for rec, cal, qry in groups:
        adapted = copy.deepcopy(source).to(device); adapted.train()
        opt = torch.optim.AdamW(adapted.parameters(), lr=1e-4, weight_decay=1e-4)
        first = None
        for step in range(args.steps):
            losses = []
            for idx, (pred, _) in forward_record(adapted, data, cal, cal, device, True):
                loss = loss_fn(pred, data["gt"][idx].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5); opt.step()
                losses.append(float(loss))
            if first is None: first = float(np.mean(losses))
        collect_record(adapted, data, (rec, cal, qry), device, stores)
        record_rows.append({"record": rec, "calibration_loss_first": first,
                            "calibration_loss_final": float(np.mean(losses))})
    adapted_metrics = metrics(stores)
    result = {"experiment_id": "q1_fewshot20_calibrated_rem_probe", "tier": "probe",
              "fold": args.fold, "protocol": "chronological first 20% supervised calibration; final 80% query",
              "test_accessed": False, "shape200_computed": False, "partitions": partition,
              "frontend": frontend, "adaptation_records": record_rows,
              "metrics": {"conventional": control, "source_rem": source_metrics, "adapted_rem": adapted_metrics}}
    result["gate"] = {"delta_vs_conventional": adapted_metrics["pcc"] - control["pcc"],
                      "delta_vs_source": adapted_metrics["pcc"] - source_metrics["pcc"],
                      "pass": adapted_metrics["pcc"] - control["pcc"] >= 0.05
                              and adapted_metrics["pcc"] - source_metrics["pcc"] >= 0.03
                              and adapted_metrics["rmse"] <= control["rmse"]
                              and adapted_metrics["r2"] >= control["r2"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"fold_{args.fold:02d}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
