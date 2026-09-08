#!/usr/bin/env python3
"""Validation-only multi-metric evaluation for controlled REM-Net development."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from scripts.train_anchor_head import match_counts, peaks_from_heatmap
from scripts.train_mmecg import _shape200_targets
from src.data.mmecg_dataset import build_loso_wsst_loaders_h5
from src.models.model_factory import build_mmecg_model
from src.utils.metrics import detect_rpeaks


def waveform_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rows = []
    for p, y in zip(pred, target):
        p, y = p.reshape(-1), y.reshape(-1)
        err = p - y
        yc, pc = y - y.mean(), p - p.mean()
        den = np.sqrt(np.sum(yc * yc) * np.sum(pc * pc))
        rows.append((np.sum(yc * pc) / max(den, 1e-8),
                     np.sqrt(np.mean(err * err)), np.mean(np.abs(err)),
                     1.0 - np.sum(err * err) / max(np.sum(yc * yc), 1e-8)))
    arr = np.asarray(rows, dtype=np.float64)
    return {name: float(np.nanmean(arr[:, i])) for i, name in enumerate(("pcc", "rmse", "mae", "r2"))}


def f1_summary(counts: dict[int, list[int]]) -> dict[str, float]:
    out = {}
    for ms, (tp, fp, fn) in counts.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        out[f"f1_{ms}ms"] = 2 * precision * recall / max(precision + recall, 1e-8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", default="best_loss.pt")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold:02d}"
    saved = json.loads((run / "config.json").read_text())
    cfg = MMECGConfig()
    for key, value in saved.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    _, loader, _ = build_loso_wsst_loaders_h5(
        args.fold, cfg.loso_h5_dir, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, balanced_sampling=False,
        target_norm=cfg.target_norm, wsst_suffix=cfg.wsst_suffix,
    )
    device = torch.device(args.device)
    model = build_mmecg_model(cfg).to(device)
    ckpt_path = run / "checkpoints" / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    waveform_pred, waveform_gt, shape_pred, shape_gt = [], [], [], []
    event_counts = {50: [0, 0, 0], 100: [0, 0, 0], 150: [0, 0, 0]}
    waveform_counts = {50: [0, 0, 0], 100: [0, 0, 0], 150: [0, 0, 0]}
    valid_cycles = total_cycles = waveform_peak_windows = 0
    with torch.no_grad():
        for batch in loader:
            radar, target = batch["radar"].to(device), batch["ecg"].to(device)
            output, events = model(radar)
            waveform_pred.append(output.cpu().numpy()[:, 0])
            waveform_gt.append(target.cpu().numpy()[:, 0])
            shape_pred.append(output.shape200_pred.cpu().numpy()[:, 0])
            shape_gt.append(_shape200_targets(target, batch["cycle_bounds"]).cpu().numpy()[:, 0])
            valid_cycles += int(batch["cycle_valid"].sum())
            total_cycles += len(target)
            masks = events[0].cpu().numpy()[:, 0]
            gt_masks = batch["rpeak"].numpy()[:, 0]
            for predicted_waveform, mask, gt_mask in zip(output.cpu().numpy()[:, 0], masks, gt_masks):
                gt_peaks = peaks_from_heatmap(gt_mask)
                event_peaks = peaks_from_heatmap(mask)
                wave_peaks = detect_rpeaks(predicted_waveform, fs=cfg.fs)
                waveform_peak_windows += int(len(wave_peaks) > 0)
                for ms in event_counts:
                    tolerance = int(round(ms * cfg.fs / 1000))
                    for store, peaks in ((event_counts, event_peaks), (waveform_counts, wave_peaks)):
                        values = match_counts(peaks, gt_peaks, tolerance)
                        for j, value in enumerate(values):
                            store[ms][j] += value
    pred, target = np.concatenate(waveform_pred), np.concatenate(waveform_gt)
    spred, starget = np.concatenate(shape_pred), np.concatenate(shape_gt)
    result = {
        "tier": "pilot", "publishable": False, "split": "validation",
        "exp_tag": args.exp_tag, "fold": args.fold,
        "checkpoint": str(ckpt_path), "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_rule": "minimum common validation base loss",
        "num_windows": int(len(pred)), "valid_cycle_coverage": valid_cycles / max(total_cycles, 1),
        "waveform_peak_detection_coverage": waveform_peak_windows / max(len(pred), 1),
        "waveform": waveform_metrics(pred, target),
        "shape200": waveform_metrics(spred, starget),
        "event_head": f1_summary(event_counts),
        "waveform_derived": f1_summary(waveform_counts),
    }
    out = run / "results_validation"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
