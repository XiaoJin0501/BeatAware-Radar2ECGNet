#!/usr/bin/env python3
"""Evaluate REM-Align shape200 heads directly.

This script evaluates only models whose returned ECG tensor carries a
`shape200_pred` attribute. It does not crop/resample the full-window ECG
prediction; instead it compares the head output to the same 200-point target
used during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from scripts.train_mmecg import _shape200_targets  # noqa: E402
from src.data.mmecg_dataset import build_loso_wsst_loaders_h5  # noqa: E402
from src.models.model_factory import build_mmecg_model  # noqa: E402


STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}
SUPPORTED_ARCHES = {
    "rem_align_mtl",
    "remalign_mtl",
    "remalign_net_mtl",
    "rem_align_mtl_refine",
    "remalign_mtl_refine",
    "remalign_net_mtl_refine",
    "rem_align_hybrid_mtl",
    "rem_align_hybrid",
    "remalign_hybrid_mtl",
    "rem_phase_mtl",
    "remnet_phase_mtl",
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return str(obj)


def _load_saved_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Saved config not found: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _apply_saved_config(cfg: MMECGConfig, saved: dict[str, Any]) -> None:
    for key, value in saved.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _load_model(
    cfg: MMECGConfig,
    run_dir: Path,
    device: torch.device,
    checkpoint_name: str = "best.pt",
) -> torch.nn.Module:
    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_mmecg_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[load_state_dict] dropped {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    if missing:
        print(f"[load_state_dict] missing {len(missing)} keys (e.g. {missing[:3]})")
    print(
        "Loaded checkpoint: "
        f"epoch={ckpt.get('epoch', '?')} "
        f"val_pcc={float(ckpt.get('val_pcc', float('nan'))):.4f}"
    )
    model.eval()
    return model


def _extract_ecg_tensor(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)) and model_out and torch.is_tensor(model_out[0]):
        return model_out[0]
    raise TypeError(f"Model output does not contain an ECG tensor: {type(model_out)!r}")


def _pcc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    pred_c = pred - float(pred.mean())
    gt_c = gt - float(gt.mean())
    denom = float(np.sqrt(np.sum(pred_c * pred_c) * np.sum(gt_c * gt_c))) + 1e-8
    return float(np.sum(pred_c * gt_c) / denom)


def _waveform_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    err = pred - gt
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((gt - float(gt.mean())) ** 2)) + 1e-8
    return {
        "pcc": _pcc(pred, gt),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    cfg = MMECGConfig()
    run_dir = Path(cfg.exp_dir) / args.exp_tag / f"fold_{args.fold_idx:02d}"
    saved = _load_saved_config(run_dir)
    _apply_saved_config(cfg, saved)

    model_arch = str(getattr(cfg, "model_arch", saved.get("model_arch", ""))).lower()
    if model_arch not in SUPPORTED_ARCHES:
        raise ValueError(
            "evaluate_remalign_shape_head.py is intended for REM-Align MTL shape heads; "
            f"got model_arch={model_arch!r}"
        )

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "results_shape_head"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = build_loso_wsst_loaders_h5(
        fold_idx=args.fold_idx,
        loso_dir=cfg.loso_h5_dir,
        batch_size=int(getattr(cfg, "batch_size", 16)),
        num_workers=int(getattr(cfg, "num_workers", 4)),
        balanced_sampling=False,
        target_norm=getattr(cfg, "target_norm", "minmax"),
        wsst_suffix=getattr(cfg, "wsst_suffix", "_wsst.h5"),
    )
    model = _load_model(cfg, run_dir, device, checkpoint_name=args.checkpoint)

    rows: list[dict[str, Any]] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    segment_ids: list[int] = []
    subject_ids: list[int] = []
    states: list[int] = []

    seg_id = 0
    with torch.no_grad():
        for batch in test_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            ecg = batch["ecg"].to(device, non_blocking=True)
            cycle_bounds = batch["cycle_bounds"].to(device, non_blocking=True)

            ecg_out = _extract_ecg_tensor(model(radar))
            if not hasattr(ecg_out, "shape200_pred"):
                raise AttributeError(
                    "Model output ECG tensor has no shape200_pred attribute; "
                    f"model_arch={model_arch!r}"
                )
            pred = ecg_out.shape200_pred
            target = _shape200_targets(ecg, cycle_bounds).to(pred.device)
            if tuple(pred.shape) != tuple(target.shape):
                raise ValueError(f"shape200_pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")

            pred_np = pred.detach().cpu().numpy()[:, 0, :]
            target_np = target.detach().cpu().numpy()[:, 0, :]
            for i in range(pred_np.shape[0]):
                subject = int(batch["subject"][i].item())
                state = int(batch["state"][i].item())
                row = {
                    "segment_id": seg_id,
                    "subject_id": subject,
                    "state": state,
                    "scene": STATE_NAMES.get(state, "UNK"),
                    "cycle_start": int(batch["cycle_bounds"][i, 0].item()),
                    "cycle_end": int(batch["cycle_bounds"][i, 1].item()),
                }
                row.update(_waveform_metrics(pred_np[i], target_np[i]))
                rows.append(row)
                preds.append(pred_np[i].astype(np.float32))
                targets.append(target_np[i].astype(np.float32))
                segment_ids.append(seg_id)
                subject_ids.append(subject)
                states.append(state)
                seg_id += 1

            if seg_id and seg_id % 100 == 0:
                print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_metrics_shape_head.csv", index=False)
    np.savez_compressed(
        out_dir / "predictions_shape_head.npz",
        pred=np.asarray(preds, dtype=np.float32),
        target=np.asarray(targets, dtype=np.float32),
        segment_id=np.asarray(segment_ids, dtype=np.int64),
        subject_id=np.asarray(subject_ids, dtype=np.int64),
        state=np.asarray(states, dtype=np.int64),
    )

    def mean_col(name: str) -> float:
        return float(df[name].mean()) if len(df) else float("nan")

    summary = {
        "protocol": "shape_head",
        "exp_tag": args.exp_tag,
        "fold_idx": int(args.fold_idx),
        "model_arch": model_arch,
        "source_protocol": saved.get("protocol", "loso"),
        "checkpoint": str(run_dir / "checkpoints" / args.checkpoint),
        "config": str(run_dir / "config.json"),
        "output_dir": str(out_dir),
        "wsst_suffix": getattr(cfg, "wsst_suffix", "_wsst.h5"),
        "num_segments": int(len(df)),
        "pcc_mean": mean_col("pcc"),
        "pcc_median": float(df["pcc"].median()) if len(df) else float("nan"),
        "rmse_mean": mean_col("rmse"),
        "mae_mean": mean_col("mae"),
        "r2_mean": mean_col("r2"),
        "note": "Direct evaluation of the returned ECG tensor's shape200_pred head against _shape200_targets(ecg, cycle_bounds).",
    }
    (out_dir / "global_summary_shape_head.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename under fold_XX/checkpoints, e.g. best.pt or best_shape200.pt.",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
