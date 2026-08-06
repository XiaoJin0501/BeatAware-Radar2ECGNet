#!/usr/bin/env python3
"""Evaluate MMECG checkpoints with record-level ECG assembly.

This script is the bridge from the current Shape-200 pilot evidence to the
radarODE-style long ECG question. It re-runs inference, keeps H5 metadata
(`record_id`, `start_idx`, `end_idx`), overlap-adds predictions within each
recording, and evaluates the assembled signal.

Supported protocols:
  - rem_full_ola: REM/BeatAware-style 1x1600 output, overlap-add full windows.
  - rem_shape_oracle_cycle_ola: REM shape200 head placed into GT center-cycle
    bounds; an oracle morphology/assembly diagnostic, not deployable final.
  - radarode_shape_oracle_cycle_ola: radarODE ECG_shape head placed into GT
    center-cycle bounds; comparable to Shape-200 morphology but assembled in
    record coordinates. This is still an oracle-cycle diagnostic until the
    predicted Anchor/PPI assembly protocol is implemented.
  - radarode_pred_anchor_ppi_cycle_ola: radarODE ECG_shape placed using the
    predicted Anchor peak and predicted PPI length. This is the first
    deployable radarODE-style assembly approximation for the processed MMECG
    H5 representation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from scripts.evaluate_mmecg_shape200 import _load_radarode_model  # noqa: E402
from scripts.test_mmecg import _load_model as load_mmecg_model  # noqa: E402
from scripts.train_mmecg import _shape200_targets  # noqa: E402
from scripts.train_radarode_mtl_ega_mmecg import normalize_01, resample_1d  # noqa: E402


FS = 200
WIN_LEN = 1600
SHAPE_LEN = 200
STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _decode(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_saved_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _apply_saved_config(cfg: MMECGConfig, saved: dict[str, Any]) -> None:
    for key, value in saved.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _h5_paths(exp_tag: str, fold_idx: int, saved: dict[str, Any], wsst_suffix: str | None) -> tuple[Path, Path, Path]:
    cfg = MMECGConfig()
    loso_h5_dir = Path(saved.get("loso_h5_dir", cfg.loso_h5_dir))
    fold_dir = loso_h5_dir / f"fold_{fold_idx:02d}"
    test_h5 = fold_dir / "test.h5"
    suffix = wsst_suffix or saved.get("wsst_suffix", "_wsst.h5")
    if exp_tag.startswith("mmecg_radarode") or suffix == "_wsst4s.h5":
        suffix = "_wsst4s.h5"
    test_wsst = test_h5.with_name(f"test{suffix}")
    run_dir = ROOT / "experiments_mmecg" / exp_tag / f"fold_{fold_idx:02d}"
    return run_dir, test_h5, test_wsst


def _minmax_ecg(ecg: np.ndarray) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32)
    lo = float(np.min(ecg))
    hi = float(np.max(ecg))
    if hi - lo < 1e-8:
        return np.zeros_like(ecg, dtype=np.float32)
    return ((ecg - lo) / (hi - lo)).astype(np.float32)


def _clean_indices(arr: Any, lo: int = 0, hi: int = WIN_LEN) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int64)
    return np.sort(np.unique(out[(out >= lo) & (out < hi)]))


def _cycle_bounds_from_rpeaks(rpeaks: np.ndarray, length: int = WIN_LEN) -> tuple[int, int]:
    rpeaks = _clean_indices(rpeaks, 0, length)
    if rpeaks.size >= 2:
        center = length // 2
        pairs = [(int(a), int(b)) for a, b in zip(rpeaks[:-1], rpeaks[1:]) if int(b) > int(a) + 10]
        if pairs:
            containing = [(a, b) for a, b in pairs if a <= center <= b]
            if containing:
                return containing[0]
            return min(pairs, key=lambda p: abs(((p[0] + p[1]) / 2.0) - center))
    return 700, 900


def _cycle_bounds(h5: h5py.File, idx: int) -> tuple[int, int]:
    if "cycle_bounds" in h5:
        a, b = np.asarray(h5["cycle_bounds"][idx], dtype=np.int64).tolist()
        if int(b) > int(a) + 10:
            return int(a), int(b)
    return _cycle_bounds_from_rpeaks(h5["rpeak_indices"][idx])


def _resample_to_len(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(resample_1d(np.asarray(x, dtype=np.float32), n), dtype=np.float32)


def _pcc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    if pred.size < 2 or gt.size < 2:
        return float("nan")
    p = pred - float(np.mean(pred))
    g = gt - float(np.mean(gt))
    denom = float(np.sqrt(np.sum(p * p) * np.sum(g * g))) + 1e-8
    return float(np.sum(p * g) / denom)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    err = pred - gt
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((gt - float(np.mean(gt))) ** 2)) + 1e-8
    return {
        "pcc": _pcc(pred, gt),
        "rmse": float(np.sqrt(np.mean(err * err))) if err.size else float("nan"),
        "mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "r2": float(1.0 - ss_res / ss_tot) if err.size else float("nan"),
    }


def _extract_ecg_tensor(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)) and model_out and torch.is_tensor(model_out[0]):
        return model_out[0]
    raise TypeError(f"Unsupported model output: {type(model_out)!r}")


class _Accumulator:
    def __init__(self, length: int) -> None:
        self.pred_sum = np.zeros(length, dtype=np.float64)
        self.gt_sum = np.zeros(length, dtype=np.float64)
        self.weight = np.zeros(length, dtype=np.float64)

    def add(self, start: int, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1)
        n = min(pred.size, gt.size)
        if n <= 0:
            return
        start = int(start)
        end = min(start + n, self.weight.size)
        if end <= start:
            return
        local_n = end - start
        self.pred_sum[start:end] += pred[:local_n]
        self.gt_sum[start:end] += gt[:local_n]
        self.weight[start:end] += 1.0

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.weight > 0
        pred = np.zeros_like(self.pred_sum, dtype=np.float32)
        gt = np.zeros_like(self.gt_sum, dtype=np.float32)
        pred[mask] = (self.pred_sum[mask] / self.weight[mask]).astype(np.float32)
        gt[mask] = (self.gt_sum[mask] / self.weight[mask]).astype(np.float32)
        return pred, gt, mask


def _record_lengths(h5: h5py.File) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for idx in range(int(h5["ecg"].shape[0])):
        rec = _decode(h5["record_id"][idx])
        end = int(h5["end_idx"][idx])
        lengths[rec] = max(lengths.get(rec, 0), end)
    return lengths


def _make_batch(indices: list[int], h5: h5py.File, wh5: h5py.File) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radar = torch.from_numpy(np.asarray(wh5["wsst"][indices], dtype=np.float32))
    ecg = np.stack([_minmax_ecg(np.asarray(h5["ecg"][i], dtype=np.float32)) for i in indices], axis=0)
    bounds = np.asarray([_cycle_bounds(h5, i) for i in indices], dtype=np.int64)
    return radar, torch.from_numpy(ecg), torch.from_numpy(bounds)


def _init_model(args: argparse.Namespace, run_dir: Path, device: torch.device):
    if args.protocol.startswith("radarode"):
        return _load_radarode_model(args.exp_tag, args.fold_idx, device)
    cfg = MMECGConfig()
    saved = _load_saved_config(run_dir)
    _apply_saved_config(cfg, saved)
    return load_mmecg_model(cfg, run_dir, device)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    saved = _load_saved_config(ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}")
    run_dir, test_h5, test_wsst = _h5_paths(args.exp_tag, args.fold_idx, saved, args.wsst_suffix)
    if not test_h5.exists():
        raise FileNotFoundError(test_h5)
    if not test_wsst.exists():
        raise FileNotFoundError(test_wsst)

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "results_long"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _init_model(args, run_dir, device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with h5py.File(test_h5, "r") as h5, h5py.File(test_wsst, "r") as wh5, torch.no_grad():
        n = int(h5["ecg"].shape[0])
        acc = {rec: _Accumulator(length) for rec, length in _record_lengths(h5).items()}
        meta: dict[str, dict[str, Any]] = {}
        extra_anchor_argmax: list[int] = []
        extra_ppi_argmax: list[int] = []

        for lo in range(0, n, args.batch_size):
            hi = min(lo + args.batch_size, n)
            indices = list(range(lo, hi))
            radar, ecg, bounds = _make_batch(indices, h5, wh5)
            radar = radar.to(device, non_blocking=True)
            ecg = ecg.to(device, non_blocking=True)
            bounds = bounds.to(device, non_blocking=True)

            if args.protocol.startswith("radarode"):
                out = model(radar)
                pred_shape = out["ECG_shape"].detach().cpu().numpy()[:, 0]
                anchor = out.get("Anchor")
                ppi = out.get("PPI")
                anchor_argmax_batch: list[int | None] = [None] * len(indices)
                ppi_argmax_batch: list[int | None] = [None] * len(indices)
                if anchor is not None:
                    anchor_np = anchor.detach().cpu().numpy()
                    anchor_argmax_batch = np.argmax(anchor_np.reshape(anchor_np.shape[0], -1), axis=1).astype(int).tolist()
                    extra_anchor_argmax.extend(anchor_argmax_batch)
                if ppi is not None:
                    ppi_np = ppi.detach().cpu().numpy()
                    ppi_argmax_batch = np.argmax(ppi_np.reshape(ppi_np.shape[0], -1), axis=1).astype(int).tolist()
                    extra_ppi_argmax.extend(ppi_argmax_batch)
                pred_full = None
                target_shape = None
            else:
                ecg_out = _extract_ecg_tensor(model(radar))
                pred_full = ecg_out.detach().cpu().numpy()[:, 0]
                if args.protocol.startswith("rem_shape"):
                    if not hasattr(ecg_out, "shape200_pred"):
                        raise AttributeError("Selected rem_shape protocol but model output has no shape200_pred attribute.")
                    pred_shape = ecg_out.shape200_pred.detach().cpu().numpy()[:, 0]
                    target_shape = _shape200_targets(ecg, bounds).detach().cpu().numpy()[:, 0]
                else:
                    pred_shape = None
                    target_shape = None

            ecg_np = ecg.detach().cpu().numpy()[:, 0]
            bounds_np = bounds.detach().cpu().numpy()
            for j, idx in enumerate(indices):
                rec = _decode(h5["record_id"][idx])
                scene = _decode(h5["physistatus"][idx])
                subject = int(h5["subject_id"][idx])
                start = int(h5["start_idx"][idx])
                end = int(h5["end_idx"][idx])
                meta.setdefault(rec, {"record_id": rec, "subject_id": subject, "scene": scene})

                if args.protocol == "rem_full_ola":
                    pred_piece = pred_full[j]
                    gt_piece = ecg_np[j]
                    piece_start = start
                elif args.protocol in {"rem_shape_oracle_cycle_ola", "radarode_shape_oracle_cycle_ola"}:
                    a, b = int(bounds_np[j, 0]), int(bounds_np[j, 1])
                    pred_piece = _resample_to_len(pred_shape[j], max(0, b - a))
                    if target_shape is not None:
                        gt_piece = _resample_to_len(target_shape[j], max(0, b - a))
                    else:
                        gt_cycle = normalize_01(ecg_np[j, a:b] if b > a else ecg_np[j, 700:900])
                        gt_piece = _resample_to_len(gt_cycle, max(0, b - a))
                    piece_start = start + a
                elif args.protocol == "radarode_pred_anchor_ppi_cycle_ola":
                    if anchor_argmax_batch[j] is None or ppi_argmax_batch[j] is None:
                        raise ValueError("radarode_pred_anchor_ppi_cycle_ola requires Anchor and PPI outputs.")
                    pred_local_start = int(400 + int(anchor_argmax_batch[j]))
                    pred_len = int(np.clip(int(ppi_argmax_batch[j]) + 1, 40, 260))
                    pred_local_end = min(WIN_LEN, pred_local_start + pred_len)
                    if pred_local_end <= pred_local_start:
                        pred_piece = np.empty((0,), dtype=np.float32)
                        gt_piece = np.empty((0,), dtype=np.float32)
                    else:
                        pred_piece = _resample_to_len(pred_shape[j], pred_local_end - pred_local_start)
                        gt_piece = ecg_np[j, pred_local_start:pred_local_end]
                    piece_start = start + pred_local_start
                else:
                    raise ValueError(f"Unknown protocol: {args.protocol}")

                acc[rec].add(piece_start, pred_piece, gt_piece)
                row = {
                    "window_id": idx,
                    "record_id": rec,
                    "subject_id": subject,
                    "scene": scene,
                    "start_idx": start,
                    "end_idx": end,
                    "cycle_start": int(bounds_np[j, 0]),
                    "cycle_end": int(bounds_np[j, 1]),
                    "piece_start": int(piece_start),
                    "piece_len": int(len(pred_piece)),
                }
                if args.protocol.startswith("radarode"):
                    row["anchor_argmax"] = (
                        int(anchor_argmax_batch[j]) if anchor_argmax_batch[j] is not None else -1
                    )
                    row["ppi_argmax"] = int(ppi_argmax_batch[j]) if ppi_argmax_batch[j] is not None else -1
                row.update(_metrics(pred_piece, gt_piece))
                rows.append(row)

            print(f"  assembled {hi}/{n} windows")

        record_rows: list[dict[str, Any]] = []
        record_ids: list[str] = []
        pred_records: list[np.ndarray] = []
        gt_records: list[np.ndarray] = []
        mask_records: list[np.ndarray] = []
        for rec, accumulator in acc.items():
            pred, gt, mask = accumulator.finalize()
            covered = int(mask.sum())
            rec_row = {
                **meta.get(rec, {"record_id": rec}),
                "num_samples": int(mask.size),
                "covered_samples": covered,
                "coverage_ratio": float(covered / max(mask.size, 1)),
            }
            if covered >= args.min_covered_samples:
                rec_row.update(_metrics(pred[mask], gt[mask]))
            else:
                rec_row.update({"pcc": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")})
            record_rows.append(rec_row)
            record_ids.append(rec)
            pred_records.append(pred.astype(np.float32))
            gt_records.append(gt.astype(np.float32))
            mask_records.append(mask.astype(np.bool_))

    window_df = pd.DataFrame(rows)
    record_df = pd.DataFrame(record_rows).sort_values(["subject_id", "scene", "record_id"])
    window_df.to_csv(out_dir / "window_metrics_long.csv", index=False)
    record_df.to_csv(out_dir / "record_metrics_long.csv", index=False)
    np.savez_compressed(
        out_dir / "predictions_long.npz",
        record_id=np.asarray(record_ids, dtype=object),
        pred=np.asarray(pred_records, dtype=object),
        gt=np.asarray(gt_records, dtype=object),
        mask=np.asarray(mask_records, dtype=object),
    )

    summary = {
        "protocol": args.protocol,
        "dataset": "MMECG",
        "exp_tag": args.exp_tag,
        "fold_idx": int(args.fold_idx),
        "checkpoint": str(run_dir / ("checkpoints/best.pt" if not args.protocol.startswith("radarode") else f"best_radarode_mtl_ega_fold{args.fold_idx:02d}.pt")),
        "test_h5": str(test_h5),
        "test_wsst": str(test_wsst),
        "output_dir": str(out_dir),
        "num_windows": int(len(window_df)),
        "num_records": int(len(record_df)),
        "record_pcc_mean": float(pd.to_numeric(record_df["pcc"], errors="coerce").mean()),
        "record_pcc_median": float(pd.to_numeric(record_df["pcc"], errors="coerce").median()),
        "record_rmse_mean": float(pd.to_numeric(record_df["rmse"], errors="coerce").mean()),
        "record_mae_mean": float(pd.to_numeric(record_df["mae"], errors="coerce").mean()),
        "record_r2_mean": float(pd.to_numeric(record_df["r2"], errors="coerce").mean()),
        "coverage_ratio_mean": float(pd.to_numeric(record_df["coverage_ratio"], errors="coerce").mean()),
        "publishable_status": "pilot_until_long_metric_contract_and_radarode_pred_anchor_protocol_are_locked",
        "note": (
            "Record-level overlap-add on processed, per-window normalized H5 data. "
            "Protocols using oracle_cycle_ola place 200-point morphology predictions into GT center-cycle bounds; "
            "they are diagnostics/upper bounds, not final deployable long ECG claims."
        ),
    }
    (out_dir / "global_summary_long.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument(
        "--protocol",
        required=True,
        choices=[
            "rem_full_ola",
            "rem_shape_oracle_cycle_ola",
            "radarode_shape_oracle_cycle_ola",
            "radarode_pred_anchor_ppi_cycle_ola",
        ],
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wsst_suffix", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--min_covered_samples", type=int, default=200)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
