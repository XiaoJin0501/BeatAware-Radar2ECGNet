#!/usr/bin/env python3
"""Evaluate MMECG models under the ECG-shape-200 protocol.

This protocol compares predicted ECG shape outputs in the paper-style
single-cycle 200-point space instead of the full 8-second 1600-point window.
It is intended for fair comparison with radarODE-MTL, whose ECG_shape head
outputs (B, 1, 200).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_radarode_mtl_ega_mmecg import (  # noqa: E402
    fold_paths,
    make_trainer,
    normalize_01,
    resample_1d,
)
from configs.mmecg_config import MMECGConfig  # noqa: E402
from scripts.test_mmecg import _load_model as load_beataware_model  # noqa: E402
from src.data.mmecg_dataset import (  # noqa: E402
    build_loso_calibration_loaders_h5,
    build_loso_loaders_h5,
    build_loso_wsst_loaders_h5,
)
from scripts.train_airecg_mmecg_strict_loso_v2 import (  # noqa: E402
    build_airecg_loaders,
    build_model_and_diffusions as build_airecg_model_and_diffusions,
    sample_batch as sample_airecg_batch,
)
from scripts.train_rssrnet_mmecg_strict_loso import (  # noqa: E402
    build_rssrnet_loaders,
    build_model as build_rssrnet_model,
    forward_to_ecg as forward_rssrnet_to_ecg,
)


FS_ORIG = 200
SHAPE_LEN = 200
STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}


def _uses_wsst_loader(model_arch: str) -> bool:
    return model_arch.lower() in {
        "stft_tcn", "wsst_tcn", "rem_event", "remalign_event",
        "rem_align", "remalign", "remalign_net",
        "rem_align_mtl", "remalign_mtl", "remalign_net_mtl",
        "rem_align_mtl_refine", "remalign_mtl_refine", "remalign_net_mtl_refine",
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return str(obj)


def _as_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _clean_indices(arr: Any, lo: int = 0, hi: int = 1600) -> np.ndarray:
    out = np.asarray(arr).astype(np.int64)
    return np.sort(np.unique(out[(out >= lo) & (out < hi)]))


def _cycle_bounds(ecg: np.ndarray, rpeaks: np.ndarray) -> tuple[int, int]:
    rpeaks = np.sort(np.unique(rpeaks))
    if rpeaks.size >= 2:
        center = 800
        pairs = [(int(a), int(b)) for a, b in zip(rpeaks[:-1], rpeaks[1:]) if int(b) > int(a) + 10]
        if pairs:
            containing = [(a, b) for a, b in pairs if a <= center <= b]
            if containing:
                return containing[0]
            return min(pairs, key=lambda p: abs(((p[0] + p[1]) / 2.0) - center))
    return 700, 900


def _map_to_shape(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    arr = _clean_indices(arr, lo=a, hi=b)
    if b <= a:
        return np.empty((0,), dtype=np.int64)
    mapped = np.rint((arr - a) / float(b - a) * (SHAPE_LEN - 1)).astype(np.int64)
    return _clean_indices(mapped, lo=0, hi=SHAPE_LEN)


def _pcc(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred - float(np.mean(pred))
    g = gt - float(np.mean(gt))
    denom = float(np.sqrt(np.sum(p * p) * np.sum(g * g))) + 1e-8
    return float(np.sum(p * g) / denom)


def _waveform_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1)
    err = pred - gt
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((gt - float(np.mean(gt))) ** 2)) + 1e-8
    return {
        "pcc_raw": _pcc(pred, gt),
        "mae_norm": float(np.mean(np.abs(err))),
        "rmse_norm": float(np.sqrt(np.mean(err * err))),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def _detect_pred_fiducials(pred: np.ndarray, fs_shape: float) -> dict[str, np.ndarray]:
    empty = np.empty((0,), dtype=np.int64)
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(pred, sampling_rate=fs_shape, method="neurokit")
        r = np.asarray(info.get("ECG_R_Peaks", empty), dtype=np.int64)
        r = _clean_indices(r, lo=0, hi=SHAPE_LEN)
        if r.size == 0:
            return {"r": empty, "q": empty, "s": empty, "t": empty}
        _, waves = nk.ecg_delineate(pred, rpeaks=r, sampling_rate=fs_shape, method="dwt")
        return {
            "r": r,
            "q": _clean_indices(waves.get("ECG_Q_Peaks", empty), lo=0, hi=SHAPE_LEN),
            "s": _clean_indices(waves.get("ECG_S_Peaks", empty), lo=0, hi=SHAPE_LEN),
            "t": _clean_indices(waves.get("ECG_T_Peaks", empty), lo=0, hi=SHAPE_LEN),
        }
    except Exception:
        return {"r": empty, "q": empty, "s": empty, "t": empty}


def _match_peaks(pred: np.ndarray, gt: np.ndarray, tol: int) -> list[tuple[int, int]]:
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    used = np.zeros(len(pred), dtype=bool)
    matches: list[tuple[int, int]] = []
    for gi, gp in enumerate(gt):
        if pred.size == 0:
            break
        d = np.abs(pred - gp)
        order = np.argsort(d)
        for pi in order:
            if not used[pi] and d[pi] <= tol:
                used[pi] = True
                matches.append((int(pi), int(gi)))
                break
    return matches


def _nearest_errors_ms(pred: np.ndarray, gt: np.ndarray, fs_shape: float) -> list[float]:
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    if pred.size == 0 or gt.size == 0:
        return []
    return [float(np.min(np.abs(pred - g)) / fs_shape * 1000.0) for g in gt]


def _beat_intervals(fid: dict[str, np.ndarray]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    q, r, s, t = fid["q"], fid["r"], fid["s"], fid["t"]
    for rv in r:
        q_before = q[q <= rv]
        s_after = s[s >= rv]
        t_after = t[t >= rv]
        row: dict[str, float] = {}
        if q_before.size and s_after.size:
            row["qrs"] = float(s_after[0] - q_before[-1])
        if q_before.size and t_after.size:
            row["qt"] = float(t_after[0] - q_before[-1])
        out[int(rv)] = row
    return out


def _peak_interval_metrics(
    pred_fid: dict[str, np.ndarray],
    gt_fid: dict[str, np.ndarray],
    fs_shape: float,
) -> dict[str, float]:
    tol = max(1, int(round(0.150 * fs_shape)))
    r_matches = _match_peaks(pred_fid["r"], gt_fid["r"], tol)
    tp = len(r_matches)
    fp = max(0, len(pred_fid["r"]) - tp)
    fn = max(0, len(gt_fid["r"]) - tp)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    row: dict[str, float] = {
        "num_r_gt": float(len(gt_fid["r"])),
        "num_r_pred": float(len(pred_fid["r"])),
        "num_matched_r": float(tp),
        "rpeak_precision_150ms": float(precision),
        "rpeak_recall_150ms": float(recall),
        "rpeak_f1_150ms": float(f1),
        "qualified_flag": float(int(len(gt_fid["r"]) >= 1 and len(pred_fid["r"]) >= 1 and tp >= 1)),
    }

    for name in ("q", "r", "s", "t"):
        errs = _nearest_errors_ms(pred_fid[name], gt_fid[name], fs_shape)
        row[f"{name}_peak_error_ms"] = float(np.mean(errs)) if errs else float("nan")

    pred_intervals = _beat_intervals(pred_fid)
    gt_intervals = _beat_intervals(gt_fid)
    qrs_errs, qt_errs = [], []
    for pi, gi in r_matches:
        pr = int(pred_fid["r"][pi])
        gr = int(gt_fid["r"][gi])
        if "qrs" in pred_intervals.get(pr, {}) and "qrs" in gt_intervals.get(gr, {}):
            qrs_errs.append(abs(pred_intervals[pr]["qrs"] - gt_intervals[gr]["qrs"]) / fs_shape * 1000.0)
        if "qt" in pred_intervals.get(pr, {}) and "qt" in gt_intervals.get(gr, {}):
            qt_errs.append(abs(pred_intervals[pr]["qt"] - gt_intervals[gr]["qt"]) / fs_shape * 1000.0)
    row["qrs_duration_error_ms"] = float(np.mean(qrs_errs)) if qrs_errs else float("nan")
    row["qt_interval_error_ms"] = float(np.mean(qt_errs)) if qt_errs else float("nan")
    return row


def _load_radarode_model(exp_tag: str, fold_idx: int, device: torch.device):
    run_dir = ROOT / "experiments_mmecg" / exp_tag / f"fold_{fold_idx:02d}"
    ckpt_path = run_dir / f"best_radarode_mtl_ega_fold{fold_idx:02d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    args = SimpleNamespace(
        lr=5e-3,
        weight_decay=5e-4,
        momentum=0.937,
        t_max=100,
        ega_temp=1.0,
        output_dir=run_dir,
        fold_idx=fold_idx,
    )
    trainer = make_trainer(args)
    model = trainer.model.to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_radarode(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    paths = fold_paths(args.fold_idx)
    wsst_path = paths.test_wsst
    out_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}" / "results_shape200"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_radarode_model(args.exp_tag, args.fold_idx, device)

    rows = []
    preds, gts = [], []
    with h5py.File(paths.test, "r") as h5, h5py.File(wsst_path, "r") as wh5, torch.no_grad():
        n = int(h5["ecg"].shape[0])
        for idx in range(n):
            x = torch.from_numpy(np.asarray(wh5["wsst"][idx: idx + 1], dtype=np.float32)).to(device)
            pred = model(x)["ECG_shape"].detach().cpu().numpy()[0, 0].astype(np.float32)

            ecg = np.asarray(h5["ecg"][idx], dtype=np.float32).squeeze()
            r = _clean_indices(h5["rpeak_indices"][idx])
            q = _clean_indices(h5["q_indices"][idx])
            s = _clean_indices(h5["s_indices"][idx])
            t = _clean_indices(h5["tpeak_indices"][idx])
            a, b = _cycle_bounds(ecg, r)
            cycle = normalize_01(ecg[a:b] if b > a else ecg[700:900])
            gt = normalize_01(resample_1d(cycle, SHAPE_LEN))
            fs_shape = SHAPE_LEN / max((b - a) / FS_ORIG, 1e-6)

            gt_fid = {
                "r": _map_to_shape(r, a, b),
                "q": _map_to_shape(q, a, b),
                "s": _map_to_shape(s, a, b),
                "t": _map_to_shape(t, a, b),
            }
            pred_fid = _detect_pred_fiducials(pred, fs_shape)

            row = {
                "segment_id": idx,
                "subject_id": int(h5["subject_id"][idx]),
                "scene": _as_str(h5["physistatus"][idx]),
                "cycle_start": int(a),
                "cycle_end": int(b),
                "fs_shape": float(fs_shape),
                "delineation_valid": int(h5["delineation_valid"][idx]),
            }
            row.update(_waveform_metrics(pred, gt))
            row.update(_peak_interval_metrics(pred_fid, gt_fid, fs_shape))
            rows.append(row)
            preds.append(pred)
            gts.append(gt)

            if (idx + 1) % 100 == 0:
                print(f"  evaluated {idx + 1}/{n}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_metrics_shape200.csv", index=False)
    np.savez_compressed(out_dir / "predictions_shape200.npz", pred=np.asarray(preds), gt=np.asarray(gts))

    def mean_col(col: str) -> float:
        return float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df else float("nan")

    summary = {
        "protocol": "shape200",
        "model_type": args.model_type,
        "exp_tag": args.exp_tag,
        "fold_idx": args.fold_idx,
        "num_segments": int(len(df)),
        "pcc_raw_mean": mean_col("pcc_raw"),
        "pcc_raw_median": float(df["pcc_raw"].median()),
        "rmse_norm_mean": mean_col("rmse_norm"),
        "mae_norm_mean": mean_col("mae_norm"),
        "r2_mean": mean_col("r2"),
        "qualified_monitoring_rate": mean_col("qualified_flag") * 100.0,
        "rpeak_f1_150ms_mean": mean_col("rpeak_f1_150ms"),
        "r_peak_error_ms_mean": mean_col("r_peak_error_ms"),
        "q_peak_error_ms_mean": mean_col("q_peak_error_ms"),
        "s_peak_error_ms_mean": mean_col("s_peak_error_ms"),
        "t_peak_error_ms_mean": mean_col("t_peak_error_ms"),
        "qrs_duration_error_ms_mean": mean_col("qrs_duration_error_ms"),
        "qt_interval_error_ms_mean": mean_col("qt_interval_error_ms"),
        "note": (
            "Shape200 compares single-cycle ECG_shape outputs. Peak/interval metrics are "
            "computed in the resampled cycle coordinate and are not identical to the old "
            "8-second full-window clinical interval protocol."
        ),
    }
    (out_dir / "global_summary_shape200.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _load_saved_config(exp_tag: str, fold_idx: int) -> dict[str, Any]:
    cfg_path = ROOT / "experiments_mmecg" / exp_tag / f"fold_{fold_idx:02d}" / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _loader_overrides(saved: dict[str, Any]) -> dict[str, Any]:
    topk = saved.get("topk_bins", 0)
    return {
        "model_arch": saved.get("model_arch", "beataware"),
        "loso_h5_dir": saved.get("loso_h5_dir", MMECGConfig().loso_h5_dir),
        "narrow_bandpass": bool(saved.get("narrow_bandpass", False)),
        "topk_bins": topk if (topk and topk > 0) else None,
        "target_norm": saved.get("target_norm", "minmax"),
        "topk_method": saved.get("topk_method", "energy"),
        "wsst_suffix": saved.get("wsst_suffix", "_wsst.h5"),
    }


def _full_ecg_to_shape200(ecg: np.ndarray, rpeaks: np.ndarray) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32).reshape(-1)
    a, b = _cycle_bounds(ecg, rpeaks)
    cycle = normalize_01(ecg[a:b] if b > a else ecg[700:900])
    return normalize_01(resample_1d(cycle, SHAPE_LEN))


def evaluate_beataware(args: argparse.Namespace) -> dict[str, Any]:
    saved = _load_saved_config(args.exp_tag, args.fold_idx)
    protocol = saved.get("protocol", "loso")
    cfg = MMECGConfig()
    cfg.batch_size = int(saved.get("batch_size", 16))
    cfg.num_workers = int(saved.get("num_workers", 4))

    overrides = _loader_overrides(saved)
    model_arch = overrides.pop("model_arch", "beataware")
    loso_h5_dir = overrides.pop("loso_h5_dir", cfg.loso_h5_dir)
    wsst_suffix = overrides.pop("wsst_suffix", "_wsst.h5")
    if protocol == "loso_calib":
        if _uses_wsst_loader(model_arch):
            raise NotImplementedError("WSST loaders are currently implemented for strict LOSO only.")
        _, _, test_loader = build_loso_calibration_loaders_h5(
            fold_idx=args.fold_idx,
            loso_dir=loso_h5_dir,
            calib_ratio=float(saved.get("calib_ratio", 0.4)),
            calib_val_ratio=float(saved.get("calib_val_ratio", 0.1)),
            calib_n_train=saved.get("calib_n_train", None),
            calib_n_val=saved.get("calib_n_val", None),
            calib_seed=int(saved.get("calib_seed", 42)),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            balanced_sampling=False,
            **overrides,
        )
    else:
        if _uses_wsst_loader(model_arch):
            _, _, test_loader = build_loso_wsst_loaders_h5(
                fold_idx=args.fold_idx,
                loso_dir=loso_h5_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                balanced_sampling=False,
                target_norm=overrides.get("target_norm", "minmax"),
                wsst_suffix=wsst_suffix,
            )
        else:
            _, _, test_loader = build_loso_loaders_h5(
                fold_idx=args.fold_idx,
                loso_dir=loso_h5_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                balanced_sampling=False,
                **overrides,
            )

    device = torch.device(args.device)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    out_dir = run_dir / "results_shape200"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_beataware_model(cfg, run_dir, device)

    rows = []
    preds, gts = [], []
    seg_id = 0
    with torch.no_grad():
        for batch in test_loader:
            radar = batch["radar"].to(device)
            pred_full, _ = model(radar)
            pred_full_np = pred_full.detach().cpu().numpy()[:, 0]
            gt_full_np = batch["ecg"].numpy()[:, 0]
            for i in range(pred_full_np.shape[0]):
                r = _clean_indices(batch["r_idx"][i])
                pred_shape = _full_ecg_to_shape200(pred_full_np[i], r)
                gt_shape = _full_ecg_to_shape200(gt_full_np[i], r)
                row = {
                    "segment_id": seg_id,
                    "subject_id": int(batch["subject"][i].item()),
                    "scene": STATE_NAMES.get(int(batch["state"][i].item()), "UNK"),
                }
                row.update(_waveform_metrics(pred_shape, gt_shape))
                rows.append(row)
                preds.append(pred_shape)
                gts.append(gt_shape)
                seg_id += 1
            if seg_id and seg_id % 100 == 0:
                print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_metrics_shape200.csv", index=False)
    np.savez_compressed(out_dir / "predictions_shape200.npz", pred=np.asarray(preds), gt=np.asarray(gts))

    summary = {
        "protocol": "shape200",
        "model_type": args.model_type,
        "exp_tag": args.exp_tag,
        "fold_idx": args.fold_idx,
        "source_protocol": protocol,
        "num_segments": int(len(df)),
        "pcc_raw_mean": float(df["pcc_raw"].mean()),
        "pcc_raw_median": float(df["pcc_raw"].median()),
        "rmse_norm_mean": float(df["rmse_norm"].mean()),
        "mae_norm_mean": float(df["mae_norm"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "note": (
            "BeatAware full-window predictions are cropped to the same GT middle-cycle "
            "bounds and resampled to 200 points for shape-only comparison."
        ),
    }
    (out_dir / "global_summary_shape200.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def evaluate_airecg(args: argparse.Namespace) -> dict[str, Any]:
    saved = _load_saved_config(args.exp_tag, args.fold_idx)
    device = torch.device(args.device)
    seed = int(saved.get("seed", 42)) + int(args.fold_idx) * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    out_dir = run_dir / "results_shape200"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(saved.get("batch_size", 16))
    num_workers = int(saved.get("num_workers", 4))
    sampling_steps = int(saved.get("sampling_steps", 50))
    _, _, test_loader = build_airecg_loaders(args.fold_idx, run_dir, batch_size, num_workers=num_workers)

    _, ema, _, diff_sample = build_airecg_model_and_diffusions(device, sampling_steps)
    ckpt_path = run_dir / "checkpoints" / "best_ema.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    ema.load_state_dict(state["ema"])
    ema.eval()

    rows = []
    preds, gts = [], []
    seg_id = 0
    with torch.no_grad():
        for batch in test_loader:
            y1 = batch["y1"].to(device, non_blocking=True)
            y2 = batch["y2"].to(device, non_blocking=True)
            pred_full = sample_airecg_batch(ema, diff_sample, y1, y2, device).detach().cpu().numpy()[:, 0]
            gt_full = batch["ecg_1600"].numpy()[:, 0]
            for i in range(pred_full.shape[0]):
                r = _clean_indices(batch["r_idx"][i])
                pred_shape = _full_ecg_to_shape200(pred_full[i], r)
                gt_shape = _full_ecg_to_shape200(gt_full[i], r)
                row = {
                    "segment_id": seg_id,
                    "subject_id": int(batch["subject"][i].item()),
                    "scene": STATE_NAMES.get(int(batch["state"][i].item()), "UNK"),
                }
                row.update(_waveform_metrics(pred_shape, gt_shape))
                rows.append(row)
                preds.append(pred_shape)
                gts.append(gt_shape)
                seg_id += 1
            if seg_id and seg_id % 100 == 0:
                print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_metrics_shape200.csv", index=False)
    np.savez_compressed(out_dir / "predictions_shape200.npz", pred=np.asarray(preds), gt=np.asarray(gts))

    summary = {
        "protocol": "shape200",
        "model_type": args.model_type,
        "exp_tag": args.exp_tag,
        "fold_idx": args.fold_idx,
        "source_protocol": saved.get("protocol", "strict_loso"),
        "num_segments": int(len(df)),
        "pcc_raw_mean": float(df["pcc_raw"].mean()),
        "pcc_raw_median": float(df["pcc_raw"].median()),
        "rmse_norm_mean": float(df["rmse_norm"].mean()),
        "mae_norm_mean": float(df["mae_norm"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "note": "AirECG full-window predictions are cropped to the GT middle-cycle and resampled to 200 points.",
    }
    (out_dir / "global_summary_shape200.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def evaluate_rssrnet(args: argparse.Namespace) -> dict[str, Any]:
    saved = _load_saved_config(args.exp_tag, args.fold_idx)
    device = torch.device(args.device)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    out_dir = run_dir / "results_shape200"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(saved.get("batch_size", 16))
    num_workers = int(saved.get("num_workers", 4))
    _, _, test_loader = build_rssrnet_loaders(args.fold_idx, batch_size, num_workers=num_workers)

    model = build_rssrnet_model(device)
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    preds, gts = [], []
    seg_id = 0
    with torch.no_grad():
        for batch in test_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            pred_full, _ = forward_rssrnet_to_ecg(model, radar)
            pred_full_np = pred_full.clamp(0.0, 1.0).detach().cpu().numpy()[:, 0]
            gt_full_np = batch["ecg"].numpy()[:, 0]
            for i in range(pred_full_np.shape[0]):
                r = _clean_indices(batch["r_idx"][i])
                pred_shape = _full_ecg_to_shape200(pred_full_np[i], r)
                gt_shape = _full_ecg_to_shape200(gt_full_np[i], r)
                row = {
                    "segment_id": seg_id,
                    "subject_id": int(batch["subject"][i].item()),
                    "scene": STATE_NAMES.get(int(batch["state"][i].item()), "UNK"),
                }
                row.update(_waveform_metrics(pred_shape, gt_shape))
                rows.append(row)
                preds.append(pred_shape)
                gts.append(gt_shape)
                seg_id += 1
            if seg_id and seg_id % 100 == 0:
                print(f"  evaluated {seg_id}/{len(test_loader.dataset)}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "segment_metrics_shape200.csv", index=False)
    np.savez_compressed(out_dir / "predictions_shape200.npz", pred=np.asarray(preds), gt=np.asarray(gts))

    summary = {
        "protocol": "shape200",
        "model_type": args.model_type,
        "exp_tag": args.exp_tag,
        "fold_idx": args.fold_idx,
        "source_protocol": saved.get("protocol", "strict_loso"),
        "num_segments": int(len(df)),
        "pcc_raw_mean": float(df["pcc_raw"].mean()),
        "pcc_raw_median": float(df["pcc_raw"].median()),
        "rmse_norm_mean": float(df["rmse_norm"].mean()),
        "mae_norm_mean": float(df["mae_norm"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "note": "RSSRNet full-window predictions are cropped to the GT middle-cycle and resampled to 200 points.",
    }
    (out_dir / "global_summary_shape200.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        choices=["radarode_mtl_ega", "beataware", "continuous", "airecg", "rssrnet"],
        default="radarode_mtl_ega",
    )
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.model_type == "radarode_mtl_ega":
        evaluate_radarode(args)
    elif args.model_type in ("beataware", "continuous"):
        evaluate_beataware(args)
    elif args.model_type == "airecg":
        evaluate_airecg(args)
    elif args.model_type == "rssrnet":
        evaluate_rssrnet(args)
    else:
        raise ValueError(args.model_type)


if __name__ == "__main__":
    main()
