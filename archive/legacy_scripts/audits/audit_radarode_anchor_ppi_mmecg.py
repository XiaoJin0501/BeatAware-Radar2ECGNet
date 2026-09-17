#!/usr/bin/env python3
"""Audit adapted radarODE-MTL Anchor/PPI outputs on MMECG.

This is a protocol-audit script, not a paper-result evaluator. It checks the
coordinate assumptions needed before using radarODE's predicted Anchor/PPI
heads for long ECG assembly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mmecg_long_assembly import (  # noqa: E402
    FS,
    WIN_LEN,
    _clean_indices,
    _cycle_bounds,
    _decode,
    _json_default,
)
from scripts.evaluate_mmecg_shape200 import _load_radarode_model  # noqa: E402
from scripts.train_radarode_mtl_ega_mmecg import fold_paths  # noqa: E402


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _gaussian_anchor_target(rpeaks: np.ndarray, length: int = 800, sigma: float = 4.0) -> np.ndarray:
    mask = np.zeros(length, dtype=np.float32)
    local = _clean_indices(np.asarray(rpeaks, dtype=np.int64) - 400, lo=0, hi=length)
    x = np.arange(length, dtype=np.float32)
    for peak in local:
        mask = np.maximum(mask, np.exp(-0.5 * ((x - float(peak)) / sigma) ** 2).astype(np.float32))
    return mask


def _nearest_error(pred: int, candidates: np.ndarray) -> tuple[int, float]:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0:
        return -1, float("nan")
    idx = int(np.argmin(np.abs(candidates - pred)))
    nearest = int(candidates[idx])
    return nearest, float(abs(pred - nearest) / FS * 1000.0)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    paths = fold_paths(args.fold_idx)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / f"fold_{args.fold_idx:02d}"
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "anchor_ppi_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_radarode_model(args.exp_tag, args.fold_idx, device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with h5py.File(paths.test, "r") as h5, h5py.File(paths.test_wsst, "r") as wh5, torch.no_grad():
        n = int(h5["ecg"].shape[0])
        for lo in range(0, n, args.batch_size):
            hi = min(lo + args.batch_size, n)
            x = torch.from_numpy(np.asarray(wh5["wsst"][lo:hi], dtype=np.float32)).to(device)
            out = model(x)
            anchor = out["Anchor"].detach().cpu().numpy()
            ppi = out["PPI"].detach().cpu().numpy()
            anchor_flat = anchor.reshape(anchor.shape[0], -1)
            ppi_flat = ppi.reshape(ppi.shape[0], -1)

            for j, idx in enumerate(range(lo, hi)):
                rpeaks = _clean_indices(h5["rpeak_indices"][idx])
                local_r = _clean_indices(rpeaks - 400, lo=0, hi=800)
                a, b = _cycle_bounds(h5, idx)
                gt_len = int(max(0, b - a))

                pred_anchor_local = int(np.argmax(anchor_flat[j]))
                pred_anchor_global = pred_anchor_local + 400
                pred_ppi_len = int(np.argmax(ppi_flat[j]) + 1)
                gt_anchor_mask = _gaussian_anchor_target(rpeaks)
                gt_anchor_local = int(np.argmax(gt_anchor_mask)) if gt_anchor_mask.max() > 0 else -1
                gt_anchor_global = gt_anchor_local + 400 if gt_anchor_local >= 0 else -1
                nearest_local, anchor_err_ms = _nearest_error(pred_anchor_local, local_r)
                nearest_global = nearest_local + 400 if nearest_local >= 0 else -1

                rows.append(
                    {
                        "window_id": int(idx),
                        "record_id": _decode(h5["record_id"][idx]),
                        "subject_id": int(h5["subject_id"][idx]),
                        "scene": _decode(h5["physistatus"][idx]),
                        "start_idx": int(h5["start_idx"][idx]),
                        "end_idx": int(h5["end_idx"][idx]),
                        "cycle_start": int(a),
                        "cycle_end": int(b),
                        "gt_cycle_len": gt_len,
                        "pred_ppi_len": pred_ppi_len,
                        "ppi_abs_error_samples": int(abs(pred_ppi_len - gt_len)),
                        "ppi_abs_error_ms": float(abs(pred_ppi_len - gt_len) / FS * 1000.0),
                        "pred_anchor_local": pred_anchor_local,
                        "pred_anchor_global": pred_anchor_global,
                        "gt_anchor_local_argmax": gt_anchor_local,
                        "gt_anchor_global_argmax": gt_anchor_global,
                        "nearest_gt_r_local": nearest_local,
                        "nearest_gt_r_global": nearest_global,
                        "anchor_nearest_error_ms": anchor_err_ms,
                        "num_gt_r_in_anchor_window": int(local_r.size),
                        "anchor_peak_value": float(anchor_flat[j, pred_anchor_local]),
                    }
                )
            print(f"  audited {hi}/{n} windows")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "anchor_ppi_audit.csv", index=False)

    summary = {
        "protocol": "radarode_anchor_ppi_audit",
        "dataset": "MMECG",
        "exp_tag": args.exp_tag,
        "fold_idx": int(args.fold_idx),
        "num_windows": int(len(df)),
        "anchor_error_ms_mean": float(pd.to_numeric(df["anchor_nearest_error_ms"], errors="coerce").mean()),
        "anchor_error_ms_median": float(pd.to_numeric(df["anchor_nearest_error_ms"], errors="coerce").median()),
        "anchor_within_50ms_rate": float((pd.to_numeric(df["anchor_nearest_error_ms"], errors="coerce") <= 50.0).mean()),
        "ppi_error_ms_mean": float(pd.to_numeric(df["ppi_abs_error_ms"], errors="coerce").mean()),
        "ppi_error_ms_median": float(pd.to_numeric(df["ppi_abs_error_ms"], errors="coerce").median()),
        "ppi_within_50ms_rate": float((pd.to_numeric(df["ppi_abs_error_ms"], errors="coerce") <= 50.0).mean()),
        "output_dir": str(out_dir),
        "publishable_status": "audit_only_not_paper_result",
        "note": (
            "Audits adapted radarODE Anchor/PPI coordinate assumptions. "
            "Use this to decide whether predicted-anchor long assembly is credible."
        ),
    }
    (out_dir / "global_summary_anchor_ppi_audit.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=160)
        axes[0].hist(pd.to_numeric(df["anchor_nearest_error_ms"], errors="coerce").dropna(), bins=30, color="#4C78A8")
        axes[0].set_title("Anchor nearest-R error")
        axes[0].set_xlabel("ms")
        axes[0].set_ylabel("Windows")
        axes[1].hist(pd.to_numeric(df["ppi_abs_error_ms"], errors="coerce").dropna(), bins=30, color="#E15759")
        axes[1].set_title("PPI length error")
        axes[1].set_xlabel("ms")
        fig.suptitle(f"radarODE Anchor/PPI Audit Fold-{args.fold_idx:02d}")
        fig.tight_layout()
        fig.savefig(out_dir / "anchor_ppi_audit_hist.png", bbox_inches="tight")
    except Exception as exc:
        summary["plot_error"] = str(exc)

    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_tag", default="mmecg_radarode_mtl_ega_strict_loso")
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    audit(args)


if __name__ == "__main__":
    main()
