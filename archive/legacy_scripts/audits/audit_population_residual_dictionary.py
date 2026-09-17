#!/usr/bin/env python3
"""Continuous population-residual dictionary ceiling and accessibility audit."""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.train_calibration_conditioned_rem import freeze_predictions  # noqa: E402
from scripts.train_continuous_ecg_pilot import waveform_metrics  # noqa: E402

OUT = ROOT / "experiments_mmecg/q1_population_residual_dictionary_audit/results.json"
RANK, FEATURE_RANK, RIDGE = 32, 64, 1.0


def pooled_features(data: dict) -> np.ndarray:
    pred = data["pred"].numpy()[:, 0].reshape(-1, 100, 16).mean(2)
    event = data["event"].numpy().reshape(-1, 2, 100, 16).mean(3).reshape(-1, 200)
    return np.concatenate([pred, event], axis=1).astype(np.float64)


def fit_feature_map(x: np.ndarray, rank: int):
    mu, sd = x.mean(0), x.std(0); sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    basis = vt[:rank]
    return mu, sd, basis, z @ basis.T


def run_fold(fold: int, device: torch.device) -> dict:
    train = freeze_predictions(fold, "train", device)
    val = freeze_predictions(fold, "val", device)
    tr_pred, tr_gt = train["pred"].numpy()[:, 0].astype(np.float64), train["gt"].numpy()[:, 0].astype(np.float64)
    va_pred, va_gt = val["pred"].numpy()[:, 0].astype(np.float64), val["gt"].numpy()[:, 0].astype(np.float64)
    residual = tr_gt - tr_pred
    mean = residual.mean(0)
    centered = residual - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dictionary = vt[:RANK]
    tr_coeff = centered @ dictionary.T

    val_residual = va_gt - va_pred
    oracle_coeff = (val_residual - mean) @ dictionary.T
    oracle = va_pred + mean + oracle_coeff @ dictionary

    fx_mu, fx_sd, fx_basis, tr_feat = fit_feature_map(pooled_features(train), FEATURE_RANK)
    va_feat = ((pooled_features(val) - fx_mu) / fx_sd) @ fx_basis.T
    design = np.concatenate([tr_feat, np.ones((len(tr_feat), 1))], axis=1)
    reg = np.eye(design.shape[1]) * RIDGE; reg[-1, -1] = 0
    weights = np.linalg.solve(design.T @ design + reg, design.T @ tr_coeff)
    va_design = np.concatenate([va_feat, np.ones((len(va_feat), 1))], axis=1)
    ridge = va_pred + mean + (va_design @ weights) @ dictionary
    arms = {"B1": va_pred, "mean": va_pred + mean, "oracle_pca32": oracle,
            "radar_ridge_pca32": ridge}
    metrics = {name: waveform_metrics(pred, va_gt) for name, pred in arms.items()}
    energy = np.square(centered @ dictionary.T).sum() / max(np.square(centered).sum(), 1e-12)
    return {"fold": fold, "n_train": len(tr_pred), "n_val": len(va_pred),
            "train_residual_energy_explained": float(energy), "metrics": metrics,
            "delta_ridge_vs_B1_pcc": metrics["radar_ridge_pca32"]["pcc"] - metrics["B1"]["pcc"],
            "delta_ridge_vs_mean_pcc": metrics["radar_ridge_pca32"]["pcc"] - metrics["mean"]["pcc"]}


def main() -> None:
    if not torch.cuda.is_available(): raise SystemExit("CUDA required for frozen B1 inference")
    rows = [run_fold(f, torch.device("cuda")) for f in (5, 9)]
    macro_delta = float(np.mean([r["delta_ridge_vs_B1_pcc"] for r in rows]))
    representable = all(r["metrics"]["oracle_pca32"]["pcc"] >= .75 for r in rows)
    accessible = all(r["delta_ridge_vs_B1_pcc"] > 0 and r["delta_ridge_vs_mean_pcc"] > 0
                     and r["metrics"]["radar_ridge_pca32"]["r2"] > r["metrics"]["B1"]["r2"] for r in rows)
    result = {"experiment_id": "q1_population_residual_dictionary_audit", "tier": "pilot",
              "test_accessed": False, "shape200_computed": False, "folds": rows,
              "gate": {"representable": representable, "radar_accessible": accessible,
                       "macro_delta_pcc": macro_delta,
                       "pass": representable and accessible and macro_delta >= .08}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
