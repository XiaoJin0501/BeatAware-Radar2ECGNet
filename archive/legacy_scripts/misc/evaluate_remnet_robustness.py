"""evaluate_remnet_robustness.py — input-perturbation robustness on the FROZEN REM-Net.

Inference-only (no retraining). Injects additive Gaussian noise into the WSST input
representation at several relative levels and reports Shape-200 PCC degradation. The
noise scale is relative to each window's own WSST standard deviation, so level 0.0
exactly reproduces the frozen final-test number and each higher level is a controlled
corruption of the model's input. Frozen seed-42 checkpoints only; traceable to the
final run directory.

Example:
  python scripts/evaluate_remnet_robustness.py --exp_tag q1_final_remnet_incumbent_mmecg_seed42 \
      --folds 1-11 --levels 0,0.05,0.1,0.2,0.3,0.5 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig  # noqa: E402
from src.data.mmecg_dataset import build_loso_wsst_loaders_h5  # noqa: E402
from scripts.train_mmecg import _shape200_targets  # noqa: E402
from scripts.evaluate_remalign_shape_head import (  # noqa: E402
    _load_saved_config, _apply_saved_config, _load_model, _extract_ecg_tensor, _pcc,
)

ROOT = Path(__file__).resolve().parent.parent


def _parse_folds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--folds", default="1-11")
    ap.add_argument("--levels", default="0,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    folds = _parse_folds(args.folds)
    levels = [float(x) for x in args.levels.split(",")]
    dev = torch.device(args.device)

    # per-level -> list of fold-macro PCCs
    fold_curves = {lv: [] for lv in levels}
    for fold in folds:
        cfg = MMECGConfig()
        run_dir = Path(cfg.exp_dir) / args.exp_tag / f"fold_{fold:02d}"
        saved = _load_saved_config(run_dir); _apply_saved_config(cfg, saved)
        _, _, test_loader = build_loso_wsst_loaders_h5(
            fold_idx=fold, loso_dir=cfg.loso_h5_dir,
            batch_size=int(getattr(cfg, "batch_size", 16)),
            num_workers=int(getattr(cfg, "num_workers", 4)),
            balanced_sampling=False,
            target_norm=getattr(cfg, "target_norm", "minmax"),
            wsst_suffix=getattr(cfg, "wsst_suffix", "_wsst.h5"),
        )
        model = _load_model(cfg, run_dir, dev, checkpoint_name=args.checkpoint)

        per_level = {lv: [] for lv in levels}
        with torch.no_grad():
            for batch in test_loader:
                radar = batch["radar"].to(dev, non_blocking=True)
                ecg = batch["ecg"].to(dev, non_blocking=True)
                cyc = batch["cycle_bounds"].to(dev, non_blocking=True)
                target = _shape200_targets(ecg, cyc)
                tgt_np = target.detach().cpu().numpy()[:, 0, :]
                std = radar.reshape(radar.shape[0], -1).std(dim=1).view(-1, 1, 1, 1)
                for lv in levels:
                    if lv == 0.0:
                        x = radar
                    else:
                        g = torch.Generator(device=dev).manual_seed(args.seed + fold)
                        noise = torch.randn(radar.shape, generator=g, device=dev) * std * lv
                        x = radar + noise
                    pred = _extract_ecg_tensor(model(x)).shape200_pred
                    pred_np = pred.detach().cpu().numpy()[:, 0, :]
                    for i in range(pred_np.shape[0]):
                        per_level[lv].append(_pcc(pred_np[i], tgt_np[i]))

        res = {"exp_tag": args.exp_tag, "fold": fold, "checkpoint": args.checkpoint,
               "noise": "additive_gaussian_wsst_relative_perwindow_std",
               "levels": levels,
               "pcc_mean_by_level": {str(lv): float(np.mean(per_level[lv])) for lv in levels},
               "n_segments": len(per_level[levels[0]])}
        out = run_dir / "results_robustness"; out.mkdir(exist_ok=True)
        (out / "robustness_shape200.json").write_text(json.dumps(res, indent=2))
        for lv in levels:
            fold_curves[lv].append(float(np.mean(per_level[lv])))
        print(f"fold_{fold:02d}: " + "  ".join(
            f"{lv}:{np.mean(per_level[lv]):.3f}" for lv in levels))

    print("\n=== AGGREGATE Shape-200 PCC vs noise level (fold-macro) ===")
    base = np.mean(fold_curves[levels[0]])
    for lv in levels:
        m = np.mean(fold_curves[lv]); s = np.std(fold_curves[lv])
        drop = 100.0 * (m - base) / base
        print(f"  level {lv:<5}: PCC {m:.3f} ± {s:.3f}   ({drop:+.1f}% vs clean)")


if __name__ == "__main__":
    main()
