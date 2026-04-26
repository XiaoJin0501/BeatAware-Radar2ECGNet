"""
test_mmecg.py — MMECG LOSO 测试脚本

用法：
  python scripts/test_mmecg.py --exp_tag mmecg_D
  python scripts/test_mmecg.py --exp_tag mmecg_D --fold_idx 0

输出：
  experiments_mmecg/<exp_tag>/fold_<N>/results/test_metrics.json
  experiments_mmecg/<exp_tag>/summary_loso.json  ← 全部折汇总
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet
from src.utils.metrics import compute_all_metrics


def test_one_fold(cfg: MMECGConfig, exp_tag: str, fold: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_dir  = Path(cfg.exp_dir) / exp_tag / f"fold_{fold}"
    ckpt_path = fold_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"[fold {fold}] Checkpoint not found: {ckpt_path}")
        return {}

    # ── 加载 config ──────────────────────────────────────────────────────────
    cfg_path = fold_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as jf:
            saved = json.load(jf)
        for k, v in saved.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # ── 数据 ─────────────────────────────────────────────────────────────────
    _, test_loader = build_loso_loaders(
        dataset_dir=cfg.dataset_dir,
        fold_idx=fold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
    )

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        use_pam=cfg.use_pam,
        use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 推理 ─────────────────────────────────────────────────────────────────
    all_pred, all_gt, all_states = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            radar  = batch["radar"].to(device)
            ecg_gt = batch["ecg"].to(device)
            state  = batch["state"]        # int tensor, 0=NB/1=IB/2=SP/3=PE

            ecg_pred, _ = model(radar)
            if not torch.isfinite(ecg_pred).all():
                continue

            all_pred.append(ecg_pred.cpu())
            all_gt.append(ecg_gt.cpu())
            all_states.extend(state.tolist())

    if not all_pred:
        print(f"[fold {fold}] No valid predictions.")
        return {}

    all_pred   = torch.cat(all_pred, dim=0)
    all_gt     = torch.cat(all_gt,   dim=0)
    all_states = torch.tensor(all_states)

    # 整体指标
    metrics = compute_all_metrics(all_pred, all_gt, compute_f1=True)

    # 按 state 分组统计
    state_names = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}
    state_metrics = {}
    for code, name in state_names.items():
        idx = (all_states == code).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        m = compute_all_metrics(all_pred[idx], all_gt[idx], compute_f1=True)
        m["n_samples"] = int(len(idx))
        state_metrics[name] = m
    metrics["per_state"] = state_metrics

    # 读取测试受试者 ID（LOSO: fold_idx = subject index）
    meta_path = Path(cfg.dataset_dir) / "metadata_mmecg.json"
    with open(meta_path) as jf:
        meta = json.load(jf)
    test_subject = meta["loso_folds"][str(fold)]["test"]

    metrics["test_subject"] = test_subject
    metrics["n_samples"]    = int(len(all_pred))

    # 保存
    result_dir = fold_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "test_metrics.json", "w") as jf:
        json.dump(metrics, jf, indent=2)

    print(f"[fold {fold}] subject={test_subject} | "
          f"PCC={metrics['pcc']:.4f} MAE={metrics['mae']:.4f} "
          f"F1={metrics.get('f1', float('nan')):.4f} PRD={metrics['prd']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag",  type=str, required=True)
    parser.add_argument("--fold_idx", type=int, default=-1)
    args = parser.parse_args()

    cfg   = MMECGConfig()
    folds = list(range(cfg.n_folds)) if args.fold_idx == -1 else [args.fold_idx]

    all_metrics = []
    for fold in folds:
        m = test_one_fold(cfg, args.exp_tag, fold)
        if m:
            all_metrics.append(m)

    if not all_metrics:
        print("No results to summarize.")
        return

    # LOSO 整体汇总
    keys = ["pcc", "mae", "rmse", "prd", "f1"]
    summary = {"per_fold": all_metrics}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m and not np.isnan(m[k])]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))

    # per-state 跨折汇总（NB / IB / SP / PE）
    state_names = ["NB", "IB", "SP", "PE"]
    state_summary = {}
    for state in state_names:
        state_vals = {}
        for k in keys:
            vals = [
                m["per_state"][state][k]
                for m in all_metrics
                if "per_state" in m and state in m["per_state"]
                and k in m["per_state"][state]
                and not np.isnan(m["per_state"][state][k])
            ]
            if vals:
                state_vals[f"{k}_mean"] = float(np.mean(vals))
                state_vals[f"{k}_std"]  = float(np.std(vals))
        if state_vals:
            state_summary[state] = state_vals
    summary["per_state_summary"] = state_summary

    summary_path = Path(cfg.exp_dir) / args.exp_tag / "summary_loso.json"
    with open(summary_path, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"\n{'='*50}")
    print(f"LOSO Summary ({len(all_metrics)} folds):")
    for k in keys:
        if f"{k}_mean" in summary:
            print(f"  {k:6s}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}")
    print(f"\nPer-state PCC:")
    for state in state_names:
        if state in state_summary and "pcc_mean" in state_summary[state]:
            s = state_summary[state]
            print(f"  {state}: {s['pcc_mean']:.4f} ± {s['pcc_std']:.4f}")
    print(f"{'='*50}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
