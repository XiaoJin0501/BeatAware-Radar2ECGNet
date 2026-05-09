"""
analyze_mmecg_lag_adjusted_pcc.py — quantify timing vs morphology failures.

For each test segment, computes:
  - raw PCC(pred, gt)
  - best PCC after shifting pred within +/- max_lag_ms
  - best lag in ms
  - PCC gain

The script reloads the best checkpoint and test loader, then writes:
  experiments_mmecg/<exp_tag>/<run_label>/results/lag_adjusted_metrics.csv
  experiments_mmecg/<exp_tag>/<run_label>/results/lag_adjusted_summary_by_subject_scene.csv
  experiments_mmecg/<exp_tag>/<run_label>/results/lag_adjusted_global_summary.json

Examples:
  python scripts/analyze_mmecg_lag_adjusted_pcc.py \
    --exp_tag mmecg_reg_samplewise_subject --protocol samplewise

  python scripts/analyze_mmecg_lag_adjusted_pcc.py \
    --exp_tag mmecg_reg_loso_subject --protocol loso --fold_idx 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.mmecg_config import MMECGConfig
from src.data.mmecg_dataset import build_loso_loaders_h5, build_samplewise_loaders_h5
from src.models.BeatAwareNet.radar2ecgnet import BeatAwareRadar2ECGNet


FS = 200
STATE_NAMES = {0: "NB", 1: "IB", 2: "SP", 3: "PE"}


def _run_label(protocol: str, fold_idx: int) -> str:
    return "samplewise" if protocol == "samplewise" else f"fold_{fold_idx:02d}"


def _load_saved_cfg(run_dir: Path) -> MMECGConfig:
    cfg = MMECGConfig()
    saved = json.loads((run_dir / "config.json").read_text())
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _load_model(run_dir: Path, cfg: MMECGConfig, device: torch.device):
    model = BeatAwareRadar2ECGNet(
        input_type="fmcw",
        n_range_bins=cfg.n_range_bins,
        C=cfg.C,
        d_state=cfg.d_state,
        dropout=cfg.dropout,
        use_pam=cfg.use_pam,
        use_emd=cfg.use_emd,
        emd_max_delay=cfg.emd_max_delay,
        use_diffusion=cfg.use_diffusion,
        diff_T=cfg.diff_T,
        diff_ddim_steps=cfg.diff_ddim_steps,
        diff_hidden=cfg.diff_hidden,
        diff_n_blocks=cfg.diff_n_blocks,
        use_output_lag_align=getattr(cfg, "use_output_lag_align", False),
        output_lag_max_samples=int(round(
            getattr(cfg, "output_lag_max_ms", 200.0) / 1000.0 * getattr(cfg, "fs", 200)
        )),
    ).to(device)
    ckpt = torch.load(run_dir / "checkpoints" / "best.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def _build_test_loader(protocol: str, fold_idx: int, cfg: MMECGConfig):
    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        balanced_sampling=False,
        narrow_bandpass=cfg.narrow_bandpass,
    )
    if protocol == "samplewise":
        _, _, loader = build_samplewise_loaders_h5(
            sw_dir=cfg.samplewise_h5_dir, **kwargs,
        )
    else:
        _, _, loader = build_loso_loaders_h5(
            fold_idx=fold_idx,
            loso_dir=cfg.loso_h5_dir,
            **kwargs,
        )
    return loader


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    den = np.sqrt(np.sum(ac * ac) * np.sum(bc * bc)) + 1e-8
    return float(np.sum(ac * bc) / den)


def _shifted_views(pred: np.ndarray, gt: np.ndarray, lag: int):
    """
    Positive lag means pred is shifted later relative to gt, so compare
    pred[lag:] with gt[:-lag]. Negative lag compares pred[:-lag] with gt[-lag:].
    """
    if lag > 0:
        return pred[lag:], gt[:-lag]
    if lag < 0:
        return pred[:lag], gt[-lag:]
    return pred, gt


def _best_shift_pcc(pred: np.ndarray, gt: np.ndarray, max_lag_samples: int):
    best_pcc = -float("inf")
    best_lag = 0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        p, g = _shifted_views(pred, gt, lag)
        if len(p) < len(pred) * 0.75:
            continue
        val = _pcc(p, g)
        if val > best_pcc:
            best_pcc = val
            best_lag = lag
    return best_pcc, best_lag


@torch.no_grad()
def _collect_rows(model, loader, device, max_lag_samples: int):
    rows = []
    seg_id = 0
    for batch in loader:
        radar = batch["radar"].to(device)
        gt_batch = batch["ecg"].numpy()[:, 0]
        pred_batch, _ = model(radar)
        pred_batch = pred_batch.cpu().numpy()[:, 0]
        subjects = batch["subject"].tolist()
        states = batch["state"].tolist()

        for i in range(len(pred_batch)):
            pred = pred_batch[i]
            gt = gt_batch[i]
            raw = _pcc(pred, gt)
            shifted, lag = _best_shift_pcc(pred, gt, max_lag_samples)
            rows.append({
                "segment_id": seg_id,
                "subject_id": int(subjects[i]),
                "scene": STATE_NAMES.get(int(states[i]), "UNK"),
                "raw_pcc": raw,
                "shifted_pcc": shifted,
                "best_lag_samples": int(lag),
                "best_lag_ms": float(lag / FS * 1000.0),
                "pcc_gain": float(shifted - raw),
                "abs_lag_ms": float(abs(lag) / FS * 1000.0),
            })
            seg_id += 1
    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        df.groupby(keys)
        .agg(
            n=("raw_pcc", "size"),
            raw_pcc_mean=("raw_pcc", "mean"),
            shifted_pcc_mean=("shifted_pcc", "mean"),
            pcc_gain_mean=("pcc_gain", "mean"),
            best_lag_ms_median=("best_lag_ms", "median"),
            abs_lag_ms_mean=("abs_lag_ms", "mean"),
            timing_failure_rate=("pcc_gain", lambda x: float((x >= 0.15).mean())),
            morphology_failure_rate=("shifted_pcc", lambda x: float((x < 0.40).mean())),
        )
        .reset_index()
    )


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", required=True)
    parser.add_argument("--protocol", choices=["samplewise", "loso"], default="samplewise")
    parser.add_argument("--fold_idx", type=int, default=1)
    parser.add_argument("--max_lag_ms", type=float, default=300.0)
    args = parser.parse_args()

    run_label = _run_label(args.protocol, args.fold_idx)
    run_dir = ROOT / "experiments_mmecg" / args.exp_tag / run_label
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_saved_cfg(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(run_dir, cfg, device)
    loader = _build_test_loader(args.protocol, args.fold_idx, cfg)
    max_lag_samples = int(round(args.max_lag_ms / 1000.0 * FS))

    df = _collect_rows(model, loader, device, max_lag_samples)
    by_subject_scene = _summarize(df, ["subject_id", "scene"])
    by_scene = _summarize(df, ["scene"])
    by_subject = _summarize(df, ["subject_id"])

    out_csv = result_dir / "lag_adjusted_metrics.csv"
    out_ss = result_dir / "lag_adjusted_summary_by_subject_scene.csv"
    out_scene = result_dir / "lag_adjusted_summary_by_scene.csv"
    out_subject = result_dir / "lag_adjusted_summary_by_subject.csv"
    df.to_csv(out_csv, index=False)
    by_subject_scene.to_csv(out_ss, index=False)
    by_scene.to_csv(out_scene, index=False)
    by_subject.to_csv(out_subject, index=False)

    global_summary = {
        "exp_tag": args.exp_tag,
        "run_label": run_label,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "checkpoint_val_pcc": ckpt.get("val_pcc", None),
        "max_lag_ms": args.max_lag_ms,
        "n_segments": int(len(df)),
        "raw_pcc_mean": float(df["raw_pcc"].mean()),
        "shifted_pcc_mean": float(df["shifted_pcc"].mean()),
        "pcc_gain_mean": float(df["pcc_gain"].mean()),
        "best_lag_ms_median": float(df["best_lag_ms"].median()),
        "abs_lag_ms_mean": float(df["abs_lag_ms"].mean()),
        "timing_failure_rate_gain_ge_0p15": float((df["pcc_gain"] >= 0.15).mean()),
        "morphology_failure_rate_shifted_pcc_lt_0p40": float((df["shifted_pcc"] < 0.40).mean()),
    }
    with open(result_dir / "lag_adjusted_global_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2, default=_json_default)

    print(json.dumps(global_summary, indent=2, default=_json_default))
    print("\nBy subject-scene:")
    print(by_subject_scene.round(4).to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
