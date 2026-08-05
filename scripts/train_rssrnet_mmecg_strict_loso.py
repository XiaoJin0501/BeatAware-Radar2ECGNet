"""
Faithful Wu et al. 2024 (RSSRnet) reproduction on MMECG (strict LOSO, ECG-only).

Model + STFT pipeline are reimplemented from paper Figs. 4-6 + Eqs. 6-14
(no source code released by authors). Only the I/O layer adapts MMECG's
50-bin FMCW @ 200 Hz to Wu's 1-channel displacement @ 100 Hz + STFT input.

Reproduction deviations from the original RSSRnet (documented in plan §9.7):
  W1 dataset      Schellenberger 25 subj 24 GHz CW → MMECG 11 subj 77 GHz FMCW
  W2 protocol     25-fold LOSO → strict LOSO 11-fold (matches AirECG/radarODE-MTL)
  W3 architecture double-branch (ECG+RSP) → single-branch (ECG only)
  W4 preprocessing ellipse-fit + DACM phase demod → top-1 cardiac-band-energy bin
  W5 sample rate   100 Hz native → 100 Hz via 200→100 Hz resample of MMECG
  W6 batch / LR    20 / unspec → 16 / 1e-3 + cosine annealing
  W7 epochs        early stop (unspec) → 200 ep, patience 30

Usage:
  python scripts/train_rssrnet_mmecg_strict_loso.py --exp_tag mmecg_rssrnet_strict_loso_v1 \
      --fold_idx 1 --mode train_test
  python scripts/train_rssrnet_mmecg_strict_loso.py --exp_tag mmecg_rssrnet_strict_loso_v1 \
      --fold_idx -1 --mode test  # loso_summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.data.mmecg_dataset import MMECGWindowedH5Dataset, _collate_with_peaks   # noqa: E402
from src.data.wu_adapter import (                                                 # noqa: E402
    radar_to_stft_input, stft_output_to_ecg, ecg_gt_to_100hz,
    SR_NATIVE, SR_TARGET, WIN_LEN_NATIVE, WIN_LEN_TARGET, compute_t_stft,
)
from src.models.baselines.rssrnet import RSSRnetECG, rssrnet_loss                # noqa: E402
from src.utils.metrics import (                                                   # noqa: E402
    compute_waveform_metrics_protocol,
    summarize_subject_metrics, summarize_global_metrics,
)
from test_mmecg import _evaluate_segment, STATE_NAMES                            # noqa: E402

LOSO_DIR = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
EXP_ROOT = PROJECT_ROOT / "experiments_mmecg"


# =============================================================================
# 1. Loader builder — MMECG with topk_bins=1 returns (1, 1600) single-bin RCG
# =============================================================================

def _custom_collate_plain(batch: list[dict]) -> dict:
    """Stack radar/ecg/subject/state for train/val (no peak indices)."""
    result = {
        "radar":   torch.stack([b["radar"] for b in batch]),
        "ecg":     torch.stack([b["ecg"] for b in batch]),
        "subject": torch.tensor([b["subject"] for b in batch], dtype=torch.long),
        "state":   torch.tensor([b["state"] for b in batch], dtype=torch.long),
    }
    return result


def build_rssrnet_loaders(
    fold_idx: int, batch_size: int, num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    fold_data = LOSO_DIR / f"fold_{fold_idx:02d}"
    train_h5, val_h5, test_h5 = fold_data / "train.h5", fold_data / "val.h5", fold_data / "test.h5"
    for p in (train_h5, val_h5, test_h5):
        if not p.exists():
            raise FileNotFoundError(p)

    common = dict(
        narrow_bandpass=False,        # let STFT see full 0.5-20 Hz dynamics
        target_norm="minmax",
        topk_bins=1,                  # single-bin RCG → Wu paper's 1D displacement
        topk_method="energy",
    )
    train_ds = MMECGWindowedH5Dataset(train_h5, include_peak_indices=False, **common)
    val_ds   = MMECGWindowedH5Dataset(val_h5,   include_peak_indices=False, **common)
    test_ds  = MMECGWindowedH5Dataset(test_h5,  include_peak_indices=True,  **common)

    # Wu paper uses no per-subject balancing; plain shuffle.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_custom_collate_plain,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_custom_collate_plain,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_with_peaks,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# 2. Model factory
# =============================================================================

def build_model(device: torch.device) -> RSSRnetECG:
    # v2: t_stft is no longer a constructor arg; model is shape-agnostic over time.
    _ = compute_t_stft()   # keep import side-effect-free
    return RSSRnetECG().to(device)


# =============================================================================
# 3. Forward through the full pipeline (radar → ECG @ 200 Hz)
# =============================================================================

def forward_to_ecg(model: torch.nn.Module, radar_1ch_200hz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pred_ecg_200hz, pred_ecg_100hz) tensors.

    100 Hz output is convenient for training loss (matches paper sample rate).
    200 Hz output is for metric computation (matches MMECG protocol).
    """
    stft_in = radar_to_stft_input(radar_1ch_200hz)
    stft_out = model(stft_in)
    ecg_200hz = stft_output_to_ecg(stft_out)
    # also keep 100 Hz prediction for training loss
    ecg_100hz = F.interpolate(ecg_200hz, size=WIN_LEN_TARGET, mode="linear", align_corners=False)
    return ecg_200hz, ecg_100hz


# =============================================================================
# 4. Training step (one epoch)
# =============================================================================

def train_one_epoch(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loader: DataLoader, device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)         # (B, 1, 1600)
        ecg_200hz = batch["ecg"].to(device, non_blocking=True)       # (B, 1, 1600) ∈ [0,1]

        # forward
        _, pred_100hz = forward_to_ecg(model, radar)                 # (B, 1, 800)
        gt_100hz = ecg_gt_to_100hz(ecg_200hz)                        # (B, 1, 800)

        loss, _ = rssrnet_loss(pred_100hz, gt_100hz, alpha_wave=0.9, alpha_spec=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


# =============================================================================
# 5. Evaluation (val = waveform metrics only; test = full 4-level)
# =============================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
    with_peaks: bool = False,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    seg_rows: list[dict] = []
    beat_rows: list[dict] = []
    seg_id = 0

    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        ecg_200hz = batch["ecg"]                                     # (B, 1, 1600) ∈ [0,1]
        pred_200hz, _ = forward_to_ecg(model, radar)                  # (B, 1, 1600)
        pred_200hz = pred_200hz.clamp(0.0, 1.0).detach().cpu()

        subj_arr = batch["subject"].tolist()
        state_arr = batch["state"].tolist()
        B = pred_200hz.shape[0]
        pred_np = pred_200hz.squeeze(1).numpy()
        gt_np = ecg_200hz.squeeze(1).numpy()

        if with_peaks:
            r_list = batch["r_idx"]; q_list = batch["q_idx"]
            s_list = batch["s_idx"]; t_list = batch["t_idx"]
            delin = batch["delin_valid"].tolist() if torch.is_tensor(batch["delin_valid"]) \
                    else batch["delin_valid"]
            for i in range(B):
                row, brows = _evaluate_segment(
                    seg_id=seg_id, pred_1d=pred_np[i], gt_1d=gt_np[i],
                    gt_r=r_list[i], gt_q=q_list[i], gt_s=s_list[i], gt_t=t_list[i],
                    subject_id=subj_arr[i], state_code=state_arr[i],
                    delin_valid=bool(delin[i]) if isinstance(delin, list) else bool(delin[i]),
                )
                seg_rows.append(row); beat_rows.extend(brows); seg_id += 1
        else:
            for i in range(B):
                m = compute_waveform_metrics_protocol(pred_np[i], gt_np[i])
                seg_rows.append({
                    "segment_id": seg_id,
                    "subject_id": subj_arr[i],
                    "scene": STATE_NAMES.get(state_arr[i], "UNK"),
                    "pcc_raw":   m["pcc_raw"],
                    "rmse_norm": m["rmse_norm"],
                    "mae_norm":  m["mae_norm"],
                    "r2":        m["r2"],
                })
                seg_id += 1
    return seg_rows, beat_rows


# =============================================================================
# 6. Train / test / loso_summary
# =============================================================================

def train_fold(args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = EXP_ROOT / args.exp_tag / f"fold_{args.fold_idx:02d}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "results").mkdir(exist_ok=True)
    with open(fold_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, _ = build_rssrnet_loaders(
        args.fold_idx, args.batch_size, num_workers=args.num_workers,
    )
    print(f"[fold {args.fold_idx:02d}] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)}", flush=True)

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fold {args.fold_idx:02d}] RSSRnetECG params = {n_params:,}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    best_pcc = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device)
        scheduler.step()
        t_train = time.time() - t0

        log_row = {"epoch": epoch, "train_loss": avg_loss, "t_train_s": t_train,
                   "lr": float(optimizer.param_groups[0]["lr"])}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            val_rows, _ = evaluate(model, val_loader, device, with_peaks=False)
            val_pcc = float(np.mean([r["pcc_raw"] for r in val_rows]))
            val_mae = float(np.mean([r["mae_norm"] for r in val_rows]))
            t_val = time.time() - t1
            log_row.update({"val_pcc": val_pcc, "val_mae": val_mae, "t_val_s": t_val})
            print(f"[fold {args.fold_idx:02d}] epoch {epoch}/{args.epochs} "
                  f"loss={avg_loss:.4f} val_pcc={val_pcc:.4f} val_mae={val_mae:.4f} "
                  f"({t_train:.1f}s train, {t_val:.1f}s val)", flush=True)

            if val_pcc > best_pcc:
                best_pcc = val_pcc
                patience_counter = 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_pcc": val_pcc,
                    "val_mae": val_mae,
                }, fold_dir / "checkpoints/best.pt")
            else:
                patience_counter += args.val_every
                if patience_counter >= args.early_stop_patience:
                    print(f"[fold {args.fold_idx:02d}] Early stop at epoch {epoch} "
                          f"(best val_pcc={best_pcc:.4f})", flush=True)
                    history.append(log_row)
                    break
        else:
            print(f"[fold {args.fold_idx:02d}] epoch {epoch}/{args.epochs} "
                  f"loss={avg_loss:.4f} ({t_train:.1f}s)", flush=True)

        history.append(log_row)

    pd.DataFrame(history).to_csv(fold_dir / "results/train_history.csv", index=False)
    print(f"[fold {args.fold_idx:02d}] training done. best val_pcc={best_pcc:.4f}", flush=True)
    return fold_dir


def test_fold(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = EXP_ROOT / args.exp_tag / f"fold_{args.fold_idx:02d}"
    ckpt_path = fold_dir / "checkpoints/best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    _, _, test_loader = build_rssrnet_loaders(
        args.fold_idx, args.batch_size, num_workers=args.num_workers,
    )
    print(f"[fold {args.fold_idx:02d}] test={len(test_loader.dataset)}", flush=True)

    model = build_model(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[fold {args.fold_idx:02d}] loaded ckpt epoch={state.get('epoch')} "
          f"val_pcc={state.get('val_pcc'):.4f}", flush=True)

    t0 = time.time()
    seg_rows, beat_rows = evaluate(model, test_loader, device, with_peaks=True)
    print(f"[fold {args.fold_idx:02d}] evaluated {len(seg_rows)} segments "
          f"in {time.time() - t0:.1f}s", flush=True)

    out_dir = fold_dir / "results"
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(seg_rows).to_csv(out_dir / "segment_metrics.csv", index=False)
    pd.DataFrame(beat_rows).to_csv(out_dir / "beat_metrics.csv", index=False)
    summarize_subject_metrics(seg_rows).to_csv(out_dir / "subject_summary.csv", index=False)
    global_dict = summarize_global_metrics(seg_rows)
    with open(out_dir / "global_summary.json", "w") as f:
        json.dump(global_dict, f, indent=2)
    print(f"[fold {args.fold_idx:02d}] PCC_raw_mean = {global_dict.get('pcc_raw_mean'):.4f}",
          flush=True)


def loso_summary(args) -> None:
    out_dir = EXP_ROOT / args.exp_tag / "loso_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seg_rows: list[dict] = []
    all_beat_rows: list[dict] = []
    folds_found = []
    for k in range(1, 12):
        seg_csv = EXP_ROOT / args.exp_tag / f"fold_{k:02d}/results/segment_metrics.csv"
        beat_csv = EXP_ROOT / args.exp_tag / f"fold_{k:02d}/results/beat_metrics.csv"
        if not seg_csv.exists():
            print(f"  fold_{k:02d}: missing segment_metrics.csv, skip")
            continue
        all_seg_rows.extend(pd.read_csv(seg_csv).to_dict(orient="records"))
        if beat_csv.exists():
            all_beat_rows.extend(pd.read_csv(beat_csv).to_dict(orient="records"))
        folds_found.append(k)
    print(f"loso_summary: found {len(folds_found)} folds: {folds_found}", flush=True)

    pd.DataFrame(all_seg_rows).to_csv(out_dir / "all_segments.csv", index=False)
    if all_beat_rows:
        pd.DataFrame(all_beat_rows).to_csv(out_dir / "all_beats.csv", index=False)
    summarize_subject_metrics(all_seg_rows).to_csv(out_dir / "subject_summary.csv", index=False)
    global_dict = summarize_global_metrics(all_seg_rows)
    global_dict["num_folds"] = len(folds_found)
    with open(out_dir / "global_summary.json", "w") as f:
        json.dump(global_dict, f, indent=2)
    print(f"loso_summary: PCC_raw_mean = {global_dict.get('pcc_raw_mean'):.4f} "
          f"({global_dict.get('num_segments')} segments, "
          f"{global_dict.get('num_subjects')} subjects)", flush=True)


# =============================================================================
# 7. CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_tag", type=str, default="mmecg_rssrnet_strict_loso_v1")
    parser.add_argument("--protocol", type=str, default="strict_loso", choices=["strict_loso"])
    parser.add_argument("--fold_idx", type=int, required=True,
                        help="1..11, or -1 to aggregate loso_summary")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="epochs without val_pcc improvement before stop")
    parser.add_argument("--mode", type=str, default="train_test",
                        choices=["train", "test", "train_test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.fold_idx == -1:
        loso_summary(args)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode in ("train", "train_test"):
        train_fold(args)
    if args.mode in ("test", "train_test"):
        test_fold(args)


if __name__ == "__main__":
    main()
