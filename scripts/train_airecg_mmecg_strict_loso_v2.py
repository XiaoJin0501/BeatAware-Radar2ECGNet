"""
Faithful AirECG (Zhao et al. 2024 IMWUT) reproduction on MMECG (strict LOSO).

Model + diffusion math are imported verbatim from ~/Projects/AirECG.
Only the I/O layer (shapes + value ranges) is adapted for MMECG.

Reproduction deviations from the original AirECG (documented for the paper):
  Δ1 clip_denoised=False at sampling                    — matches AirECG default (v3 restored)
  Δ2 y2 strategy (vs. paper same-subject calib ECG): strict LOSO disallows target-subject info
        train: per-sample random draw from train-fold ECG pool (decorrelates y2 from target so
               the model can't learn subject-specific shortcuts through y2);
        eval : fixed pointwise-median template of the train pool (deterministic; metrics aren't
               drowned by y2 noise) — v3.1 fix; v3 used random y2 at eval which masked results
  Δ3 batch_size=16 epochs=200 (vs. paper 64 / 1400)     — RTX 4080 16GB + small MMECG train set;
                                                          optimizer steps ≥ paper
  Δ4 ECG 1600 → 1024 (vs. paper native 1024)            — match 32x32 patching, metrics @ 1600
  Δ5 mm_channels = 8 (vs. paper 8)                      — top-8 range bins per (subject, scene)
                                                          by 0.8-3.5 Hz cardiac-band energy
                                                          (matches paper's 8-channel mmWave input)

v3 (2026-05-17): Fixed v2 collapse (PCC=0.0005). See plan §1 for diagnosis.
  F1 = topk_bins=8 (was 50 raw)        — match AirECG's input modality
  F2 = per-sample y2 from train pool   — was constant template (dead cross-attn path)
  F3 = clip_denoised=False             — was True (over-restrictive on undertrained model)
Use --exp_tag mmecg_airecg_strict_loso_v3 to keep v3 outputs separate from v2.

Usage:
  python scripts/train_airecg_mmecg_strict_loso_v2.py --exp_tag mmecg_airecg_strict_loso_v3 \
      --fold_idx 1 --mode train_test
  python scripts/train_airecg_mmecg_strict_loso_v2.py --exp_tag mmecg_airecg_strict_loso_v3 \
      --fold_idx -1 --mode test  # loso_summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── AirECG source-of-truth (unmodified) ──────────────────────────────────────
AIRECG_ROOT = Path("/home/qhh2237/Projects/AirECG")
sys.path.insert(0, str(AIRECG_ROOT))
from models import AirECG_model                                 # noqa: E402
from diffusion import create_diffusion                          # noqa: E402

# ── BeatAware-Radar2ECGNet utilities ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from src.data.mmecg_dataset import MMECGWindowedH5Dataset, _minmax_ecg  # noqa: E402
from test_mmecg import _evaluate_segment, STATE_NAMES, FS, WIN_LEN      # noqa: E402
from src.utils.metrics import summarize_subject_metrics, summarize_global_metrics  # noqa: E402

LOSO_DIR = Path("/home/qhh2237/Datasets/MMECG/processed/loso")
EXP_ROOT = PROJECT_ROOT / "experiments_mmecg"
LATENT_SIZE = 32        # AirECG patching: 32×32 = 1024 ECG samples
PATCH_LEN = 1024        # 32×32 reshaped from 1024-length signal
MM_CHANNELS = 8         # v3: match AirECG paper's 8-channel mmWave input (was 50 in v2)
TOPK_METHOD = "energy"  # 0.8-3.5 Hz cardiac-band energy per range bin (per subject, scene)


# =============================================================================
# 1. EMA helper (copied verbatim from ~/Projects/AirECG/train.py:34-44)
# =============================================================================

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# =============================================================================
# 2. Training-pool reference ECG bank (v3: per-sample y2 via random draw)
# =============================================================================

def build_ref_pool(train_h5_path: Path, cache_path: Path) -> torch.Tensor:
    """Return (M, 1, 32, 32) float32 ∈ [-1, 1]; all training-fold ECGs in AirECG format.

    v3 replaces v2's fold-level median template: AirECG cross-attn ingests y2 at every
    layer, so a constant y2 collapses the conditioning path. Drawing y2 at random from
    the train pool per sample preserves strict-LOSO (no target subject leak) while
    giving cross-attn a varying morphological prior.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return torch.from_numpy(np.load(cache_path))

    with h5py.File(train_h5_path, "r") as f:
        ecg_raw = f["ecg"][:]   # (N, 1, 1600) float32 — z-score domain

    # match dataset's per-window min-max normalization
    ecg_01 = np.stack([_minmax_ecg(ecg_raw[i]) for i in range(len(ecg_raw))])  # (N,1,1600) ∈ [0,1]

    pool = np.empty((len(ecg_01), 1, LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
    for i in range(len(ecg_01)):
        # resample 1600 → 1024 (Fourier method to preserve waveform)
        ecg_1024 = scipy.signal.resample(ecg_01[i, 0], PATCH_LEN).astype(np.float32)
        pool[i, 0] = (ecg_1024 * 2.0 - 1.0).reshape(LATENT_SIZE, LATENT_SIZE)

    np.save(cache_path, pool)
    return torch.from_numpy(pool)


# =============================================================================
# 3. AirECG wrapper Dataset (I/O adaptation only)
# =============================================================================

class AirECGMMECGDataset(Dataset):
    """Adapt MMECGWindowedH5Dataset outputs to AirECG's expected shapes/ranges.

    v3 changes:
      - rcg shape is (MM_CHANNELS, 1600) because base dataset selects top-K bins (was 50 raw)
      - y2 strategy depends on `eval_mode`:
            False (train): random pool sample per __getitem__ (decorrelates from target,
                           so model can't take subject-specific shortcut)
            True (val/test): SAME fixed reference for every sample (median of pool),
                             so eval is deterministic and metrics aren't drowned by y2 noise
    """

    def __init__(
        self, base: MMECGWindowedH5Dataset, ref_pool: torch.Tensor,
        seed: int = 42, eval_mode: bool = False,
    ):
        self.base = base
        self.ref_pool = ref_pool                                 # (M, 1, 32, 32) ∈ [-1, 1]
        self.rng = np.random.RandomState(seed)
        self.eval_mode = eval_mode
        # fixed eval reference: pointwise median of the pool — deterministic, single tensor
        # (median across 1500+ ECGs in 32×32 space; smoother than any single sample)
        self.eval_ref = ref_pool.median(dim=0).values            # (1, 32, 32) ∈ [-1, 1]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]

        rcg = item["radar"].float()                              # (MM_CHANNELS, 1600)
        ecg = item["ecg"].float()                                # (1, 1600) ∈ [0, 1]

        # resample 1600 → 1024
        rcg_1024 = F.interpolate(
            rcg.unsqueeze(0), size=PATCH_LEN, mode="linear", align_corners=False
        ).squeeze(0)                                             # (MM_CHANNELS, 1024)
        ecg_1024 = F.interpolate(
            ecg.unsqueeze(0), size=PATCH_LEN, mode="linear", align_corners=False
        ).squeeze(0)                                             # (1, 1024) ∈ [0, 1]

        # reshape to AirECG's spatial format
        y1 = rcg_1024.reshape(MM_CHANNELS, LATENT_SIZE, LATENT_SIZE)      # (8, 32, 32)
        x = (ecg_1024 * 2.0 - 1.0).reshape(1, LATENT_SIZE, LATENT_SIZE)   # (1, 32, 32) ∈ [-1, 1]

        # v3.1: train = per-sample random pool draw (decorrelates y2 from target);
        #       eval = fixed median template (deterministic across epochs/runs)
        if self.eval_mode:
            y2 = self.eval_ref.clone()                           # (1, 32, 32) ∈ [-1, 1]
        else:
            ref_idx = int(self.rng.randint(0, len(self.ref_pool)))
            y2 = self.ref_pool[ref_idx].clone()                  # (1, 32, 32) ∈ [-1, 1]

        out = {
            "x": x,
            "y1": y1,
            "y2": y2,
            "ecg_1600": ecg,                                     # (1, 1600) ∈ [0, 1]  — for metrics
            "subject": int(item["subject"]),
            "state": int(item["state"]),
        }
        if "r_idx" in item:    # test set
            out["r_idx"] = np.asarray(item["r_idx"], dtype=np.int32)
            out["q_idx"] = np.asarray(item["q_idx"], dtype=np.int32)
            out["s_idx"] = np.asarray(item["s_idx"], dtype=np.int32)
            out["t_idx"] = np.asarray(item["t_idx"], dtype=np.int32)
            out["delin_valid"] = bool(item["delin_valid"])
        return out


def _airecg_collate(batch: list[dict]) -> dict:
    """Custom collate: stack stackable, list-of-arrays for vlen peak indices."""
    keys_tensor = ("x", "y1", "y2", "ecg_1600")
    keys_int = ("subject", "state")
    keys_list = ("r_idx", "q_idx", "s_idx", "t_idx")

    result: dict = {}
    for k in keys_tensor:
        result[k] = torch.stack([b[k] for b in batch])
    for k in keys_int:
        result[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    if "r_idx" in batch[0]:
        for k in keys_list:
            result[k] = [b[k] for b in batch]
        result["delin_valid"] = torch.tensor(
            [b["delin_valid"] for b in batch], dtype=torch.bool
        )
    return result


# =============================================================================
# 4. Loader builder
# =============================================================================

def build_airecg_loaders(
    fold_idx: int, fold_dir: Path, batch_size: int, num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    fold_data = LOSO_DIR / f"fold_{fold_idx:02d}"
    train_h5, val_h5, test_h5 = fold_data / "train.h5", fold_data / "val.h5", fold_data / "test.h5"
    for p in (train_h5, val_h5, test_h5):
        if not p.exists():
            raise FileNotFoundError(p)

    ref_pool = build_ref_pool(train_h5, fold_dir / "ref_pool_1024.npy")
    print(f"[fold {fold_idx:02d}] ref pool: {len(ref_pool)} ECGs, "
          f"shape={tuple(ref_pool.shape[1:])}", flush=True)

    common = dict(
        narrow_bandpass=False, target_norm="minmax",
        topk_bins=MM_CHANNELS, topk_method=TOPK_METHOD,         # v3 F1: 8 bins by cardiac energy
    )
    train_base = MMECGWindowedH5Dataset(train_h5, include_peak_indices=False, **common)
    val_base = MMECGWindowedH5Dataset(val_h5, include_peak_indices=False, **common)
    test_base = MMECGWindowedH5Dataset(test_h5, include_peak_indices=True, **common)

    # v3 F2 + v3.1: train uses random per-sample y2; val/test use fixed median template
    train_ds = AirECGMMECGDataset(train_base, ref_pool, seed=42, eval_mode=False)
    val_ds   = AirECGMMECGDataset(val_base,   ref_pool, seed=43, eval_mode=True)
    test_ds  = AirECGMMECGDataset(test_base,  ref_pool, seed=44, eval_mode=True)

    # AirECG paper uses plain shuffle (no subject balancing)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=_airecg_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_airecg_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_airecg_collate,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# 5. Model + diffusion factories
# =============================================================================

def build_model_and_diffusions(
    device: torch.device, sampling_steps: int,
) -> tuple[torch.nn.Module, torch.nn.Module, object, object]:
    model = AirECG_model(input_size=LATENT_SIZE, mm_channels=MM_CHANNELS).to(device)
    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False
    update_ema(ema, model, decay=0.0)    # sync initial weights

    diffusion_train = create_diffusion(timestep_respacing="")
    diffusion_sample = create_diffusion(timestep_respacing=f"ddim{sampling_steps}")
    return model, ema, diffusion_train, diffusion_sample


# =============================================================================
# 6. Training step (one epoch)
# =============================================================================

def train_one_epoch(
    model: torch.nn.Module, ema: torch.nn.Module, diffusion,
    optimizer: torch.optim.Optimizer, loader: DataLoader, device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y1 = batch["y1"].to(device, non_blocking=True)
        y2 = batch["y2"].to(device, non_blocking=True)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        loss_dict = diffusion.training_losses(model, x, t, dict(y1=y1, y2=y2))
        loss = loss_dict["loss"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema, model, decay=0.9999)
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(1, n)


# =============================================================================
# 7. Sampling
# =============================================================================

@torch.no_grad()
def sample_batch(
    ema: torch.nn.Module, diffusion_sample, y1: torch.Tensor, y2: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return (B, 1, 1600) float32 ∈ [0, 1]."""
    B = y1.shape[0]
    z = torch.randn(B, 1, LATENT_SIZE, LATENT_SIZE, device=device)
    samples = diffusion_sample.p_sample_loop(
        ema.forward, z.shape, z,
        clip_denoised=False,                                      # v3 F3: AirECG paper default
        model_kwargs=dict(y1=y1, y2=y2),
        progress=False, device=device,
    )                                                             # (B, 1, 32, 32)
    # safety: clamp to expected x0 range even though clip_denoised=False; out-of-range
    # samples then get squashed to [0, 1] for metric computation
    samples = samples.clamp(-1.0, 1.0)
    samples_1024 = samples.reshape(B, 1, PATCH_LEN)
    samples_01 = (samples_1024 + 1.0) / 2.0                       # → [0, 1]
    samples_1600 = F.interpolate(
        samples_01, size=WIN_LEN, mode="linear", align_corners=False,
    )                                                             # (B, 1, 1600) ∈ [0, 1]
    return samples_1600


# =============================================================================
# 8. Evaluate (val without peaks, test with full 4-level metrics)
# =============================================================================

def evaluate(
    ema: torch.nn.Module, diffusion_sample, loader: DataLoader, device: torch.device,
    with_peaks: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Return (segment_rows, beat_rows). Beat rows is empty if with_peaks=False."""
    ema.eval()
    seg_rows: list[dict] = []
    beat_rows: list[dict] = []
    seg_id = 0
    for batch in loader:
        y1 = batch["y1"].to(device, non_blocking=True)
        y2 = batch["y2"].to(device, non_blocking=True)
        pred_1600 = sample_batch(ema, diffusion_sample, y1, y2, device)   # (B,1,1600) ∈ [0,1]
        gt_1600 = batch["ecg_1600"]                                       # (B,1,1600) ∈ [0,1]
        subj_arr = batch["subject"].tolist()
        state_arr = batch["state"].tolist()

        B = pred_1600.shape[0]
        pred_np = pred_1600.squeeze(1).detach().cpu().numpy()             # (B, 1600)
        gt_np = gt_1600.squeeze(1).detach().cpu().numpy()                 # (B, 1600)

        if with_peaks:
            r_list = batch["r_idx"]; q_list = batch["q_idx"]
            s_list = batch["s_idx"]; t_list = batch["t_idx"]
            delin = batch["delin_valid"].tolist()

            for i in range(B):
                row, brows = _evaluate_segment(
                    seg_id=seg_id, pred_1d=pred_np[i], gt_1d=gt_np[i],
                    gt_r=r_list[i], gt_q=q_list[i], gt_s=s_list[i], gt_t=t_list[i],
                    subject_id=subj_arr[i], state_code=state_arr[i],
                    delin_valid=delin[i],
                )
                seg_rows.append(row)
                beat_rows.extend(brows)
                seg_id += 1
        else:
            # val: just L1 (PCC / RMSE / MAE) per sample
            from src.utils.metrics import compute_waveform_metrics_protocol
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
# 9. Train / test / loso_summary
# =============================================================================

def train_fold(args) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = EXP_ROOT / args.exp_tag / f"fold_{args.fold_idx:02d}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "results").mkdir(exist_ok=True)
    with open(fold_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, _ = build_airecg_loaders(
        args.fold_idx, fold_dir, args.batch_size, num_workers=args.num_workers,
    )
    print(f"[fold {args.fold_idx:02d}] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)}", flush=True)

    model, ema, diff_train, diff_sample = build_model_and_diffusions(device, args.sampling_steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fold {args.fold_idx:02d}] AirECG params = {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    best_pcc = -float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, ema, diff_train, optimizer, train_loader, device)
        t_train = time.time() - t0

        log_row = {"epoch": epoch, "train_loss": avg_loss, "t_train_s": t_train}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            val_rows, _ = evaluate(ema, diff_sample, val_loader, device, with_peaks=False)
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
                    "ema":   ema.state_dict(),
                    "epoch": epoch,
                    "val_pcc": val_pcc,
                    "val_mae": val_mae,
                }, fold_dir / "checkpoints/best_ema.pt")
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
    ckpt_path = fold_dir / "checkpoints/best_ema.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    _, _, test_loader = build_airecg_loaders(
        args.fold_idx, fold_dir, args.batch_size, num_workers=args.num_workers,
    )
    print(f"[fold {args.fold_idx:02d}] test={len(test_loader.dataset)}", flush=True)

    _, ema, _, diff_sample = build_model_and_diffusions(device, args.sampling_steps)
    state = torch.load(ckpt_path, map_location=device)
    ema.load_state_dict(state["ema"])
    print(f"[fold {args.fold_idx:02d}] loaded ckpt epoch={state.get('epoch')} "
          f"val_pcc={state.get('val_pcc'):.4f}", flush=True)

    t0 = time.time()
    seg_rows, beat_rows = evaluate(ema, diff_sample, test_loader, device, with_peaks=True)
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
# 10. CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp_tag", type=str, default="mmecg_airecg_strict_loso_v3")
    parser.add_argument("--protocol", type=str, default="strict_loso",
                        choices=["strict_loso"])
    parser.add_argument("--fold_idx", type=int, required=True,
                        help="1..11, or -1 to aggregate loso_summary")
    parser.add_argument("--epochs", type=int, default=200)        # v3: 150 → 200
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=50,  # v3: 30 → 50
                        help="epochs without val_pcc improvement before stop")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="DDIM sampling steps at val/test")
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
