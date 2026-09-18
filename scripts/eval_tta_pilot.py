#!/usr/bin/env python3
"""Zero-label scalar-shift TTA for the frozen Fold-2 continuous REM-Net.

The adaptation phase consumes WSST and the model's own waveform/event outputs
only.  The held-out ECG is loaded only after ``adapt_subject_shift`` returns,
solely to report the frozen post-hoc evaluation.  This is a pilot, not a
deployable or publishable result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wsst8s_subject_pool import SubjectWSSTPool  # noqa: E402
from src.models.continuous.ce_net import CENet  # noqa: E402
from src.utils.atomic_state import atomic_json  # noqa: E402

FS = 200
WIN = 1600
FOLD = 2
MAX_SHIFT_SAMPLES = 20.0  # registered +/-100 ms diagnostic support
CHECKPOINT = ROOT / (
    "experiments_mmecg/q1_remnet_continuous_strict_loso_seed42/"
    "L3/fold_02/checkpoints/best_composite.pt"
)
SPLIT_H5 = Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_02/test.h5")
POOL = Path("/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool")
OUTPUT = ROOT / "experiments_mmecg/q1_zero_label_tta_fold02/result.json"


def fractional_circular_shift(x: torch.Tensor, shift_samples: torch.Tensor) -> torch.Tensor:
    """Differentiably delay ``x`` by ``shift_samples`` using Fourier phase."""
    if x.ndim != 3 or x.shape[-1] != WIN:
        raise ValueError(f"expected [B,C,{WIN}], got {tuple(x.shape)}")
    frequency = torch.fft.rfftfreq(WIN, d=1.0, device=x.device).to(x.dtype)
    phase = torch.exp(
        -2j * torch.pi * frequency.to(torch.complex64) * shift_samples.to(torch.complex64)
    )
    return torch.fft.irfft(torch.fft.rfft(x, dim=-1) * phase.view(1, 1, -1), n=WIN, dim=-1)


def event_phase_loss(shifted_wave: torch.Tensor, event_probability: torch.Tensor) -> torch.Tensor:
    gradient_energy = torch.sqrt(torch.diff(shifted_wave, dim=-1).square() + 1e-8)
    gradient_energy = F.pad(gradient_energy, (0, 1))
    event = event_probability[:, :1].clamp_min(0.0)
    numerator = (event * gradient_energy).sum(-1)
    denominator = event.norm(dim=-1) * gradient_energy.norm(dim=-1)
    return 1.0 - (numerator / denominator.clamp_min(1e-8)).mean()


def cyclic_prominence_loss(shifted_wave: torch.Tensor) -> torch.Tensor:
    """Negative soft autocorrelation prominence over plausible RR lags.

    A circular shift leaves this term invariant in exact arithmetic.  It is
    retained because it is part of the requested hypothesis, and its near-zero
    gradient is explicitly reported rather than silently treated as evidence.
    """
    centred = shifted_wave - shifted_wave.mean(-1, keepdim=True)
    centred = centred / centred.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
    lags = range(int(0.4 * FS), int(1.5 * FS) + 1)
    autocorr = torch.stack(
        [(centred * torch.roll(centred, shifts=lag, dims=-1)).mean(-1) for lag in lags],
        dim=-1,
    )
    soft_peak = torch.logsumexp(20.0 * autocorr, dim=-1) / 20.0
    prominence = soft_peak - autocorr.mean(-1)
    return -prominence.mean()


def adapt_subject_shift(
    model: torch.nn.Module,
    calibration_wsst: torch.Tensor,
    *,
    steps: int = 20,
    lr: float = 1e-2,
) -> tuple[float, list[dict[str, float]]]:
    """Adapt one scalar from radar/model outputs; no ECG argument exists."""
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        waveform, event_logits = model(calibration_wsst)
        event_probability = torch.sigmoid(event_logits).detach()
        waveform = waveform.detach()

    raw_shift = torch.nn.Parameter(torch.zeros((), device=waveform.device))
    optimizer = torch.optim.Adam([raw_shift], lr=lr)
    history: list[dict[str, float]] = []
    best = (float("inf"), 0.0)
    for step in range(1, steps + 1):
        shift = MAX_SHIFT_SAMPLES * torch.tanh(raw_shift)
        shifted = fractional_circular_shift(waveform, shift)
        event_loss = event_phase_loss(shifted, event_probability)
        cyclic_loss = cyclic_prominence_loss(shifted)
        total = event_loss + 0.5 * cyclic_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if raw_shift.grad is None or not torch.isfinite(raw_shift.grad):
            raise RuntimeError("TTA scalar has no finite gradient")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen model received a gradient during TTA")
        optimizer.step()
        current_shift = float((MAX_SHIFT_SAMPLES * torch.tanh(raw_shift)).detach())
        record = {
            "step": step,
            "shift_samples": current_shift,
            "shift_ms": current_shift * 1000.0 / FS,
            "total": float(total.detach()),
            "event": float(event_loss.detach()),
            "cyclic": float(cyclic_loss.detach()),
            "raw_shift_gradient": float(raw_shift.grad.detach()),
        }
        history.append(record)
        if record["total"] < best[0]:
            best = record["total"], current_shift
    return best[1], history


def pcc_per_window(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(1, keepdims=True)
    ground_truth = ground_truth - ground_truth.mean(1, keepdims=True)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(ground_truth, axis=1)
    return (prediction * ground_truth).sum(1) / np.maximum(denominator, 1e-12)


@torch.no_grad()
def infer(model: torch.nn.Module, wsst: np.ndarray, device: torch.device, batch_size: int = 16):
    prediction, event = [], []
    for start in range(0, len(wsst), batch_size):
        x = torch.from_numpy(wsst[start:start + batch_size].astype(np.float32)).to(device)
        y, logits = model(x)
        prediction.append(y[:, 0].cpu())
        event.append(torch.sigmoid(logits[:, 0]).cpu())
    return torch.cat(prediction).numpy(), torch.cat(event).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    if args.k != 2 or args.steps not in range(15, 21) or args.lr != 1e-2:
        raise SystemExit("Registered pilot is fixed to K=2, 15-20 steps, Adam lr=1e-2")

    device = torch.device(args.device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if checkpoint.get("fold") != FOLD or checkpoint.get("arm") != "B1":
        raise RuntimeError("checkpoint identity mismatch")
    model = CENet("B1").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)

    # Radar-only phase. SubjectWSSTPool uses split metadata to restore row
    # order; it never opens the ECG dataset. No ground-truth tensor exists yet.
    wsst = SubjectWSSTPool(POOL).load_for_split(SPLIT_H5)
    calibration = torch.from_numpy(wsst[:args.k].astype(np.float32)).to(device)
    started = time.perf_counter()
    shift_samples, history = adapt_subject_shift(
        model, calibration, steps=args.steps, lr=args.lr,
    )
    adaptation_seconds = time.perf_counter() - started

    prediction, _ = infer(model, wsst, device)
    prediction_tensor = torch.from_numpy(prediction[:, None]).to(device)
    shift_tensor = torch.tensor(shift_samples, device=device)
    corrected = fractional_circular_shift(prediction_tensor, shift_tensor)[:, 0].cpu().numpy()

    # Evaluation-only phase starts here. GT is opened after tau is immutable;
    # it never enters autograd, the loss, optimizer, or model selection.
    frozen_shift_samples = float(shift_samples)
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        with h5py.File(SPLIT_H5, "r") as handle:
            ground_truth = np.asarray(handle["ecg"][:, 0], dtype=np.float32)
    query = slice(args.k, None)
    baseline_pcc = float(np.nanmean(pcc_per_window(prediction[query], ground_truth[query])))
    corrected_pcc = float(np.nanmean(pcc_per_window(corrected[query], ground_truth[query])))

    result = {
        "experiment_id": "q1_zero_label_tta_fold02",
        "status": "pilot_nonpublishable",
        "fold": FOLD,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "adaptation": {
            "inputs": "first 2 test WSST windows and frozen model outputs only",
            "gt_ecg_loaded": False,
            "k_windows": args.k,
            "steps": args.steps,
            "lr": args.lr,
            "seconds": adaptation_seconds,
            "tau_samples": frozen_shift_samples,
            "tau_ms": frozen_shift_samples * 1000.0 / FS,
            "history": history,
        },
        "evaluation": {
            "gt_loaded_only_after_adaptation": True,
            "query_excludes_first_k_windows": True,
            "n_query_windows": len(ground_truth) - args.k,
            "baseline_pcc_at_0": baseline_pcc,
            "tta_pcc_at_0": corrected_pcc,
            "delta_pcc": corrected_pcc - baseline_pcc,
        },
        "caveat": "test-subject pilot; GT is evaluation-only; no promotion or claim authorization",
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
