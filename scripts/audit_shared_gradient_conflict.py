#!/usr/bin/env python3
"""Read-only gradient-conflict audit for completed B1/L3 checkpoints.

The L3 model is CENet(B1), whose true shared path is `wsst_stem` plus
`decoder.tcn`. The waveform and event output heads are excluded. This script
never opens a test split and never calls an optimizer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_continuous_ecg_pilot import (  # noqa: E402
    BUDGET,
    EVENT_W,
    Fold,
    LOSS_WEIGHTS,
    MRSTFT_CONFIG,
    _set_split_root,
    fit_scaler,
    set_seed,
)
from src.losses.losses import MultiResolutionSTFTLoss  # noqa: E402
from src.models.continuous.ce_net import CENet  # noqa: E402


def shared_named_parameters(model: CENet):
    allowed = ("wsst_stem.", "decoder.tcn.")
    return [(name, p) for name, p in model.named_parameters()
            if p.requires_grad and name.startswith(allowed)]


def parameter_group(name: str) -> str:
    if name.startswith("wsst_stem."):
        return "wsst_stem"
    parts = name.split(".")
    if len(parts) >= 4 and parts[:3] == ["decoder", "tcn", "blocks"]:
        return ".".join(parts[:4])
    raise ValueError(f"unregistered shared parameter group: {name}")


def cosine(parts_a, parts_b) -> tuple[float, float, float]:
    dot = sum((a.double() * b.double()).sum() for a, b in zip(parts_a, parts_b))
    na2 = sum(a.double().square().sum() for a in parts_a)
    nb2 = sum(b.double().square().sum() for b in parts_b)
    na, nb = na2.sqrt(), nb2.sqrt()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan"), float(na), float(nb)
    return float(dot / (na * nb)), float(na), float(nb)


def bootstrap_ci(values: list[float], seed: int, draws: int = 10000) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def summarize(values: list[float], seed: int) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "n_batches": int(len(x)),
        "mean": float(x.mean()) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "negative_fraction": float((x < 0).mean()) if len(x) else float("nan"),
        "le_minus_0_30_fraction": float((x <= -0.30).mean()) if len(x) else float("nan"),
        "bootstrap_95ci_mean": bootstrap_ci(x.tolist(), seed),
    }


def audit_fold(fold: int, checkpoint: Path, device: torch.device,
               batch_size: int, max_batches: int | None) -> tuple[list[dict], dict]:
    set_seed(BUDGET["seed"] + fold)
    scaler = fit_scaler(fold)
    dataset = Fold(fold, "val", "B1", scaler)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = CENet("B1").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    mrstft = MultiResolutionSTFTLoss(**MRSTFT_CONFIG).to(device)
    named = shared_named_parameters(model)
    if not named:
        raise RuntimeError("no shared parameters selected")
    params = [p for _, p in named]
    rows, by_group = [], defaultdict(list)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        wsst = batch["wsst"].to(device)
        target = batch["y"].to(device)
        event_target = batch["event"].to(device)
        pred, event_logits = model(wsst)
        wave_loss = LOSS_WEIGHTS["L3"] * mrstft(pred, target)
        event_loss = EVENT_W * F.binary_cross_entropy_with_logits(event_logits, event_target)
        gw = torch.autograd.grad(wave_loss, params, retain_graph=True, allow_unused=True)
        ge = torch.autograd.grad(event_loss, params, allow_unused=True)
        if any(g is None for g in (*gw, *ge)):
            missing = [name for (name, _), a, b in zip(named, gw, ge)
                       if a is None or b is None]
            raise RuntimeError(f"shared parameters without both gradients: {missing}")
        gw = [g.detach() for g in gw]
        ge = [g.detach() for g in ge]
        global_cos, wave_norm, event_norm = cosine(gw, ge)
        row = {
            "fold": fold,
            "batch": batch_index,
            "n_windows": int(wsst.shape[0]),
            "global_cosine": global_cos,
            "wave_grad_norm": wave_norm,
            "event_grad_norm": event_norm,
            "wave_event_norm_ratio": wave_norm / event_norm if event_norm else float("nan"),
        }
        grouped_w, grouped_e = defaultdict(list), defaultdict(list)
        for (name, _), a, b in zip(named, gw, ge):
            group = parameter_group(name)
            grouped_w[group].append(a)
            grouped_e[group].append(b)
        for group in sorted(grouped_w):
            c, _, _ = cosine(grouped_w[group], grouped_e[group])
            row[f"cosine::{group}"] = c
            by_group[group].append(c)
        rows.append(row)
        del pred, event_logits, wave_loss, event_loss, gw, ge

    summary = {
        "fold": fold,
        "global": summarize([r["global_cosine"] for r in rows], 1000 + fold),
        "groups": {g: summarize(v, 2000 + fold) for g, v in sorted(by_group.items())},
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 5, 9])
    parser.add_argument("--split_root", default="loso_v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="smoke/debug only; omission is required for the formal audit")
    parser.add_argument("--source_root", type=Path,
                        default=ROOT / "experiments_mmecg/q1_continuous_ecg_loss_ablation/L3")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/audits/Q1_SHARED_GRADIENT_CONFLICT_AUDIT.json")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    _set_split_root(args.split_root)
    device = torch.device(args.device)
    all_rows, fold_summaries = [], []
    for fold in args.folds:
        checkpoint = args.source_root / f"fold_{fold:02d}/checkpoints/best_composite.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        rows, summary = audit_fold(fold, checkpoint, device, args.batch_size, args.max_batches)
        all_rows.extend(rows)
        fold_summaries.append(summary)

    group_names = sorted({k for r in all_rows for k in r if k.startswith("cosine::")})
    global_ok = (all(s["global"]["mean"] < 0 for s in fold_summaries)
                 and sum(s["global"]["bootstrap_95ci_mean"][1] < 0
                         for s in fold_summaries) >= 2)
    localized = {
        g: sum(s["groups"][g]["mean"] <= -0.30 for s in fold_summaries)
        for g in (name.removeprefix("cosine::") for name in group_names)
    }
    localized_ok = any(count >= 2 for count in localized.values())
    formal = args.max_batches is None and sorted(args.folds) == [1, 5, 9]
    result = {
        "experiment_id": "q1_shared_gradient_conflict_audit",
        "formal": formal,
        "device": str(device),
        "split_root": args.split_root,
        "test_accessed": False,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "shared_parameter_names": [name for name, _ in shared_named_parameters(CENet("B1"))],
        "folds": fold_summaries,
        "decision": {
            "global_requirement": global_ok,
            "localized_counts_le_minus_0_30": localized,
            "localized_requirement": localized_ok,
            "conflict_supported": bool(global_ok and localized_ok) if formal else None,
            "note": "A limited-batch run is smoke/debug evidence only." if not formal else "Applied preregistered rule.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(result["decision"], indent=2))
    print(f"wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
