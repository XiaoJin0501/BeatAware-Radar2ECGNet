#!/usr/bin/env python3
"""Run the registered folds 1/5/9 direct Q/R/S/T feasibility gate.

Validation alone selects a checkpoint.  The test dataset is not constructed
until the state has irreversibly entered CHECKPOINT_FROZEN.  Epoch state is
saved atomically and includes optimizer, patience, history and all RNG states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_continuous_ecg_pilot import Fold, fit_scaler  # noqa: E402
from src.data.mmecg_dataset import _gaussian_mask  # noqa: E402
from src.models.continuous.structured_event_rem import FIDUCIALS, StructuredEventREM  # noqa: E402
from src.utils.atomic_state import (  # noqa: E402
    atomic_json, atomic_torch_save, capture_rng_state, restore_rng_state,
)
from src.utils.clinical_endpoints import delineate_signal, match_beats  # noqa: E402

TAG = "q1_structured_event_gate_v1"
SPEC = ROOT / "experiments/specs/q1_structured_event_gate_v1.json"
REGISTRY = ROOT / "experiments_mmecg/EXPERIMENT_REGISTRY.csv"
OUT = ROOT / "experiments_mmecg" / TAG
BASELINE = ROOT / "docs/audits/Q1_STRUCTURED_EVENT_GATE_BASELINE.json"
FOLDS = (1, 5, 9)
FS, WIN, SIGMA, MATCH_TOL = 200, 1600, 4.0, 30
MAX_EPOCHS, PATIENCE, BATCH = 100, 15, 16
SEED = 42
STAGES = ("TRAINING", "CHECKPOINT_FROZEN", "TEST_EVALUATING", "COMPLETE")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def capture(*cmd: str) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transition(path: Path, new_stage: str, **extra) -> dict:
    old = json.loads(path.read_text()) if path.exists() else None
    if new_stage not in STAGES:
        raise ValueError(new_stage)
    if old:
        oi, ni = STAGES.index(old["stage"]), STAGES.index(new_stage)
        if ni < oi or ni > oi + 1:
            raise RuntimeError(f"illegal transition {old['stage']} -> {new_stage}")
        if old["stage"] == "COMPLETE":
            raise RuntimeError("COMPLETE is terminal")
    state = {**(old or {}), "stage": new_stage, **extra}
    atomic_json(path, state)
    return state


def require_preflight() -> tuple[str, str]:
    spec = json.loads(SPEC.read_text())
    if not spec["status"].startswith("PREFLIGHT_PASSED"):
        raise SystemExit(f"spec is not launchable: {spec['status']}")
    dirty = capture("git", "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit("refusing launch with tracked changes:\n" + dirty)
    commit, branch = capture("git", "rev-parse", "HEAD"), capture("git", "branch", "--show-current")
    expected = spec.get("commit_id")
    if not expected or expected != commit:
        raise SystemExit(f"spec commit {expected!r} != HEAD {commit}")
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            p = Path(f"/home/qhh2237/Datasets/MMECG/processed/loso_v2/fold_{fold:02d}/{split}.h5")
            if not p.exists():
                raise SystemExit(f"missing {p}")
    return commit, branch


def update_registry(commit: str, branch: str, status: str, reason: str) -> None:
    with REGISTRY.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    target = next(r for r in rows if r["tag"] == TAG)
    target.update(commit_id=commit, branch=branch, status=status, reason=reason)
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    tmp = REGISTRY.with_name(f".{REGISTRY.name}.tmp")
    tmp.write_text(buf.getvalue()); os.replace(tmp, REGISTRY)


def extract_points(ecg: np.ndarray) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return delineate_signal(ecg, fs=FS, method="prominence")


class EventFold(Dataset):
    """Wrap the frozen full-support loader and add fixed Q/R/S/T targets."""

    constructions: list[str] = []  # regression-test witness for test-access timing

    def __init__(self, fold: int, split: str, allow_test: bool = False, limit=None):
        EventFold.constructions.append(split)
        self.base = Fold(fold, split, "B1", fit_scaler(fold), limit=limit,
                         allow_test=allow_test, wsst_source="subject_pool")
        self.points = [extract_points(x) for x in self.base.ecg]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        target = np.stack([
            _gaussian_mask(np.asarray(self.points[index][p], float)[
                np.isfinite(self.points[index][p])].astype(int), WIN, SIGMA)
            for p in FIDUCIALS
        ]).astype(np.float32)
        return {"wsst": item["wsst"], "event": torch.from_numpy(target),
                "index": index}


def peaks(probability: np.ndarray) -> dict[str, np.ndarray]:
    return {p: find_peaks(probability[i], height=.5, distance=40)[0].astype(float)
            for i, p in enumerate(FIDUCIALS)}


def score(probabilities: np.ndarray, references: list[dict]) -> dict:
    sufficient = {p: {"gt": 0, "pred": 0, "match": 0, "errors": []} for p in FIDUCIALS}
    complete = violations = 0
    for prob, ref in zip(probabilities, references):
        pred = peaks(prob)
        for p in FIDUCIALS:
            gt = np.asarray(ref[p], float); gt = gt[np.isfinite(gt)]
            pairs = match_beats(gt, pred[p], MATCH_TOL)
            row = sufficient[p]
            row["gt"] += len(gt); row["pred"] += len(pred[p]); row["match"] += len(pairs)
            row["errors"].extend(abs(float(gt[i] - pred[p][j])) * 1000 / FS
                                 for i, j in pairs.items())
        # An unconstrained nearest-neighbour association exposes rather than enforces order.
        for r in pred["r"]:
            chosen = {}
            for p in ("q", "s", "t"):
                if len(pred[p]):
                    chosen[p] = float(pred[p][np.argmin(abs(pred[p] - r))])
            if len(chosen) == 3:
                complete += 1
                violations += int(not (chosen["q"] < r < chosen["s"] < chosen["t"]))
    out = {}
    for p, row in sufficient.items():
        e = np.asarray(row.pop("errors"), float)
        out[p] = {**row,
                  "f1_100ms": 2 * row["match"] / (row["gt"] + row["pred"])
                  if row["gt"] + row["pred"] else None,
                  "coverage": row["match"] / row["gt"] if row["gt"] else None,
                  "medae_ms": float(np.median(e)) if e.size else None}
    out["macro_f1_100ms"] = float(np.mean([out[p]["f1_100ms"] for p in FIDUCIALS]))
    out["macro_medae_ms"] = float(np.mean([out[p]["medae_ms"] for p in FIDUCIALS]))
    out["order_complete"] = complete
    out["order_violations"] = violations
    out["order_violation_rate"] = violations / complete if complete else None
    return out


@torch.no_grad()
def evaluate(model, dataset: EventFold, device: torch.device, batch_size: int = BATCH):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); chunks = []
    for batch in loader:
        chunks.append(torch.sigmoid(model(batch["wsst"].to(device))).cpu().numpy())
    probability = np.concatenate(chunks)
    return score(probability, dataset.points), probability


def checkpoint_payload(model, optimizer, epoch, patience_count, best_score, best_medae, history):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "patience_count": patience_count, "best_score": best_score,
            "best_medae": best_medae, "history": history, "rng": capture_rng_state()}


def train_fold(fold: int, device: torch.device, max_epochs: int = MAX_EPOCHS,
               patience: int = PATIENCE, limit=None, interrupt_after: int | None = None) -> None:
    fold_dir = OUT / f"fold_{fold:02d}"; ckpt = fold_dir / "checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(exist_ok=True)
    state_path, last_path, best_path = (fold_dir / "run_state.json",
                                        ckpt / "last_state.pt", ckpt / "best_event.pt")
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and state["stage"] == "COMPLETE":
        print(f"[skip] fold {fold:02d} complete", flush=True); return
    if state and state["stage"] in ("CHECKPOINT_FROZEN", "TEST_EVALUATING"):
        if not best_path.exists(): raise RuntimeError("frozen state lacks best checkpoint")
        run_test(fold, device, limit); return

    set_seed(SEED + fold)
    model = StructuredEventREM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    start, patience_count, best_score, best_medae, history = 1, 0, -np.inf, np.inf, []
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        restore_rng_state(saved["rng"])
        start, patience_count = saved["epoch"] + 1, saved["patience_count"]
        best_score, best_medae, history = saved["best_score"], saved["best_medae"], saved["history"]
        print(f"[resume] fold {fold:02d} epoch {start}", flush=True)
    if not state:
        transition(state_path, "TRAINING", fold=fold, test_accessed=False)
    train = EventFold(fold, "train", limit=limit)
    val = EventFold(fold, "val", limit=limit)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED + fold))
    for epoch in range(start, max_epochs + 1):
        model.train(); total = batches = 0
        for batch in loader:
            logits = model(batch["wsst"].to(device)); target = batch["event"].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()); batches += 1
        metrics, _ = evaluate(model, val, device)
        record = {"epoch": epoch, "train_loss": total / max(batches, 1), **metrics}
        history.append(record)
        improved = (metrics["macro_f1_100ms"] > best_score + 1e-12 or
                    (abs(metrics["macro_f1_100ms"] - best_score) <= 1e-12 and
                     metrics["macro_medae_ms"] < best_medae))
        if improved:
            best_score, best_medae, patience_count = (metrics["macro_f1_100ms"],
                                                       metrics["macro_medae_ms"], 0)
            atomic_torch_save(best_path, {"model": model.state_dict(), "epoch": epoch,
                                          "validation": metrics})
        else:
            patience_count += 1
        atomic_torch_save(last_path, checkpoint_payload(
            model, optimizer, epoch, patience_count, best_score, best_medae, history))
        atomic_json(fold_dir / "history.json", history)
        print(f"fold{fold:02d} ep{epoch:03d} loss={record['train_loss']:.4f} "
              f"valF1={metrics['macro_f1_100ms']:.3f} valMedAE={metrics['macro_medae_ms']:.1f}",
              flush=True)
        if interrupt_after == epoch:
            raise RuntimeError("SIMULATED_INTERRUPTION")
        if patience_count >= patience:
            break
    transition(state_path, "CHECKPOINT_FROZEN", best_validation_f1=best_score,
               best_validation_medae_ms=best_medae, test_accessed=False)
    run_test(fold, device, limit)


def run_test(fold: int, device: torch.device, limit=None) -> None:
    fold_dir = OUT / f"fold_{fold:02d}"; state_path = fold_dir / "run_state.json"
    state = json.loads(state_path.read_text())
    if state["stage"] == "CHECKPOINT_FROZEN":
        transition(state_path, "TEST_EVALUATING", test_accessed=True)
    elif state["stage"] != "TEST_EVALUATING":
        raise RuntimeError(f"test access from {state['stage']}")
    model = StructuredEventREM().to(device)
    saved = torch.load(fold_dir / "checkpoints/best_event.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    test = EventFold(fold, "test", allow_test=True, limit=limit)
    metrics, probability = evaluate(model, test, device)
    atomic_json(fold_dir / "test_metrics.json", metrics)
    atomic_npz(fold_dir / "test_event_probability.npz", probability=probability.astype(np.float16))
    transition(state_path, "COMPLETE", test_accessed=True,
               test_metrics_sha256=sha256(fold_dir / "test_metrics.json"))


def baseline_fold(fold: int) -> dict:
    return json.loads(BASELINE.read_text())["folds"][str(fold)]


def aggregate() -> dict:
    event = {}; baseline = {}
    for fold in FOLDS:
        event[fold] = json.loads((OUT / f"fold_{fold:02d}/test_metrics.json").read_text())
        baseline[fold] = baseline_fold(fold)
    e_median = {p: float(np.median([event[f][p]["medae_ms"] for f in FOLDS])) for p in FIDUCIALS}
    b_median = {p: float(np.median([baseline[f][p]["medae_ms"] for f in FOLDS])) for p in FIDUCIALS}
    wins = [p for p in FIDUCIALS if e_median[p] < b_median[p]]
    event_macro = [np.mean([event[f][p]["medae_ms"] for p in FIDUCIALS]) for f in FOLDS]
    base_macro = [np.mean([baseline[f][p]["medae_ms"] for p in FIDUCIALS]) for f in FOLDS]
    reduction = 1 - np.mean(event_macro) / np.mean(base_macro)
    coverage_ok = all(np.median([event[f][p]["coverage"] for f in FOLDS]) >=
                      np.median([baseline[f][p]["coverage"] for f in FOLDS]) - .03 for p in wins)
    ordering_ok = all(event[f]["order_violation_rate"] is not None and
                      event[f]["order_violation_rate"] <= .05 for f in FOLDS)
    gate = {"accuracy": len(wins) >= 3 and reduction >= .10,
            "fold_consistency": sum(e < b for e, b in zip(event_macro, base_macro)) >= 2,
            "coverage": coverage_ok, "ordering": ordering_ok}
    gate["overall_pass"] = all(gate.values())
    result = {"event": event, "baseline": baseline, "event_median": e_median,
              "baseline_median": b_median, "event_metrics_won": wins,
              "macro_relative_reduction": reduction, "gate": gate}
    atomic_json(OUT / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--interrupt_after", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.aggregate_only:
        print(json.dumps(aggregate()["gate"], indent=2)); return
    commit, branch = require_preflight()
    update_registry(commit, branch, "PREFLIGHT PASSED" if args.preflight_only else "RUNNING",
                    "Validation-only checkpoint selection; test opens once after freeze.")
    if args.preflight_only:
        print(f"PASS {TAG} {commit} {branch}"); return
    device = torch.device(args.device)
    folds = (args.fold,) if args.fold else FOLDS
    for fold in folds:
        train_fold(fold, device, args.max_epochs, PATIENCE, args.limit, args.interrupt_after)
    if all((OUT / f"fold_{f:02d}/test_metrics.json").exists() for f in FOLDS):
        result = aggregate()
        update_registry(commit, branch,
                        "COMPLETED — GATE PASSED" if result["gate"]["overall_pass"] else "COMPLETED — GATE FAILED",
                        "See experiments_mmecg/q1_structured_event_gate_v1/summary.json")


if __name__ == "__main__":
    main()
