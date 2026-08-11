#!/usr/bin/env python3
"""Run the frozen REM-Net final-candidate Shape-200 experiment.

This is the formal RQ1 supervised morphology run selected from F0:
peak06 = beta_peak 0.6, lambda_shape200 0.8, shape200_pcc_weight 0.5,
hidden 128, no alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
OUT_DIR = ROOT / "experiments_mmecg" / "q1_final_remnet_peak06"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    protocol: str
    folds: tuple[int, ...]
    loso_h5_dir: str
    n_range_bins: int
    narrow_bandpass: bool


DATASETS = {
    "mmecg": DatasetSpec(
        key="mmecg",
        label="MMECG public",
        protocol="11-fold strict LOSO REM-Net final candidate",
        folds=tuple(range(1, 12)),
        loso_h5_dir="/home/qhh2237/Datasets/MMECG/processed/loso",
        n_range_bins=50,
        narrow_bandpass=True,
    ),
    "crrvs": DatasetSpec(
        key="crrvs",
        label="CR_RVS / Nature",
        protocol="5-fold subject-independent REM-Net final candidate",
        folds=tuple(range(1, 6)),
        loso_h5_dir="/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold",
        n_range_bins=1,
        narrow_bandpass=False,
    ),
}


def run_capture(cmd: list[str]) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    result.check_returncode()
    return result.stdout.strip()


def run_stream(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def git_clean() -> bool:
    return run_capture(["git", "status", "--short"]) == ""


def git_commit() -> str:
    return run_capture(["git", "rev-parse", "--short", "HEAD"])


def git_branch() -> str:
    return run_capture(["git", "branch", "--show-current"])


def load_registry() -> list[dict[str, str]]:
    with REGISTRY.open(newline="") as f:
        return list(csv.DictReader(f))


def write_registry(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty registry.")
    fieldnames = list(rows[0].keys())
    with REGISTRY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_registry_row(tag: str, dataset: DatasetSpec, commit: str, branch: str) -> None:
    rows = load_registry()
    if any(row.get("tag") == tag for row in rows):
        return
    rows.append(
        {
            "tag": tag,
            "dataset": dataset.label,
            "protocol": dataset.protocol,
            "category": "final_candidate",
            "publishable": "false",
            "commit_id": commit,
            "branch": branch,
            "role": "Frozen supervised REM-Net peak06 morphology run",
            "status": "registered",
            "reason": (
                "Final-candidate full-fold run from F0: beta_peak=0.6, "
                "lambda_shape200=0.8, shape200_pcc_weight=0.5, hidden=128, "
                "use_alignment=false. Awaiting metric audit before promotion."
            ),
        }
    )
    write_registry(rows)


def update_registry_status(tag: str, status: str, reason: str) -> None:
    rows = load_registry()
    found = False
    for row in rows:
        if row.get("tag") == tag:
            row["status"] = status
            row["reason"] = reason
            found = True
    if not found:
        raise KeyError(tag)
    write_registry(rows)


def tag_for(dataset: DatasetSpec, seed: int) -> str:
    return f"q1_final_remnet_peak06_{dataset.key}_seed{seed}"


def train_and_eval(dataset: DatasetSpec, seed: int, epochs: int, patience: int, checkpoint: str) -> None:
    tag = tag_for(dataset, seed)
    for fold in dataset.folds:
        fold_dir = ROOT / "experiments_mmecg" / tag / f"fold_{fold:02d}"
        summary_path = fold_dir / "results_shape_head" / "global_summary_shape_head.json"
        if summary_path.exists():
            print(f"[skip] {tag} fold {fold:02d} already evaluated", flush=True)
            continue
        run_stream(
            [
                sys.executable,
                "scripts/train_mmecg.py",
                "--exp_tag",
                tag,
                "--fold_idx",
                str(fold),
                "--loso_h5_dir",
                dataset.loso_h5_dir,
                "--n_range_bins",
                str(dataset.n_range_bins),
                "--model_arch",
                "rem_align_mtl",
                "--epochs",
                str(epochs),
                "--early_stop_patience",
                str(patience),
                "--lr",
                "0.0001",
                "--weight_decay",
                "0.0001",
                "--narrow_bandpass",
                str(dataset.narrow_bandpass).lower(),
                "--target_norm",
                "minmax",
                "--wsst_suffix",
                "_wsst.h5",
                "--beta_peak",
                "0.6",
                "--use_shape200_loss",
                "true",
                "--lambda_shape200",
                "0.8",
                "--shape200_pcc_weight",
                "0.5",
                "--baseline_hidden",
                "128",
                "--use_alignment",
                "false",
                "--seed",
                str(seed),
            ]
        )
        run_stream(
            [
                sys.executable,
                "scripts/evaluate_remalign_shape_head.py",
                "--exp_tag",
                tag,
                "--fold_idx",
                str(fold),
                "--checkpoint",
                checkpoint,
            ]
        )


def aggregate(tags: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tag in tags:
        for summary_path in sorted((ROOT / "experiments_mmecg" / tag).glob("fold_*/results_shape_head/global_summary_shape_head.json")):
            summary = json.loads(summary_path.read_text())
            fold_dir = summary_path.parents[1]
            rows.append(
                {
                    "tag": tag,
                    "dataset": "crrvs" if "_crrvs_" in tag else "mmecg",
                    "seed": int(tag.split("seed")[-1]),
                    "fold": int(fold_dir.name.split("_")[-1]),
                    "pcc": float(summary["pcc_mean"]),
                    "rmse": float(summary["rmse_mean"]),
                    "mae": float(summary["mae_mean"]),
                    "r2": float(summary["r2_mean"]),
                    "n_segments": int(summary["n_segments"]),
                }
            )
    if not rows:
        raise RuntimeError("No shape-head summaries found.")

    detail_path = OUT_DIR / "final_fold_metrics.csv"
    with detail_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row["pcc"]) for row in rows if row["dataset"] == dataset]
        summary_rows.append(
            {
                "dataset": dataset,
                "mean_pcc": mean(vals),
                "std_pcc": pstdev(vals) if len(vals) > 1 else 0.0,
                "n_folds": len(vals),
            }
        )
    with (OUT_DIR / "final_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {detail_path}", flush=True)
    print(f"Wrote {OUT_DIR / 'final_summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mmecg", "crrvs"], choices=sorted(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()

    tags = [tag_for(DATASETS[key], args.seed) for key in args.datasets]
    if args.aggregate_only:
        aggregate(tags)
        return

    if not git_clean():
        raise SystemExit("Refusing to run final-candidate training from a dirty worktree.")

    commit = git_commit()
    branch = git_branch()
    for key in args.datasets:
        ensure_registry_row(tag_for(DATASETS[key], args.seed), DATASETS[key], commit, branch)

    for key in args.datasets:
        dataset = DATASETS[key]
        tag = tag_for(dataset, args.seed)
        update_registry_status(tag, "running", "Final-candidate full-fold training/eval is running.")
        try:
            train_and_eval(dataset, args.seed, args.epochs, args.early_stop_patience, args.checkpoint)
        except Exception as exc:
            update_registry_status(tag, "failed", f"Final-candidate run failed: {exc}")
            raise
        update_registry_status(tag, "completed", "Final-candidate full-fold training/eval completed; awaiting metric audit.")

    aggregate(tags)


if __name__ == "__main__":
    main()
