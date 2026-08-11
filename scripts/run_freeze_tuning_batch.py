#!/usr/bin/env python3
"""Run the F0 supervised REM-Net freeze-tuning batch.

This runner is intentionally conservative:
- refuses to run from a dirty git worktree;
- registers every candidate before training;
- trains/evaluates sequentially to avoid GPU contention;
- aggregates fold-level shape-head summaries after completion.
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
OUT_SUMMARY = ROOT / "experiments_mmecg" / "q1_freeze_f0_supervised_tuning"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    folds: tuple[int, ...]
    loso_h5_dir: str
    n_range_bins: int
    narrow_bandpass: bool


@dataclass(frozen=True)
class Candidate:
    name: str
    beta_peak: float
    lambda_shape200: float
    shape200_pcc_weight: float
    baseline_hidden: int


DATASETS = {
    "mmecg": DatasetSpec(
        key="mmecg",
        label="MMECG public",
        folds=(1, 5, 9),
        loso_h5_dir="/home/qhh2237/Datasets/MMECG/processed/loso",
        n_range_bins=50,
        narrow_bandpass=True,
    ),
    "crrvs": DatasetSpec(
        key="crrvs",
        label="CR_RVS / Nature",
        folds=(1, 3, 5),
        loso_h5_dir="/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold",
        n_range_bins=1,
        narrow_bandpass=False,
    ),
}


CANDIDATES = (
    Candidate("incumbent", 0.3, 0.8, 0.5, 128),
    Candidate("shape12", 0.3, 1.2, 0.5, 128),
    Candidate("pcc08", 0.3, 0.8, 0.8, 128),
    Candidate("peak01", 0.1, 0.8, 0.5, 128),
    Candidate("peak06", 0.6, 0.8, 0.5, 128),
    Candidate("hidden192", 0.3, 0.8, 0.5, 192),
)


def run_capture(cmd: list[str], *, cwd: Path = ROOT) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    result.check_returncode()
    return result.stdout.strip()


def run_stream(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


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


def ensure_registry_row(tag: str, dataset: DatasetSpec, commit: str, branch: str, reason: str) -> None:
    rows = load_registry()
    if any(row.get("tag") == tag for row in rows):
        return
    rows.append(
        {
            "tag": tag,
            "dataset": dataset.label,
            "protocol": "F0 dev-fold tuning; strict subject-independent split",
            "category": "pilot",
            "publishable": "false",
            "commit_id": commit,
            "branch": branch,
            "role": "Final REM-Net freeze tuning",
            "status": "registered",
            "reason": reason,
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


def train_and_eval(dataset: DatasetSpec, candidate: Candidate, seed: int, epochs: int) -> None:
    tag = f"q1_freeze_f0_{dataset.key}_{candidate.name}_seed{seed}"
    common = [
        sys.executable,
        "scripts/train_mmecg.py",
        "--exp_tag",
        tag,
        "--fold_idx",
        "-1",
        "--loso_h5_dir",
        dataset.loso_h5_dir,
        "--n_range_bins",
        str(dataset.n_range_bins),
        "--model_arch",
        "rem_align_mtl",
        "--epochs",
        str(epochs),
        "--early_stop_patience",
        "100",
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
        str(candidate.beta_peak),
        "--use_shape200_loss",
        "true",
        "--lambda_shape200",
        str(candidate.lambda_shape200),
        "--shape200_pcc_weight",
        str(candidate.shape200_pcc_weight),
        "--baseline_hidden",
        str(candidate.baseline_hidden),
        "--use_alignment",
        "false",
        "--seed",
        str(seed),
    ]

    # The training script's -1 means all folds, so run specific dev folds one by one.
    for fold in dataset.folds:
        fold_dir = ROOT / "experiments_mmecg" / tag / f"fold_{fold:02d}"
        summary_path = fold_dir / "results_shape_head" / "global_summary_shape_head.json"
        if summary_path.exists():
            print(f"[skip] {tag} fold {fold:02d} already evaluated", flush=True)
            continue
        cmd = common.copy()
        cmd[cmd.index("-1")] = str(fold)
        run_stream(cmd)
        run_stream(
            [
                sys.executable,
                "scripts/evaluate_remalign_shape_head.py",
                "--exp_tag",
                tag,
                "--fold_idx",
                str(fold),
                "--checkpoint",
                "best.pt",
            ]
        )


def aggregate(tags: list[str]) -> None:
    OUT_SUMMARY.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tag in tags:
        for summary_path in sorted((ROOT / "experiments_mmecg" / tag).glob("fold_*/results_shape_head/global_summary_shape_head.json")):
            summary = json.loads(summary_path.read_text())
            cfg = json.loads((summary_path.parents[2] / "config.json").read_text())
            rows.append(
                {
                    "tag": tag,
                    "dataset": "crrvs" if "_crrvs_" in tag else "mmecg",
                    "candidate": tag.split("_seed")[0].split("_")[-1],
                    "seed": int(tag.split("seed")[-1]),
                    "fold": int(summary_path.parents[2].name.split("_")[-1]),
                    "pcc": float(summary["pcc_mean"]),
                    "rmse": float(summary["rmse_mean"]),
                    "mae": float(summary["mae_mean"]),
                    "r2": float(summary["r2_mean"]),
                    "beta_peak": cfg.get("beta_peak"),
                    "lambda_shape200": cfg.get("lambda_shape200"),
                    "shape200_pcc_weight": cfg.get("shape200_pcc_weight"),
                    "baseline_hidden": cfg.get("baseline_hidden"),
                }
            )

    if not rows:
        raise RuntimeError("No summaries found to aggregate.")

    fold_csv = OUT_SUMMARY / "f0_fold_metrics.csv"
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["candidate"])), []).append(float(row["pcc"]))

    summary_rows = []
    for (dataset, candidate), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "candidate": candidate,
                "mean_pcc": mean(vals),
                "std_pcc": pstdev(vals) if len(vals) > 1 else 0.0,
                "n_folds": len(vals),
            }
        )

    summary_csv = OUT_SUMMARY / "f0_candidate_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({"fold_csv": str(fold_csv), "summary_csv": str(summary_csv)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--candidates", nargs="+", default=[c.name for c in CANDIDATES])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--allow_dirty", action="store_true")
    args = parser.parse_args()

    if not args.allow_dirty and not git_clean():
        raise SystemExit("Refusing to launch training from a dirty worktree. Commit first.")

    commit = git_commit()
    branch = git_branch()
    selected_candidates = [c for c in CANDIDATES if c.name in set(args.candidates)]
    if not selected_candidates:
        raise SystemExit("No candidates selected.")

    tags: list[str] = []
    for dataset_key in args.datasets:
        dataset = DATASETS[dataset_key]
        for candidate in selected_candidates:
            tag = f"q1_freeze_f0_{dataset.key}_{candidate.name}_seed{args.seed}"
            tags.append(tag)
            ensure_registry_row(
                tag,
                dataset,
                commit,
                branch,
                (
                    f"F0 supervised freeze tuning on dev folds {dataset.folds}: "
                    f"beta={candidate.beta_peak}, lambda_shape200={candidate.lambda_shape200}, "
                    f"shape_pcc={candidate.shape200_pcc_weight}, hidden={candidate.baseline_hidden}."
                ),
            )

    for dataset_key in args.datasets:
        dataset = DATASETS[dataset_key]
        for candidate in selected_candidates:
            tag = f"q1_freeze_f0_{dataset.key}_{candidate.name}_seed{args.seed}"
            update_registry_status(tag, "running", "F0 supervised freeze tuning is running.")
            try:
                train_and_eval(dataset, candidate, args.seed, args.epochs)
            except Exception as exc:
                update_registry_status(tag, "failed", f"F0 tuning failed: {exc}")
                raise
            update_registry_status(tag, "completed", "F0 tuning train/eval completed; aggregate summary generated by runner.")

    aggregate(tags)


if __name__ == "__main__":
    main()
