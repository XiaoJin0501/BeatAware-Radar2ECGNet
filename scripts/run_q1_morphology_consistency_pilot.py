#!/usr/bin/env python3
"""Run the registered B0/B1/B2 validation-only development pilot."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments_mmecg" / "EXPERIMENT_REGISTRY.csv"
ARMS = {"b0": (0.0, "head"), "b1": (0.05, "head"), "b2": (0.05, "reference")}
DATASETS = {
    "mmecg": ("/home/qhh2237/Datasets/MMECG/processed/loso", 50, "1,5,9", True),
    "crrvs": ("/home/qhh2237/Datasets/CR_RVS/processed/q1_shape_kfold", 1, "1,3,5", False),
}


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def update(tag: str, status: str, reason: str) -> None:
    with REGISTRY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    found = False
    for row in rows:
        if row["tag"] == tag:
            row["status"], row["reason"], found = status, reason, True
    if not found:
        raise KeyError(f"Unregistered experiment: {tag}")
    with REGISTRY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["mmecg", "crrvs", "both"], default="both")
    ap.add_argument("--arms", default="b0,b1,b2")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    datasets = DATASETS if args.dataset == "both" else {args.dataset: DATASETS[args.dataset]}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(arm)
    for dataset, (root, channels, folds_text, bandpass) in datasets.items():
        for arm in arms:
            tag = f"q1_morph_consistency_pilot_{dataset}_{arm}"
            weight, target = ARMS[arm]
            update(tag, "running", "Registered development pilot is running; validation-only decision.")
            for fold in [int(x) for x in folds_text.split(",")]:
                run([
                    sys.executable, "scripts/train_mmecg.py", "--exp_tag", tag,
                    "--fold_idx", str(fold), "--loso_h5_dir", root,
                    "--n_range_bins", str(channels), "--model_arch", "rem_align_mtl",
                    "--epochs", str(args.epochs), "--early_stop_patience", str(args.patience),
                    "--lr", "0.0001", "--weight_decay", "0.0001",
                    "--narrow_bandpass", str(bandpass).lower(), "--target_norm", "minmax",
                    "--wsst_suffix", "_wsst.h5", "--beta_peak", "0.3",
                    "--use_shape200_loss", "true", "--lambda_shape200", "0.8",
                    "--shape200_pcc_weight", "0.5", "--baseline_hidden", "128",
                    "--use_alignment", "false", "--seed", "42",
                    "--lambda_morphology_consistency", str(weight),
                    "--morphology_consistency_target", target,
                ])
                run([sys.executable, "scripts/evaluate_remnet_validation.py",
                     "--exp_tag", tag, "--fold", str(fold),
                     "--checkpoint", "best_loss.pt", "--device", args.device])
            update(tag, "completed", "Three development folds trained and evaluated on validation only; awaiting audit.")
    print("Pilot runs completed; aggregate validation artifacts before any promotion.")


if __name__ == "__main__":
    main()
