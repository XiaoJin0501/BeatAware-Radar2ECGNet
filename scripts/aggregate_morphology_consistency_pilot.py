#!/usr/bin/env python3
"""Aggregate the registered morphology-consistency validation pilot."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments_mmecg"
OUT_ROOT = RUN_ROOT / "q1_morphology_consistency_pilot_summary"
FOLDS = {"mmecg": (1, 5, 9), "crrvs": (1, 3, 5)}
ARMS = ("b0", "b1", "b2")
METRICS = {
    "waveform_pcc": ("waveform", "pcc", True),
    "waveform_rmse": ("waveform", "rmse", False),
    "waveform_mae": ("waveform", "mae", False),
    "waveform_r2": ("waveform", "r2", True),
    "shape200_pcc": ("shape200", "pcc", True),
    "event_f1_50ms": ("event_head", "f1_50ms", True),
    "waveform_f1_50ms": ("waveform_derived", "f1_50ms", True),
    "waveform_f1_100ms": ("waveform_derived", "f1_100ms", True),
    "waveform_f1_150ms": ("waveform_derived", "f1_150ms", True),
}


def load_result(dataset: str, arm: str, fold: int) -> dict:
    path = (
        RUN_ROOT
        / f"q1_morph_consistency_pilot_{dataset}_{arm}"
        / f"fold_{fold:02d}"
        / "results_validation"
        / "metrics.json"
    )
    result = json.loads(path.read_text())
    if result["split"] != "validation" or result["publishable"]:
        raise ValueError(f"Unexpected evidence tier in {path}")
    return result


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    raw = {}
    rows = []
    for dataset, folds in FOLDS.items():
        raw[dataset] = {}
        for arm in ARMS:
            results = [load_result(dataset, arm, fold) for fold in folds]
            raw[dataset][arm] = results
            row = {"dataset": dataset, "arm": arm, "n_folds": len(folds)}
            for label, (section, key, _) in METRICS.items():
                values = np.asarray([r[section][key] for r in results], dtype=float)
                row[f"{label}_mean"] = float(values.mean())
                row[f"{label}_std"] = float(values.std(ddof=1))
            rows.append(row)

    comparisons = []
    for dataset, folds in FOLDS.items():
        for candidate, reference in (("b1", "b0"), ("b1", "b2")):
            item = {"dataset": dataset, "candidate": candidate, "reference": reference}
            for label, (section, key, higher_is_better) in METRICS.items():
                candidate_values = np.asarray(
                    [r[section][key] for r in raw[dataset][candidate]], dtype=float
                )
                reference_values = np.asarray(
                    [r[section][key] for r in raw[dataset][reference]], dtype=float
                )
                delta = candidate_values - reference_values
                wins = delta > 0 if higher_is_better else delta < 0
                item[f"{label}_delta_mean"] = float(delta.mean())
                item[f"{label}_wins"] = int(wins.sum())
                item[f"{label}_n"] = len(folds)
            comparisons.append(item)

    fieldnames = list(rows[0])
    with (OUT_ROOT / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tier": "pilot",
        "publishable": False,
        "split": "validation",
        "checkpoint_rule": "minimum common validation base loss",
        "aggregates": rows,
        "paired_comparisons": comparisons,
    }
    (OUT_ROOT / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
