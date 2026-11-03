"""Shared waveform metrics, delegated to the frozen continuous evaluator.

The A3 scope needs only pair metrics and subject aggregation. Later A/B code
must import these functions rather than reproduce their formulas.
"""
from __future__ import annotations

import warnings

import numpy as np

from scripts.train_continuous_ecg_pilot import waveform_metrics

FISHER_EPS = 1e-7


def waveform_pair(pred, reference):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if pred.shape != reference.shape or len(pred) < 2:
        raise ValueError("Waveforms must have identical support and length >=2")
    if not np.isfinite(pred).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite waveform: refusing silent exclusion")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        measured = waveform_metrics([pred], [reference])
    return {"pcc": float(measured["pcc"]) if np.isfinite(measured["pcc"]) else None,
            "rmse": float(measured["rmse"])}


def fisher_mean(values):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("NaN must be explicitly represented and counted")
    return float(np.tanh(np.mean(np.arctanh(np.clip(
        values, -1 + FISHER_EPS, 1 - FISHER_EPS)))))


def distribution(values):
    values = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite subject summary")
    q1, median, q3 = np.percentile(values, [25, 50, 75], method="linear")
    return {"n": len(values), "median": float(median), "q1": float(q1),
            "q3": float(q3), "mean": float(values.mean())}


def aggregate_subjects(rows):
    groups = {}
    for row in rows:
        key = row["dataset"], row["rr_bin"], int(row["subject_id"])
        groups.setdefault(key, []).append(row)
    subjects = []
    for (dataset, rr_bin, subject), observations in sorted(groups.items()):
        pccs = [x["pcc"] for x in observations]
        subjects.append({"dataset": dataset, "rr_bin": rr_bin, "subject_id": subject,
            "n_beats": len(observations),
            "n_undefined_pcc": sum(x is None for x in pccs),
            "pcc_fisher_mean": fisher_mean(pccs),
            "rmse_mean": float(np.mean([x["rmse"] for x in observations]))})
    return subjects
