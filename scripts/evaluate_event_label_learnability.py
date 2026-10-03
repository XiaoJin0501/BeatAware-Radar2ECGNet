#!/usr/bin/env python3
"""Gate 0: fixed-algorithm Q/R/S/T label completeness and repeatability audit."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.clinical_endpoints import delineate_signal, match_beats

SPEC = ROOT / "experiments/specs/q1_event_label_learnability_v1.json"
OUT = ROOT / "docs/audits/q1_event_label_learnability_v1"
POOL = Path("/home/qhh2237/Datasets/MMECG/processed/wsst8s_subject_pool")
FS, EDGE, TOL = 200, 60, 30  # samples: 300 ms edge; 150 ms matching
POINTS = ("q", "r", "s", "t")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save(path: Path, value) -> None:
    atomic(path, json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def digest_bytes(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def points_in(points: dict, name: str, lo: int, hi: int, start: int) -> np.ndarray:
    x = np.asarray(points[name], float)
    x = x[np.isfinite(x)] + start
    return np.sort(x[(x >= lo) & (x < hi)])


def paired_repeatability(a: np.ndarray, b: np.ndarray) -> tuple[int, int, list[float]]:
    pairs = match_beats(a, b, TOL)
    errors = [abs(float(a[i] - b[j])) * 1000 / FS for i, j in pairs.items()]
    # Symmetric denominator: matching all observations once gives coverage 1.
    return len(a) + len(b), 2 * len(errors), errors


def subject_job(path_text: str) -> str:
    path = Path(path_text)
    dest = OUT / f"{path.stem}.json"
    with h5py.File(path, "r") as h:
        ecg = np.asarray(h["ecg"][:, 0], np.float32)
        subject = np.asarray(h["subject_id"][:], np.int32)
        record = np.asarray(h["record_id"][:]).astype("S")
        start = np.asarray(h["start_idx"][:], np.int64)
        end = np.asarray(h["end_idx"][:], np.int64)
    fingerprint = digest_bytes([ecg, subject, record, start, end])
    if dest.exists():
        old = json.loads(dest.read_text())
        if old["ecg_metadata_sha256"] != fingerprint:
            raise RuntimeError(f"input fingerprint changed: {path}")
        return str(dest)

    detections = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in ecg:
            detections.append(delineate_signal(x, fs=FS, method="prominence"))

    available = {p: 0 for p in POINTS}
    eligible = {p: 0 for p in POINTS}
    assessable = violations = 0
    for pts in detections:
        n_r = len(pts["r"])
        for p in POINTS:
            eligible[p] += n_r
            available[p] += int(np.isfinite(pts[p]).sum())
        if n_r:
            matrix = np.stack([np.asarray(pts[p], float) for p in POINTS], axis=1)
            ok = np.all(np.isfinite(matrix), axis=1)
            assessable += int(ok.sum())
            violations += int(np.sum(ok & ~(
                (matrix[:, 0] < matrix[:, 1]) & (matrix[:, 1] < matrix[:, 2]) &
                (matrix[:, 2] < matrix[:, 3]))))

    rep_total = {p: 0 for p in POINTS}
    rep_matched = {p: 0 for p in POINTS}
    rep_errors = {p: [] for p in POINTS}
    order = np.lexsort((start, record))
    for ia, ib in zip(order[:-1], order[1:]):
        if record[ia] != record[ib] or start[ib] >= end[ia]:
            continue
        lo = max(int(start[ia]), int(start[ib])) + EDGE
        hi = min(int(end[ia]), int(end[ib])) - EDGE
        if hi <= lo:
            continue
        for p in POINTS:
            a = points_in(detections[ia], p, lo, hi, int(start[ia]))
            b = points_in(detections[ib], p, lo, hi, int(start[ib]))
            total, matched, errors = paired_repeatability(a, b)
            rep_total[p] += total
            rep_matched[p] += matched
            rep_errors[p].extend(errors)

    sid = int(np.unique(subject).item())
    metrics = {}
    for p in POINTS:
        e = np.asarray(rep_errors[p], float)
        metrics[p] = {
            "availability": available[p] / eligible[p] if eligible[p] else None,
            "available": available[p], "eligible": eligible[p],
            "repeatability_coverage": rep_matched[p] / rep_total[p] if rep_total[p] else None,
            "repeatability_matched": rep_matched[p], "repeatability_total": rep_total[p],
            "repeatability_medae_ms": np.median(e) if e.size else None,
            "repeatability_p95_ms": np.percentile(e, 95) if e.size else None,
        }
    save(dest, {"subject": sid, "source": str(path), "ecg_metadata_sha256": fingerprint,
                "windows": len(ecg), "order_assessable": assessable,
                "order_violations": violations,
                "order_violation_rate": violations / assessable if assessable else None,
                "metrics": metrics})
    return str(dest)


def summarize(paths: list[Path]) -> dict:
    rows = [json.loads(path.read_text()) for path in paths]
    metrics = {}
    completeness_pass = repeatability_pass = True
    for p in POINTS:
        a = np.asarray([r["metrics"][p]["availability"] for r in rows], float)
        c = np.asarray([r["metrics"][p]["repeatability_coverage"] for r in rows], float)
        m = np.asarray([r["metrics"][p]["repeatability_medae_ms"] for r in rows], float)
        q = np.asarray([r["metrics"][p]["repeatability_p95_ms"] for r in rows], float)
        cp = bool(np.median(a) >= .90 and np.min(a) >= .80)
        rp = bool(np.median(c) >= .90 and np.median(m) <= 10 and np.median(q) <= 20)
        completeness_pass &= cp
        repeatability_pass &= rp
        metrics[p] = {"availability_subject_median": np.median(a), "availability_subject_min": np.min(a),
                      "repeatability_coverage_subject_median": np.median(c),
                      "repeatability_medae_subject_median_ms": np.median(m),
                      "repeatability_p95_subject_median_ms": np.median(q),
                      "completeness_pass": cp, "repeatability_pass": rp}
    o = np.asarray([r["order_violation_rate"] for r in rows], float)
    order_pass = bool(np.median(o) <= .01 and np.max(o) <= .05)
    gate = {"completeness_pass": bool(completeness_pass),
            "repeatability_pass": bool(repeatability_pass), "ordering_pass": order_pass}
    gate["overall_pass"] = all(gate.values())
    return {"subjects": len(rows), "windows": sum(r["windows"] for r in rows),
            "metrics": metrics,
            "ordering": {"subject_median_violation_rate": np.median(o),
                         "subject_max_violation_rate": np.max(o), "pass": order_pass},
            "gate": gate}


def write_report(summary: dict) -> None:
    lines = ["# Q1 event-label learnability Gate 0", "",
             "Fixed `prominence` delineation on unique MMECG ECG windows. This audits computational label completeness and overlap repeatability; it does not establish clinical validity.", "",
             "| Fiducial | Availability median | Minimum subject | Repeatability coverage | Repeatability MedAE (ms) | Repeatability P95 (ms) | Complete | Repeatable |",
             "|---|---:|---:|---:|---:|---:|:---:|:---:|"]
    for p in POINTS:
        x = summary["metrics"][p]
        lines.append(f"| {p.upper()} | {x['availability_subject_median']:.3f} | {x['availability_subject_min']:.3f} | {x['repeatability_coverage_subject_median']:.3f} | {x['repeatability_medae_subject_median_ms']:.1f} | {x['repeatability_p95_subject_median_ms']:.1f} | {'PASS' if x['completeness_pass'] else 'FAIL'} | {'PASS' if x['repeatability_pass'] else 'FAIL'} |")
    lines += ["", f"Q<R<S<T violation rate: subject median {summary['ordering']['subject_median_violation_rate']:.4f}, maximum {summary['ordering']['subject_max_violation_rate']:.4f} — {'PASS' if summary['ordering']['pass'] else 'FAIL'}.", "",
              f"**Gate 0: {'PASS' if summary['gate']['overall_pass'] else 'FAIL'}**", "",
              "A pass authorises design of a separate three-fold Event-Head experiment only. It is not manuscript evidence and does not validate Q/S peaks clinically."]
    atomic(OUT / "REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    spec = json.loads(SPEC.read_text())
    if not spec["status"].startswith("APPROVED"):
        raise SystemExit("spec is not approved")
    files = sorted(POOL.glob("subject_*.h5"))
    if len(files) != 11:
        raise SystemExit(f"expected 11 subject files, found {len(files)}")
    import neurokit2
    source_hashes = {}
    for p in (SPEC, Path(__file__), ROOT / "src/utils/clinical_endpoints.py"):
        source_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    contract = {"version": "q1_event_label_learnability_v1", "neurokit2": neurokit2.__version__,
                "source_hashes": source_hashes, "input_files": [str(p) for p in files]}
    manifest = OUT / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise RuntimeError("run contract changed; refusing to mix artifacts")
    if not manifest.exists():
        save(manifest, contract)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(subject_job, str(p)) for p in files]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"subject {i}/11: {future.result()}", flush=True)
    result = summarize(sorted(OUT.glob("subject_*.json")))
    save(OUT / "summary.json", result)
    write_report(result)
    # Compact subject table for inspecting failures without treating windows as independent.
    subject_rows = []
    for path in sorted(OUT.glob("subject_*.json")):
        row = json.loads(path.read_text())
        flat = {"subject": row["subject"], "windows": row["windows"],
                "order_violation_rate": row["order_violation_rate"]}
        for p in POINTS:
            for key in ("availability", "repeatability_coverage", "repeatability_medae_ms", "repeatability_p95_ms"):
                flat[f"{p}_{key}"] = row["metrics"][p][key]
        subject_rows.append(flat)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(subject_rows[0]))
    writer.writeheader(); writer.writerows(subject_rows)
    atomic(OUT / "subject_metrics.csv", buf.getvalue())
    print(json.dumps(result["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
