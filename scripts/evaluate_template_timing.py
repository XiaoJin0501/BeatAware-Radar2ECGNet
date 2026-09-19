#!/usr/bin/env python3
"""Template-timing diagnostics: is the continuous-ECG gap timing or morphology?

Registered analyses `q1_template_timing_diagnostics_seed42` (MMECG) and
`q1_crrvs_diagnostics_seed42` (CR_RVS, A3-A5). A fixed beat template, built from
TRAINING-fold labels only, is placed at different sources of R-peak timing and
scored with the frozen evaluator functions, exactly like the models:

  T0  ground-truth R peaks (oracle timing -- an upper bound, never achievable);
  T1  REM-Net's event-head peaks (radar-only);
  T2  peaks detected on REM-Net's reconstructed waveform (radar-only);
  T3  peaks detected on Chen-TCN's reconstructed waveform (radar-only);
  T4  peaks detected on WaveGRU-Net's reconstructed waveform (radar-only);
  T1c / T3c  T1 / T3 shifted by each test subject's own median offset to the GT
             peaks (ORACLE per-subject latency calibration).

It also decomposes the signed timing error of every radar-derived peak source into
a global bias, a between-subject bias spread and within-subject jitter.

Nothing is trained or selected.

Usage:
  python scripts/evaluate_template_timing.py                  # MMECG
  python scripts/evaluate_template_timing.py --dataset crrvs  # CR_RVS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean, pstdev

import h5py
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_anchor_head import peaks_from_heatmap  # noqa: E402
from scripts.train_continuous_ecg_pilot import (  # noqa: E402
    region_errors, smoothing_diagnostics, timing_metrics, waveform_metrics,
)
from src.utils.atomic_state import atomic_csv, atomic_json  # noqa: E402
from src.utils.metrics import detect_rpeaks  # noqa: E402

FS = 200
WIN, TEMPLATE_LEN = 1600, 200
MATCH_TOL = int(round(0.150 * FS))          # +/-150 ms for the offset decomposition
METRICS = ("pcc", "rmse", "nrmse", "mae", "r2", "amplitude_variance_ratio",
           "f1_50ms", "f1_100ms", "qrs_region_mae", "stt_region_mae")

EXPERIMENTS = ROOT / "experiments_mmecg"
CONFIG = {
    "mmecg": {
        "tag": "q1_template_timing_diagnostics_seed42",
        "splits": Path("/home/qhh2237/Datasets/MMECG/processed/loso_v2"),
        "folds": list(range(1, 12)),
        "rem_net": EXPERIMENTS / "q1_remnet_continuous_strict_loso_seed42" / "L3",
        "baselines": EXPERIMENTS / "q1_matched_baselines_continuous_strict_loso_seed42",
    },
    "crrvs": {
        "tag": "q1_template_timing_crrvs_seed42",
        "splits": Path("/home/qhh2237/Datasets/CR_RVS/processed/q1_continuous_kfold"),
        "folds": list(range(1, 6)),
        "rem_net": EXPERIMENTS / "q1_remnet_continuous_crrvs_seed42" / "L3",
        "baselines": EXPERIMENTS / "q1_matched_baselines_continuous_crrvs_seed42",
    },
}
MODEL_OUTPUTS = ("remnet_output", "chen_tcn_output", "wavegru_output")
TIMED = ("T0_gt", "T1_remnet_event_head", "T2_remnet_waveform", "T3_chen_tcn_waveform",
         "T4_wavegru_waveform", "T1c_event_head_subject_calibrated",
         "T3c_chen_tcn_subject_calibrated")
PEAK_SOURCES = ("T1_remnet_event_head", "T2_remnet_waveform", "T3_chen_tcn_waveform",
                "T4_wavegru_waveform")


def _clean(peaks) -> np.ndarray:
    r = np.sort(np.unique(np.asarray(peaks, np.int64)))
    return r[(r >= 0) & (r < WIN)]


def train_template(splits: Path, fold: int) -> tuple[np.ndarray, float, float]:
    """Mean R-to-R cycle of the TRAINING fold, its median RR and its mean level."""
    grid = np.linspace(0, 1, TEMPLATE_LEN)
    cycles, rrs = [], []
    with h5py.File(splits / f"fold_{fold:02d}" / "train.h5", "r") as h:
        ecg, peaks = h["ecg"][:, 0], h["rpeak_indices"][:]
    for e, p in zip(ecg, peaks):
        r = _clean(p)
        for a, b in zip(r[:-1], r[1:]):
            if b - a > 10:
                cycles.append(np.interp(grid, np.linspace(0, 1, b - a), e[a:b]))
                rrs.append(b - a)
    return np.mean(cycles, axis=0), float(np.median(rrs)), float(ecg.mean())


def synthesise(peaks: np.ndarray, template: np.ndarray, fallback_rr: float,
               level: float) -> np.ndarray:
    """Place the template between consecutive peaks; pad the ends with whole cycles."""
    out = np.full(WIN, level, dtype=np.float32)
    if len(peaks) == 0:
        return out
    grid = np.linspace(0, 1, TEMPLATE_LEN)

    def cycle(n: int) -> np.ndarray:
        return np.interp(np.linspace(0, 1, n), grid, template)

    rr = float(np.median(np.diff(peaks))) if len(peaks) >= 2 else fallback_rr
    n = max(int(round(rr)), 11)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b > a:
            out[a:b] = cycle(b - a)
    first, last = int(peaks[0]), int(peaks[-1])
    if first > 0:                                   # a cycle ending on the first peak
        c = cycle(n)
        k = min(first, n)
        out[first - k:first] = c[n - k:]
    if last < WIN:                                  # a cycle starting on the last peak
        c = cycle(n)
        k = min(WIN - last, n)
        out[last:last + k] = c[:k]
        if last + k < WIN:
            out[last + k:] = c[k - 1]
    return out


def score(pred: np.ndarray, gt: np.ndarray, rpeaks) -> dict:
    """The frozen evaluator's metrics, as `evaluate()` computes them."""
    rps = [np.asarray(r, float) for r in rpeaks]
    m = waveform_metrics(pred, gt)
    m.update(smoothing_diagnostics(pred, gt))
    m.update(timing_metrics(pred, np.zeros_like(pred), rps))
    m.update(region_errors(pred, gt, rps))
    return {k: m[k] for k in METRICS}


def signed_offsets(radar_sets, gt_sets) -> tuple[list[np.ndarray], int, int]:
    """Per window, offsets (samples) of the nearest radar peak to each GT peak."""
    offsets, n_gt, n_radar = [], 0, 0
    for rp, gp in zip(radar_sets, gt_sets):
        n_gt += len(gp); n_radar += len(rp)
        if len(rp) == 0 or len(gp) == 0:
            offsets.append(np.empty(0))
            continue
        d = rp[None, :] - gp[:, None]                  # (gt, radar)
        nearest = d[np.arange(len(gp)), np.abs(d).argmin(axis=1)]
        offsets.append(nearest[np.abs(nearest) <= MATCH_TOL].astype(float))
    return offsets, n_gt, n_radar


def decompose(offsets: list[np.ndarray], subjects: np.ndarray) -> dict:
    """Global bias, between-subject bias spread and within-subject jitter, in ms."""
    to_ms = 1000.0 / FS
    per_subject = {}
    for off, s in zip(offsets, subjects):
        per_subject.setdefault(int(s), []).append(off)
    biases, jitters = {}, {}
    for s, parts in per_subject.items():
        v = np.concatenate(parts) if parts else np.empty(0)
        if len(v) >= 5:
            biases[s] = float(np.median(v) * to_ms)
            jitters[s] = float(1.4826 * np.median(np.abs(v - np.median(v))) * to_ms)
    allv = np.concatenate([o for o in offsets if len(o)]) if any(len(o) for o in offsets) \
        else np.empty(0)
    return {"global_bias_ms": float(np.median(allv) * to_ms) if len(allv) else float("nan"),
            "global_abs_error_ms": float(np.median(np.abs(allv)) * to_ms) if len(allv)
            else float("nan"),
            "between_subject_bias_sd_ms": float(np.std(list(biases.values())))
            if biases else float("nan"),
            "within_subject_jitter_ms": float(np.median(list(jitters.values())))
            if jitters else float("nan"),
            "per_subject_bias_ms": biases, "per_subject_jitter_ms": jitters}


def calibrated(peak_sets, offsets, subjects) -> list[np.ndarray]:
    """ORACLE: shift each window's peaks by its subject's median offset to GT."""
    per_subject = {}
    for off, s in zip(offsets, subjects):
        per_subject.setdefault(int(s), []).append(off)
    shift = {s: (float(np.median(np.concatenate(v))) if sum(len(x) for x in v) else 0.0)
             for s, v in per_subject.items()}
    return [_clean(np.round(p - shift[int(s)])) for p, s in zip(peak_sets, subjects)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(CONFIG), default="mmecg")
    cfg = CONFIG[ap.parse_args().dataset]
    out_dir = EXPERIMENTS / cfg["tag"]

    variants = MODEL_OUTPUTS + TIMED
    per = {v: [] for v in variants}
    failures = {v: 0 for v in TIMED}
    all_offsets = {src: [] for src in PEAK_SOURCES}
    all_subjects, counts = [], {src: [0, 0] for src in PEAK_SOURCES}
    rows = []
    for fold in cfg["folds"]:
        fd = f"fold_{fold:02d}"
        rem = np.load(cfg["rem_net"] / fd / "test_predictions.npz", allow_pickle=True)
        chen = np.load(cfg["baselines"] / "mmecg_tcn" / fd / "test_predictions.npz",
                       allow_pickle=True)
        wave = np.load(cfg["baselines"] / "wavegru_net" / fd / "test_predictions.npz",
                       allow_pickle=True)
        gt, rpeaks = rem["ground_truth"], rem["rpeaks"]
        for other in (chen, wave):
            if not np.array_equal(gt, other["ground_truth"]):
                raise SystemExit(f"fold {fold}: model test windows differ")
        with h5py.File(cfg["splits"] / fd / "test.h5", "r") as h:
            subjects = h["subject_id"][:]
            if not np.allclose(h["ecg"][:, 0], gt):
                raise SystemExit(f"fold {fold}: predictions are not row-aligned with test.h5")
        template, fallback_rr, level = train_template(cfg["splits"], fold)

        gt_sets = [_clean(r) for r in rpeaks]
        sources = {
            "T0_gt": gt_sets,
            "T1_remnet_event_head": [_clean(peaks_from_heatmap(p[None]))
                                     for p in rem["event_probability"]],
            "T2_remnet_waveform": [_clean(detect_rpeaks(p)) for p in rem["prediction"]],
            "T3_chen_tcn_waveform": [_clean(detect_rpeaks(p)) for p in chen["prediction"]],
            "T4_wavegru_waveform": [_clean(detect_rpeaks(p)) for p in wave["prediction"]],
        }
        fold_offsets = {}
        for src in PEAK_SOURCES:
            off, n_gt, n_radar = signed_offsets(sources[src], gt_sets)
            fold_offsets[src] = off
            all_offsets[src].extend(off)
            counts[src][0] += n_gt
            counts[src][1] += n_radar
        all_subjects.extend(subjects.tolist())
        sources["T1c_event_head_subject_calibrated"] = calibrated(
            sources["T1_remnet_event_head"], fold_offsets["T1_remnet_event_head"], subjects)
        sources["T3c_chen_tcn_subject_calibrated"] = calibrated(
            sources["T3_chen_tcn_waveform"], fold_offsets["T3_chen_tcn_waveform"], subjects)

        results = {"remnet_output": score(rem["prediction"], gt, rpeaks),
                   "chen_tcn_output": score(chen["prediction"], gt, rpeaks),
                   "wavegru_output": score(wave["prediction"], gt, rpeaks)}
        for name in TIMED:
            failures[name] += sum(len(p) == 0 for p in sources[name])
            synth = np.stack([synthesise(p, template, fallback_rr, level) for p in sources[name]])
            results[name] = score(synth, gt, rpeaks)
        for name, r in results.items():
            per[name].append(r)
            rows.append({"variant": name, "fold": fold, **r})
        print(f"fold {fold:2d}: " + "  ".join(f"{n[:3]} {r['pcc']:+.3f}"
                                              for n, r in results.items()), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(out_dir / "template_timing_fold_metrics.csv", list(rows[0]), rows)
    summary = {v: {k: {"mean": mean(r[k] for r in rs), "std_population": pstdev(r[k] for r in rs)}
                   for k in METRICS} for v, rs in per.items()}
    ref = per["remnet_output"]
    comparisons = {}
    for v in TIMED + ("chen_tcn_output", "wavegru_output"):
        comparisons[v] = {}
        for k in ("pcc", "r2", "nrmse", "f1_100ms", "qrs_region_mae"):
            a = np.array([r[k] for r in per[v]]); b = np.array([r[k] for r in ref])
            try:
                p = float(wilcoxon(a, b).pvalue)
            except ValueError:
                p = float("nan")
            comparisons[v][k] = {"variant_minus_remnet": float((a - b).mean()),
                                 "wilcoxon_p": p, "variant_higher_folds": int(((a - b) > 0).sum())}
    subjects_arr = np.asarray(all_subjects)
    timing = {}
    for src in PEAK_SOURCES:
        d = decompose(all_offsets[src], subjects_arr)
        matched = sum(len(o) for o in all_offsets[src])
        d.update({"recall_within_150ms": matched / max(counts[src][0], 1),
                  "radar_peaks_per_gt_peak": counts[src][1] / max(counts[src][0], 1)})
        timing[src] = d
    t0 = summary["T0_gt"]["r2"]["mean"]
    recovery = {}
    for raw, cal in (("T1_remnet_event_head", "T1c_event_head_subject_calibrated"),
                     ("T3_chen_tcn_waveform", "T3c_chen_tcn_subject_calibrated")):
        r_raw, r_cal = summary[raw]["r2"]["mean"], summary[cal]["r2"]["mean"]
        recovery[raw] = {"r2_raw": r_raw, "r2_subject_calibrated": r_cal, "r2_gt_timing": t0,
                         "fraction_of_gap_recovered": (r_cal - r_raw) / (t0 - r_raw)
                         if t0 != r_raw else float("nan")}
    atomic_json(out_dir / "template_timing_summary.json", {
        "experiment_id": cfg["tag"],
        "template": "training-fold mean R-to-R cycle, stored ECG units",
        "aggregation": f"window-macro within fold, fold-macro over {len(cfg['folds'])} folds, "
                       "population std",
        "windows_without_peaks": failures, "variants": summary,
        "versus_remnet_output": comparisons,
        "timing_error_decomposition": timing,
        "subject_calibration_recovery": recovery})

    print(f"\n{'variant':34s}" + "".join(f"{k:>16s}" for k in ("pcc", "r2", "nrmse", "f1_100ms",
                                                              "qrs_region_mae", "var_ratio")))
    for v, s in summary.items():
        print(f"{v:34s}" + "".join(
            f"  {s[k]['mean']:+.3f}±{s[k]['std_population']:.3f}"
            for k in ("pcc", "r2", "nrmse", "f1_100ms", "qrs_region_mae",
                      "amplitude_variance_ratio")))
    print(f"\nwindows with no peaks: {failures}")
    print("\ntiming error of radar-derived peaks (matched within ±150 ms):")
    for src, d in timing.items():
        print(f"  {src:24s} recall {d['recall_within_150ms']:.3f}  bias {d['global_bias_ms']:+6.1f} ms"
              f"  |err| {d['global_abs_error_ms']:5.1f} ms  between-subject bias SD "
              f"{d['between_subject_bias_sd_ms']:5.1f} ms  within-subject jitter "
              f"{d['within_subject_jitter_ms']:5.1f} ms")
    print("\noracle per-subject latency calibration (R²):")
    for src, r in recovery.items():
        print(f"  {src:24s} raw {r['r2_raw']:+.3f} -> calibrated {r['r2_subject_calibrated']:+.3f}"
              f"  (GT timing {r['r2_gt_timing']:+.3f}; gap recovered "
              f"{100 * r['fraction_of_gap_recovered']:.1f} %)")


if __name__ == "__main__":
    main()
