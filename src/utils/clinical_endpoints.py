"""Versioned, provisional clinical endpoints. Coordinates are samples, errors ms.

No reference data is accepted by the prediction delineator. NaNs mean unknown,
not confirmed physiological absence. No GT alignment or amplitude correction.
"""
import numpy as np

FIELDS = {"q": "ECG_Q_Peaks", "s": "ECG_S_Peaks", "t": "ECG_T_Peaks",
          "p_on": "ECG_P_Onsets", "qrs_on": "ECG_R_Onsets",
          "qrs_off": "ECG_R_Offsets", "t_off": "ECG_T_Offsets"}
INTERVALS = {"qrs": ("qrs_on", "qrs_off"), "qt": ("qrs_on", "t_off"),
             "pr": ("p_on", "qrs_on")}


def delineate_signal(signal, fs=200, method="dwt"):
    """Independently detect R peaks and delineate this signal, never using GT.

    ``dwt`` remains the default so the completed exploratory v1 artifacts keep
    their exact meaning.  Registered evaluations must record ``method``.
    """
    import neurokit2 as nk
    x = np.asarray(signal, float).reshape(-1)
    try:
        clean = nk.ecg_clean(x, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    except (ValueError, IndexError, ZeroDivisionError):
        return {k: np.array([], float) for k in (*FIELDS, "r")}
    r = np.asarray(info["ECG_R_Peaks"], float)
    out = {k: np.full(len(r), np.nan) for k in FIELDS}
    out["r"] = r
    try:
        _, waves = nk.ecg_delineate(clean, rpeaks=r.astype(int),
                                    sampling_rate=fs, method=method)
    except (ValueError, IndexError, ZeroDivisionError):
        return out
    for k, name in FIELDS.items():
        a = np.asarray(waves.get(name, []), float)
        if len(a) == len(r):
            out[k] = np.where((a >= 0) & (a < len(x)), a, np.nan)
    return out


def match_beats(ref, pred, tolerance):
    """Order-preserving maximum-cardinality match, then minimum total error."""
    n, m = len(ref), len(pred)
    score = [[(0, 0.) for _ in range(m + 1)] for _ in range(n + 1)]
    path = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [(score[i-1][j], (i-1, j)), (score[i][j-1], (i, j-1))]
            d = abs(ref[i-1] - pred[j-1])
            if d <= tolerance:
                c, e = score[i-1][j-1]
                choices.append(((c + 1, e - d), (i-1, j-1)))
            score[i][j], path[i, j] = max(choices, key=lambda x: x[0])
    pairs = []
    i, j = n, m
    while i and j:
        a, b = path[i, j]
        if a == i-1 and b == j-1:
            pairs.append((i-1, j-1))
        i, j = a, b
    return dict(reversed(pairs))


def score_endpoints(ref, pred, fs=200, tolerance_ms=150):
    """Return per-window sufficient observations, retaining missing predictions.

Interval eligibility is endpoint-specific: finite, positive reference duration.
No normal-range filter excludes pathological intervals. RR requires adjacency
on BOTH sides. Event detection success additionally requires <= tolerance_ms.
"""
    for points in (ref, pred):
        r = np.asarray(points["r"])
        if not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0):
            raise ValueError("R coordinates must be finite, strictly increasing")
        if any(len(points[k]) != len(r) for k in FIELDS):
            raise ValueError("Every fiducial must have one entry per detected R")
    pairs = match_beats(ref["r"], pred["r"], tolerance_ms * fs / 1000)
    rows = []
    for name in ("q", "r", "s", "t", "qq", "rr", "ss", "tt", "qrs", "qt", "qtc", "pr"):
        event = name in ("q", "r", "s", "t")
        errors, relative = [], []
        eligible = success = 0
        for i in range(len(ref["r"])):
            j = pairs.get(i)
            if event:
                actual = ref[name][i]
                estimated = pred[name][j] if j is not None else np.nan
            elif name in ("qq", "rr", "ss", "tt"):
                if i == 0:
                    continue
                point = name[0]
                actual = ref[point][i] - ref[point][i-1]
                estimated = (pred[point][j] - pred[point][j-1]
                             if j is not None and j > 0 and pairs.get(i-1) == j-1 else np.nan)
            elif name == "qtc":
                if i == 0:
                    continue
                actual_qt = ref["t_off"][i] - ref["qrs_on"][i]
                actual_rr = ref["r"][i] - ref["r"][i-1]
                actual = actual_qt / np.cbrt(actual_rr / fs) if actual_rr > 0 else np.nan
                if j is not None and j > 0 and pairs.get(i-1) == j-1:
                    estimated_qt = pred["t_off"][j] - pred["qrs_on"][j]
                    estimated_rr = pred["r"][j] - pred["r"][j-1]
                    estimated = (estimated_qt / np.cbrt(estimated_rr / fs)
                                 if estimated_rr > 0 else np.nan)
                else:
                    estimated = np.nan
            else:
                a, b = INTERVALS[name]
                actual = ref[b][i] - ref[a][i]
                estimated = pred[b][j] - pred[a][j] if j is not None else np.nan
            if not np.isfinite(actual) or (not event and actual <= 0):
                continue
            eligible += 1
            if not np.isfinite(estimated) or (not event and estimated <= 0):
                continue
            error = abs(estimated - actual) * 1000 / fs
            errors.append(float(error))
            if event:
                success += int(error <= tolerance_ms)
            else:
                relative.append(float(100 * abs(estimated - actual) / actual))
        predicted = int(np.isfinite(pred[name]).sum()) if event else None
        total = (max(len(ref["r"])-1, 0)
                 if name in ("qq", "rr", "ss", "tt", "qtc") else len(ref["r"]))
        rows.append(dict(metric=name, reference_candidates=total,
                         reference_unassessable=total-eligible,
                         eligible=eligible, evaluable=len(errors),
                         errors_ms=errors, relative_errors_pct=relative,
                         coverage=len(errors)/eligible if eligible else None,
                         unavailable_rate=1-len(errors)/eligible if eligible else None,
                         mae_ms=float(np.mean(errors)) if errors else None,
                         medae_ms=float(np.median(errors)) if errors else None,
                         p95_ms=float(np.percentile(errors, 95)) if errors else None,
                         mre_pct=float(np.mean(relative)) if relative else None,
                         detection_tp=success if event else None,
                         predicted_count=predicted,
                         f1=2*success/(eligible+predicted) if event and eligible+predicted else None))
    return rows
