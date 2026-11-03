"""Single metric entry point for the requested A diagnostics and later B logs.

Reuse the existing waveform, Shape-200, matching and delineation functions.
No model, loader, optimizer, sampling or RNG operations belong in this module.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from experiments.development.structured_decoder_v0_20261017_ab.waveform_metrics_shared import (
    distribution, fisher_mean, waveform_pair,
)
from scripts.run_dual_head_interaction import shape_metrics
from src.utils.clinical_endpoints import delineate_signal, match_beats


def shape_head_metrics(prediction, reference):
    """Historical Shape-200 arithmetic window mean, one target per window."""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[1] != 200:
        raise ValueError("Expected matching [number_of_windows,200] shape arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite Shape-200 input")
    return shape_metrics(prediction, reference)


def continuous_pair(prediction, reference):
    """Same PCC/RMSE evaluator; new contract explicitly scores flat signals 0."""
    result = waveform_pair(prediction, reference)
    result["undefined_pcc"] = result["pcc"] is None
    if result["undefined_pcc"]:
        result["pcc"] = 0.0
    result["failed"] = result["pcc"] < 0.5
    return result


def beat_metrics(prediction, reference, gt_r, valid_start, valid_stop):
    """All complete reference R-R cycles, no prediction-dependent selection."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.ndim != 1 or prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must share the record time axis")
    if not (0 <= valid_start < valid_stop <= len(reference)):
        raise ValueError("Invalid record scoring support")
    gt_r = np.asarray(gt_r)
    if not np.isfinite(gt_r).all() or np.any(gt_r != gt_r.astype(int)) or np.any(np.diff(gt_r) <= 0):
        raise ValueError("Reference R peaks must be strictly increasing sample indices")
    rows = []
    for a, b in zip(gt_r[:-1].astype(int), gt_r[1:].astype(int)):
        if a < valid_start or b > valid_stop:
            continue
        if b-a < 2:
            raise ValueError("GT cycle too short for correlation")
        rows.append({"start": int(a), "stop": int(b), "samples": int(b-a),
                     **continuous_pair(prediction[a:b], reference[a:b])})
    return rows


def window_metrics(prediction, reference, starts, width=1024):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.ndim != 1 or prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must share the record time axis")
    rows = []
    for start in starts:
        start = int(start)
        if start < 0 or start+width > len(reference):
            raise ValueError("Window exceeds shared record support")
        rows.append({"start": start, "stop": start+width,
                     **continuous_pair(prediction[start:start+width], reference[start:start+width])})
    return rows


def summarize_subjects(rows):
    """Input rows contain subject_id and native beat/window PCC/RMSE."""
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["subject_id"]), []).append(row)
    subjects = []
    for subject, group in sorted(grouped.items()):
        subjects.append({"subject_id": subject, "n": len(group),
            "pcc_fisher_mean": fisher_mean([x["pcc"] for x in group]),
            "rmse_mean": float(np.mean([x["rmse"] for x in group])),
            "failure_rate": sum(x["failed"] for x in group)/len(group),
            "undefined_pcc_count": sum(x["undefined_pcc"] for x in group)})
    return {"subjects": subjects,
        "pcc": distribution([x["pcc_fisher_mean"] for x in subjects]),
        "rmse": distribution([x["rmse_mean"] for x in subjects]),
        "subject_failure_rates": distribution([x["failure_rate"] for x in subjects]),
        "n_scored": len(rows),
        "pooled_failure_rate": sum(x["failed"] for x in rows)/len(rows) if rows else None}


def match_events(reference, predicted, tolerance):
    """Exact original matcher on disconnected temporal components.

    A gap > tolerance in the sorted UNION excludes every cross-gap feasible
    pair, so the original maximum-cardinality/minimum-cost DP factorizes.
    This avoids quadratic record-sized Python matrices for long CW records.
    """
    reference,predicted=np.asarray(reference,float),np.asarray(predicted,float)
    if not len(reference) or not len(predicted): return {}
    union=np.sort(np.concatenate([reference,predicted]))
    splits=union[np.where(np.diff(union)>tolerance)[0]+1]
    r0=p0=0; out={}
    for bound in [*splits,np.inf]:
        r1=int(np.searchsorted(reference,bound,side='left'))
        p1=int(np.searchsorted(predicted,bound,side='left'))
        out.update({i+r0:j+p0 for i,j in
                    match_beats(reference[r0:r1],predicted[p0:p1],tolerance).items()})
        r0,p0=r1,p1
    return out


def event_metrics(predicted, reference, fs=200, tolerance_ms=150):
    """Reuse order-preserving one-to-one matching, retaining missed events."""
    result = {}
    for name in ("q", "r", "s", "t"):
        p, y = np.asarray(predicted[name], float), np.asarray(reference[name], float)
        p, y = p[np.isfinite(p)], y[np.isfinite(y)]
        if np.any(np.diff(p) < 0) or np.any(np.diff(y) < 0):
            raise ValueError("Event locations must be ordered")
        pairs = match_events(y, p, tolerance_ms*fs/1000)
        tp, nr, npred = len(pairs), len(y), len(p)
        result[name] = {"reference_count": nr, "prediction_count": npred,
            "tp": tp, "fn": nr-tp, "fp": npred-tp,
            "se": tp/nr if nr else 0.0,
            "ppv": tp/npred if npred else 0.0,
            "f1": 2*tp/(nr+npred) if nr+npred else 0.0,
            "mae_ms": float(np.mean([abs(y[i]-p[j])*1000/fs for i,j in pairs.items()])) if pairs else None,
            "within_tolerance_fraction": tp/nr if nr else None}
    return result


def waveform_events(signal, fs=200):
    """The same frozen NeuroKit2 prominence pipeline on each waveform."""
    return delineate_signal(signal, fs=fs, method="prominence")


def summarize_events(rows):
    """Pool record counts within subject, then summarize subjects equally."""
    grouped={}
    for row in rows: grouped.setdefault(int(row['subject_id']),[]).append(row['events'])
    subjects=[]
    for subject,group in sorted(grouped.items()):
        values={}
        for k in ('q','r','s','t'):
            counts={c:sum(g[k][c] for g in group) for c in
                    ('tp','fn','fp','reference_count','prediction_count')}
            tp,nr,npred=counts['tp'],counts['reference_count'],counts['prediction_count']
            denom=sum(g[k]['tp'] for g in group if g[k]['mae_ms'] is not None)
            values[k]={**counts,'se':tp/nr if nr else 0.,'ppv':tp/npred if npred else 0.,
                'f1':2*tp/(nr+npred) if nr+npred else 0.,
                'mae_ms':sum(g[k]['mae_ms']*g[k]['tp'] for g in group if g[k]['mae_ms'] is not None)/denom if denom else None}
        subjects.append(dict(subject_id=subject,**values))
    return dict(subjects=subjects,**{k:{m:distribution([s[k][m] for s in subjects])
        for m in ('se','ppv','f1','mae_ms')} for k in ('q','r','s','t')})


def epoch_head_metrics(shapes, targets, probabilities, references):
    """B2 telemetry: same Shape-200 function and same event matcher as A.

    Window-micro head metrics are explicitly distinct from record-native
    waveform detection. The historical independent F1@100 selection is NOT
    replaced by any field returned here.
    """
    from scripts.run_dual_head_interaction import structured_decode, score_events
    rows=[]; differences=[]
    for p,ref in zip(probabilities,references):
        decoded=structured_decode(p)
        ref={k:np.sort(np.asarray(ref[k])[np.isfinite(ref[k])]) for k in ('q','r','s','t')}
        rows.append(dict(subject_id=1,events=event_metrics(decoded,ref)))
        differences.append(len(decoded['r'])-len(ref['r']))
    pooled=summarize_events(rows)['subjects'][0]
    events={k:dict(**pooled[k],within_tolerance_fraction=pooled[k]['se']) for k in ('q','r','s','t')}
    independent=score_events(probabilities,references,False)
    structured=score_events(probabilities,references,True)
    return dict(shape200=shape_head_metrics(shapes,targets),anchor=events,
        anchor_readout='structured head, window-micro, tolerance 150ms (NOT waveform R-F1)',
        independent_order_violation_rate=independent['order_violation_rate'],
        structured_order_violation_rate=structured['order_violation_rate'],
        mean_detected_minus_gt_beats=float(np.mean(differences)),
        amplitude_ratio_mean=float(np.mean(np.std(shapes,axis=1)/np.maximum(np.std(targets,axis=1),1e-8))),
        amplitude_ratio_space='Shape-200, not continuous ECG',
        event_probability_mean_peak=float(np.mean(np.max(probabilities,axis=-1))))


def judge_dataset(o4_pcc_median, r_f1_subject_median, pooled_beat_failure_rate,
                  o1_pcc_median, o2_pcc_median, o3_pcc_median):
    """Fixed author thresholds. Check joint bottleneck before either branch.

    Decimal comparisons prevent binary-roundoff changing strict boundary cases.
    This accepts one dataset only; callers may not pool MMECG and CR_RVS.
    """
    D = lambda x: Decimal(str(x))
    pcc, f1, failure, o1, o2, o3 = map(D, (o4_pcc_median, r_f1_subject_median,
        pooled_beat_failure_rate, o1_pcc_median, o2_pcc_median, o3_pcc_median))
    if not all(x.is_finite() for x in (pcc,f1,failure,o1,o2,o3)):
        raise ValueError("Complete finite dataset metrics are required")
    timing, morphology = o1-o3, o1-o2
    if timing > D(.10) and morphology > D(.10) and abs(timing-morphology) < D(.03):
        bottleneck = "joint"
    elif timing > D(.10) and timing > morphology:
        bottleneck = "timing"
    elif morphology > D(.10) and morphology > timing:
        bottleneck = "morphology"
    else:
        bottleneck = "inconclusive"
    return {"o4_pass": pcc >= D(.80) and f1 >= D(.95) and failure <= D(.10),
            "timing_drop": float(timing), "morphology_drop": float(morphology),
            "bottleneck": bottleneck, "automatic_training_authorized": False}
