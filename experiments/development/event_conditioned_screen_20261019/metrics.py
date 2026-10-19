"""One metric interface for all four arms; delegates to audited A primitives."""
import environment as env
import numpy as np
from unified_metrics import (continuous_pair, beat_metrics, window_metrics,
    summarize_subjects, waveform_events, match_events, fisher_mean, distribution)
from structured import stitch_anchors
from scripts import evaluate_formal_anchor_intervals as formal

# Same factored maximum-cardinality/minimum-error DP as A; no metric truncation.
formal.match_beats = match_events


def stitch_wave(length, starts, outputs, segments):
    lookup = {int(s): i for i, s in enumerate(starts)}
    wave = np.zeros(length, np.float32)
    coverage = np.zeros(length, np.int8)
    for seg in segments:
        a, b, offset = seg['start'], seg['stop'], seg['input_start']
        wave[a:b] = outputs[lookup[offset], a-offset:b-offset]
        coverage[a:b] += 1
    if not np.all(coverage[288:length-288] == 1):
        raise AssertionError('Coverage gap/duplicate in native reference coordinates')
    if not np.isfinite(wave).all():
        raise ValueError('Nonfinite wave cannot be excluded')
    return wave


def waveform_r(signal, valid_start, valid_stop):
    """The ONLY prediction-R detector for M1 and waveform R-F1."""
    return np.asarray(waveform_events(np.asarray(signal)[valid_start:valid_stop])['r'], float)+valid_start


def r_aligned_morphology(prediction, reference, gt_r, valid_start, valid_stop, detected_r=None):
    """M1: reference-RR-sized windows centered separately on matched R peaks.

    Coverage denominator: ALL reference R in retained support. Following RR is
    used; the final record R uses preceding RR only if no following R exists.
    Boundary pairs lacking FULL pred/ref support are counted matched but cannot
    contribute morphology; their count is explicit, never silently called missed.
    Linear interpolation at 200 endpoint-inclusive fractional coordinates, with
    no extrapolation, polarity flip, gain fit, or post-extraction lag search.
    """
    p, y = np.asarray(prediction), np.asarray(reference)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite native shared record coordinates required')
    gt_r = np.asarray(gt_r, float)
    if np.any(np.diff(gt_r) <= 0):
        raise ValueError('Strictly ordered GT R required')
    owned = np.flatnonzero((gt_r >= valid_start) & (gt_r < valid_stop))
    pred_r = waveform_r(p, valid_start, valid_stop) if detected_r is None else np.asarray(detected_r, float)
    pred_r = pred_r[(pred_r >= valid_start) & (pred_r < valid_stop)]
    pairs = match_events(gt_r[owned], pred_r, 30)
    rows, boundary, rr_missing = [], 0, 0
    native = np.arange(len(y))
    for compact_i, j in pairs.items():
        i = owned[compact_i]
        rr = (gt_r[i+1]-gt_r[i] if i+1 < len(gt_r) else
              gt_r[i]-gt_r[i-1] if i else None)
        if rr is None or rr <= 0:
            rr_missing += 1
            continue
        offset = np.linspace(-.25*rr, .75*rr, 200)
        yg, pg = gt_r[i]+offset, pred_r[j]+offset
        if min(yg[0], pg[0]) < valid_start or max(yg[-1], pg[-1]) >= valid_stop:
            boundary += 1
            continue
        score = continuous_pair(np.interp(pg, native, p), np.interp(yg, native, y))
        rows.append(dict(gt_r=float(gt_r[i]), predicted_r=float(pred_r[j]), rr_reference_samples=float(rr), **score))
    tp, nr, npred = len(pairs), len(owned), len(pred_r)
    return dict(rows=rows, reference_count=nr, prediction_count=npred, matched=tp,
                scored=len(rows), boundary_excluded=boundary, rr_unavailable=rr_missing,
                coverage=tp/nr if nr else 0., scored_coverage=len(rows)/nr if nr else 0.,
                waveform_r_f1=2*tp/(nr+npred) if nr+npred else 0.,
                waveform_r_se=tp/nr if nr else 0., waveform_r_ppv=tp/npred if npred else 0.)


def window_cross_correlation(prediction, reference, starts):
    """M3: maximum PCC over integer lags -10..10 on FIXED1004sample support.

    Positive lag means predicted waveform is delayed relative to reference.
    Ties choose smallest |lag|, then negative lag. No shift is applied to M2.
    """
    rows = []
    for start in starts:
        p = np.asarray(prediction)[start:start+1024]
        y = np.asarray(reference)[start:start+1024]
        if len(p) != 1024 or len(y) != 1024:
            raise ValueError('M3 requires complete1024-sample blocks')
        candidates = [(continuous_pair(p[10+lag:1014+lag], y[10:1014]), lag) for lag in range(-10, 11)]
        best, lag = max(candidates, key=lambda v: (v[0]['pcc'], -abs(v[1]), -v[1]))
        rows.append(dict(start=int(start), stop=int(start+1024), lag_samples=lag, lag_ms=lag*5, **best))
    return rows


def supplementary_record(wave, reference, gt_r, valid_start, valid_stop, subject_id, detected=None):
    """Shared supplement for A assemblies and ALL new continuous arms."""
    detected = waveform_r(wave, valid_start, valid_stop) if detected is None else detected
    m1 = r_aligned_morphology(wave, reference, gt_r, valid_start, valid_stop, detected)
    m1['rows'] = [dict(subject_id=subject_id, **r) for r in m1['rows']]
    starts = list(range(valid_start, valid_stop-1024+1, 1024))
    m2 = [dict(subject_id=subject_id, **r) for r in window_metrics(wave, reference, starts)]
    m3 = [dict(subject_id=subject_id, **r) for r in window_cross_correlation(wave, reference, starts)]
    return dict(subject_id=subject_id, m1=m1, m2=m2, m3=m3)


def aggregate_supplement(records):
    subjects = sorted({r['subject_id'] for r in records})
    m1_rows = [r for v in records for r in v['m1']['rows']]
    # Preserve subjects with NO matches as explicitly unavailable, not absent.
    subjects_m1 = []
    for subject in subjects:
        group = [v['m1'] for v in records if v['subject_id'] == subject]
        scored = [r for g in group for r in g['rows']]
        counts = {k: sum(g[k] for g in group) for k in
                  ('reference_count', 'prediction_count', 'matched', 'scored', 'boundary_excluded', 'rr_unavailable')}
        nr, npred, tp = counts['reference_count'], counts['prediction_count'], counts['matched']
        subjects_m1.append(dict(subject_id=subject, **counts,
            pcc_fisher_mean=fisher_mean([r['pcc'] for r in scored]),
            coverage=tp/nr if nr else 0., scored_coverage=counts['scored']/nr if nr else 0.,
            waveform_r_f1=2*tp/(nr+npred) if nr+npred else 0.,
            waveform_r_se=tp/nr if nr else 0., waveform_r_ppv=tp/npred if npred else 0.))
    m3_rows = [r for v in records for r in v['m3']]
    return dict(M1=dict(subjects=subjects_m1,
        pcc=distribution([s['pcc_fisher_mean'] for s in subjects_m1]),
        coverage=distribution([s['coverage'] for s in subjects_m1]),
        scored_coverage=distribution([s['scored_coverage'] for s in subjects_m1]),
        unavailable_subjects=[s['subject_id'] for s in subjects_m1 if s['pcc_fisher_mean'] is None],
        reference_count=sum(s['reference_count'] for s in subjects_m1),
        matched=sum(s['matched'] for s in subjects_m1), scored=len(m1_rows)),
        M2=summarize_subjects([r for v in records for r in v['m2']]),
        M3=dict(**summarize_subjects(m3_rows),
            lag_histogram_ms={str(lag*5): sum(r['lag_samples'] == lag for r in m3_rows) for lag in range(-10, 11)},
            lag_distribution_ms=distribution([r['lag_ms'] for r in m3_rows])))


def native_record(row, predictions, probs, raw, stats):
    n = row['samples']
    target = ((raw['ecg']-stats['median'])/stats['iqr']).astype(np.float32)
    wave = stitch_wave(n, row['starts'], predictions, row['segments'])
    subject = row['subject_id']
    beats = [dict(subject_id=subject, **r) for r in
             beat_metrics(wave, target, raw['event_r'], 288, n-288)]
    windows = [dict(subject_id=subject, **r) for r in
               window_metrics(wave, target, range(288, n-288-1024+1, 1024))]
    # Shape-200 here counts ALL GT R-R beats, not one representative per input.
    shapes = []
    for beat in beats:
        a, b = beat['start'], beat['stop']
        grid = np.linspace(0, b-a-1, 200)
        p = np.interp(grid, np.arange(b-a), wave[a:b])
        y = np.interp(grid, np.arange(b-a), target[a:b])
        shapes.append(dict(subject_id=subject, **continuous_pair(p, y)))
    anchor_beats, _ = stitch_anchors(n, row['starts'], probs, env.core.structured_decode)
    anchor_beats = [b for b in anchor_beats if 288 <= b['r'] < n-288]
    pred = {k: np.asarray([b[k] for b in anchor_beats]) for k in 'qrst'}
    owned = (raw['event_r'] >= 288) & (raw['event_r'] < n-288)
    ref = {k: raw[f'event_{k}'][owned] for k in 'qrst'}
    bucket = formal.empty_bucket()
    formal.accumulate_window(bucket, ref, pred)
    rmatch = match_events(ref['r'], pred['r'], 30)
    event = dict(bucket=bucket, r_reference=len(ref['r']), r_prediction=len(pred['r']), r_tp=len(rmatch))
    complete = violations = candidates = 0
    for seg in row['segments']:
        i = row['starts'].index(seg['input_start'])
        points = env.core.independent_decode(probs[i])
        for r in points['r']:
            if not seg['start'] <= r+seg['input_start'] < seg['stop']:
                continue
            candidates += 1
            if all(len(points[k]) for k in 'qst'):
                q, s, t = [points[k][np.argmin(abs(points[k]-r))] for k in 'qst']
                complete += 1
                violations += not(q < r < s < t)
    # GT-free diagnostic pairing of waveform R vs predicted head R, same tolerance.
    detected = waveform_r(wave, 288, n-288)
    supplement = supplementary_record(wave, target, raw['event_r'], 288, n-288, subject, detected)
    pairs = match_events(pred['r'], detected, 30)
    offsets = [(detected[j]-pred['r'][i])*5 for i, j in pairs.items()]
    return dict(subject_id=subject, supplement=supplement, beat=beats, window=windows, shape=shapes, event=event,
                order_complete=complete, order_violations=int(violations),
                order_r_candidates=candidates, consistency_offsets_ms=offsets,
                consistency_head_r=len(pred['r']), consistency_wave_r=len(detected))


def aggregate(records):
    output = {k: summarize_subjects([r for v in records for r in v[k]])
              for k in ('beat', 'window', 'shape')}
    events = []
    for subject in sorted({r['subject_id'] for v in records for r in v['window']}):
        group = [v for v in records if v['window'][0]['subject_id'] == subject]
        combined = formal.empty_bucket()
        for v in group:
            for k, item in v['event']['bucket'].items():
                for key in ('eligible', 'evaluable'):
                    combined[k][key] += item[key]
                for key in ('errors_ms', 'mre_pct'):
                    combined[k][key].extend(item[key])
        tp = sum(v['event']['r_tp'] for v in group)
        nr = sum(v['event']['r_reference'] for v in group)
        npred = sum(v['event']['r_prediction'] for v in group)
        complete = sum(v['order_complete'] for v in group)
        offsets = [x for v in group for x in v['consistency_offsets_ms']]
        events.append(dict(subject_id=subject, fiducials=formal.finalise(combined),
            r_se=tp/nr if nr else 0., r_ppv=tp/npred if npred else 0.,
            r_f1=2*tp/(nr+npred) if nr+npred else 0.,
            r_tp=tp, r_reference=nr, r_prediction=npred,
            order_violation_rate=sum(v['order_violations'] for v in group)/complete if complete else None,
            order_complete=complete, order_r_candidates=sum(v['order_r_candidates'] for v in group),
            consistency_mae_ms=float(np.mean(np.abs(offsets))) if offsets else None,
            consistency_signed_median_ms=float(np.median(offsets)) if offsets else None,
            consistency_pairs=len(offsets),
            consistency_head_r=sum(v['consistency_head_r'] for v in group),
            consistency_wave_r=sum(v['consistency_wave_r'] for v in group)))
    output['events_by_subject'] = events
    output.update(aggregate_supplement([r['supplement'] for r in records]))
    output['selection_window_pcc'] = output['window']['pcc']['mean']
    output['reference_space'] = 'A raw-record train-only median/IQR; no per-window normalization'
    output['shape_definition'] = 'All complete GT R-R cycles; native output resampled to 200, NOT old representative-head metric'
    return output


def screening_decision(rows):
    """Author's 36-run descriptive screening rule, NOT statistical evidence."""
    index = {(r['fold'], r['seed'], r['arm']): r for r in rows}
    required = {(f, s, a) for f in (1, 2, 3) for s in (42, 43, 44) for a in (1, 2, 3, 4)}
    if len(index) != 36 or set(index) != required:
        raise ValueError('Exactly36 unique preregistered fold/seed/arm rows required')
    pairs = []
    for fold in (1, 2, 3):
        for seed in (42, 43, 44):
            a, b = index[fold, seed, 3], index[fold, seed, 2]
            pairs.append(dict(fold=fold, seed=seed,
                m2_delta=a['M2']-b['M2'],
                m1_delta=a['M1']-b['M1'] if a['M1'] is not None and b['M1'] is not None else None,
                **{f'{k}_mae_delta_ms': a[f'{k}_mae_ms']-b[f'{k}_mae_ms']
                   if a[f'{k}_mae_ms'] is not None and b[f'{k}_mae_ms'] is not None else None for k in 'qrst'}))
    subject_means = []
    for fold in (1, 2, 3):
        g = [r for r in pairs if r['fold'] == fold]
        subject_means.append(dict(fold=fold, **{key: float(np.mean([r[key] for r in g]))
            if all(r[key] is not None for r in g) else None
            for key in ('m2_delta', 'm1_delta', *(f'{k}_mae_delta_ms' for k in 'qrst'))}))
    means = {key: float(np.mean([r[key] for r in subject_means]))
        if all(r[key] is not None for r in subject_means) else None
        for key in ('m2_delta', 'm1_delta', *(f'{k}_mae_delta_ms' for k in 'qrst'))}
    positive = sum(r['m2_delta'] > 0 for r in pairs)
    guards = {k: means[f'{k}_mae_delta_ms'] is not None and means[f'{k}_mae_delta_ms'] <= 5. for k in 'qrst'}
    return dict(screening_only=True, statistical_evidence=False,
        pairs=pairs, subject_seed_means=subject_means, overall_subject_mean_deltas=means,
        positive_m2_pairs=positive,
        conditions=dict(at_least8of9_m2_positive=positive >= 8,
            mean_m2_gain_at_least_001=means['m2_delta'] >= .01,
            mean_m1_gain_at_least_001=means['m1_delta'] is not None and means['m1_delta'] >= .01,
            anchor_mae_guards=guards),
        screening_pass=positive >= 8 and means['m2_delta'] >= .01 and
            means['m1_delta'] is not None and means['m1_delta'] >= .01 and all(guards.values()),
        confirmation_11fold_authorized=False)


def paired_gate(rows):
    """Average seeds WITHIN subject, never pretend repeated subjects are independent."""
    from scipy.stats import wilcoxon
    groups = {}
    for r in rows:
        if r['arm'] in (2, 3):
            groups.setdefault((r['subject_id'], r['arm']), []).append(r)
    subjects = sorted({key[0] for key in groups})
    if any(len(groups.get((s, a), [])) != 3 for s in subjects for a in (2, 3)):
        raise ValueError('Three seeds required for every subject and arm')
    results = {}
    for metric in ('window_pcc', 'shape_pcc'):
        delta = np.asarray([np.mean([r[metric] for r in groups[s, 3]])-
                            np.mean([r[metric] for r in groups[s, 2]]) for s in subjects])
        p = float(wilcoxon(delta, alternative='greater', method='exact').pvalue) if np.any(delta) else 1.
        results[metric] = dict(p_one_sided=p, mean_delta=float(delta.mean()), n_subjects=len(subjects))
    noninferior = {}
    for k in 'qrst':
        if any(r[f'{k}_mae_ms'] is None for r in rows if r['arm'] in (2, 3)):
            noninferior[k] = False
        else:
            delta = [np.mean([r[f'{k}_mae_ms'] for r in groups[s, 3]])-
                     np.mean([r[f'{k}_mae_ms'] for r in groups[s, 2]]) for s in subjects]
            # Conservative explicit interpretation: +5 ms guard holds for EACH subject.
            noninferior[k] = bool(np.all(np.asarray(delta) <= 5.))
    return dict(**results, anchor_noninferior_by_endpoint=noninferior,
                supports_event_conditioning=all(r['p_one_sided'] < .05 and r['mean_delta'] > 0
                    for r in results.values()) and all(noninferior.values()),
                independent_subjects=len(subjects), seeds_are_not_independent_samples=True)


