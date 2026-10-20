"""Append-only unified metrics extension. Existing v2 metrics are never edited."""
import bridge
import numpy as np


def matched_regions(wave, reference, m1):
    """Exactly the scored M1 pairs; no new matching, gain fitting, or lag search."""
    native = np.arange(len(reference))
    result = []
    for pair in m1['rows']:
        r, pr, rr = pair['gt_r'], pair['predicted_r'], pair['rr_reference_samples']
        # Keep T support inside M1's +0.75 RR support for short cycles.
        scale = min(1., .75*rr/90.)
        offsets = dict(qrs=np.arange(-12., 13.), t=np.linspace(30*scale, 90*scale, 61))
        scores = {}
        for key, offset in offsets.items():
            p = np.interp(pr+offset, native, wave)
            y = np.interp(r+offset, native, reference)
            scores[key+'_pcc'] = bridge.continuous_pair(p, y)['pcc']
        full = np.linspace(-.25*rr, .75*rr, 200)
        p = np.interp(pr+full, native, wave)
        y = np.interp(r+full, native, reference)
        denominator = float(np.std(y))
        scores['amplitude_ratio'] = float(np.std(p)/denominator) if denominator > 1e-12 else None
        result.append(dict(subject_id=pair['subject_id'], gt_r=r, predicted_r=pr,
                           t_scale=scale, **scores))
    return result


def native_record(row, predictions, probs, raw, stats):
    result = bridge.old_metrics.native_record(row, predictions, probs, raw, stats)
    wave = bridge.old_metrics.stitch_wave(row['samples'], row['starts'], predictions, row['segments'])
    target = ((raw['ecg']-stats['median'])/stats['iqr']).astype(np.float32)
    result['regional'] = matched_regions(wave, target, result['supplement']['m1'])
    return result


def aggregate_regions(records):
    subjects = sorted({r['subject_id'] for r in records})
    rows = [r for v in records for r in v['regional']]
    summaries = []
    for subject in subjects:
        group = [r for r in rows if r['subject_id'] == subject]
        summaries.append(dict(subject_id=subject, scored=len(group),
            amplitude_ratio_unavailable=sum(r['amplitude_ratio'] is None for r in group),
            qrs_pcc=bridge.fisher_mean([r['qrs_pcc'] for r in group]),
            t_pcc=bridge.fisher_mean([r['t_pcc'] for r in group]),
            amplitude_ratio=float(np.mean([r['amplitude_ratio'] for r in group if r['amplitude_ratio'] is not None]))
                if any(r['amplitude_ratio'] is not None for r in group) else None))
    return dict(subjects=summaries, amplitude_ratio_unavailable=sum(s['amplitude_ratio_unavailable'] for s in summaries),
                **{key:bridge.distribution([r[key] for r in summaries])
                 for key in ('qrs_pcc', 't_pcc', 'amplitude_ratio')})


def aggregate(records):
    result = bridge.old_metrics.aggregate(records)
    result['regional'] = aggregate_regions(records)
    result.update(bridge.v2_metrics.aggregate_record_lags([r['record_optimal'] for r in records]))
    result['selection'] = bridge.v2_metrics.checkpoint_score(result['M1'])
    return result
