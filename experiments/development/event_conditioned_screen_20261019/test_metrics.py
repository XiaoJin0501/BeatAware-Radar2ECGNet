import numpy as np
from metrics import (supplementary_record, aggregate_supplement,
                     waveform_r, screening_decision)


def synthetic_ecg():
    n = 12000
    rs = np.arange(500, 11500, 200)
    t = np.arange(n)
    y = np.zeros(n)
    for r in rs:
        for offset, amp, sigma in [(-35, .12, 5), (-6, -.2, 1.6), (0, 1., 1.8),
                                   (7, -.25, 1.8), (52, .3, 8)]:
            y += amp*np.exp(-.5*((t-r-offset)/sigma)**2)
    return y, rs


def values(p, y, r):
    record = supplementary_record(p, y, r, 288, len(y)-288, 1)
    result = aggregate_supplement([record])
    return record, result


def test_identity_M1_M2_M3_are_one_with_real_neurokit_detector():
    y, r = synthetic_ecg()
    record, m = values(y, y, r)
    assert all(abs(v['pcc']-1.) < 1e-10 for v in record['m1']['rows'])
    assert all(abs(v['pcc']-1.) < 1e-10 for v in record['m2'])
    assert all(abs(v['pcc']-1.) < 1e-10 for v in record['m3'])
    assert m['M1']['coverage']['mean'] == 1.
    assert set(v['lag_samples'] for v in record['m3']) == {0}


def test_20ms_delay_corrected_only_by_M1_and_M3():
    y, r = synthetic_ecg()
    pred = np.concatenate((np.zeros(4), y[:-4]))
    record, m = values(pred, y, r)
    assert m['M1']['pcc']['mean'] > .995
    assert m['M3']['pcc']['mean'] > .995
    assert m['M2']['pcc']['mean'] < .8
    assert set(v['lag_samples'] for v in record['m3']) == {4}


def test_deleting_ten_percent_preserves_conditional_M1_but_lowers_coverage():
    y, r = synthetic_ecg()
    pred = y.copy()
    # Six of55 beats=10.91%; zero whole local cycle, leave other beats unchanged.
    for i in (5, 14, 23, 32, 41, 50):
        pred[r[i]-50:r[i]+150] = 0
    _, baseline = values(y, y, r)
    record, m = values(pred, y, r)
    assert abs(m['M1']['coverage']['mean']-.9) < .025
    assert abs(m['M1']['pcc']['mean']-baseline['M1']['pcc']['mean']) < .002
    assert m['M1']['matched'] == 49 and m['M1']['reference_count'] == 55


def test_M1_reports_boundary_pairs_without_dropping_coverage_denominator():
    from metrics import r_aligned_morphology
    y, _ = synthetic_ecg()
    r = np.array([300., 500., 11500.])
    m = r_aligned_morphology(y, y, r, 288, 11712, detected_r=r)
    assert m['reference_count'] == m['matched'] == 3
    assert m['boundary_excluded'] > 0 and m['scored'] < 3
    assert m['coverage'] == 1.


def test_screening_requires_all_conditions_without_significance_claim():
    rows = [dict(fold=f, seed=s, arm=a, M1=.6+.02*(a == 3), M2=.3+.02*(a == 3),
        **{f'{k}_mae_ms': 20.+4*(a == 3) for k in 'qrst'})
        for f in (1, 2, 3) for s in (42, 43, 44) for a in (1, 2, 3, 4)]
    result = screening_decision(rows)
    assert result['screening_pass'] and result['positive_m2_pairs'] == 9
    assert not result['statistical_evidence'] and not result['confirmation_11fold_authorized']
    rows[2]['q_mae_ms'] = 100.
    assert not screening_decision(rows)['screening_pass']


def test_real_pool_preload_preserves_input_and_labels_bitwise():
    from data import TrainWindows
    dataset = TrainWindows(1, limit=2)
    old = [dataset[i] for i in range(2)]
    dataset.preload()
    for i in range(2):
        new = dataset[i]
        for name in ('wsst', 'target', 'events'):
            np.testing.assert_array_equal(old[i][name].numpy(), new[name].numpy())
