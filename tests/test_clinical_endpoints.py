import numpy as np
from src.utils.clinical_endpoints import FIELDS, score_endpoints


def beats(r=(100, 300, 500), shift=0):
    r = np.array(r, dtype=float) + shift
    offsets = dict(q=-5, s=5, t=40, p_on=-30, qrs_on=-10, qrs_off=10, t_off=60)
    return dict(r=r, **{k: r+offsets[k] for k in FIELDS})


def scored(a, b):
    return {r["metric"]: r for r in score_endpoints(a, b)}


def test_common_shift_cancels_intervals_not_events():
    d = scored(beats(), beats(shift=10))
    assert d["r"]["mae_ms"] == 50
    assert all(d[k]["mae_ms"] == 0 for k in ("qq", "rr", "ss", "tt", "qrs", "qt", "qtc", "pr"))


def test_same_fiducial_intervals_require_adjacent_matched_beats():
    ref = beats((100, 300, 500))
    pred = beats((110, 520))
    d = scored(ref, pred)
    assert all(d[k]["eligible"] == 2 for k in ("qq", "rr", "ss", "tt"))
    assert all(d[k]["evaluable"] == 0 for k in ("qq", "rr", "ss", "tt"))


def test_standard_boundaries_not_q_s_peaks():
    p = beats(); p["q"] += 3; p["s"] -= 3
    d = scored(beats(), p)
    assert d["q"]["mae_ms"] == 15
    assert d["qrs"]["mae_ms"] == 0


def test_missed_middle_beat_cannot_bridge_rr():
    d = scored(beats(), beats((100, 500)))
    assert d["rr"]["eligible"] == 2
    assert d["rr"]["evaluable"] == 0
    assert d["rr"]["mae_ms"] is None


def test_missing_p_does_not_remove_qt():
    p = beats(); p["p_on"][:] = np.nan
    d = scored(beats(), p)
    assert d["pr"]["unavailable_rate"] == 1
    assert d["qt"]["coverage"] == 1


def test_relative_error_and_mae_are_means():
    p = beats(); p["qrs_off"] += [0, 2, 10]
    d = scored(beats(), p)["qrs"]
    assert d["mae_ms"] == 20
    assert d["medae_ms"] == 10
    assert d["mre_pct"] == 20


def test_extra_beat_counts_against_detection():
    d = scored(beats(), beats((100, 200, 300, 500)))["r"]
    assert d["detection_tp"] == 3
    assert d["predicted_count"] == 4
    assert d["f1"] == 6/7


def test_empty_prediction_is_failure_not_zero_error():
    d = scored(beats(), beats(()))
    assert d['r']['f1'] == 0
    assert d['qt']['coverage'] == 0
    assert d['qt']['mae_ms'] is None


def test_unknown_reference_is_not_perfect_detection():
    r=beats(); r['q'][:]=np.nan
    d=scored(r,beats())['q']
    assert d['eligible']==0 and d['reference_unassessable']==3
    assert d['mae_ms'] is None


def test_rr_extra_prediction_blocks_adjacency():
    d=scored(beats(),beats((100,200,300,500)))['rr']
    assert d['eligible']==2 and d['evaluable']==1


def test_pr_uses_onset_not_p_peak():
    p=beats(); p['p_on']-=2
    d=scored(beats(),p)['pr']
    assert d['mae_ms']==10
    assert d['mre_pct']==10
