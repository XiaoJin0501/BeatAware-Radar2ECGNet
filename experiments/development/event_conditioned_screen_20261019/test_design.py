import environment
import numpy as np
import pytest
import torch
from model import initialize_arm, parameter_counts, regional_masks, objective
from metrics import stitch_wave, paired_gate, native_record
from layout import approved_layout

torch.set_num_threads(1)


def test_capacity_and_no_shape_head():
    models = [initialize_arm(a, 43) for a in (1, 2, 3, 4)]
    counts = [parameter_counts(m) for m in models]
    assert abs(counts[1]['total']-counts[2]['total'])/counts[2]['total'] < .02
    assert abs(counts[1]['decoder']-counts[2]['decoder'])/counts[2]['decoder'] < .02
    assert counts[2] == counts[3]
    assert all(not any('shape' in n for n, _ in m.named_parameters()) for m in models)
    for prefix in ('stem.', 'backbone.', 'anchor_head.', 'global_path.'):
        for k, v in models[0].state_dict().items():
            if k.startswith(prefix):
                assert all(torch.equal(v, m.state_dict()[k]) for m in models[1:])
    for k, v in models[1].state_dict().items():
        if not k.startswith('feature_gate.'):
            assert torch.equal(v, models[2].state_dict()[k])


def test_waveform_gradient_stops_at_event_logits_but_event_loss_trains_head():
    m = initialize_arm(3, 43).eval()
    h = torch.randn(1, 128, 1600, requires_grad=True)
    logits = m.anchor_head(h)
    logits.retain_grad()
    out = m.decode(h, logits)
    wave = out['wave'].square().mean()
    wave.backward(retain_graph=True)
    assert logits.grad is None
    assert all(p.grad is None for p in m.anchor_head.parameters())
    assert h.grad is not None and torch.count_nonzero(h.grad) > 0
    m.zero_grad(set_to_none=True)
    loss, _ = objective(out, torch.randn_like(out['wave']), torch.zeros_like(logits))
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0
    assert all(p.grad is not None for p in m.anchor_head.parameters())


def test_forward_shapes_and_oracle_isolation():
    x = torch.randn(2, 50, 9, 13)
    gt = torch.zeros(2, 4, 1600)
    for arm in (1, 2, 3, 4):
        m = initialize_arm(arm, 43).eval()
        with torch.no_grad():
            o = m(x, gt if arm == 4 else None)
        assert o['wave'].shape == (2, 1, 1600)
        assert o['anchor_logits'].shape == (2, 4, 1600)
        assert torch.isfinite(o['wave']).all()
        if arm != 4:
            with pytest.raises(PermissionError):
                m(x, gt)
        else:
            with pytest.raises(PermissionError):
                m(x)


def test_masks_are_bounded_regions_and_GT_gradient_is_blocked():
    p = torch.zeros(1, 4, 1600)
    p[:, 1, 800] = 1
    p[:, 3, 1000] = 1
    g = regional_masks(p)
    assert g.min() >= 0 and g.max() <= 1.000001
    assert g[0, 0, 800] > .99 and g[0, 1, 1000] > .99
    assert g[0, 0, 700] == 0 and g[0, 1, 900] == 0
    assert torch.count_nonzero(regional_masks(torch.zeros_like(p))) == 0
    m = initialize_arm(4, 43).eval()
    h = torch.randn(1, 128, 1600, requires_grad=True)
    p.requires_grad_()
    m.decode(h, m.anchor_head(h), p)['wave'].mean().backward()
    assert p.grad is None


def test_detach_keeps_event_conditioning_effect_in_forward():
    m = initialize_arm(3, 43).eval()
    h = torch.randn(1, 128, 1600)
    with torch.no_grad():
        low = m.decode(h, torch.full((1, 4, 1600), -15.))['wave']
        high = m.decode(h, torch.full((1, 4, 1600), 15.))['wave']
    assert not torch.allclose(low, high)


@pytest.mark.parametrize('n', [1600, 2624, 3000, 35505])
def test_stitch_matches_native_sample_indices_including_tail(n):
    segments = approved_layout(n)
    starts = sorted({s['input_start'] for s in segments})
    outputs = np.asarray([np.arange(s, s+1600) for s in starts])
    pred = stitch_wave(n, starts, outputs, segments)
    np.testing.assert_array_equal(pred[288:n-288], np.arange(288, n-288))


def test_exact_requested_loss_full_window_and_BCE_mean():
    pred = torch.arange(1600, dtype=torch.float32).reshape(1, 1, -1)/1600
    target = pred.clone()
    logits = torch.zeros(1, 4, 1600)
    output = {'wave': pred, 'anchor_logits': logits}
    total, pieces = objective(output, target, torch.zeros_like(logits))
    assert abs(pieces['l1'].item()) < 1e-7
    assert abs(pieces['derivative'].item()) < 1e-7
    assert abs(total.item()-.3*np.log(2)) < 1e-6
    changed = target.clone()
    changed[..., -1] += 1
    _, q = objective(output, changed, torch.zeros_like(logits))
    assert q['l1'] > 0 and q['derivative'] > 0  # Loss includes context edges.


def test_perfect_native_record_roundtrip_and_direct_event_metrics():
    from data import event_targets, local_reference
    n = 3648
    r = np.arange(100, n-90, 200, dtype=float)
    raw = {f'event_{k}': r+offset for k, offset in zip('qrst', (-6, 0, 6, 45))}
    time = np.arange(n)
    wave = np.zeros(n)
    for center in r:
        wave += np.exp(-.5*((time-center)/2)**2)
        wave -= .15*np.exp(-.5*((time-center+6)/2)**2)
        wave -= .2*np.exp(-.5*((time-center-6)/2)**2)
        wave += .3*np.exp(-.5*((time-center-45)/10)**2)
    raw['ecg'] = wave
    segments = approved_layout(n)
    starts = sorted({s['input_start'] for s in segments})
    row = dict(samples=n, segments=segments, starts=starts, subject_id=2)
    output = np.stack([wave[s:s+1600] for s in starts])
    p = np.stack([event_targets(local_reference(raw, s)) for s in starts])
    measured = native_record(row, output, p, raw, {'median': 0., 'iqr': 1.})
    assert all(v['pcc'] > .999999 for v in measured['beat'])
    assert all(v['rmse'] < 1e-7 for v in measured['window'])
    assert all(v['pcc'] > .999999 for v in measured['shape'])
    event = measured['event']
    assert event['r_tp'] == event['r_prediction'] == event['r_reference']
    assert all(v == 0 for v in event['bucket']['r']['errors_ms'])
    assert measured['order_violations'] == 0


def test_validation_preserves_modes_and_all_rng_states(tmp_path, monkeypatch):
    import random
    import run
    from src.utils.atomic_state import capture_rng_state
    # Real model/loss forward, instrumented metric sink; no outside artifacts.
    feature = tmp_path/'features.npz'
    np.savez(feature, features=np.zeros((1, 50, 9, 13), np.float32))
    raw = dict(ecg=np.sin(np.arange(1600)/20),
               **{f'event_{k}': np.array([600.]) for k in 'qrst'})
    monkeypatch.setattr(run, 'reference', lambda rid: raw)
    monkeypatch.setattr(run, 'native_record', lambda *args: {})
    monkeypatch.setattr(run, 'aggregate', lambda records: {})
    monkeypatch.setattr(run, 'record_metric', lambda *args: {})
    monkeypatch.setattr(run, 'aggregate_record_lags', lambda records: {'M1': {}})
    monkeypatch.setattr(run, 'checkpoint_score', lambda m1: {'score': 0.})
    row = dict(features=str(feature), starts=[0], record_id='synthetic')
    m = initialize_arm(3, 43).train()
    before = capture_rng_state()
    run.validate(m, [row], {'median': 0., 'iqr': 1.}, 'cpu')
    after = capture_rng_state()
    assert m.training
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])
    assert all(torch.equal(a, b) for a, b in zip(before['torch_cuda'], after['torch_cuda']))


def test_seeds_are_not_pseudoreplicated_and_three_subject_gate_cannot_pass():
    rows = [dict(subject_id=s, seed=seed, arm=a,
        window_pcc=.6+.1*(a == 3)+s*.001*(a == 3),
        shape_pcc=.7+.1*(a == 3)+s*.001*(a == 3),
        **{f'{k}_mae_ms': 20. for k in 'qrst'})
        for s in (2, 6, 10) for seed in (42, 123, 2024) for a in (2, 3)]
    r = paired_gate(rows)
    assert r['independent_subjects'] == 3
    assert r['window_pcc']['p_one_sided'] == .125
    assert not r['supports_event_conditioning']


