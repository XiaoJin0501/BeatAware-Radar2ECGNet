import numpy as np

from scripts.run_dual_head_interaction import score_events, structured_decode, shape_target


def test_structured_decoder_enforces_qrst_order_without_reference():
    probability = np.zeros((4, 1600), np.float32)
    probability[1, [400, 800]] = 1
    probability[0, [390, 790]] = 1
    probability[2, [410, 810]] = 1
    probability[3, [470, 870]] = 1
    out = structured_decode(probability)
    assert len(out["r"]) == 2
    assert all(q < r < s < t for q, r, s, t in zip(
        out["q"], out["r"], out["s"], out["t"]))


def test_shape_target_is_200_point_minmax_cycle():
    ecg = np.sin(np.linspace(0, 20, 1600)).astype(np.float32)
    out = shape_target(ecg, np.array([600, 850, 1100]))
    assert out.shape == (200,)
    assert np.isclose(out.min(), 0)
    assert np.isclose(out.max(), 1)


def test_empty_early_predictions_do_not_crash_metric_aggregation():
    probability = np.zeros((1, 4, 1600), np.float32)
    reference = [{p: np.array([400.0]) for p in ("q", "r", "s", "t")}]
    result = score_events(probability, reference, structured=False)
    assert result["macro_medae_ms"] is None
    assert result["macro_f1_100ms"] == 0.0
