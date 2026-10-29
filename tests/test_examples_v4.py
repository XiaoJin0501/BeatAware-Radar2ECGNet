import numpy as np
import pytest
from scripts.figures import plot_examples_v4 as ex

def test_centre_labels_keep_exact_predicted_times_and_order():
    a={'pred_r':np.array([100.,500.,900.]),'pred_q':np.array([90.,490.,890.]),
       'pred_s':np.array([105.,505.,905.]),'pred_t':np.array([160.,560.,960.])}
    before={k:v.copy() for k,v in a.items()}
    assert ex.centre_events(a)==dict(q=490.,r=500.,s=505.,t=560.)
    assert all(np.array_equal(before[k],v) for k,v in a.items())

def test_no_complete_group_is_not_silently_replaced_by_waveform_peaks():
    a={'pred_r':np.array([500.]),'pred_q':np.array([490.]),
       'pred_s':np.array([505.]),'pred_t':np.array([np.nan])}
    with pytest.raises(ValueError):ex.centre_events(a)
