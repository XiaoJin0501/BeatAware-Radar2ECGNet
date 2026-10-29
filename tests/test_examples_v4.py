import numpy as np
import pytest
from scripts.figures import plot_examples_v4 as ex

def test_centre_labels_keep_exact_stored_reference_times_and_same_beat():
    a={'ref_r':np.array([100.,500.,900.]),'ref_q':np.array([90.,490.,890.]),
       'ref_s':np.array([105.,505.,905.]),'ref_t':np.array([160.,560.,960.])}
    before={k:v.copy() for k,v in a.items()}
    assert ex.centre_events(a)==dict(q=490.,r=500.,s=505.,t=560.)
    assert all(np.array_equal(before[k],v) for k,v in a.items())

def test_missing_reference_endpoints_are_not_replaced_by_waveform_search():
    a={'ref_r':np.array([500.]),'ref_q':np.array([490.]),
       'ref_s':np.array([505.]),'ref_t':np.array([np.nan])}
    with pytest.raises(ValueError):ex.centre_events(a)
