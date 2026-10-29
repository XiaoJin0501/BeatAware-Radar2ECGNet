import importlib.util
from pathlib import Path
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/figures/plot_localization_cdf_all.py'
s=importlib.util.spec_from_file_location('localization_cdf_all',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_display_crop_preserves_full_cdf_denominator_and_zero_mass():
    x,f=m.visible_step_nodes([0,10,200],[.1,.5,1.])
    assert np.array_equal(x,[0,10,150])
    assert np.array_equal(f,[.1,.5,.5])

def test_exact_boundary_probability_is_not_interpolated():
    x,f=m.visible_step_nodes([0,100,150,200],[0,.4,.8,1])
    assert x[-1]==150 and f[-1]==.8

def test_invalid_cdf_is_not_silently_sorted_or_smoothed():
    with pytest.raises(ValueError):m.visible_step_nodes([0,10,20],[0,.8,.6])
