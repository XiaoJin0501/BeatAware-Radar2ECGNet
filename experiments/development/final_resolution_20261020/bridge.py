"""Read-only reuse of the frozen, completed v2 implementation."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESOURCE = Path('/home/qhh2237/Projects/BeatAware-Radar2ECGNet')
SOURCE = Path('/home/qhh2237/Projects/BeatAware-Radar2ECGNet-continuous-screen-v2/experiments/development/event_conditioned_screen_20261019')
OUT = RESOURCE/'experiments/development/final_resolution_20261020'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
import environment as old_environment
import model as old_model
import data as old_data
import metrics as old_metrics
import run as old_run
import v2_metrics
import freeze as old_freeze
old_freeze.verify()
for module in (old_model, old_data, old_metrics, old_run, v2_metrics):
    assert Path(module.__file__).resolve().parent == SOURCE
os.environ['MPLCONFIGDIR'] = str(OUT/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(OUT/'.cache/numba')
import numba
numba.config.CACHE_DIR = str(OUT/'.cache/numba')
sys.path.insert(0, str(HERE))
from rawio import record_data, radar_window, split_rows
from unified_metrics import continuous_pair, fisher_mean, distribution
from src.utils.atomic_state import atomic_json, atomic_text, atomic_npz, atomic_torch_save, capture_rng_state, restore_rng_state
