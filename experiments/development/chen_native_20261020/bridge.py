"""Read-only imports of the ACTUAL frozen v2 evaluator, never its local copy."""
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESOURCE=Path('/home/qhh2237/Projects/BeatAware-Radar2ECGNet')
SOURCE=Path('/home/qhh2237/Projects/BeatAware-Radar2ECGNet-continuous-screen-v2/experiments/development/event_conditioned_screen_20261019')
OUT=RESOURCE/'experiments/development/chen_native_20261020'
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(OUT/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(OUT/'.cache/numba')
sys.path.insert(0,str(SOURCE))
import environment as frozen_environment
# Frozen environment sets its own cache defaults; override in THIS CPU process
# before invoking any evaluator, without changing files or the training service.
os.environ['MPLCONFIGDIR']=str(OUT/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR']=str(OUT/'.cache/numba')
import numba
numba.config.CACHE_DIR=str(OUT/'.cache/numba')
import metrics as frozen_metrics
import v2_metrics as frozen_v2
import freeze
assert Path(frozen_metrics.__file__).resolve()==SOURCE/'metrics.py'
assert Path(frozen_v2.__file__).resolve()==SOURCE/'v2_metrics.py'
freeze.verify()
# environment/bootstrap prepend legacy roots; restore this package for local imports.
sys.path.insert(0,str(HERE))
from rawio import split_rows
from layout import approved_layout
from unified_metrics import event_metrics,summarize_events
from src.utils.atomic_state import atomic_json,atomic_text,atomic_npz

def verify_frozen():
    freeze.verify()
    return freeze.sha(frozen_environment.OUT/'METRIC_FREEZE.json')
