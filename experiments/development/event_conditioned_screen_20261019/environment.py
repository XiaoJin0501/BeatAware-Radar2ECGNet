"""Screen source is committed separately; historical references are read-only."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESOURCE_ROOT = Path(os.environ.get('REM_REFERENCE_ROOT', '/home/qhh2237/Projects/BeatAware-Radar2ECGNet'))
A = RESOURCE_ROOT/'experiments/development/structured_decoder_v0_20261017_ab'
FROZEN = Path('/tmp/rem_dual_head_formal')
OUT = Path(os.environ.get('REM_SCREEN_OUT', RESOURCE_ROOT/'experiments/development/event_conditioned_screen_20261019/results'))
os.environ['MPLCONFIGDIR'] = str(OUT/'.cache/matplotlib')
os.environ['NUMBA_CACHE_DIR'] = str(OUT/'.cache/numba')
os.environ.setdefault('SSQ_PARALLEL', '0')
sys.dont_write_bytecode = True
sys.path[:0] = [str(FROZEN), str(A), str(RESOURCE_ROOT)]
import bootstrap
core = bootstrap.activate()

