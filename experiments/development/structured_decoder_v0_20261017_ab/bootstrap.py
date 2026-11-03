"""Pin read-only historical model implementations; all writes stay HERE."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = Path('/tmp/rem_dual_head_formal')
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.cache/matplotlib'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE/'.cache/numba'))
os.environ.setdefault('SSQ_PARALLEL', '0')
sys.path[:0] = [str(FROZEN), str(ROOT), str(HERE)]

def activate():
    # Importing the original training module prepends its own matching root.
    import scripts.run_dual_head_interaction as core
    return core
