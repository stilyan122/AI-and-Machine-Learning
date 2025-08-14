"""
Seed utilities for reproducible runs.
Call set_seed(42) at the top of notebooks/scripts.
"""

import os
import random

try:
    import numpy as np
except Exception:  
    np = None

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)       # stable hashing
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
        
    if deterministic:
        os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")