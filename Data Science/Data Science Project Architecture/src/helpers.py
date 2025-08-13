"""
General data preparation helpers.

Pure, vectorized utilities for dropping columns, ordering columns by groups,
and scaling continuous features. Functions are DF-in -> DF-out and parameterized.
"""

import numpy as np
import pandas as pd

def drop_and_dedup(df, drop_cols):
    """
    Drop only the columns that exist; remove exact duplicate rows
    """
    
    present = [c for c in drop_cols if c in df.columns]
    out = df.drop(columns=present).drop_duplicates(ignore_index=True)
    return out, present

def build_column_order(df, continuous_cols, small_codes, binary_cols, target):

    """
    Order columns for modeling/reporting:

    leftovers (not in any group) +
    continuous (in GIVEN order) +
    small_codes (in GIVEN order) +
    binary (in GIVEN order) +
    target (last).
    """
    cols = list(df.columns)

    # Build the tail in EXACT given order, include only present cols, avoid dups
    seen = {target}
    core = []
    for c in list(continuous_cols) + list(small_codes) + list(binary_cols):
        if c in cols and c not in seen:
            core.append(c)
            seen.add(c)

    # Leftovers = everything else except target and anything already in core
    leftovers = [c for c in cols if c not in seen]

    # Final order: leftovers + core + target
    return leftovers + core + [target]
def scale_continuous(df, cols, mode="z"):
    """
    Scale only 'cols'.
    mode: 'z' -> (x-mean)/std ; 'minmax' -> (x-min)/(max-min)
    """
    
    out = df.copy()
    use = [c for c in cols if c in out.columns]
    params = {"mode": mode, "cols": use}

    if not use:
        return out, params

    if mode == "minmax":
        mins = out[use].min()
        maxs = out[use].max()
        span = (maxs - mins).replace(0, 1.0)
        out[use] = (out[use] - mins) / span
        params["mins"] = mins.to_dict()
        params["maxs"] = maxs.to_dict()
    else:
        means = out[use].mean()
        stds  = out[use].std(ddof=0).replace(0, 1.0)
        out[use] = (out[use] - means) / stds
        params["means"] = means.to_dict()
        params["stds"]  = stds.to_dict()

    return out, params