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
    Order: continuous + small_codes + binaries + target (last); keep leftovers first
    """
    
    continuous = [c for c in continuous_cols if c in df.columns]
    small = [c for c in small_codes if c in df.columns]
    bins = [c for c in binary_cols if c in df.columns and c != target]
    
    core = continuous + small + bins
    
    leftovers = [c for c in df.columns if c not in core + [target]]
    order = leftovers + core + ([target] if target in df.columns else [])
    
    return order

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