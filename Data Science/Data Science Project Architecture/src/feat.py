"""
Feature engineering utilities.

Adds composite indices, curated interactions, one-hot encodings, and performs
basic redundancy pruning (zero-variance, high correlation). Outputs float-only tables.
"""

import numpy as np
import pandas as pd

def fe_add_composites(df):
    """
    Function used in feature engineering - add composite columns - columns that compress related signals
    """
    
    out = df.copy()

    resp = [c for c in ["Wheezing", "ShortnessOfBreath", "ChestTightness"] if c in out.columns]
    atopy = [c for c in ["HistoryOfAllergies", "HayFever", "PetAllergy"] if c in out.columns]
    expos = [c for c in ["PollutionExposure", "PollenExposure", "DustExposure"] if c in out.columns]

    if resp:
        out["RespiratorySymptomCount"] = out[resp].sum(axis=1)        # 0..3
    if atopy:
        out["AtopyIndex"] = out[atopy].sum(axis=1)                    # 0..3
    if expos:
        out["ExposureIndex"] = out[expos].mean(axis=1)                # stays on z-ish scale

    if {"LungFunctionFEV1","LungFunctionFVC"}.issubset(out.columns):
        out["LungFunctionDiff"] = out["LungFunctionFEV1"] - out["LungFunctionFVC"]
        out["LungFunctionMin"]  = np.minimum(out["LungFunctionFEV1"], out["LungFunctionFVC"])

    symptoms_all = [
        "Smoking", "PetAllergy", "FamilyHistoryAsthma", "HistoryOfAllergies", "Eczema", "HayFever",
        "GastroesophagealReflux", "Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing",
        "NighttimeSymptoms", "ExerciseInduced"
    ]
    
    use_sym = [c for c in symptoms_all if c in out.columns]
    
    if use_sym:
        out["SymptomCount"] = out[use_sym].sum(axis=1)               

    return out

def fe_add_interactions(df):
    """
    Function to add interaction between different types of columns
    """
    
    out = df.copy()

    # binary x binary (AND)
    for a, b in [
        ("ExerciseInduced", "Wheezing"),
        ("Wheezing", "NighttimeSymptoms"),
        ("ShortnessOfBreath", "ChestTightness"),
        ("HistoryOfAllergies", "HayFever"),
    ]:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = (out[a] * out[b]).astype(float)

    # binary x continuous
    bx = ["GastroesophagealReflux"]
    cont = [
        "PollutionExposure", "PollenExposure", "DustExposure",
        "BMI", "Age", "LungFunctionFEV1", "LungFunctionFVC",
        "PhysicalActivity", "DietQuality", "SleepQuality"
    ]
    
    for b in bx:
        if b in out.columns:
            for c in [x for x in cont if x in out.columns]:
                out[f"{b}__x__{c}"] = out[b] * out[c]

    # continuous x continuous
    for a, b in [
        ("PollutionExposure","PollenExposure"),
        ("LungFunctionFEV1","LungFunctionFVC"),
        ("BMI","PhysicalActivity"),
    ]:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a] * out[b]

    return out

def fe_one_hot_codes(df, cols, drop_first=False, prefix_sep="="):
    """
    Function to add one-hot encoding to a dataset
    """
    
    have = [c for c in cols if c in df.columns]
    if not have:
        return df.copy(), []
        
    dummies = pd.get_dummies(df[have].astype(int), columns=have,
                             drop_first=drop_first, prefix_sep=prefix_sep, dtype=float)
    
    out = pd.concat([df.drop(columns=have), dummies], axis=1)
    new_cols = [c for c in out.columns if c not in df.columns]
    
    return out, new_cols

def fe_variance_filter(df, exclude, eps=1e-12):
    """
    Drop zero-variance features.
    - For numeric columns: use variance.
    - For non-numeric columns: drop if nunique==1 (constant label).
    """
    
    out = df.copy()
    cols = [c for c in out.columns if c not in exclude]

    # split columns by dtype
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(out[c])]
    non_num_cols = [c for c in cols if c not in num_cols]

    # numeric: keep those with variance > eps
    keep_num = [c for c in num_cols if out[c].var() > eps]
    drop_const_num = [c for c in num_cols if c not in keep_num]

    # non-numeric: keep those with more than one unique value
    keep_non = []
    drop_const_non = []
    for c in non_num_cols:
        if out[c].nunique(dropna=False) > 1:
            keep_non.append(c)
        else:
            drop_const_non.append(c)

    kept = keep_num + keep_non
    dropped = sorted(drop_const_num + drop_const_non)

    return pd.concat([out[exclude], out[kept]], axis=1), dropped

def fe_drop_high_corr(df, exclude, threshold=0.98):
    """
    Drop one of any pair of highly correlated NUMERIC features (|r| >= threshold).
    Non-numeric columns are kept as-is.
    """
    
    cols = [c for c in df.columns if c not in exclude]
    cols = sorted(cols)  # deterministic

    # only correlate numeric columns
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    non_num_cols = [c for c in cols if c not in num_cols]

    if len(num_cols) <= 1:
        kept = non_num_cols + num_cols
        return pd.concat([df[exclude], df[kept]], axis=1), []

    C = df[num_cols].corr().abs()
    to_drop = set()
    for i, a in enumerate(num_cols):
        if a in to_drop:
            continue
        for b in num_cols[i+1:]:
            if b in to_drop:
                continue
            if C.loc[a, b] >= threshold:
                to_drop.add(b)

    kept_num = [c for c in num_cols if c not in to_drop]
    kept = non_num_cols + kept_num
    return pd.concat([df[exclude], df[kept]], axis=1), sorted(list(to_drop))

def fe_enforce_float_and_order(df, target):
    """
    Ensure all predictors are float dtype and keep target last.
    Non-numeric predictors are dropped in this final feature table.
    """
    out = df.copy()
    preds = [c for c in out.columns if c != target]

    # keep only numeric predictors
    num_preds = [c for c in preds if pd.api.types.is_numeric_dtype(out[c])]

    # cast numeric predictors to float
    for c in num_preds:
        out[c] = out[c].astype(float)

    # return predictors (float-only) + target last
    return out[num_preds + [target]]

def fe_build_feature_table(df, target="Diagnosis", code_cols=None, corr_threshold=0.98, drop_first=False):
    """
    Function to create feature table
    """
    
    if code_cols is None:
        code_cols = ["Gender","Ethnicity","EducationLevel"]

    pipeline = {"added_by_composites": [], "added_by_interactions": [], "one_hot_new_cols": [],
                "dropped_zero_var": [], "dropped_high_corr": []}

    d0_cols = set(df.columns)

    x = fe_add_composites(df);                 
    pipeline["added_by_composites"] = sorted(set(x.columns) - d0_cols)
    
    d1_cols = set(x.columns)

    x = fe_add_interactions(x);               
    pipeline["added_by_interactions"] = sorted(set(x.columns) - d1_cols)

    x, new_ohe = fe_one_hot_codes(x, code_cols, drop_first=drop_first); 
    pipeline["one_hot_new_cols"] = new_ohe

    x, dropped_const = fe_variance_filter(x, exclude=[target]);         
    pipeline["dropped_zero_var"] = dropped_const

    x, dropped_corr = fe_drop_high_corr(x, exclude=[target], threshold=corr_threshold); 
    pipeline["dropped_high_corr"] = dropped_corr

    x = fe_enforce_float_and_order(x, target)

    return x, pipeline

