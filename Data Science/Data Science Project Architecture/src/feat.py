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
    Function to clear a values where variance is 0
    """
    
    cols = [c for c in df.columns if c not in exclude]
    keep = [c for c in cols if df[c].var() > eps]
    dropped = sorted(set(cols) - set(keep))
    
    return pd.concat([df[exclude], df[keep]], axis=1), dropped

def fe_drop_high_corr(df, exclude, threshold=0.98):
    """
    Function to drop features with high correlation between them
    """
    
    cols = sorted([c for c in df.columns if c not in exclude])
    
    if len(cols) <= 1:
        return df.copy(), []
        
    C = df[cols].corr().abs()
    
    to_drop = set()
    
    for i, a in enumerate(cols):
        if a in to_drop: 
            continue
        for b in cols[i+1:]:
            if b in to_drop:
                continue
            if C.at[a, b] >= threshold:
                to_drop.add(b)
                
    kept = [c for c in cols if c not in to_drop]
    return pd.concat([df[exclude], df[kept]], axis=1), sorted(to_drop)

def fe_enforce_float_and_order(df, target):
    """
    Function to ensure numeric columns and correct order
    """
    
    cols = [c for c in df.columns if c != target]
    out = df.copy()
    
    for c in cols:
        out[c] = out[c].astype(float)
        
    return out[cols + [target]]

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

