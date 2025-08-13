import numpy as np
import pandas as pd

from src.feat import (
    fe_add_composites, fe_add_interactions, fe_one_hot_codes,
    fe_variance_filter, fe_drop_high_corr, fe_enforce_float_and_order,
    fe_build_feature_table
)  # :contentReference[oaicite:5]{index=5}

TARGET = "Diagnosis"

def test_composites(df_min):
    out = fe_add_composites(df_min)
    # Created columns exist
    for c in ["RespiratorySymptomCount","AtopyIndex","ExposureIndex","LungFunctionDiff","LungFunctionMin","SymptomCount"]:
        assert c in out.columns
        
    # Respiratory count equals sum of its flags on a row
    i = 1
    expect = df_min.loc[i, ["Wheezing","ShortnessOfBreath","ChestTightness"]].sum()
    assert out.loc[i, "RespiratorySymptomCount"] == expect

def test_interactions(df_min):
    x = fe_add_interactions(df_min)
    # A binary×binary AND example
    assert "ExerciseInduced__x__Wheezing" in x.columns
    
    # A binary×continuous example (GERD x Age)
    assert "GastroesophagealReflux__x__Age" in x.columns
    
    # Continuous×continuous example
    assert "PollutionExposure__x__PollenExposure" in x.columns
    
    # Value check
    j = 3
    assert x.loc[j, "ExerciseInduced__x__Wheezing"] == float(df_min.loc[j, "ExerciseInduced"] * df_min.loc[j, "Wheezing"])

def test_one_hot_codes(df_min):
    x, new = fe_one_hot_codes(df_min, ["Gender","Ethnicity"], drop_first=True, prefix_sep="=")
    
    # original codes removed
    assert "Gender" not in x.columns and "Ethnicity" not in x.columns
    
    # with drop_first, exactly one dummy for Gender (because it has 2 levels)
    assert any(c.startswith("Gender=") for c in x.columns)
    
    # every row has at most one 1 among Gender dummies
    gd = [c for c in x.columns if c.startswith("Gender=")]
    assert (x[gd].sum(axis=1) <= 1).all()

def test_variance_and_corr_prune(df_min):
    x, dropped_const = fe_variance_filter(df_min, exclude=[TARGET])
    assert "CONST" in dropped_const and "CONST" not in x.columns

    # create exact duplicate of a column to guarantee |r|=1
    x["BMI_dup"] = x["BMI"]
    pruned, dropped_corr = fe_drop_high_corr(x, exclude=[TARGET], threshold=0.9999)
    assert ("BMI_dup" in dropped_corr) ^ ("BMI" in dropped_corr)  # drop one of the pair

def test_float_and_order(df_min):
    x = df_min.copy()
    out = fe_enforce_float_and_order(x, TARGET)
    
    # target last
    assert out.columns[-1] == TARGET
    
    # all predictors are float
    assert all(np.issubdtype(out[c].dtype, np.floating) for c in out.columns[:-1])

def test_build_feature_table_e2e(df_min):
    # Simulate that continuous were already scaled; not required for correctness here
    final, info = fe_build_feature_table(df_min, target=TARGET, code_cols=["Gender","Ethnicity","EducationLevel"], corr_threshold=0.98, drop_first=False)
    
    # sanity: target last, no NaNs, float predictors
    assert final.columns[-1] == TARGET
    assert not final.isna().any().any()
    assert all(np.issubdtype(final[c].dtype, np.number) for c in final.columns[:-1])
    
    # audit info contains expected keys
    for k in ["added_by_composites","added_by_interactions","one_hot_new_cols","dropped_zero_var","dropped_high_corr"]:
        assert k in info