import numpy as np
import pandas as pd

from src.helpers import drop_and_dedup, build_column_order, scale_continuous  # :contentReference[oaicite:3]{index=3}

def test_drop_and_dedup(df_min):
    out, dropped = drop_and_dedup(df_min, ["PatientID","DoctorInCharge","MISSING"])
    assert set(dropped) == {"PatientID","DoctorInCharge"}
    
    # one duplicate row removed (PatientID duplicated)
    assert len(out) == 3
    assert "PatientID" not in out.columns and "DoctorInCharge" not in out.columns

def test_scale_continuous_z(df_min):
    x, p = scale_continuous(df_min, ["Age","BMI"], mode="z")
    assert set(p["cols"]) == {"Age","BMI"} and p["mode"] == "z"
    
    # ddof=0 in your impl, so std = 1 exactly for non-constant
    assert abs(x["Age"].mean()) < 1e-9
    assert np.isclose(x["Age"].std(ddof=0), 1.0, atol=1e-9)

def test_scale_continuous_minmax(df_min):
    x, p = scale_continuous(df_min, ["Age","CONST"], mode="minmax")
    assert p["mode"] == "minmax"
    
    # minmax -> range in [0,1]; constant becomes zeros
    assert x["Age"].min() == 0 and x["Age"].max() == 1
    assert (x["CONST"] == 0).all()