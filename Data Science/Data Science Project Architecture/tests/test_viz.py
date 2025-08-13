import os, tempfile
import numpy as np
import pandas as pd

from src.viz import (
    plot_target_distribution, plot_hist, boxplot_by_target,
    top_corr_with_target, positive_rate_by_binary
)  # :contentReference[oaicite:6]{index=6}

def _tmpfile(suffix=".png"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

def test_plot_target_distribution(df_min):
    save = _tmpfile()
    plot_target_distribution(df_min["Diagnosis"], save=save)
    assert os.path.getsize(save) > 0

def test_plot_hist_and_box(df_min):
    hsave = _tmpfile(); bsave = _tmpfile()
    plot_hist(df_min, "Age", save=hsave, bins=5)
    boxplot_by_target(df_min, "BMI", "Diagnosis", save=bsave)
    assert os.path.getsize(hsave) > 0 and os.path.getsize(bsave) > 0

def test_top_corr_and_pos_rate(df_min):
    csave = _tmpfile(); rsave = _tmpfile()
    s = top_corr_with_target(df_min, "Diagnosis", k=5, save=csave)
    assert 1 <= len(s) <= 5
    pr = positive_rate_by_binary(df_min, ["Smoking","Wheezing","Gender"], "Diagnosis", top=3, save=rsave)
    # function returns only binary-like features; Gender here is binary-coded so it is allowed by your implementation
    assert set(pr.columns) == {"feature","pos_rate"}
    assert len(pr) <= 3
    assert os.path.getsize(csave) > 0 and os.path.getsize(rsave) > 0