import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from src.helpers import scale_continuous  # :contentReference[oaicite:4]{index=4}

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=5, max_size=50))
def test_z_scale_properties(xs):
    s = pd.Series(xs, name="X")
    df = pd.DataFrame({"X": s})
    xz, _ = scale_continuous(df, ["X"], mode="z")
    if np.std(xs) == 0:
        # constant: becomes zeros
        assert (xz["X"] == 0).all()
    else:
        assert abs(xz["X"].mean()) < 1e-6
        assert abs(xz["X"].std(ddof=0) - 1.0) < 1e-5